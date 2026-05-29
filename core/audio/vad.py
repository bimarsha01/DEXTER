import queue
import os
import tempfile
import uuid
import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
import time
from utils.logger import get_logger
from utils.config import get_config

logger = get_logger("vad")

class VADListener:
    def __init__(
        self,
        sample_rate=16000,
        chunk_size=512,
        vad_threshold: float | None = None,
        min_speech_duration_ms: int | None = None,
        min_silence_duration_ms: int | None = None,
        min_utterance_duration_ms: int | None = None,
        speech_pad_ms: int | None = None,
        max_speech_duration_s: int | None = None,
    ):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.q = queue.Queue()
        self._ambient_rms = 0.0
        # Timestamp until which VAD should ignore input (seconds since epoch)
        self._ignore_until = 0.0
        self._clap_active = False
        self._clap_start = 0.0
        self._last_clap_time = 0.0

        cfg = get_config()
        audio_cfg = getattr(cfg, "audio_settings", None)
        self.vad_threshold = float(
            vad_threshold if vad_threshold is not None else getattr(audio_cfg, "vad_threshold", 0.3)
        )
        self.min_speech_duration_ms = int(
            min_speech_duration_ms if min_speech_duration_ms is not None else getattr(audio_cfg, "min_speech_duration_ms", 100)
        )
        self.min_silence_duration_ms = int(
            min_silence_duration_ms if min_silence_duration_ms is not None else getattr(audio_cfg, "min_silence_duration_ms", 1100)
        )
        self.min_utterance_duration_ms = int(
            min_utterance_duration_ms if min_utterance_duration_ms is not None else getattr(audio_cfg, "min_utterance_duration_ms", 350)
        )
        self.speech_pad_ms = int(
            speech_pad_ms if speech_pad_ms is not None else getattr(audio_cfg, "speech_pad_ms", 400)
        )
        self.max_speech_duration_s = int(
            max_speech_duration_s if max_speech_duration_s is not None else getattr(audio_cfg, "max_speech_duration_s", 30)
        )
        
        logger.info("vad_initializing")
        # Load Silero VAD from PyTorch Hub
        self.model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )
        (self.get_speech_timestamps, 
         self.save_audio,
         self.read_audio,
         self.VADIterator,
         self.collect_chunks) = utils
        
        logger.info("vad_initialized")

    def _audio_callback(self, indata, frames, time_info, status):
        """This is called for each audio block by sounddevice."""
        if status:
            logger.warning("audio_input_status", detail=str(status))
        # Put the raw numpy array chunk into our queue
        self.q.put(indata.copy())

    def _resolve_output_path(self, output_file: str | None) -> str:
        if output_file:
            return output_file
        filename = f"dexter_mic_{uuid.uuid4().hex}.wav"
        return os.path.join(tempfile.gettempdir(), filename)

    def listen(
        self,
        output_file=None,
        silence_threshold: float | None = None,
        on_speech_start=None,
        on_clap=None,
        clap_sensitivity: float = 3.0,
        clap_max_ms: int = 150,
        clap_pair_window: float = 1.5,
    ):
        """
        Listens to the microphone continuously. 
        Only records when VAD detects a human voice.
        Stops recording after `silence_threshold` seconds of silence.
        
        When interrupted (on_speech_start called while TTS is playing):
        - Uses shorter silence_threshold (0.8s) to respond quickly
        - Drops very short utterances to avoid accidental triggers
        """
        recording = []
        is_speaking = False
        silence_start_time = None
        speech_start_time = None
        speech_ms = 0.0
        output_file = self._resolve_output_path(output_file)
        was_interrupted = False
        pre_buffer = []
        pre_buffer_samples = 0
        pad_samples = int(self.sample_rate * (self.speech_pad_ms / 1000.0))
        base_silence_s = (
            float(silence_threshold)
            if silence_threshold is not None
            else float(self.min_silence_duration_ms) / 1000.0
        )
        
        logger.info("vad_listening_started")

        while not self.q.empty():
            try:
                self.q.get_nowait()
            except Exception as e:
                logger.debug("vad_queue_drain_stopped", error=str(e))
                break
        
        try:
            with sd.InputStream(samplerate=self.sample_rate, 
                                channels=1, 
                                callback=self._audio_callback,
                                blocksize=self.chunk_size):
                while True:
                    chunk = self.q.get()
                    # If suppression is active (e.g., TTS playback), ignore incoming chunks
                    if time.time() < getattr(self, "_ignore_until", 0.0):
                        # Drain/skip while suppressed
                        continue
                    if on_clap is not None:
                        rms = float(np.sqrt(np.mean(chunk**2))) if chunk.size else 0.0
                        self._detect_clap(
                            rms,
                            clap_sensitivity=clap_sensitivity,
                            clap_max_seconds=clap_max_ms / 1000.0,
                            clap_pair_window=clap_pair_window,
                            on_clap=on_clap,
                        )
                    
                    # Convert chunk to a PyTorch tensor (required by Silero)
                    tensor_chunk = torch.from_numpy(chunk).float().flatten()
                    
                    # Pad chunk if it's too short
                    if len(tensor_chunk) < 512:
                        continue
                        
                    # Calculate probability that this chunk is human speech
                    speech_prob = self.model(tensor_chunk, self.sample_rate).item()
                    chunk_duration_ms = (len(tensor_chunk) / self.sample_rate) * 1000.0

                    if not is_speaking:
                        pre_buffer.append(chunk)
                        pre_buffer_samples += len(chunk)
                        while pre_buffer_samples > pad_samples and pre_buffer:
                            removed = pre_buffer.pop(0)
                            pre_buffer_samples -= len(removed)

                    if speech_prob > self.vad_threshold:
                        speech_ms += chunk_duration_ms
                        if not is_speaking and speech_ms >= self.min_speech_duration_ms:
                            logger.debug("vad_speech_started")
                            is_speaking = True
                            speech_start_time = time.time()
                            was_interrupted = True  # Mark that we detected speech during listening
                            if pre_buffer:
                                recording.extend(pre_buffer)
                                pre_buffer = []
                                pre_buffer_samples = 0
                            if on_speech_start:
                                try:
                                    on_speech_start()
                                except Exception as e:
                                    logger.warning(
                                        "vad_speech_start_callback_failed",
                                        error=str(e),
                                        exc_info=True,
                                    )
                        if is_speaking:
                            recording.append(chunk)
                        silence_start_time = None  # Reset silence timer
                    else:
                        speech_ms = 0.0
                        if is_speaking:
                            recording.append(chunk)
                            if silence_start_time is None:
                                silence_start_time = time.time()
                            
                            # If interrupted (speech detected during playback): shorter timeout
                            # Otherwise: normal timeout
                            effective_threshold = min(0.8, base_silence_s) if was_interrupted else base_silence_s
                            
                            # If they have been silent for X seconds, stop recording
                            if time.time() - silence_start_time > effective_threshold:
                                logger.debug("vad_speech_ended", was_interrupted=was_interrupted, silence_duration=(time.time() - silence_start_time))
                                break

                    if is_speaking and speech_start_time is not None:
                        if time.time() - speech_start_time > float(self.max_speech_duration_s):
                            logger.info("vad_max_speech_duration_reached", seconds=self.max_speech_duration_s)
                            break
        except Exception as e:
            logger.error("vad_microphone_error", error=str(e), exc_info=True)
            return None

        # If we captured audio, save it to a file
        if recording:
            if speech_start_time is not None:
                utterance_duration_ms = (time.time() - speech_start_time) * 1000.0
                if utterance_duration_ms < float(self.min_utterance_duration_ms):
                    logger.info(
                        "vad_utterance_too_short",
                        duration_ms=round(utterance_duration_ms, 2),
                        min_duration_ms=self.min_utterance_duration_ms,
                    )
                    return None
            audio_data = np.concatenate(recording, axis=0)
            sf.write(output_file, audio_data, self.sample_rate)
            return output_file
            
        return None

    def _detect_clap(
        self,
        rms: float,
        clap_sensitivity: float,
        clap_max_seconds: float,
        clap_pair_window: float,
        on_clap,
    ) -> None:
        now = time.time()
        if self._ambient_rms <= 0.0:
            self._ambient_rms = max(rms, 1e-6)
            return

        threshold = max(self._ambient_rms * clap_sensitivity, 1e-6)
        if rms > threshold:
            if not self._clap_active:
                self._clap_active = True
                self._clap_start = now
            return

        if self._clap_active:
            duration = now - self._clap_start
            self._clap_active = False
            if duration <= clap_max_seconds:
                if self._last_clap_time and (now - self._last_clap_time) <= clap_pair_window:
                    self._last_clap_time = 0.0
                    logger.info("clap_activation_detected")
                    try:
                        on_clap()
                    except Exception as e:
                        logger.warning("clap_callback_failed", error=str(e), exc_info=True)
                else:
                    self._last_clap_time = now

        self._ambient_rms = (self._ambient_rms * 0.9) + (rms * 0.1)

    def suppress_for(self, seconds: float) -> None:
        """Temporarily ignore VAD input for `seconds` seconds.

        Useful to avoid picking up TTS playback or other known audio sources.
        """
        try:
            self._ignore_until = time.time() + max(0.0, float(seconds))
        except Exception:
            self._ignore_until = time.time()
