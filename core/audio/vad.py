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

logger = get_logger("vad")

class VADListener:
    def __init__(self, sample_rate=16000, chunk_size=512):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.q = queue.Queue()
        
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

    def listen(self, output_file=None, silence_threshold=1.5, on_speech_start=None):
        """
        Listens to the microphone continuously. 
        Only records when VAD detects a human voice.
        Stops recording after `silence_threshold` seconds of silence.
        """
        recording = []
        is_speaking = False
        silence_start_time = None
        output_file = self._resolve_output_path(output_file)
        
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
                    
                    # Convert chunk to a PyTorch tensor (required by Silero)
                    tensor_chunk = torch.from_numpy(chunk).float().flatten()
                    
                    # Pad chunk if it's too short
                    if len(tensor_chunk) < 512:
                        continue
                        
                    # Calculate probability that this chunk is human speech
                    speech_prob = self.model(tensor_chunk, self.sample_rate).item()

                    if speech_prob > 0.5:
                        if not is_speaking:
                            logger.debug("vad_speech_started")
                            is_speaking = True
                            if on_speech_start:
                                try:
                                    on_speech_start()
                                except Exception as e:
                                    logger.warning(
                                        "vad_speech_start_callback_failed",
                                        error=str(e),
                                        exc_info=True,
                                    )
                        recording.append(chunk)
                        silence_start_time = None  # Reset silence timer
                    else:
                        if is_speaking:
                            recording.append(chunk)
                            if silence_start_time is None:
                                silence_start_time = time.time()
                            
                            # If they have been silent for X seconds, stop recording
                            if time.time() - silence_start_time > silence_threshold:
                                logger.debug("vad_speech_ended")
                                break
        except Exception as e:
            logger.error("vad_microphone_error", error=str(e), exc_info=True)
            return None

        # If we captured audio, save it to a file
        if recording:
            audio_data = np.concatenate(recording, axis=0)
            sf.write(output_file, audio_data, self.sample_rate)
            return output_file
            
        return None
