import inspect
import os
import threading
import time
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from utils.logger import get_logger
from utils.vocabulary import build_whisper_vocabulary
from utils.config import get_config

logger = get_logger("transcriber")


DEFAULT_INITIAL_PROMPT = (
    "Common Windows assistant commands and app names: open, close, start, launch, "
    "play, watch, search, find, what is, what's, weather, forecast, time, date, "
    "take screenshot, screen capture, read clipboard, copy to clipboard, "
    "Chrome, Google Chrome, Edge, Microsoft Edge, Firefox, Brave, VS Code, Visual Studio Code, "
    "Spotify, Discord, Notepad, Calculator, Settings, File Explorer, PowerShell, Command Prompt, "
    "Windows Terminal, Explorer, Outlook, Word, Excel, PowerPoint."
)

DEFAULT_WAKE_PROMPT = "Dexter,"


class DexterTranscriber:
    def __init__(
        self,
        model_size: str | None = None,
        beam_size: int = 5,
        best_of: int = 5,
        temperature: float = 0.0,
        patience: float = 1.0,
        log_prob_threshold: float = -1.0,
        no_speech_threshold: float = 0.6,
        condition_on_previous_text: bool = False,
        initial_prompt: str = "",
    ):
        """
        Initialize Whisper speech-to-text model.
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large-v2)
            beam_size: Beam search width. Lower = faster, Higher = more accurate.
                       1 = greedy (fastest), 5 = thorough (default whisper).
        """
        self.beam_size = int(beam_size)
        self.best_of = int(best_of)
        # Handle both single float and list of floats
        if isinstance(temperature, list):
            self.temperature = temperature
        elif temperature is None:
            self.temperature = 0.0
        else:
            self.temperature = float(temperature)
        self.patience = float(patience)
        self.log_prob_threshold = float(log_prob_threshold)
        self.no_speech_threshold = float(no_speech_threshold)
        self.condition_on_previous_text = bool(condition_on_previous_text)
        # Build a dynamic initial prompt from the user's workspace where possible.
        if initial_prompt and initial_prompt.strip():
            self.initial_prompt = initial_prompt.strip()
        else:
            try:
                cfg = get_config()
                vocab_prompt = build_whisper_vocabulary(cfg)
                # Fall back to configured STT initial prompt or default
                if vocab_prompt and len(vocab_prompt.strip()) > 10:
                    self.initial_prompt = vocab_prompt.strip()
                else:
                    self.initial_prompt = self._build_initial_prompt(cfg)
            except Exception:
                self.initial_prompt = DEFAULT_INITIAL_PROMPT
        
        self._initial_prompt = getattr(self, 'initial_prompt', DEFAULT_INITIAL_PROMPT)
        cfg = get_config()
        self.model_size = (model_size or cfg.hardware.whisper_model or "tiny").strip()
        self._model: WhisperModel | None = None
        self._model_lock = threading.Lock()
        self._batch_size_supported: bool | None = None
        logger.info(
            "transcriber_loading",
            model_size=model_size,
            beam_size=self.beam_size,
            best_of=self.best_of,
            temperature=self.temperature,
        )
        self._ensure_model()

    def warm_up(self) -> None:
        """Preload the Whisper model so the first transcription doesn't time out."""
        try:
            logger.info("transcriber_warmup_started", model_size=self.model_size)
            self._ensure_model()
            logger.info("transcriber_warmup_completed", model_size=self.model_size)
        except Exception as e:
            logger.warning("transcriber_warmup_failed", error=str(e))

    def _ensure_model(self) -> WhisperModel:
        """Load Whisper model with GPU/CPU fallback while keeping compute types stable."""
        if self._model is not None:
            return self._model

        with self._model_lock:
            if self._model is not None:
                return self._model

            cfg = get_config()
            hardware = cfg.hardware
            device = (hardware.device or "cpu").strip().lower()
            compute_type = (hardware.whisper_compute_type or "float32").strip().lower()
            try:
                logger.info("whisper_model_downloading", model=self.model_size)
                self._model = WhisperModel(self.model_size, device=device, compute_type=compute_type)
                logger.info(f"Whisper loaded: {self.model_size} on {device} ({compute_type})")
                if device == "cuda":
                    try:
                        _dummy_audio = np.zeros(16000, dtype=np.float32)
                        _segments, _info = self._model.transcribe(_dummy_audio, beam_size=1)
                        list(_segments)
                        logger.info("Whisper CUDA validation passed")
                        del _dummy_audio, _segments, _info
                    except Exception as e:
                        logger.error(f"Whisper CUDA validation failed: {e} — falling back to CPU")
                        self._model = WhisperModel(
                            self.model_size,
                            device="cpu",
                            compute_type="float32"
                        )
                        hardware.device = "cpu"
                        hardware.whisper_compute_type = "float32"
                        hardware.embedding_device = "cpu"
                        logger.warning("Whisper reloaded on CPU — DEXTER running in CPU mode for this session")
            except Exception as e:
                logger.warning("transcriber_model_load_failed", device=device, compute_type=compute_type, error=str(e), exc_info=True)
                fallback_device = "cpu"
                fallback_compute_type = "float32"
                self._model = WhisperModel(self.model_size, device=fallback_device, compute_type=fallback_compute_type)
                hardware.device = fallback_device
                hardware.whisper_compute_type = fallback_compute_type
                hardware.embedding_device = fallback_device
                logger.info(f"Whisper loaded: {self.model_size} on {fallback_device} ({fallback_compute_type})")

        return self._model

    def _supports_batch_size(self, model: WhisperModel) -> bool:
        """Return True when the Whisper backend supports the batch_size argument."""
        if self._batch_size_supported is not None:
            return self._batch_size_supported
        try:
            params = inspect.signature(model.transcribe).parameters
            self._batch_size_supported = "batch_size" in params
        except Exception:
            self._batch_size_supported = False
        return self._batch_size_supported

    def _build_initial_prompt(self, config) -> str:
        from pathlib import Path
        import os
        
        base = [
            "Dexter", "Chrome", "Spotify", 
            "YouTube", "UserAuth", "office reporting",
            "web practical", "Bimarsha", "Kathmandu",
            "Spring Boot", "Java", "Python",
            "theatre", "booking", "payment",
            "screenshot", "volume", "weather"
        ]
        
        terms = set(base)
        
        roots = getattr(
            config, 'rag', None
        )
        personal_roots = getattr(
            roots, 'personal_roots', []
        ) if roots else []
        
        for root_str in personal_roots[:3]:
            root = Path(
                os.path.expandvars(
                    root_str.replace(
                        '%USERPROFILE%', 
                        str(Path.home())
                    )
                )
            )
            if root.exists():
                try:
                    for item in root.iterdir():
                        if item.is_dir():
                            name = item.name.replace(
                                '-', ' '
                            ).replace('_', ' ')
                            terms.add(name)
                except PermissionError:
                    pass
        
        return ", ".join(sorted(terms)[:60])

    def _normalize_audio(self, path: str) -> str:
        try:
            data, sr = sf.read(path)
            if getattr(data, "size", 0) == 0:
                return path
            if len(getattr(data, "shape", [])) > 1:
                data = np.mean(data, axis=1)

            rms = np.sqrt(np.mean(data**2))
            if rms < 0.01 and rms > 0:
                data = np.clip(
                    data * (0.1 / rms), -1.0, 1.0
                )
                norm_path = path.replace(
                    '.wav', '_norm.wav'
                )
                sf.write(norm_path, data, sr)
                logger.debug("audio_normalized",
                             original_rms=float(rms))
                return norm_path
            return path
        except Exception as e:
            logger.warning("audio_norm_failed",
                           error=str(e))
            return path

    def transcribe(self, audio_file: str, on_partial=None) -> str:
        """
        Transcribe an audio file with Whisper and return the final text.

        Logs audio duration, transcription duration, and transcript stats to
        make long-utterance failures easier to diagnose.
        """
        if not os.path.exists(audio_file):
            logger.error("transcriber_audio_missing", path=audio_file)
            return ""

        normalized_path = self._normalize_audio(audio_file)
        audio_duration_ms = None
        try:
            info = sf.info(normalized_path)
            if info.frames and info.samplerate:
                audio_duration_ms = (info.frames / info.samplerate) * 1000.0
        except Exception as exc:
            logger.debug("transcribe_audio_info_failed", error=str(exc))

        logger.info(
            "transcribe_started",
            path=audio_file,
            beam_size=self.beam_size,
            audio_duration_ms=round(audio_duration_ms, 2) if audio_duration_ms is not None else None,
        )
        model = self._ensure_model()
        prompt = DEFAULT_WAKE_PROMPT
        extra_prompt = (self.initial_prompt or "").strip()
        if extra_prompt:
            if extra_prompt.lower().startswith("dexter"):
                prompt = extra_prompt
            else:
                prompt = f"{DEFAULT_WAKE_PROMPT} {extra_prompt}"
        cfg = get_config()
        audio_cfg = getattr(cfg, "audio_settings", None)
        whisper_batch_size = int(getattr(audio_cfg, "whisper_batch_size", 16) or 16)
        vad_params = {
            "threshold": 0.25,
            "min_speech_duration_ms": 100,
            "min_silence_duration_ms": 900,
            "speech_pad_ms": 400,
        }
        transcribe_kwargs = {
            "beam_size": self.beam_size,
            "best_of": self.best_of,
            "temperature": self.temperature,
            "condition_on_previous_text": self.condition_on_previous_text,
            "initial_prompt": self._initial_prompt,
            "vad_filter": True,
            "vad_parameters": vad_params,
        }

        if whisper_batch_size > 0 and self._supports_batch_size(model):
            transcribe_kwargs["batch_size"] = whisper_batch_size

        transcribe_start = time.perf_counter()
        try:
            segments, info = model.transcribe(normalized_path, **transcribe_kwargs)
        except TypeError as exc:
            if "batch_size" not in str(exc):
                raise
            self._batch_size_supported = False
            logger.warning("transcribe_batch_size_unsupported", error=str(exc))
            transcribe_kwargs.pop("batch_size", None)
            segments, info = model.transcribe(normalized_path, **transcribe_kwargs)
        except Exception as exc:
            logger.error(
                "transcribe_failed",
                error=str(exc),
                audio_duration_ms=round(audio_duration_ms, 2) if audio_duration_ms is not None else None,
            )
            raise

        # Join all spoken segments
        collected = []
        for segment in segments:
            collected.append(segment.text)
            if on_partial:
                try:
                    on_partial(" ".join(collected).strip())
                except Exception as e:
                    logger.warning(
                        "transcriber_partial_callback_failed",
                        error=str(e),
                        exc_info=True,
                    )
        text = " ".join(collected).strip()
        duration_ms = (time.perf_counter() - transcribe_start) * 1000.0
        transcript_length = len(text)

        privacy_cfg = getattr(get_config(), "privacy", None)
        include_transcript = bool(getattr(privacy_cfg, "debug_log_transcripts", False))
        logger.info(
            "transcribe_completed",
            transcribe_duration_ms=round(duration_ms, 2),
            audio_duration_ms=round(audio_duration_ms, 2) if audio_duration_ms is not None else None,
            transcript_length=transcript_length,
            transcript=text if include_transcript else None,
            transcript_preview=text[:200] if text and not include_transcript else None,
        )
        return text
