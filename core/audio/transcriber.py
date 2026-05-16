import os
import threading
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
        model_size: str = "small.en",
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
            model_size: Whisper model size (tiny.en, base.en, small.en, medium.en)
            beam_size: Beam search width. Lower = faster, Higher = more accurate.
                       1 = greedy (fastest), 5 = thorough (default whisper).
        """
        self.beam_size = int(beam_size)
        self.best_of = int(best_of)
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
        self.model_size = model_size
        self._model: WhisperModel | None = None
        self._model_lock = threading.Lock()
        logger.info(
            "transcriber_loading",
            model_size=model_size,
            beam_size=self.beam_size,
            best_of=self.best_of,
            temperature=self.temperature,
        )

    def warm_up(self) -> None:
        """Preload the Whisper model so the first transcription doesn't time out."""
        try:
            logger.info("transcriber_warmup_started", model_size=self.model_size)
            self._ensure_model()
            logger.info("transcriber_warmup_completed", model_size=self.model_size)
        except Exception as e:
            logger.warning("transcriber_warmup_failed", error=str(e))

    def _ensure_model(self) -> WhisperModel:
        if self._model is not None:
            return self._model

        with self._model_lock:
            if self._model is not None:
                return self._model

            try:
                # Use float16 for RTX GPUs to maximize speed and save VRAM
                self._model = WhisperModel(self.model_size, device="cuda", compute_type="float16")
                logger.info("transcriber_device_ready", device="cuda", compute_type="float16")
            except Exception as e:
                logger.warning("transcriber_gpu_fallback", error=str(e), exc_info=True)
                self._model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
                logger.info("transcriber_device_ready", device="cpu", compute_type="int8")

        return self._model

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
        Takes an audio filepath and returns the transcribed text.
        Uses the configured beam_size for speed/accuracy tradeoff.
        """
        if not os.path.exists(audio_file):
            logger.error("transcriber_audio_missing", path=audio_file)
            return ""

        normalized_path = self._normalize_audio(audio_file)
        logger.debug("transcriber_run_started", path=audio_file, beam_size=self.beam_size)
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
        vad_params = {
            "threshold": 0.25,
            "min_speech_duration_ms": 100,
            "min_silence_duration_ms": 900,
            "speech_pad_ms": 400,
        }
        segments, info = model.transcribe(
            normalized_path,
            beam_size=self.beam_size,
            best_of=5,
            temperature=[0.0, 0.2, 0.4],
            condition_on_previous_text=False,
            initial_prompt=self._initial_prompt,
            vad_filter=True,
            vad_parameters=vad_params,
        )

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
        text = " ".join(collected)
        return text.strip()
