from faster_whisper import WhisperModel
from utils.logger import get_logger

logger = get_logger("transcriber")
import os


class DexterTranscriber:
    def __init__(self, model_size="small.en", beam_size=1):
        """
        Initialize Whisper speech-to-text model.
        
        Args:
            model_size: Whisper model size (tiny.en, base.en, small.en, medium.en)
            beam_size: Beam search width. Lower = faster, Higher = more accurate.
                       1 = greedy (fastest), 5 = thorough (default whisper).
        """
        self.beam_size = beam_size
        logger.info("transcriber_loading", model_size=model_size, beam_size=beam_size)
        
        try:
            # Use float16 for RTX GPUs to maximize speed and save VRAM
            self.model = WhisperModel(model_size, device="cuda", compute_type="float16")
            logger.info("transcriber_device_ready", device="cuda", compute_type="float16")
        except Exception as e:
            logger.warning("transcriber_gpu_fallback", error=str(e), exc_info=True)
            self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
            logger.info("transcriber_device_ready", device="cpu", compute_type="int8")

    def transcribe(self, audio_file: str, on_partial=None) -> str:
        """
        Takes an audio filepath and returns the transcribed text.
        Uses the configured beam_size for speed/accuracy tradeoff.
        """
        if not os.path.exists(audio_file):
            logger.error("transcriber_audio_missing", path=audio_file)
            return ""

        logger.debug("transcriber_run_started", path=audio_file, beam_size=self.beam_size)
        segments, info = self.model.transcribe(
            audio_file,
            beam_size=self.beam_size,
            vad_filter=True  # Skip silence segments for speed
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
