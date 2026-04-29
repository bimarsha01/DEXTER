from faster_whisper import WhisperModel
from utils.logger import logger
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
        logger.info(f"Loading faster-whisper ({model_size}) | beam_size={beam_size}...")
        
        try:
            # Use float16 for RTX GPUs to maximize speed and save VRAM
            self.model = WhisperModel(model_size, device="cuda", compute_type="float16")
            logger.info(f"Transcriber loaded on GPU (CUDA) ✓")
        except Exception as e:
            logger.warning(f"GPU loading failed: {e}. Falling back to CPU.")
            self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
            logger.info("Transcriber loaded on CPU (int8) ✓")

    def transcribe(self, audio_file: str, on_partial=None) -> str:
        """
        Takes an audio filepath and returns the transcribed text.
        Uses the configured beam_size for speed/accuracy tradeoff.
        """
        if not os.path.exists(audio_file):
            logger.error(f"Audio file not found: {audio_file}")
            return ""

        logger.debug(f"Transcribing {audio_file} (beam={self.beam_size})...")
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
                except Exception:
                    pass
        text = " ".join(collected)
        return text.strip()
