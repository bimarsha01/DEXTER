"""
Dexter TTS Speaker — Converts text to speech using Edge-TTS
and plays audio using pygame for low-latency playback, with fallbacks.
"""
import edge_tts
import asyncio
import io
import os
import time
import uuid
import threading
from utils.logger import get_logger
from utils.metrics import metrics

logger = get_logger("speaker")

_PYGAME_READY = False
_PYGAME_LOCK = threading.Lock()


class TTSManager:
    def __init__(self, voice: str = "en-GB-RyanNeural"):
        self.voice = voice
        self._cancel_event = threading.Event()
        self._global_cancel = threading.Event()
        self._channel = None
        self._lock = asyncio.Lock()

    def stop(self) -> None:
        self._global_cancel.set()
        self._cancel_event.set()
        if self._channel is not None:
            try:
                self._channel.stop()
            except Exception as e:
                logger.debug("tts_channel_stop_failed", error=str(e))

    async def speak(self, text: str, interrupt: bool = True) -> None:
        if not text or not text.strip():
            return

        async with self._lock:
            if interrupt:
                self.stop()
                self._global_cancel.clear()
            if self._global_cancel.is_set():
                return
            self._cancel_event = threading.Event()
            cancel_event = self._cancel_event

        preview = text[:80] + "..." if len(text) > 80 else text
        logger.info("tts_speak_started", text_preview=preview, text_length=len(text))
        audio_file = os.path.join(os.path.dirname(__file__), "..", "..", f"temp_response_{uuid.uuid4().hex}.mp3")
        audio_file = os.path.abspath(audio_file)

        try:
            synth_start = time.perf_counter()
            communicate = edge_tts.Communicate(text, self.voice)
            await communicate.save(audio_file)
            metrics.record_latency("tts_synth_ms", (time.perf_counter() - synth_start) * 1000)

            with open(audio_file, "rb") as audio_handle:
                audio_bytes = audio_handle.read()
            logger.debug("tts_audio_loaded_to_memory", bytes=len(audio_bytes))
            _safe_delete(audio_file)

            play_start = time.perf_counter()
            await _play_audio_bytes(audio_bytes, cancel_event, self)
            metrics.record_latency("tts_play_ms", (time.perf_counter() - play_start) * 1000)

        except Exception as e:
            logger.error("tts_synthesis_failed", error=str(e), exc_info=True)
        finally:
            _safe_delete(audio_file)
            self._channel = None


async def speak(text: str, voice: str = "en-GB-RyanNeural"):
    """Backward-compatible helper for simple speech output."""
    manager = TTSManager(voice=voice)
    await manager.speak(text)


def _ensure_pygame_ready() -> None:
    global _PYGAME_READY
    if _PYGAME_READY:
        return
    with _PYGAME_LOCK:
        if _PYGAME_READY:
            return
        import pygame

        pygame.mixer.pre_init(frequency=48000, size=-16, channels=1, buffer=512)
        pygame.mixer.init()
        _PYGAME_READY = True


async def _play_audio_bytes(audio_bytes: bytes, cancel_event: threading.Event, manager: TTSManager) -> None:
    _ensure_pygame_ready()
    import pygame

    sound = pygame.mixer.Sound(io.BytesIO(audio_bytes))
    logger.debug("tts_playback_started", duration_estimate_ms=int(sound.get_length() * 1000))
    channel = sound.play()
    if channel is None:
        raise RuntimeError("pygame_failed_to_start_playback")

    manager._channel = channel
    try:
        while channel.get_busy():
            if cancel_event.is_set() or manager._global_cancel.is_set():
                channel.stop()
                break
            await asyncio.sleep(0.05)
        logger.info("tts_playback_complete")
    finally:
        if manager._channel is channel:
            manager._channel = None


def _safe_delete(filepath: str):
    """Safely delete a file, ignoring errors if it's locked."""
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
    except OSError as e:
        logger.debug("tts_temp_file_delete_failed", path=filepath, error=str(e))
