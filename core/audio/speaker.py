"""
Dexter TTS Speaker — Converts text to speech using Edge-TTS
and plays audio using pygame for low-latency playback, with fallbacks.
"""
import edge_tts
import asyncio
import io
import math
import os
import struct
import time
import threading
import wave
import tempfile
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
        self._interrupt_cooldown = 0.0  # Prevent rapid re-interrupts

    def stop(self) -> None:
        """Immediately stops TTS playback. Thread-safe."""
        current_time = time.time()
        if current_time < self._interrupt_cooldown:
            return  # Skip if called too soon after last interrupt
        
        self._interrupt_cooldown = current_time + 0.1  # 100ms cooldown
        self._global_cancel.set()
        self._cancel_event.set()
        
        if self._channel is not None:
            try:
                # Forcefully stop the channel
                self._channel.stop()
                logger.info("tts_interrupted_channel_stopped")
            except Exception as e:
                logger.debug("tts_channel_stop_failed", error=str(e))
        
        logger.debug("tts_stop_requested")

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
        audio_file = None

        try:
            synth_start = time.perf_counter()
            audio_bytes = await _synthesize_edge_tts_bytes(text, self.voice)
            metrics.record_latency("tts_synth_ms", (time.perf_counter() - synth_start) * 1000)

            logger.debug("tts_audio_loaded_to_memory", bytes=len(audio_bytes))

            play_start = time.perf_counter()
            await _play_audio_bytes(audio_bytes, cancel_event, self, track_channel=True)
            metrics.record_latency("tts_play_ms", (time.perf_counter() - play_start) * 1000)

        except Exception as e:
            logger.error("tts_synthesis_failed", error=str(e), exc_info=True)
        finally:
            if audio_file:
                _safe_delete(audio_file)
            self._channel = None

    async def play_chime(self) -> None:
        try:
            audio_bytes = _load_chime_bytes()
            await _play_audio_bytes(audio_bytes, threading.Event(), self, track_channel=False)
        except Exception as e:
            logger.debug("tts_chime_failed", error=str(e))


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


async def _play_audio_bytes(
    audio_bytes: bytes,
    cancel_event: threading.Event,
    manager: TTSManager,
    track_channel: bool = True,
) -> None:
    _ensure_pygame_ready()
    import pygame

    try:
        sound = pygame.mixer.Sound(buffer=audio_bytes)
    except Exception:
        sound = pygame.mixer.Sound(io.BytesIO(audio_bytes))
    logger.debug("tts_playback_started", duration_estimate_ms=int(sound.get_length() * 1000))
    channel = sound.play()
    if channel is None:
        raise RuntimeError("pygame_failed_to_start_playback")

    if track_channel:
        manager._channel = channel
    try:
        while channel.get_busy():
            if cancel_event.is_set() or manager._global_cancel.is_set():
                channel.stop()
                break
            await asyncio.sleep(0.05)
        logger.info("tts_playback_complete")
    finally:
        if track_channel and manager._channel is channel:
            manager._channel = None


async def _synthesize_edge_tts_bytes(text: str, voice: str) -> bytes:
    """Prefer in-memory synthesis. Fall back to a temp file only if streaming fails."""
    communicate = edge_tts.Communicate(text, voice)
    audio_chunks: list[bytes] = []

    try:
        async for chunk in communicate.stream():
            if not isinstance(chunk, dict):
                continue
            if chunk.get("type") == "audio":
                data = chunk.get("data")
                if data:
                    audio_chunks.append(data)
        if audio_chunks:
            return b"".join(audio_chunks)
    except Exception as e:
        logger.debug("tts_stream_synthesis_failed", error=str(e), exc_info=True)

    temp_handle = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    temp_handle.close()
    try:
        await communicate.save(temp_handle.name)
        with open(temp_handle.name, "rb") as audio_handle:
            return audio_handle.read()
    finally:
        _safe_delete(temp_handle.name)


def _load_chime_bytes() -> bytes:
    chime_path = _get_chime_path()
    if os.path.exists(chime_path):
        with open(chime_path, "rb") as handle:
            return handle.read()

    os.makedirs(os.path.dirname(chime_path), exist_ok=True)
    audio_bytes = _generate_chime_wav_bytes()
    with open(chime_path, "wb") as handle:
        handle.write(audio_bytes)
    logger.info("activation_chime_generated", path=chime_path)
    return audio_bytes


def _get_chime_path() -> str:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(base_dir, "assets", "sounds", "activate.mp3")


def _generate_chime_wav_bytes(
    duration_s: float = 0.3,
    frequency: float = 880.0,
    sample_rate: int = 48000,
    volume: float = 0.25,
) -> bytes:
    frames = int(duration_s * sample_rate)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        for i in range(frames):
            sample = volume * math.sin(2 * math.pi * frequency * (i / sample_rate))
            wav.writeframes(struct.pack("<h", int(sample * 32767)))
    return buffer.getvalue()


def _safe_delete(filepath: str):
    """Safely delete a file, ignoring errors if it's locked."""
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
    except OSError as e:
        logger.debug("tts_temp_file_delete_failed", path=filepath, error=str(e))
