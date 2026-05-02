"""
Dexter TTS Speaker — Converts text to speech using Edge-TTS
and plays audio using pygame for low-latency playback, with fallbacks.
"""
import edge_tts
import asyncio
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
        self._current_process = None
        self._lock = asyncio.Lock()

    def stop(self) -> None:
        self._global_cancel.set()
        self._cancel_event.set()
        if self._current_process and self._current_process.returncode is None:
            try:
                self._current_process.terminate()
            except Exception as e:
                logger.debug("tts_process_terminate_failed", error=str(e))

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

            play_start = time.perf_counter()
            await _play_audio_windows(audio_file, cancel_event, self)
            metrics.record_latency("tts_play_ms", (time.perf_counter() - play_start) * 1000)

        except Exception as e:
            logger.error("tts_synthesis_failed", error=str(e), exc_info=True)
        finally:
            _safe_delete(audio_file)


async def speak(text: str, voice: str = "en-GB-RyanNeural"):
    """Backward-compatible helper for simple speech output."""
    manager = TTSManager(voice=voice)
    await manager.speak(text)


async def _play_audio_windows(filepath: str, cancel_event: threading.Event, manager: TTSManager):
    """
    Play an audio file using Windows' built-in capabilities.
    Tries multiple approaches for maximum compatibility.
    """
    # Approach 1: pygame for low-latency playback
    try:
        await asyncio.to_thread(_play_audio_pygame, filepath, cancel_event, manager)
        return
    except ImportError:
        logger.debug("tts_playback_fallback", stage="pygame", reason="not_installed")
    except Exception as e:
        logger.debug("tts_playback_fallback", stage="pygame", reason="playback_failed", error=str(e))

    # Approach 2: Use PowerShell with .NET MediaPlayer (legacy fallback)
    try:
        ps_script = (
            f"$player = New-Object System.Media.SoundPlayer; "
            f"Add-Type -AssemblyName presentationCore; "
            f"$mediaPlayer = New-Object System.Windows.Media.MediaPlayer; "
            f"$mediaPlayer.Open('{filepath}'); "
            f"Start-Sleep -Milliseconds 500; "
            f"$mediaPlayer.Play(); "
            f"Start-Sleep -Milliseconds 500; "
            f"while ($mediaPlayer.Position -lt $mediaPlayer.NaturalDuration.TimeSpan) {{ "
            f"  Start-Sleep -Milliseconds 200 "
            f"}}; "
            f"Start-Sleep -Milliseconds 300; "
            f"$mediaPlayer.Close()"
        )

        process = await asyncio.create_subprocess_exec(
            "powershell", "-NoProfile", "-Command", ps_script,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        manager._current_process = process
        await _wait_or_cancel(process, cancel_event, manager)
        return
    except Exception as e:
        logger.debug("tts_playback_fallback", stage="powershell", error=str(e))

    # Approach 3: Use ffplay if available (from ffmpeg)
    try:
        process = await asyncio.create_subprocess_exec(
            "ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", filepath,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        manager._current_process = process
        await _wait_or_cancel(process, cancel_event, manager)
        return
    except FileNotFoundError:
        logger.debug("tts_playback_fallback", stage="ffplay", reason="not_found")

    # Approach 4: Use Windows start command (opens default media player)
    try:
        process = await asyncio.create_subprocess_shell(
            f'start /wait "" "{filepath}"',
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        manager._current_process = process
        await _wait_or_cancel(process, cancel_event, manager)
    except Exception as e:
        logger.error("tts_all_playback_methods_failed", error=str(e), exc_info=True)


def _ensure_pygame_ready() -> None:
    global _PYGAME_READY
    if _PYGAME_READY:
        return
    with _PYGAME_LOCK:
        if _PYGAME_READY:
            return
        import pygame

        pygame.mixer.init()
        _PYGAME_READY = True


def _play_audio_pygame(filepath: str, cancel_event: threading.Event, manager: TTSManager) -> None:
    _ensure_pygame_ready()
    import pygame

    pygame.mixer.music.load(filepath)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        if cancel_event.is_set() or manager._global_cancel.is_set():
            pygame.mixer.music.stop()
            break
        time.sleep(0.05)


async def _wait_or_cancel(process: asyncio.subprocess.Process, cancel_event: threading.Event, manager: TTSManager) -> None:
    while True:
        if cancel_event.is_set() or manager._global_cancel.is_set():
            try:
                process.terminate()
            except Exception as e:
                logger.debug("tts_process_terminate_failed", error=str(e))
            break
        if process.returncode is not None:
            break
        await asyncio.sleep(0.1)
    try:
        await process.wait()
    except Exception as e:
        logger.debug("tts_process_wait_failed", error=str(e))
    manager._current_process = None


def _safe_delete(filepath: str):
    """Safely delete a file, ignoring errors if it's locked."""
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
    except OSError as e:
        logger.debug("tts_temp_file_delete_failed", path=filepath, error=str(e))
