"""
Dexter TTS Speaker — Converts text to speech using Edge-TTS
and plays audio using sounddevice (no pygame dependency needed).
"""
import edge_tts
import asyncio
import os
import time
import uuid
import threading
from utils.logger import logger
from utils.metrics import metrics


class TTSManager:
    def __init__(self, voice: str = "en-GB-RyanNeural"):
        self.voice = voice
        self._cancel_event = threading.Event()
        self._current_process = None
        self._lock = asyncio.Lock()

    def stop(self) -> None:
        self._cancel_event.set()
        if self._current_process and self._current_process.returncode is None:
            try:
                self._current_process.terminate()
            except Exception:
                pass

    async def speak(self, text: str) -> None:
        if not text or not text.strip():
            return

        async with self._lock:
            self.stop()
            self._cancel_event = threading.Event()
            cancel_event = self._cancel_event

        logger.info(f"Speaking: '{text[:80]}...'") if len(text) > 80 else logger.info(f"Speaking: '{text}'")
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
            logger.error(f"TTS Error: {e}")
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
    # Approach 1: Use PowerShell with .NET MediaPlayer (most reliable)
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
        logger.debug(f"PowerShell MediaPlayer failed: {e}, trying fallback...")

    # Approach 2: Use ffplay if available (from ffmpeg)
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
        logger.debug("ffplay not found, trying next fallback...")

    # Approach 3: Use Windows start command (opens default media player)
    try:
        process = await asyncio.create_subprocess_shell(
            f'start /wait "" "{filepath}"',
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        manager._current_process = process
        await _wait_or_cancel(process, cancel_event, manager)
    except Exception as e:
        logger.error(f"All audio playback methods failed: {e}")


async def _wait_or_cancel(process: asyncio.subprocess.Process, cancel_event: threading.Event, manager: TTSManager) -> None:
    while True:
        if cancel_event.is_set():
            try:
                process.terminate()
            except Exception:
                pass
            break
        if process.returncode is not None:
            break
        await asyncio.sleep(0.1)
    try:
        await process.wait()
    except Exception:
        pass
    manager._current_process = None


def _safe_delete(filepath: str):
    """Safely delete a file, ignoring errors if it's locked."""
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
    except OSError:
        pass
