import base64
import ctypes
import io
import os
from dataclasses import dataclass
from typing import Optional

from PIL import Image, ImageGrab

try:
    import win32gui  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    win32gui = None
from utils.logger import get_logger

logger = get_logger("vision_tools")
from utils.config import get_workspace_root


class _RECT(ctypes.Structure):
    _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long), ("right", ctypes.c_long), ("bottom", ctypes.c_long)]


@dataclass
class ScreenCaptureResult:
    image_bytes: bytes
    foreground_window: str
    capture_mode: str


def read_workspace_file(relative_path: str, max_chars: int = 4000) -> str:
    """
    Reads a text file within the workspace. Rejects absolute paths or parent traversal.
    Returns a truncated string if the file is large.
    """
    if not relative_path or not relative_path.strip():
        return "You must provide a workspace-relative file path."

    clean_path = relative_path.strip().replace("\\", "/")
    if os.path.isabs(clean_path) or ":" in clean_path:
        return "Absolute paths are not allowed. Provide a relative path from the workspace root."
    if ".." in clean_path.split("/"):
        return "Parent path traversal is not allowed. Provide a safe relative path."

    root = get_workspace_root()
    full_path = os.path.abspath(os.path.join(root, clean_path))
    if not full_path.startswith(root):
        return "Path is outside the workspace."

    if not os.path.exists(full_path):
        return f"File not found: {clean_path}"

    try:
        with open(full_path, "r", encoding="utf-8", errors="replace") as file:
            data = file.read(max_chars + 1)
        if "\x00" in data:
            return "This file appears to be binary and cannot be read as text."
        if len(data) > max_chars:
            return data[:max_chars] + "\n...[truncated]"
        return data
    except Exception as e:
        logger.error("workspace_file_read_failed", path=clean_path, error=str(e), exc_info=True)
        return f"I could not read {clean_path}."


import time

# IDE window title keywords — used as fallback when win32process is unavailable
_IDE_TITLE_KEYWORDS = [
    "visual studio code", "- code", "vscode",
    "pycharm", "intellij", "android studio",
    "command prompt", "powershell", "windows terminal",
    "python", "cmd.exe", "dexter",
]

def _is_ide_window(hwnd) -> bool:
    """Check if a window handle belongs to an IDE or terminal, using both process name and title."""
    # Strategy 1: Check process name (most reliable)
    try:
        import win32process
        import psutil
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        process_name = psutil.Process(pid).name().lower()
        ide_processes = {"code.exe", "windowsterminal.exe", "cmd.exe", "powershell.exe", "python.exe", "pythonw.exe", "cursor.exe", "pycharm64.exe", "idea64.exe"}
        if process_name in ide_processes:
            logger.debug("ide_detected_by_process", process=process_name)
            return True
    except Exception as e:
        logger.debug("ide_process_check_failed", error=str(e))

    # Strategy 2: Check window title (fallback)
    try:
        title = win32gui.GetWindowText(hwnd).lower() if win32gui else ""
        for keyword in _IDE_TITLE_KEYWORDS:
            if keyword in title:
                logger.debug("ide_detected_by_title", title=title, keyword=keyword)
                return True
    except Exception:
        pass

    return False


def hide_ide_if_foreground() -> Optional[int]:
    """Minimizes the IDE/terminal if it is the foreground window. Returns hwnd to restore later."""
    try:
        if win32gui is None:
            logger.warning("hide_ide_skipped_no_win32gui")
            return None

        hwnd = win32gui.GetForegroundWindow()
        if not hwnd:
            return None

        if _is_ide_window(hwnd):
            logger.info("minimizing_ide_for_capture", title=win32gui.GetWindowText(hwnd))
            # Use ctypes as a robust fallback for ShowWindow
            SW_MINIMIZE = 6
            ctypes.windll.user32.ShowWindow(hwnd, SW_MINIMIZE)
            time.sleep(0.5)  # Wait for minimize animation to fully complete
            return hwnd

    except Exception as e:
        logger.error("hide_ide_failed", error=str(e), exc_info=True)
    return None


def restore_ide(hwnd: Optional[int]):
    """Restores a previously minimized IDE window."""
    if not hwnd:
        return
    try:
        SW_RESTORE = 9
        ctypes.windll.user32.ShowWindow(hwnd, SW_RESTORE)
        ctypes.windll.user32.SetForegroundWindow(hwnd)
    except Exception as e:
        logger.debug("restore_ide_failed", error=str(e))


def capture_screen_for_vision(max_dimension: int = 1280) -> ScreenCaptureResult:
    """Capture the user's actual viewport: minimizes IDE if needed, captures full screen."""
    hidden_hwnd = hide_ide_if_foreground()
    try:
        # After minimizing the IDE, always capture the FULL SCREEN.
        # This shows exactly what the user sees on their monitor.
        image = ImageGrab.grab(all_screens=True)
        capture_mode = "full_screen"

        # Also grab the foreground window title for context
        title, _ = _get_foreground_window_bbox()

        image = _resize_image(image, max_dimension=max_dimension)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        logger.info(
            "screen_capture_prepared",
            foreground_window=title or "",
            capture_mode=capture_mode,
            was_ide_hidden=hidden_hwnd is not None,
            width=image.width,
            height=image.height,
        )
        return ScreenCaptureResult(
            image_bytes=buffer.getvalue(),
            foreground_window=title or "",
            capture_mode=capture_mode,
        )
    except Exception as e:
        logger.error(
            "screen_capture_failed",
            error=str(e),
            exc_info=True,
        )
        raise
    finally:
        restore_ide(hidden_hwnd)


def capture_screen(max_dimension: int = 1280) -> str:
    """Captures the screen in memory and returns a base64-encoded PNG string."""
    try:
        result = capture_screen_for_vision(max_dimension=max_dimension)
        encoded = base64.b64encode(result.image_bytes).decode("ascii")
        return f"image/png;base64,{encoded}"
    except Exception as e:
        logger.error("screen_capture_base64_failed", error=str(e), exc_info=True)
        return "I was unable to capture the screen."


def _resize_image(image: Image.Image, max_dimension: int) -> Image.Image:
    width, height = image.size
    if max(width, height) <= max_dimension:
        return image
    scale = max_dimension / float(max(width, height))
    new_size = (int(width * scale), int(height * scale))
    return image.resize(new_size, Image.LANCZOS)


def _get_foreground_window_bbox() -> tuple[str, Optional[tuple[int, int, int, int]]]:
    try:
        if win32gui is not None:
            hwnd = win32gui.GetForegroundWindow()
            title = win32gui.GetWindowText(hwnd) or ""
            rect = win32gui.GetWindowRect(hwnd)
            if rect and len(rect) == 4:
                left, top, right, bottom = rect
                if right > left and bottom > top:
                    return title, (left, top, right, bottom)
            return title, None
    except Exception:
        logger.debug("foreground_window_win32gui_failed", exc_info=True)

    try:
        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return "", None
        length = user32.GetWindowTextLengthW(hwnd)
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)
        rect = _RECT()
        if user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            bbox = (rect.left, rect.top, rect.right, rect.bottom)
            if rect.right > rect.left and rect.bottom > rect.top:
                return buffer.value, bbox
        return buffer.value, None
    except Exception:
        logger.error("foreground_window_ctypes_failed", exc_info=True)
        return "", None
