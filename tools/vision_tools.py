import asyncio
import base64
import ctypes
import io
import os
from functools import lru_cache
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


def _summarize_foreground_window(title: str) -> str:
    clean_title = (title or "").strip()
    if not clean_title:
        return "I can capture the screen, but I can't tell what's open from the window title alone."

    lower_title = clean_title.lower()
    app_suffixes = {
        "brave": "Brave",
        "google chrome": "Chrome",
        "chrome": "Chrome",
        "microsoft edge": "Microsoft Edge",
        "edge": "Microsoft Edge",
        "firefox": "Firefox",
        "word": "Word",
        "excel": "Excel",
        "powerpoint": "PowerPoint",
        "notepad": "Notepad",
        "visual studio code": "Visual Studio Code",
        "vscode": "Visual Studio Code",
        "code": "Visual Studio Code",
    }

    for marker, app_name in app_suffixes.items():
        if lower_title.endswith(f" - {marker}") or lower_title == marker:
            content = clean_title[: -(len(marker) + 3)].strip(" -") if lower_title.endswith(f" - {marker}") else ""
            if content:
                return f"I can see {app_name} is open on {content}."
            return f"I can see {app_name} is open."

    if " - " in clean_title:
        parts = [part.strip() for part in clean_title.split(" - ") if part.strip()]
        if len(parts) >= 2:
            page_title = parts[0]
            app_name = parts[-1]
            return f"I can see {app_name} is open on {page_title}."

    return f"I can see a window titled {clean_title}."


@lru_cache(maxsize=1)
def _get_local_vision_captioner():
    try:
        import torch
        from transformers import pipeline

        device = 0 if torch.cuda.is_available() else -1
        return pipeline(
            "image-to-text",
            model="Salesforce/blip-image-captioning-base",
            device=device,
        )
    except Exception as e:
        logger.debug("local_vision_captioner_unavailable", error=str(e), exc_info=True)
        return None


def _caption_screen_locally(capture: ScreenCaptureResult) -> str:
    captioner = _get_local_vision_captioner()
    if captioner is None:
        return ""

    try:
        image = Image.open(io.BytesIO(capture.image_bytes)).convert("RGB")
        result = captioner(image)
        if isinstance(result, list) and result:
            first_item = result[0]
            if isinstance(first_item, dict):
                caption = first_item.get("generated_text") or first_item.get("caption") or ""
                return str(caption).strip()
        if isinstance(result, dict):
            caption = result.get("generated_text") or result.get("caption") or ""
            return str(caption).strip()
    except Exception as e:
        logger.debug("local_screen_caption_failed", error=str(e), exc_info=True)

    return ""


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

# Extended IDE process names for more comprehensive detection
_IDE_PROCESSES = {
    "code.exe", "code - insiders.exe",
    "cursor.exe",
    "windowsterminal.exe", "windowsapp.exe",
    "cmd.exe", "powershell.exe", "pwsh.exe",
    "python.exe", "pythonw.exe",
    "pycharm64.exe", "pycharmapp.exe",
    "idea64.exe", "ideaapp.exe",
    "studio64.exe",
    "notepad++.exe",
    "sublime_text.exe",
    "atom.exe",
    "vim.exe",
    "nvim.exe",
    "conhost.exe",
}

def _is_ide_window(hwnd) -> bool:
    """Check if a window handle belongs to an IDE or terminal, using both process name and title."""
    if not hwnd:
        return False

    # Strategy 1: Check process name (most reliable)
    try:
        import win32process
        import psutil
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        process_name = psutil.Process(pid).name().lower()
        if process_name in _IDE_PROCESSES:
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
    except Exception as e:
        logger.debug("ide_title_check_failed", error=str(e))

    return False


def hide_ide_if_foreground() -> Optional[int]:
    """Hides the IDE/terminal if it is the foreground window. Returns hwnd to restore later."""
    try:
        if win32gui is None:
            logger.warning("hide_ide_skipped_no_win32gui")
            return None

        hwnd = win32gui.GetForegroundWindow()
        if not hwnd:
            logger.debug("hide_ide_no_foreground_window")
            return None

        if not _is_ide_window(hwnd):
            logger.debug("hide_ide_not_ide_window")
            return None

        window_title = win32gui.GetWindowText(hwnd)
        logger.info("hiding_ide_for_capture", title=window_title, hwnd=hwnd)

        # Method 1: Use SetWindowPos with SWP_HIDEWINDOW for true invisibility
        try:
            SWP_HIDEWINDOW = 0x0080
            user32 = ctypes.windll.user32
            result = user32.SetWindowPos(hwnd, None, 0, 0, 0, 0, SWP_HIDEWINDOW)
            if result:
                time.sleep(0.3)  # Wait for hide to complete
                logger.info("ide_hidden_with_setwindowpos", hwnd=hwnd)
                return hwnd
            else:
                logger.warning("setwindowpos_failed", hwnd=hwnd)
        except Exception as e:
            logger.debug("setwindowpos_failed_exception", error=str(e))

        # Method 2: Fallback to minimize
        try:
            SW_MINIMIZE = 6
            result = ctypes.windll.user32.ShowWindow(hwnd, SW_MINIMIZE)
            if result:
                time.sleep(0.5)  # Wait for minimize animation
                logger.info("ide_minimized_as_fallback", hwnd=hwnd)
                return hwnd
            else:
                logger.warning("minimize_failed", hwnd=hwnd)
        except Exception as e:
            logger.debug("minimize_failed_exception", error=str(e))

        return None

    except Exception as e:
        logger.error("hide_ide_failed", error=str(e), exc_info=True)
    return None


def restore_ide(hwnd: Optional[int]):
    """Restores a previously hidden IDE window."""
    if not hwnd:
        return
    try:
        user32 = ctypes.windll.user32
        
        # Try to restore from hidden state first
        try:
            SWP_SHOWWINDOW = 0x0040
            SWP_NOZORDER = 0x0004
            result = user32.SetWindowPos(hwnd, None, 0, 0, 0, 0, SWP_SHOWWINDOW | SWP_NOZORDER)
            if result:
                logger.debug("ide_restored_with_setwindowpos", hwnd=hwnd)
                time.sleep(0.2)
                user32.SetForegroundWindow(hwnd)
                return
        except Exception as e:
            logger.debug("restore_setwindowpos_failed", error=str(e))

        # Fallback to restore from minimized state
        SW_RESTORE = 9
        result = user32.ShowWindow(hwnd, SW_RESTORE)
        if result:
            logger.debug("ide_restored_with_showwindow", hwnd=hwnd)
            time.sleep(0.2)
            user32.SetForegroundWindow(hwnd)
        else:
            logger.warning("restore_failed", hwnd=hwnd)
    except Exception as e:
        logger.debug("restore_ide_failed", error=str(e))


def capture_screen_for_vision(max_dimension: int = 1280) -> ScreenCaptureResult:
    """Capture the user's actual viewport: hides IDE if needed, captures full screen."""
    hidden_hwnd = hide_ide_if_foreground()
    try:
        # Add extra sleep if IDE was hidden to ensure it's fully gone from frame buffer
        if hidden_hwnd:
            time.sleep(0.5)
        
        # After hiding the IDE, always capture the FULL SCREEN.
        # This shows exactly what the user sees on their monitor (without the IDE).
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


async def describe_screen(user_question: str, gemini_client) -> str:
    """Capture the screen in memory and describe it with Gemini vision."""
    try:
        capture = capture_screen_for_vision()

        from google.genai import types

        response = await gemini_client.aio.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                types.Content(parts=[
                    types.Part.from_bytes(data=capture.image_bytes, mime_type="image/png"),
                    types.Part(text=(
                        f"The user asked: {user_question}\n\n"
                        f"The foreground window title is: {capture.foreground_window}\n\n"
                        "Describe what is on the screen in 2-3 sentences. Be specific about what content is visible — "
                        "website names, video titles, document content, app names. Do not mention that you received a screenshot. "
                        "Just describe what you see naturally."
                    )),
                ])
            ],
        )

        return response.text or ""
    except Exception as e:
        logger.error("screen_describe_failed", error=str(e), exc_info=True)
        if "capture" in locals():
            summary = _summarize_foreground_window(capture.foreground_window)
            return f"{summary} I can't inspect the pixels right now because my vision service is rate-limited."
        return "I can't inspect the screen contents right now because my vision service is rate-limited."


def describe_screen_without_vision() -> str:
    """Return a local fallback description when Gemini vision is unavailable."""
    try:
        capture = capture_screen_for_vision()
        local_caption = _caption_screen_locally(capture)
        if local_caption:
            return (
                f"I can see a screenshot that looks like {local_caption}. "
                f"{_summarize_foreground_window(capture.foreground_window)}"
            ).strip()
        summary = _summarize_foreground_window(capture.foreground_window)
        if capture.foreground_window.strip():
            return f"{summary} I can't inspect the pixels without Gemini vision."
        return "I can capture the screen, but I can't inspect the pixels without Gemini vision."
    except Exception as e:
        logger.error("screen_fallback_failed", error=str(e), exc_info=True)
        return "I can't inspect the screen contents right now."


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
