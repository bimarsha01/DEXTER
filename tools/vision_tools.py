from __future__ import annotations

import asyncio
import base64
import ctypes
import hashlib
import io
import os
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Optional

from PIL import Image, ImageGrab

try:
    import psutil
except Exception:
    psutil = None

try:
    import win32con
    import win32gui
    import win32process
except Exception:
    win32con = None
    win32gui = None
    win32process = None

from utils.config import get_config, get_workspace_root
from utils.logger import get_logger

logger = get_logger("vision_tools")

_EVENT_BUS = None


class _RECT(ctypes.Structure):
    _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long), ("right", ctypes.c_long), ("bottom", ctypes.c_long)]


class VisionCaptureError(RuntimeError):
    pass


class VisionCaptureTimeoutError(VisionCaptureError):
    pass


class VisionCaptureRestoreError(VisionCaptureError):
    pass


@dataclass
class ScreenCaptureResult:
    image_bytes: bytes
    foreground_window: str
    capture_mode: str


@dataclass
class WindowSnapshot:
    hwnd: int
    title: str
    rect: tuple[int, int, int, int] | None
    visible: bool
    placement: tuple[Any, Any, Any, Any, Any] | None = None
    z_index: int = 0


@dataclass
class HiddenWindowState:
    hwnd: int
    title: str
    rect: tuple[int, int, int, int] | None
    was_visible: bool
    pre_hide_hash: str | None = None


def set_event_bus(event_bus) -> None:
    global _EVENT_BUS
    _EVENT_BUS = event_bus


def _emit_vision_capture_event(status: str, duration_ms: float, **fields: Any) -> None:
    if _EVENT_BUS is None:
        return
    try:
        _EVENT_BUS.emit("vision_capture", {"status": status, "duration_ms": duration_ms, **fields})
    except Exception:
        logger.debug("vision_capture_event_emit_failed", status=status, exc_info=True)


def _capture_timeout_seconds() -> float:
    try:
        return max(0.1, float(get_config().vision.capture_timeout))
    except Exception:
        return 5.0


def _virtual_screen_origin() -> tuple[int, int]:
    try:
        user32 = ctypes.windll.user32
        return int(user32.GetSystemMetrics(76)), int(user32.GetSystemMetrics(77))
    except Exception:
        return 0, 0


def _hash_image(image: Image.Image) -> str:
    rgb = image.convert("RGB")
    return hashlib.sha256(rgb.tobytes()).hexdigest()


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

    try:
        if win32process is not None and psutil is not None:
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            process_name = psutil.Process(pid).name().lower()
            if process_name in _IDE_PROCESSES:
                logger.debug("ide_detected_by_process", process=process_name)
                return True
    except Exception as e:
        logger.debug("ide_process_check_failed", error=str(e))

    try:
        title = win32gui.GetWindowText(hwnd).lower() if win32gui else ""
        for keyword in _IDE_TITLE_KEYWORDS:
            if keyword in title:
                logger.debug("ide_detected_by_title", title=title, keyword=keyword)
                return True
    except Exception as e:
        logger.debug("ide_title_check_failed", error=str(e))

    return False


def _snapshot_windows() -> list[WindowSnapshot]:
    snapshots: list[WindowSnapshot] = []
    if win32gui is None:
        return snapshots

    def _callback(hwnd, result_list):
        try:
            rect = win32gui.GetWindowRect(hwnd)
            placement = None
            try:
                placement = win32gui.GetWindowPlacement(hwnd)
            except Exception:
                placement = None
            result_list.append(
                WindowSnapshot(
                    hwnd=hwnd,
                    title=str(win32gui.GetWindowText(hwnd) or ""),
                    rect=tuple(rect) if rect and len(rect) == 4 else None,
                    visible=bool(win32gui.IsWindowVisible(hwnd)),
                    placement=placement,
                    z_index=len(result_list),
                )
            )
        except Exception:
            pass

    try:
        win32gui.EnumWindows(_callback, snapshots)
    except Exception as e:
        logger.debug("window_snapshot_failed", error=str(e), exc_info=True)

    return snapshots


def _restore_window_snapshot(snapshot: WindowSnapshot) -> bool:
    if win32gui is None or not snapshot.hwnd:
        return True

    try:
        if not win32gui.IsWindow(snapshot.hwnd):
            return True

        if snapshot.visible:
            show_cmd = 5
            if snapshot.placement and len(snapshot.placement) > 1:
                show_cmd = int(snapshot.placement[1] or 5)
                if show_cmd in {0, 2}:
                    show_cmd = 9
            try:
                win32gui.ShowWindow(snapshot.hwnd, show_cmd)
            except Exception:
                win32gui.ShowWindow(snapshot.hwnd, 9)
        else:
            try:
                win32gui.ShowWindow(snapshot.hwnd, 0)
            except Exception:
                pass

        if snapshot.rect and len(snapshot.rect) == 4:
            left, top, right, bottom = snapshot.rect
            width = max(0, right - left)
            height = max(0, bottom - top)
            try:
                win32gui.SetWindowPos(
                    snapshot.hwnd,
                    None,
                    left,
                    top,
                    width,
                    height,
                    win32con.SWP_NOACTIVATE if win32con is not None else 0,
                )
            except Exception:
                pass

        return True
    except Exception:
        return False


def _restore_window_z_order(snapshots: list[WindowSnapshot]) -> bool:
    if win32gui is None:
        return True

    ordered = [snapshot for snapshot in snapshots if snapshot.hwnd and win32gui.IsWindow(snapshot.hwnd)]
    if not ordered:
        return True

    restore_sequence = list(reversed(ordered))
    ok = True
    for index, snapshot in enumerate(restore_sequence):
        try:
            insert_after = restore_sequence[index - 1].hwnd if index > 0 else getattr(win32con, "HWND_BOTTOM", None)
            win32gui.SetWindowPos(
                snapshot.hwnd,
                insert_after,
                0,
                0,
                0,
                0,
                (getattr(win32con, "SWP_NOMOVE", 0) | getattr(win32con, "SWP_NOSIZE", 0) | getattr(win32con, "SWP_NOACTIVATE", 0)),
            )
        except Exception:
            ok = False
    return ok


def _restore_windows(snapshots: list[WindowSnapshot]) -> bool:
    ok = True
    for snapshot in snapshots:
        if not _restore_window_snapshot(snapshot):
            ok = False
    if not _restore_window_z_order(snapshots):
        ok = False

    try:
        if snapshots:
            foreground = next((snapshot for snapshot in snapshots if snapshot.visible), None)
            if foreground and win32gui is not None and win32gui.IsWindow(foreground.hwnd):
                try:
                    win32gui.SetForegroundWindow(foreground.hwnd)
                except Exception:
                    pass
    except Exception:
        ok = False

    return ok


def _find_foreground_ide_window() -> tuple[int | None, str, tuple[int, int, int, int] | None]:
    if win32gui is None:
        return None, "", None

    try:
        hwnd = win32gui.GetForegroundWindow()
        if not hwnd or not _is_ide_window(hwnd):
            return None, "", None
        title = str(win32gui.GetWindowText(hwnd) or "")
        rect = win32gui.GetWindowRect(hwnd)
        bbox = tuple(rect) if rect and len(rect) == 4 else None
        return hwnd, title, bbox
    except Exception as e:
        logger.debug("foreground_ide_lookup_failed", error=str(e), exc_info=True)
        return None, "", None


def _hide_window(hwnd: int) -> bool:
    if not hwnd:
        return False
    try:
        user32 = ctypes.windll.user32
        swp_hidewindow = 0x0080
        if user32.SetWindowPos(hwnd, None, 0, 0, 0, 0, swp_hidewindow):
            time.sleep(0.1)
            return True
    except Exception as e:
        logger.debug("setwindowpos_hide_failed", error=str(e))

    try:
        if ctypes.windll.user32.ShowWindow(hwnd, 6):
            time.sleep(0.1)
            return True
    except Exception as e:
        logger.debug("showwindow_minimize_failed", error=str(e))

    return False


def _is_hidden(hwnd: int) -> bool:
    if win32gui is None or not hwnd:
        return False
    try:
        if not win32gui.IsWindow(hwnd):
            return True
        if win32gui.IsWindowVisible(hwnd):
            return False
        if win32gui.GetForegroundWindow() == hwnd:
            return False
        return True
    except Exception:
        return False


def _capture_region_hash(rect: tuple[int, int, int, int] | None) -> str | None:
    if not rect:
        return None
    try:
        image = ImageGrab.grab(bbox=rect)
        return _hash_image(image)
    except Exception as e:
        logger.debug("capture_region_hash_failed", error=str(e), exc_info=True)
        return None


class CaptureContext:
    def __init__(self, capture_timeout: float | None = None) -> None:
        self.capture_timeout = float(capture_timeout or _capture_timeout_seconds())
        self.started_at = 0.0
        self.window_snapshots: list[WindowSnapshot] = []
        self.hidden_window: HiddenWindowState | None = None
        self.restore_failed = False

    def __enter__(self):
        self.started_at = time.monotonic()
        self.window_snapshots = _snapshot_windows()
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if not _restore_windows(self.window_snapshots):
                self.restore_failed = True
                logger.warning("capture_context_restore_failed", hidden_hwnd=getattr(self.hidden_window, "hwnd", None))
        except Exception as restore_exc:
            self.restore_failed = True
            logger.warning("capture_context_restore_failed", error=str(restore_exc), exc_info=True)
        return False

    def elapsed_ms(self) -> float:
        return (time.monotonic() - self.started_at) * 1000.0

    def check_timeout(self) -> None:
        if self.elapsed_ms() > self.capture_timeout * 1000.0:
            raise VisionCaptureTimeoutError(f"Vision capture exceeded timeout of {self.capture_timeout:.2f}s")

    def hide_foreground_ide(self) -> HiddenWindowState | None:
        hwnd, title, rect = _find_foreground_ide_window()
        if not hwnd:
            return None

        pre_hide_hash = _capture_region_hash(rect)
        logger.info("hiding_ide_for_capture", title=title, hwnd=hwnd)
        if not _hide_window(hwnd):
            raise VisionCaptureError(f"Unable to hide IDE window '{title or hwnd}' before capture")

        state = HiddenWindowState(
            hwnd=hwnd,
            title=title,
            rect=rect,
            was_visible=True,
            pre_hide_hash=pre_hide_hash,
        )
        self.hidden_window = state
        return state

    def ensure_hidden(self) -> None:
        if self.hidden_window is None:
            return
        if not _is_hidden(self.hidden_window.hwnd):
            raise VisionCaptureError(
                f"IDE window '{self.hidden_window.title or self.hidden_window.hwnd}' was not hidden before capture"
            )

    def verify_post_capture(self, screenshot: Image.Image) -> None:
        if self.hidden_window is None or not self.hidden_window.rect or not self.hidden_window.pre_hide_hash:
            return

        origin_x, origin_y = _virtual_screen_origin()
        left, top, right, bottom = self.hidden_window.rect
        crop_box = (left - origin_x, top - origin_y, right - origin_x, bottom - origin_y)
        crop_box = (
            max(0, crop_box[0]),
            max(0, crop_box[1]),
            min(screenshot.width, crop_box[2]),
            min(screenshot.height, crop_box[3]),
        )
        if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
            raise VisionCaptureError("Unable to verify the screenshot because the IDE region is outside the captured frame")

        cropped = screenshot.crop(crop_box)
        post_hash = _hash_image(cropped)
        if post_hash == self.hidden_window.pre_hide_hash:
            raise VisionCaptureError("Screenshot verification failed: IDE pixels still appear in the captured region")


def hide_ide_if_foreground() -> Optional[int]:
    """Hides the IDE/terminal if it is the foreground window. Returns hwnd to restore later."""
    try:
        hwnd, _, _ = _find_foreground_ide_window()
        if not hwnd:
            return None
        if _hide_window(hwnd):
            logger.info("ide_hidden_for_capture", hwnd=hwnd)
            return hwnd
    except Exception as e:
        logger.error("hide_ide_failed", error=str(e), exc_info=True)
    return None


def restore_ide(hwnd: Optional[int]):
    """Restores a previously hidden IDE window."""
    if not hwnd:
        return
    try:
        user32 = ctypes.windll.user32
        try:
            swp_showwindow = 0x0040
            swp_nozorder = 0x0004
            if user32.SetWindowPos(hwnd, None, 0, 0, 0, 0, swp_showwindow | swp_nozorder):
                time.sleep(0.1)
                user32.SetForegroundWindow(hwnd)
                return
        except Exception as e:
            logger.debug("restore_setwindowpos_failed", error=str(e))

        if user32.ShowWindow(hwnd, 9):
            time.sleep(0.1)
            user32.SetForegroundWindow(hwnd)
        else:
            logger.warning("restore_failed", hwnd=hwnd)
    except Exception as e:
        logger.debug("restore_ide_failed", error=str(e))


def _resize_image(image: Image.Image, max_dimension: int) -> Image.Image:
    width, height = image.size
    if max(width, height) <= max_dimension:
        return image
    scale = max_dimension / float(max(width, height))
    new_size = (int(width * scale), int(height * scale))
    return image.resize(new_size, Image.LANCZOS)


def capture_screen_for_vision(max_dimension: int = 1280) -> ScreenCaptureResult:
    """Capture the user's actual viewport atomically and fail loudly on any verification problem."""
    timeout_seconds = _capture_timeout_seconds()
    with CaptureContext(capture_timeout=timeout_seconds) as capture_context:
        status = "success"
        try:
            capture_context.check_timeout()
            hidden_state = capture_context.hide_foreground_ide()
            capture_context.check_timeout()
            if hidden_state is not None:
                capture_context.ensure_hidden()
                capture_context.check_timeout()

            image = ImageGrab.grab(all_screens=True)
            capture_context.check_timeout()
            capture_context.verify_post_capture(image)
            capture_context.check_timeout()

            title, _ = _get_foreground_window_bbox()
            capture_mode = "full_screen"
            resized = _resize_image(image, max_dimension=max_dimension)

            buffer = io.BytesIO()
            resized.save(buffer, format="PNG")
            logger.info(
                "screen_capture_prepared",
                foreground_window=title or "",
                capture_mode=capture_mode,
                was_ide_hidden=hidden_state is not None,
                width=resized.width,
                height=resized.height,
                capture_timeout_seconds=timeout_seconds,
            )

            result = ScreenCaptureResult(
                image_bytes=buffer.getvalue(),
                foreground_window=title or "",
                capture_mode=capture_mode,
            )

            if capture_context.restore_failed:
                status = "restore_failed"
                raise VisionCaptureRestoreError("Capture state restore failed after screenshot generation")

            _emit_vision_capture_event(status, capture_context.elapsed_ms())
            return result
        except VisionCaptureTimeoutError:
            status = "timeout"
            _emit_vision_capture_event(status, capture_context.elapsed_ms())
            raise
        except Exception:
            status = "restore_failed"
            _emit_vision_capture_event(status, capture_context.elapsed_ms())
            raise


def capture_screen(max_dimension: int = 1280) -> str:
    """Captures the screen in memory and returns a base64-encoded PNG string."""
    try:
        result = capture_screen_for_vision(max_dimension=max_dimension)
        encoded = base64.b64encode(result.image_bytes).decode("ascii")
        return f"image/png;base64,{encoded}"
    except Exception as e:
        logger.error("screen_capture_base64_failed", error=str(e), exc_info=True)
        return f"I was unable to capture the screen: {str(e)}"


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
