from __future__ import annotations

import subprocess
import time
from typing import Any

try:
    import pyautogui as _pyautogui
except Exception:
    _pyautogui = None

try:
    import win32con
    import win32gui
except Exception:
    win32con = None
    win32gui = None

from utils.config import AutomationConfig, get_config
from utils.logger import get_logger

logger = get_logger("input_tools")

_EVENT_BUS = None


class AutomationError(RuntimeError):
    pass


class AutomationFocusError(AutomationError):
    pass


def set_event_bus(event_bus) -> None:
    global _EVENT_BUS
    _EVENT_BUS = event_bus


def _emit_automation_action(action: str, target: str, status: str, **fields: Any) -> None:
    if _EVENT_BUS is None:
        return
    try:
        payload = {"action": action, "target": target, "status": status, **fields}
        _EVENT_BUS.emit("automation_action", payload)
    except Exception:
        logger.debug("automation_action_emit_failed", action=action, target=target, status=status, exc_info=True)


def _automation_settings() -> AutomationConfig:
    try:
        return get_config().automation
    except Exception:
        return AutomationConfig()


def _foreground_title() -> str:
    if win32gui is None:
        return ""
    try:
        hwnd = win32gui.GetForegroundWindow()
        if not hwnd:
            return ""
        return str(win32gui.GetWindowText(hwnd) or "")
    except Exception:
        return ""


def verify_foreground(expected_title_fragment: str | None) -> tuple[bool, str]:
    if win32gui is None:
        return False, ""

    title = _foreground_title()
    if not title:
        return False, title

    needle = (expected_title_fragment or "").strip().lower()
    if needle and needle not in title.lower():
        return False, title

    return True, title


def _focus_window_for_app(app_name: str, wait_ms: int) -> bool:
    if win32gui is None:
        return False

    needle = app_name.lower().strip()
    if not needle:
        return False

    results: list[int] = []

    def enum_windows_callback(hwnd, window_results):
        try:
            title = win32gui.GetWindowText(hwnd).lower()
            if needle in title:
                window_results.append(hwnd)
        except Exception:
            pass

    try:
        win32gui.EnumWindows(enum_windows_callback, results)
    except Exception:
        return False

    if not results:
        return False

    hwnd = results[0]
    try:
        if win32con is not None:
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        win32gui.SetForegroundWindow(hwnd)
        if wait_ms > 0:
            time.sleep(wait_ms / 1000.0)
        return True
    except Exception:
        return False


def _ensure_foreground(expected_title_fragment: str | None, *, focus_target: str | None = None) -> str:
    settings = _automation_settings()
    focus_wait_ms = max(0, int(settings.focus_wait_ms or 0))
    max_retries = max(0, int(settings.max_focus_retries or 0))

    matched, title = verify_foreground(expected_title_fragment)
    if matched:
        return title

    for _ in range(max_retries):
        if focus_target:
            _focus_window_for_app(focus_target, focus_wait_ms)
        if focus_wait_ms > 0:
            time.sleep(focus_wait_ms / 1000.0)
        matched, title = verify_foreground(expected_title_fragment)
        if matched:
            return title

    actual_title = _foreground_title()
    expected_label = (expected_title_fragment or focus_target or "any foreground window").strip() or "any foreground window"
    raise AutomationFocusError(f"Foreground window check failed for '{expected_label}'. Current foreground title: '{actual_title or '<none>'}'")


def _mouse_position() -> tuple[int, int] | None:
    if _pyautogui is None:
        return None
    try:
        pos = _pyautogui.position()
        return int(pos.x), int(pos.y)
    except Exception:
        return None


def _read_clipboard_text() -> str | None:
    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "Get-Clipboard"],
            capture_output=True,
            text=True,
            check=True,
        )
        return completed.stdout.strip()
    except Exception:
        return None


def _post_action_verified(action: str, before_cursor: tuple[int, int] | None, before_clipboard: str | None) -> bool:
    settings = _automation_settings()
    if not settings.post_action_verify:
        return True

    after_cursor = _mouse_position()
    after_clipboard = _read_clipboard_text()

    if action == "type":
        return after_cursor is not None

    if action == "shortcut":
        if after_clipboard is not None and before_clipboard is not None:
            return after_clipboard != before_clipboard or bool(after_clipboard)
        return after_clipboard is not None or after_cursor is not None

    return after_cursor is not None or after_clipboard is not None


def _mark_unverified(action: str, target: str, reason: str, **fields: Any) -> str:
    logger.warning("automation_action_unverified", action=action, target=target, reason=reason, **fields)
    _emit_automation_action(action, target, "unverified", reason=reason, **fields)
    return reason


def type_text(text: str, app_name: str | None = None, wait_seconds: float = 2.0) -> str:
    """
    Simulates keyboard typing to input text into the currently active window.
    """
    logger.info("keyboard_type_started", text_length=len(text), app_name=app_name or "")
    if _pyautogui is None:
        return "Keyboard automation is unavailable: missing pyautogui module."

    target = app_name or "foreground"
    try:
        foreground_title = _ensure_foreground(app_name, focus_target=app_name)
    except AutomationFocusError as exc:
        logger.error("keyboard_type_focus_failed", error=str(exc), app_name=app_name or "")
        _emit_automation_action("type", target, "focus_failed", error=str(exc), foreground_title=_foreground_title())
        raise

    try:
        before_cursor = _mouse_position()
        before_clipboard = _read_clipboard_text()
        _pyautogui.write(text, interval=0.01)
        verified = _post_action_verified("type", before_cursor, before_clipboard)
        if verified:
            _emit_automation_action("type", target, "success", foreground_title=foreground_title)
            return "Successfully typed the text into the active application."
        reason = _mark_unverified("type", target, "Text was typed, but post-action verification could not confirm it.", foreground_title=foreground_title)
        return reason
    except Exception as e:
        logger.error("keyboard_type_failed", error=str(e), exc_info=True)
        _emit_automation_action("type", target, "unverified", error=str(e), foreground_title=foreground_title)
        return f"Error typing text: {str(e)}"


def press_shortcut(keys: str) -> str:
    """
    Executes a keyboard shortcut (e.g., 'ctrl c', 'alt tab', 'win r').
    Expects a single space-separated string of the keys to press in sequence.
    """
    key_list = keys.split()
    logger.info("keyboard_shortcut_started", key_count=len(key_list))
    if _pyautogui is None:
        return "Keyboard automation is unavailable: missing pyautogui module."

    target = keys.strip() or "shortcut"
    try:
        foreground_title = _ensure_foreground(None)
    except AutomationFocusError as exc:
        logger.error("keyboard_shortcut_focus_failed", error=str(exc), keys=keys)
        _emit_automation_action("shortcut", target, "focus_failed", error=str(exc), foreground_title=_foreground_title())
        raise

    try:
        before_cursor = _mouse_position()
        before_clipboard = _read_clipboard_text()
        _pyautogui.hotkey(*key_list)
        verified = _post_action_verified("shortcut", before_cursor, before_clipboard)
        if verified:
            _emit_automation_action("shortcut", target, "success", foreground_title=foreground_title)
            return f"Successfully executed shortcut: {keys}"
        reason = _mark_unverified("shortcut", target, "Shortcut executed, but post-action verification could not confirm it.", foreground_title=foreground_title)
        return reason
    except Exception as e:
        logger.error("keyboard_shortcut_failed", error=str(e), exc_info=True)
        _emit_automation_action("shortcut", target, "unverified", error=str(e), foreground_title=foreground_title)
        return f"Could not perform shortcut: {str(e)}"


def enter_key() -> str:
    """
    Presses the 'Enter' key on the keyboard to submit forms or new lines.
    """
    logger.info("keyboard_enter_requested")
    if _pyautogui is None:
        return "Keyboard automation is unavailable: missing pyautogui module."

    target = "enter"
    try:
        foreground_title = _ensure_foreground(None)
    except AutomationFocusError as exc:
        logger.error("keyboard_enter_focus_failed", error=str(exc))
        _emit_automation_action("shortcut", target, "focus_failed", error=str(exc), foreground_title=_foreground_title())
        raise

    try:
        before_cursor = _mouse_position()
        before_clipboard = _read_clipboard_text()
        _pyautogui.press("enter")
        verified = _post_action_verified("shortcut", before_cursor, before_clipboard)
        if verified:
            _emit_automation_action("shortcut", target, "success", foreground_title=foreground_title)
            return "Pressed the Enter key successfully."
        reason = _mark_unverified("shortcut", target, "Enter key was pressed, but post-action verification could not confirm it.", foreground_title=foreground_title)
        return reason
    except Exception as e:
        logger.error("keyboard_enter_failed", error=str(e), exc_info=True)
        _emit_automation_action("shortcut", target, "unverified", error=str(e), foreground_title=foreground_title)
        return f"Error pressing enter: {str(e)}"


def minimize_all_windows() -> str:
    """
    Uses the Windows + D shortcut to minimize all active windows to the desktop.
    """
    logger.info("desktop_show_requested")
    if _pyautogui is None:
        return "Desktop control unavailable: missing pyautogui module."

    target = "desktop"
    try:
        foreground_title = _ensure_foreground(None)
    except AutomationFocusError as exc:
        logger.error("desktop_show_focus_failed", error=str(exc))
        _emit_automation_action("shortcut", target, "focus_failed", error=str(exc), foreground_title=_foreground_title())
        raise

    try:
        before_cursor = _mouse_position()
        before_clipboard = _read_clipboard_text()
        _pyautogui.hotkey("win", "d")
        verified = _post_action_verified("shortcut", before_cursor, before_clipboard)
        if verified:
            _emit_automation_action("shortcut", target, "success", foreground_title=foreground_title)
            return "Windows minimized."
        reason = _mark_unverified("shortcut", target, "Desktop shortcut executed, but post-action verification could not confirm it.", foreground_title=foreground_title)
        return reason
    except Exception as e:
        logger.error("desktop_show_failed", error=str(e), exc_info=True)
        _emit_automation_action("shortcut", target, "unverified", error=str(e), foreground_title=foreground_title)
        return f"Error minimizing windows: {str(e)}"
