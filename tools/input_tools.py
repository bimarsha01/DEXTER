import subprocess
import time

try:
    import pyautogui as _pyautogui
except Exception:
    _pyautogui = None

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

from utils.logger import get_logger

logger = get_logger("input_tools")

def _foreground_matches_app(app_name: str) -> bool:
    if win32gui is None:
        return False

    hwnd = win32gui.GetForegroundWindow()
    if not hwnd:
        return False

    title = win32gui.GetWindowText(hwnd).lower()
    needle = app_name.lower().strip()
    if needle and needle in title:
        return True

    if win32process is not None and psutil is not None:
        try:
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            process_name = psutil.Process(pid).name().lower()
            if needle and needle in process_name:
                return True
        except Exception:
            pass

    return False


def _focus_window_for_app(app_name: str) -> bool:
    if win32gui is None:
        return False

    needle = app_name.lower().strip()
    results: list[int] = []

    def enum_windows_callback(hwnd, window_results):
        try:
            title = win32gui.GetWindowText(hwnd).lower()
            if needle and needle in title:
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
        time.sleep(0.5)
        return True
    except Exception:
        return False


def type_text(text: str, app_name: str | None = None, wait_seconds: float = 2.0) -> str:
    """
    Simulates keyboard typing to input text into the currently active window.
    """
    logger.info("keyboard_type_started", text_length=len(text), app_name=app_name or "")
    if _pyautogui is None:
        return "Keyboard automation is unavailable: missing pyautogui module."
    try:
        if app_name:
            deadline = time.time() + 5.0
            found = False
            while time.time() < deadline:
                if _foreground_matches_app(app_name):
                    found = True
                    break
                time.sleep(0.3)

            if not found:
                _focus_window_for_app(app_name)
        else:
            # Give the opened app time to gain focus before typing.
            time.sleep(wait_seconds)

        _pyautogui.write(text, interval=0.01)
        return "Successfully typed the text into the active application."
    except Exception as e:
        logger.error("keyboard_type_failed", error=str(e), exc_info=True)
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
    try:
        # Unpack the list as arguments
        _pyautogui.hotkey(*key_list)
        return f"Successfully executed shortcut: {keys}"
    except Exception as e:
        logger.error("keyboard_shortcut_failed", error=str(e), exc_info=True)
        return f"Could not perform shortcut: {str(e)}"

def enter_key() -> str:
    """
    Presses the 'Enter' key on the keyboard to submit forms or new lines.
    """
    logger.info("keyboard_enter_requested")
    if _pyautogui is None:
        return "Keyboard automation is unavailable: missing pyautogui module."
    try:
        _pyautogui.press('enter')
        return "Pressed the Enter key successfully."
    except Exception as e:
        logger.error("keyboard_enter_failed", error=str(e), exc_info=True)
        return f"Error pressing enter: {str(e)}"
         
def minimize_all_windows() -> str:
    """
    Uses the Windows + D shortcut to minimize all active windows to the desktop.
    """
    logger.info("desktop_show_requested")
    if _pyautogui is None:
        return "Desktop control unavailable: missing pyautogui module."
    try:
         _pyautogui.hotkey('win', 'd')
         return "Windows minimized."
    except Exception as e:
        logger.error("desktop_show_failed", error=str(e), exc_info=True)
        return f"Error minimizing windows: {str(e)}"
