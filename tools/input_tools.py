try:
    import pyautogui as _pyautogui
except Exception:
    _pyautogui = None
import time
from utils.logger import get_logger

logger = get_logger("input_tools")

def type_text(text: str) -> str:
    """
    Simulates keyboard typing to input text into the currently active window.
    """
    logger.info("keyboard_type_started", text_length=len(text))
    if _pyautogui is None:
        return "Keyboard automation is unavailable: missing pyautogui module."
    try:
        # Give the user a brief second to focus the window if requested via voice
        time.sleep(1)
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
         return "Windows minimized, sir."
    except Exception as e:
        logger.error("desktop_show_failed", error=str(e), exc_info=True)
        return f"Error minimizing windows: {str(e)}"
