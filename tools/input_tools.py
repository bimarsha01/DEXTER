import pyautogui
import time
from utils.logger import logger

def type_text(text: str) -> str:
    """
    Simulates keyboard typing to input text into the currently active window.
    """
    logger.info(f"Typing text into the active window: '{text}'...")
    try:
        # Give the user a brief second to focus the window if requested via voice
        time.sleep(1)
        pyautogui.write(text, interval=0.01)
        return "Successfully typed the text into the active application."
    except Exception as e:
        logger.error(f"Failed to type text: {e}")
        return f"Error typing text: {str(e)}"

def press_shortcut(keys: str) -> str:
    """
    Executes a keyboard shortcut (e.g., 'ctrl c', 'alt tab', 'win r').
    Expects a single space-separated string of the keys to press in sequence.
    """
    key_list = keys.split()
    logger.info(f"Pressing shortcut combination: {key_list}")
    try:
        # Unpack the list as arguments
        pyautogui.hotkey(*key_list)
        return f"Successfully executed shortcut: {keys}"
    except Exception as e:
        logger.error(f"Failed shortcut: {e}")
        return f"Could not perform shortcut: {str(e)}"

def enter_key() -> str:
    """
    Presses the 'Enter' key on the keyboard to submit forms or new lines.
    """
    logger.info("Pressing the Enter key.")
    try:
        pyautogui.press('enter')
        return "Pressed the Enter key successfully."
    except Exception as e:
         return f"Error pressing enter: {str(e)}"
         
def minimize_all_windows() -> str:
    """
    Uses the Windows + D shortcut to minimize all active windows to the desktop.
    """
    logger.info("Minimizing all windows to desktop.")
    try:
         pyautogui.hotkey('win', 'd')
         return "Windows minimized, sir."
    except Exception as e:
         return f"Error minimizing windows: {str(e)}"
