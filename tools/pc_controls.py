import os
import subprocess
import ctypes
from utils.logger import logger
from utils.config import load_config


APP_MAP = {
    "notepad": "notepad.exe",
    "calculator": "calc.exe",
    "calc": "calc.exe",
    "cmd": "cmd.exe",
    "command prompt": "cmd.exe",
    "terminal": "wt.exe",
    "powershell": "powershell.exe",
    "explorer": "explorer.exe",
    "file explorer": "explorer.exe",
    "paint": "mspaint.exe",
    "settings": "ms-settings:",
    "control panel": "control.exe",
    "task manager": "taskmgr.exe",
    "word": "winword.exe",
    "excel": "excel.exe",
    "powerpoint": "powerpnt.exe",
    "outlook": "outlook.exe",
    "chrome": "chrome.exe",
    "google chrome": "chrome.exe",
    "firefox": "firefox.exe",
    "edge": "msedge.exe",
    "microsoft edge": "msedge.exe",
    "brave": "brave.exe",
    "vscode": "code.exe",
    "visual studio code": "code.exe",
    "vs code": "code.exe",
    "spotify": "spotify.exe",
    "discord": "discord.exe",
    "teams": "teams.exe",
    "microsoft teams": "teams.exe",
    "snipping tool": "snippingtool.exe",
    "snip": "snippingtool.exe",
    "clock": "ms-clock:",
    "alarm": "ms-clock:",
    "camera": "microsoft.windows.camera:",
    "maps": "bingmaps:",
    "store": "ms-windows-store:",
    "photos": "ms-photos:",
    "mail": "outlookmail:",
    "calendar": "outlookcal:",
}


DEFAULT_ALLOWED_APPS = set(APP_MAP.keys())


def open_application(app_name: str) -> str:
    """Opens a Windows application by name (e.g., 'notepad', 'calculator', 'chrome', 'spotify')."""
    app_name = app_name.lower().strip()
    logger.info(f"Attempting to open application: {app_name}")

    if any(ch in app_name for ch in ["&", "|", ";"]):
        return "Application name contains unsafe characters."

    allowed = _get_allowed_apps()
    if allowed and app_name not in allowed:
        return f"'{app_name}' is not in the allowed applications list."

    # Comprehensive map of common app names to executables
    command = APP_MAP.get(app_name)
    if command:
        try:
            os.startfile(command)
            return f"Successfully opened {app_name}, sir."
        except Exception as e:
            logger.error(f"Failed to open {app_name}: {e}")
            return f"Error opening {app_name}: {str(e)}"

    # Fallback: try to launch it via Windows Start
    try:
        subprocess.Popen(["cmd", "/c", "start", "", app_name])
        return f"Attempted to open {app_name} via Windows, sir."
    except Exception as e:
        return f"I could not find or launch '{app_name}', sir."


def close_application(app_name: str) -> str:
    """Closes a running application by its process name (e.g., 'notepad', 'chrome')."""
    app_name = app_name.lower().strip()
    logger.info(f"Attempting to close application: {app_name}")

    # Map friendly names to process names
    process_map = {
        "notepad": "notepad.exe",
        "chrome": "chrome.exe",
        "google chrome": "chrome.exe",
        "firefox": "firefox.exe",
        "edge": "msedge.exe",
        "spotify": "spotify.exe",
        "discord": "discord.exe",
        "word": "winword.exe",
        "excel": "excel.exe",
        "vscode": "code.exe",
        "vs code": "code.exe",
        "calculator": "calculator.exe",
        "calc": "calculator.exe",
        "paint": "mspaint.exe",
    }

    process = process_map.get(app_name, f"{app_name}.exe")

    try:
        result = subprocess.run(
            ["taskkill", "/IM", process, "/F"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return f"Successfully closed {app_name}, sir."
        else:
            return f"Could not find a running instance of {app_name}."
    except Exception as e:
        logger.error(f"Failed to close {app_name}: {e}")
        return f"Error closing {app_name}: {str(e)}"


def set_system_volume(level: int) -> str:
    """Sets the Windows system master volume to an exact percentage between 0 and 100."""
    if level < 0 or level > 100:
        return "Volume level must be between 0 and 100 percent, sir."

    logger.info(f"Setting system volume to {level}%")

    # Try using pycaw for precise volume control
    try:
        from ctypes import cast, POINTER
        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        # SetMasterVolumeLevelScalar takes 0.0 to 1.0
        volume.SetMasterVolumeLevelScalar(level / 100.0, None)
        return f"Volume set to {level}%, sir."

    except ImportError:
        logger.warning("pycaw not installed. Falling back to key simulation.")
    except Exception as e:
        logger.warning(f"pycaw volume control failed: {e}. Falling back to key simulation.")

    # Fallback: simulate volume keys via PowerShell
    try:
        # First mute by pressing volume down many times, then press up to target
        steps = level // 2  # Each key press is ~2%
        ps_cmd = (
            "$wshell = New-Object -ComObject wscript.shell; "
            "for($i=0; $i -lt 50; $i++) { $wshell.SendKeys([char]174) }; "
            f"for($i=0; $i -lt {steps}; $i++) {{ $wshell.SendKeys([char]175) }}"
        )
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_cmd],
            capture_output=True, timeout=15
        )
        return f"Volume set to approximately {level}%, sir."
    except Exception as e:
        logger.error(f"Volume fallback also failed: {e}")
        return "I was unable to adjust the volume, sir."


def lock_workstation() -> str:
    """Locks the Windows computer screen immediately."""
    logger.info("Locking workstation...")
    try:
        ctypes.windll.user32.LockWorkStation()
        return "Workstation locked successfully, sir."
    except Exception as e:
        return f"Failed to lock workstation: {str(e)}"


def _get_allowed_apps() -> set:
    config = load_config()
    security = config.get("security", {})
    allowed = security.get("allowed_apps")
    if isinstance(allowed, list) and allowed:
        return {str(item).lower().strip() for item in allowed}
    return DEFAULT_ALLOWED_APPS
