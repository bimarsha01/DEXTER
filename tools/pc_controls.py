import os
import subprocess
import ctypes
import shutil
from utils.logger import get_logger

logger = get_logger("pc_controls")
from utils.config import get_config

try:
    import winreg
except ImportError:  # pragma: no cover - only available on Windows
    winreg = None

try:
    from rapidfuzz import process, fuzz
except ImportError:
    process = None
    fuzz = None

try:
    import win32com.client
except ImportError:
    win32com = None

_APP_CACHE = {}
_APP_CACHE_LAST_REFRESH = 0

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


import time

def _search_registry_for_app(app_name: str) -> str:
    if winreg is None:
        return ""
    app_name_exe = f"{app_name}.exe".lower()
    for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        try:
            with winreg.OpenKey(hive, r"Software\Microsoft\Windows\CurrentVersion\App Paths") as key:
                for i in range(winreg.QueryInfoKey(key)[0]):
                    subkey_name = winreg.EnumKey(key, i)
                    if subkey_name.lower() == app_name_exe or subkey_name.lower().startswith(app_name.lower()):
                        try:
                            with winreg.OpenKey(key, subkey_name) as subkey:
                                value, _ = winreg.QueryValueEx(subkey, "")
                                if value and os.path.exists(value):
                                    return value
                        except OSError:
                            pass
        except OSError:
            pass
    return ""

def _search_common_directories(app_name: str) -> str:
    if not process or not fuzz:
        return ""
    
    user_profile = os.environ.get("USERPROFILE", "")
    program_files = os.environ.get("ProgramFiles", "C:\\Program Files")
    program_files_x86 = os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)")
    
    search_dirs = [
        program_files,
        program_files_x86,
    ]
    if user_profile:
        search_dirs.extend([
            os.path.join(user_profile, "AppData", "Roaming"),
            os.path.join(user_profile, "AppData", "Local"),
            os.path.join(user_profile, "AppData", "Local", "Microsoft", "WindowsApps"),
        ])
        
    candidates = []
    
    for base_dir in search_dirs:
        if not os.path.exists(base_dir):
            continue
        try:
            for root, dirs, files in os.walk(base_dir):
                depth = root[len(base_dir):].count(os.sep)
                if depth > 3:
                    dirs[:] = []
                    continue
                
                for file in files:
                    if file.lower().endswith(".exe"):
                        candidates.append(os.path.join(root, file))
        except Exception:
            pass

    if not candidates:
        return ""

    filenames = [os.path.basename(c).lower().replace(".exe", "") for c in candidates]
    best = process.extractOne(app_name.lower(), filenames, scorer=fuzz.WRatio)
    if best and best[1] > 85:
        return candidates[best[2]]
        
    return ""

def _search_start_menu(app_name: str) -> str:
    if not process or not fuzz or not win32com:
        return ""
    
    user_profile = os.environ.get("USERPROFILE", "")
    search_dirs = [
        r"C:\ProgramData\Microsoft\Windows\Start Menu\Programs"
    ]
    if user_profile:
        search_dirs.append(os.path.join(user_profile, r"AppData\Roaming\Microsoft\Windows\Start Menu\Programs"))
        
    candidates = []
    
    for base_dir in search_dirs:
        if not os.path.exists(base_dir):
            continue
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file.lower().endswith(".lnk"):
                    candidates.append(os.path.join(root, file))
                    
    if not candidates:
        return ""

    filenames = [os.path.basename(c).lower().replace(".lnk", "") for c in candidates]
    best = process.extractOne(app_name.lower(), filenames, scorer=fuzz.WRatio)
    if best and best[1] > 85:
        lnk_path = candidates[best[2]]
        try:
            shell = win32com.client.Dispatch("WScript.Shell")
            shortcut = shell.CreateShortCut(lnk_path)
            target = shortcut.Targetpath
            if target and os.path.exists(target):
                return target
        except Exception as e:
            logger.error("shortcut_resolution_failed", path=lnk_path, error=str(e))
            
    return ""

def open_application(app_name: str) -> str:
    """Opens a Windows application by name (e.g., 'notepad', 'calculator', 'chrome', 'spotify')."""
    app_name = app_name.lower().strip()
    logger.info("app_open_requested", app_name=app_name)

    if any(ch in app_name for ch in ["&", "|", ";"]):
        return "Application name contains unsafe characters."

    allowed = _get_allowed_apps()
    if allowed and app_name not in allowed:
        return f"'{app_name}' is not in the allowed applications list."

    global _APP_CACHE, _APP_CACHE_LAST_REFRESH
    if time.time() - _APP_CACHE_LAST_REFRESH > 86400:
        _APP_CACHE.clear()
        _APP_CACHE_LAST_REFRESH = time.time()
        
    if app_name in _APP_CACHE:
        cached_path = _APP_CACHE[app_name]
        if os.path.exists(cached_path):
            try:
                subprocess.Popen([cached_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return f"Successfully opened {app_name} (from cache), sir."
            except Exception as e:
                logger.error("app_open_cached_failed", app_name=app_name, error=str(e))

    command = APP_MAP.get(app_name) or app_name
    resolved = _resolve_command(command)
    if resolved and (os.path.exists(resolved) or resolved == command):
        try:
            if resolved.endswith(".exe") or os.path.exists(resolved):
                subprocess.Popen([resolved], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                _APP_CACHE[app_name] = resolved
            else:
                os.startfile(command)
            return f"Successfully opened {app_name}, sir."
        except Exception as e:
            logger.debug("app_open_s1_failed", error=str(e))

    reg_path = _search_registry_for_app(app_name)
    if reg_path:
        try:
            subprocess.Popen([reg_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            _APP_CACHE[app_name] = reg_path
            return f"Successfully opened {app_name} (registry), sir."
        except Exception as e:
            logger.debug("app_open_s2_failed", error=str(e))

    lnk_path = _search_start_menu(app_name)
    if lnk_path:
        try:
            subprocess.Popen([lnk_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            _APP_CACHE[app_name] = lnk_path
            return f"Successfully opened {app_name} (start menu), sir."
        except Exception as e:
            logger.debug("app_open_s5_failed", error=str(e))

    fuzz_path = _search_common_directories(app_name)
    if fuzz_path:
        try:
            subprocess.Popen([fuzz_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            _APP_CACHE[app_name] = fuzz_path
            return f"Successfully opened {app_name} (fuzzy search), sir."
        except Exception as e:
            logger.debug("app_open_s3_failed", error=str(e))

    try:
        subprocess.Popen(["cmd", "/c", "start", "", app_name])
        return f"Attempted to open {app_name} via Windows Shell, sir."
    except Exception as e:
        logger.error("app_open_start_failed", app_name=app_name, error=str(e), exc_info=True)

    return f"I could not find {app_name} installed on your system sir. Would you like me to search for it online instead?"


def close_application(app_name: str) -> str:
    """Closes a running application by its process name (e.g., 'notepad', 'chrome')."""
    app_name = app_name.lower().strip()
    logger.info("app_close_requested", app_name=app_name)

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
        logger.error("app_close_failed", app_name=app_name, error=str(e), exc_info=True)
        return f"Error closing {app_name}: {str(e)}"


def set_system_volume(level: int) -> str:
    """Sets the Windows system master volume to an exact percentage between 0 and 100."""
    if level < 0 or level > 100:
        return "Volume level must be between 0 and 100 percent, sir."

    logger.info("system_volume_set_requested", level_percent=level)

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
        logger.warning("volume_control_fallback", reason="pycaw_not_installed")
    except Exception as e:
        logger.warning("volume_control_fallback", reason="pycaw_failed", error=str(e), exc_info=True)

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
        logger.error("volume_key_simulation_failed", error=str(e), exc_info=True)
        return "I was unable to adjust the volume, sir."


def lock_workstation() -> str:
    """Locks the Windows computer screen immediately."""
    logger.info("workstation_lock_requested")
    try:
        ctypes.windll.user32.LockWorkStation()
        return "Workstation locked successfully, sir."
    except Exception as e:
        logger.error("workstation_lock_failed", error=str(e), exc_info=True)
        return f"Failed to lock workstation: {str(e)}"


def _get_allowed_apps() -> set:
    security = get_config().security
    allowed = security.allowed_apps
    if isinstance(allowed, list) and allowed:
        return {str(item).lower().strip() for item in allowed}
    return DEFAULT_ALLOWED_APPS


def _resolve_command(command: str) -> str:
    if not command.lower().endswith(".exe"):
        return command

    which_path = shutil.which(command)
    if which_path:
        return which_path

    if winreg is not None:
        reg_path = rf"Software\Microsoft\Windows\CurrentVersion\App Paths\{command}"
        for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
            try:
                with winreg.OpenKey(hive, reg_path) as key:
                    value, _ = winreg.QueryValueEx(key, "")
                    if value and os.path.exists(value):
                        return value
            except OSError:
                pass

    return command
