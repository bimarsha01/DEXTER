"""
Dexter System Tools — Jarvis-style utilities for time, weather, system info,
clipboard, screenshots, and power management.
"""
import os
import subprocess
import time
import urllib.request
import urllib.parse
from datetime import datetime
from utils.logger import get_logger
from utils.config import get_config
from utils.metrics import metrics

logger = get_logger("system_tools")


def get_current_datetime() -> str:
    """Returns the current date, day of the week, and time. Use this when the user asks what time or date it is."""
    now = datetime.now()
    return (
        f"Today is {now.strftime('%A, %B %d, %Y')}. "
        f"The current time is {now.strftime('%I:%M %p')}."
    )


def get_weather(city: str) -> str:
    """Gets the current weather conditions for a given city name. Use this when the user asks about weather."""
    try:
        safe_city = urllib.parse.quote(city.strip())
        url = f"https://wttr.in/{safe_city}?format=%C+%t+Humidity:+%h+Wind:+%w"
        req = urllib.request.Request(url, headers={"User-Agent": "Dexter-AI-Assistant/1.0"})
        response = urllib.request.urlopen(req, timeout=8)
        weather_data = response.read().decode("utf-8").strip()
        return f"Weather in {city}: {weather_data}"
    except urllib.error.URLError:
        return f"I could not reach the weather service for {city}, sir. Please check your internet connection."
    except Exception as e:
        logger.error("weather_fetch_failed", error=str(e), exc_info=True)
        return f"I was unable to retrieve the weather for {city} at this time."


def get_system_status() -> str:
    """Returns current PC system information including CPU usage, RAM usage, and battery level. Use when the user asks about system status or performance."""
    info_parts = []

    # CPU Usage
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Processor).LoadPercentage"],
            capture_output=True, text=True, timeout=8
        )
        cpu = result.stdout.strip()
        if cpu:
            info_parts.append(f"CPU Usage: {cpu}%")
    except Exception as e:
        logger.debug("system_status_cpu_unavailable", error=str(e))
        info_parts.append("CPU: unavailable")

    # RAM Usage
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "$m = Get-CimInstance Win32_OperatingSystem; "
             "$used = [math]::Round(($m.TotalVisibleMemorySize - $m.FreePhysicalMemory) / 1MB, 1); "
             "$total = [math]::Round($m.TotalVisibleMemorySize / 1MB, 1); "
             "\"$used GB used of $total GB total\""],
            capture_output=True, text=True, timeout=8
        )
        ram = result.stdout.strip()
        if ram:
            info_parts.append(f"RAM: {ram}")
    except Exception as e:
        logger.debug("system_status_ram_unavailable", error=str(e))
        info_parts.append("RAM: unavailable")

    # Battery
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "$b = Get-CimInstance Win32_Battery; "
             "if ($b) { \"$($b.EstimatedChargeRemaining)% - $($b.BatteryStatus)\" } else { 'No battery' }"],
            capture_output=True, text=True, timeout=8
        )
        battery = result.stdout.strip()
        if battery:
            info_parts.append(f"Battery: {battery}")
    except Exception as e:
        logger.debug("system_status_battery_unavailable", error=str(e))

    return "System Status: " + " | ".join(info_parts) if info_parts else "Could not retrieve system information."


def read_clipboard() -> str:
    """Reads and returns the current text content from the Windows clipboard."""
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "Get-Clipboard"],
            capture_output=True, text=True, timeout=5
        )
        content = result.stdout.strip()
        if content:
            return f"Clipboard contains: {content}"
        return "The clipboard is currently empty, sir."
    except Exception as e:
        logger.error("clipboard_read_failed", error=str(e), exc_info=True)
        return "I was unable to read the clipboard."


def copy_to_clipboard(text: str) -> str:
    """Copies the given text string to the Windows clipboard so the user can paste it."""
    try:
        # Use PowerShell Set-Clipboard with stdin to avoid quoting issues
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", "Set-Clipboard -Value $input"],
            input=text,
            capture_output=True,
            text=True,
            timeout=5,
        )
        preview = text[:80] + "..." if len(text) > 80 else text
        return f"Copied to clipboard: {preview}"
    except Exception as e:
        logger.error("clipboard_write_failed", error=str(e), exc_info=True)
        return "I was unable to copy the text to the clipboard."


def take_screenshot() -> str:
    """Takes a screenshot of the entire screen and saves it to the user's Desktop."""
    try:
        import pyautogui
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        filepath = os.path.join(desktop, f"dexter_screenshot_{timestamp}.png")

        screenshot = pyautogui.screenshot()
        screenshot.save(filepath)
        logger.info("screenshot_saved", path=filepath)
        return f"Screenshot saved to your Desktop as dexter_screenshot_{timestamp}.png"
    except Exception as e:
        logger.error("screenshot_failed", error=str(e), exc_info=True)
        return f"I was unable to capture the screenshot: {str(e)}"


def shutdown_pc(confirm: bool = False) -> str:
    """Shuts down the computer after a 30-second delay. The user can cancel with 'cancel shutdown'."""
    logger.info("system_shutdown_scheduled", delay_seconds=30)
    try:
        if _requires_confirm() and not confirm:
            return "Shutdown requested. Please confirm by saying 'confirm shutdown'."
        os.system("shutdown /s /t 30 /c \"Dexter is shutting down the system in 30 seconds.\"")
        return "Initiating shutdown in 30 seconds, sir. Say 'cancel shutdown' if you change your mind."
    except Exception as e:
        logger.error("system_shutdown_failed", error=str(e), exc_info=True)
        return f"Failed to initiate shutdown: {str(e)}"


def restart_pc(confirm: bool = False) -> str:
    """Restarts the computer after a 30-second delay. The user can cancel with 'cancel shutdown'."""
    logger.info("system_restart_scheduled", delay_seconds=30)
    try:
        if _requires_confirm() and not confirm:
            return "Restart requested. Please confirm by saying 'confirm restart'."
        os.system("shutdown /r /t 30 /c \"Dexter is restarting the system in 30 seconds.\"")
        return "Initiating restart in 30 seconds, sir. Say 'cancel shutdown' to abort."
    except Exception as e:
        logger.error("system_restart_failed", error=str(e), exc_info=True)
        return f"Failed to initiate restart: {str(e)}"


def cancel_shutdown() -> str:
    """Cancels any pending shutdown or restart command."""
    logger.info("system_shutdown_cancel_requested")
    try:
        os.system("shutdown /a")
        return "Shutdown cancelled successfully, sir."
    except Exception as e:
        logger.error("system_shutdown_cancel_failed", error=str(e), exc_info=True)
        return f"Could not cancel shutdown: {str(e)}"


def sleep_pc(confirm: bool = False) -> str:
    """Puts the computer into sleep mode immediately."""
    logger.info("system_sleep_requested")
    try:
        if _requires_confirm() and not confirm:
            return "Sleep requested. Please confirm by saying 'confirm sleep'."
        os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")
        return "Putting the system to sleep now, sir."
    except Exception as e:
        logger.error("system_sleep_failed", error=str(e), exc_info=True)
        return f"Failed to sleep the system: {str(e)}"


def get_health_report() -> str:
    """Returns Dexter's current health report with latency and provider status."""
    return metrics.get_health_report()


def _requires_confirm() -> bool:
    return bool(get_config().security.require_confirm_power_actions)
