from __future__ import annotations

import os
import platform
import threading
import time
from pathlib import Path
from typing import Any

import psutil

from core.event_bus import EventBus
from utils.config import get_workspace_root
from utils.logger import get_logger

logger = get_logger("hardware_watchdog")


class HardwareWatchdog:
    def __init__(self, config, event_bus: EventBus | None, stop_event: threading.Event):
        self.config = config
        self.event_bus = event_bus
        self._stop_event = stop_event
        self._shutdown_event = threading.Event()
        self._cpu_temp_over_since: float | None = None
        self._gpu_temp_over_since: float | None = None
        self._thread: threading.Thread | None = None
        self._emergency_latched = False
        self._poll_interval_warned = False
        _configured_interval = float(getattr(self.config, "poll_interval_sec", 3.0) or 3.0)
        if _configured_interval < 2.0:
            logger.warning(
                f"watchdog.poll_interval_sec={_configured_interval} is below the 2-second minimum — clamped to 2 seconds to prevent excessive CPU usage"
            )
            self._poll_interval_seconds = 2.0
        else:
            self._poll_interval_seconds = _configured_interval
        self._disk_root = self._resolve_disk_root()

    def _poll_interval(self) -> float:
        return float(self._poll_interval_seconds)

    def _resolve_disk_root(self) -> str:
        configured = getattr(self.config, "data_root", None)
        if configured:
            try:
                root = Path(str(configured)).expanduser()
                anchor = root.anchor or str(root)
                return anchor or str(root)
            except Exception:
                pass
        try:
            root = Path(get_workspace_root())
            anchor = root.anchor or str(root)
            return anchor or str(root)
        except Exception:
            return Path.cwd().anchor or str(Path.cwd())

    def start(self) -> None:
        if not bool(getattr(self.config, "watchdog_enabled", True)):
            logger.info("hardware_watchdog_disabled")
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._shutdown_event.clear()
        self._thread = threading.Thread(target=self._watch_loop, name="hardware_watchdog", daemon=True)
        self._thread.start()
        logger.info("hardware_watchdog_started", poll_interval_sec=self._poll_interval())

    def stop(self) -> None:
        self._shutdown_event.set()

    def join(self, timeout: float | None = None) -> None:
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)

    def _watch_loop(self) -> None:
        while not self._shutdown_event.is_set():
            if self._stop_event.is_set():
                self._wait_for_safe_resume()
                continue
            self._check_all()
            self._shutdown_event.wait(self._poll_interval())

    def _wait_for_safe_resume(self) -> None:
        safe_since: float | None = None
        while not self._shutdown_event.is_set() and self._stop_event.is_set():
            overall = self._check_all()
            if overall == "ok":
                if safe_since is None:
                    safe_since = time.time()
                elif time.time() - safe_since >= 60.0:
                    self._stop_event.clear()
                    self._emergency_latched = False
                    logger.info("Hardware returned to safe levels — resuming")
                    return
            else:
                safe_since = None
            self._shutdown_event.wait(10.0)

    def _check_all(self) -> str:
        statuses = [
            self._check_cpu_temp(),
            self._check_gpu_temp(),
            self._check_ram(),
            self._check_vram(),
            self._check_disk(),
        ]
        if "critical" in statuses:
            reason = ", ".join(status for status in statuses if status == "critical") or "hardware threshold exceeded"
            self._trigger_emergency_stop(reason)
            return "critical"
        if "warn" in statuses:
            return "warn"
        return "ok"

    def _check_cpu_temp(self) -> str:
        warn_c = float(getattr(self.config, "cpu_temp_warn_c", 85.0) or 85.0)
        critical_c = float(getattr(self.config, "cpu_temp_critical_c", 95.0) or 95.0)
        critical_duration = float(getattr(self.config, "cpu_temp_critical_duration_sec", 30.0) or 30.0)
        temp = self._read_cpu_temp_c()
        if temp is None:
            return "ok"
        if temp >= critical_c:
            if self._cpu_temp_over_since is None:
                self._cpu_temp_over_since = time.time()
            if time.time() - self._cpu_temp_over_since >= critical_duration:
                return "critical"
        else:
            if temp < warn_c:
                self._cpu_temp_over_since = None
        if temp >= warn_c:
            logger.warning(f"CPU temp high: {temp:.0f}°C")
            return "warn"
        return "ok"

    def _check_gpu_temp(self) -> str:
        warn_c = float(getattr(self.config, "gpu_temp_warn_c", 83.0) or 83.0)
        critical_c = float(getattr(self.config, "gpu_temp_critical_c", 92.0) or 92.0)
        critical_duration = float(getattr(self.config, "gpu_temp_critical_duration_sec", 30.0) or 30.0)
        temp = self._read_gpu_temp_c()
        if temp is None:
            return "ok"
        if temp >= critical_c:
            if self._gpu_temp_over_since is None:
                self._gpu_temp_over_since = time.time()
            if time.time() - self._gpu_temp_over_since >= critical_duration:
                return "critical"
        else:
            if temp < warn_c:
                self._gpu_temp_over_since = None
        if temp >= warn_c:
            logger.warning(f"GPU temp high: {temp:.0f}°C")
            return "warn"
        return "ok"

    def _check_ram(self) -> str:
        pct = float(psutil.virtual_memory().percent)
        if pct >= float(getattr(self.config, "ram_critical_pct", 96.0) or 96.0):
            return "critical"
        if pct >= float(getattr(self.config, "ram_warn_pct", 88.0) or 88.0):
            logger.warning(f"RAM usage high: {pct:.0f}%")
            return "warn"
        return "ok"

    def _check_vram(self) -> str:
        info = self._read_vram_info()
        if info is None:
            return "ok"
        used_pct = info[0]
        if used_pct >= float(getattr(self.config, "vram_critical_pct", 98.0) or 98.0):
            return "critical"
        if used_pct >= float(getattr(self.config, "vram_warn_pct", 90.0) or 90.0):
            logger.warning(f"VRAM usage high: {used_pct:.0f}%")
            return "warn"
        return "ok"

    def _check_disk(self) -> str:
        disk = psutil.disk_usage(self._disk_root)
        free_pct = (float(disk.free) / float(disk.total)) * 100.0 if disk.total else 0.0
        if free_pct <= float(getattr(self.config, "disk_free_critical_pct", 2.0) or 2.0):
            return "critical"
        if free_pct <= float(getattr(self.config, "disk_free_warn_pct", 5.0) or 5.0):
            logger.warning(f"Disk space low: {free_pct:.1f}% free")
            return "warn"
        return "ok"

    def _read_cpu_temp_c(self) -> float | None:
        try:
            temps = psutil.sensors_temperatures(fahrenheit=False)
            if temps:
                values: list[float] = []
                for entries in temps.values():
                    for entry in entries:
                        current = getattr(entry, "current", None)
                        if current is not None:
                            values.append(float(current))
                if values:
                    return max(values)
        except Exception:
            pass

        if platform.system().lower() == "windows":
            try:
                import wmi

                raw_values: list[float] = []
                for processor in wmi.WMI(namespace="root\wmi").MSAcpi_ThermalZoneTemperature():
                    raw = getattr(processor, "CurrentTemperature", None)
                    if raw is None:
                        continue
                    raw_values.append((float(raw) / 10.0) - 273.15)
                if raw_values:
                    return max(raw_values)
            except Exception:
                try:
                    import wmi

                    processors = wmi.WMI().Win32_Processor()
                    if processors:
                        raw = getattr(processors[0], "CurrentTemperature", None)
                        if raw is not None:
                            return (float(raw) / 10.0) - 273.15
                except Exception:
                    return None
        return None

    def _read_gpu_temp_c(self) -> float | None:
        try:
            import pynvml

            pynvml.nvmlInit()
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                return float(temp)
            finally:
                try:
                    pynvml.nvmlShutdown()
                except Exception:
                    pass
        except Exception:
            return None

    def _read_vram_info(self) -> tuple[float, float] | None:
        try:
            import pynvml

            pynvml.nvmlInit()
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                total = float(info.total)
                used = float(info.used)
                if total <= 0:
                    return None
                return (used / total) * 100.0, total
            finally:
                try:
                    pynvml.nvmlShutdown()
                except Exception:
                    pass
        except Exception:
            return None

    def _trigger_emergency_stop(self, reason: str) -> None:
        if self._emergency_latched:
            return
        self._emergency_latched = True
        logger.critical(f"HARDWARE SAFETY STOP: {reason} — signaling pipeline to halt")
        if self.event_bus is not None:
            try:
                self.event_bus.emit(
                    "hardware_emergency_stop",
                    {"reason": reason, "ts": time.time()},
                )
            except Exception:
                logger.debug("hardware_emergency_stop_emit_failed", error=reason, exc_info=True)
        self._stop_event.set()
        logger.critical("Stop signal sent. Waiting for pipeline to confirm stop...")
