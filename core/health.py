from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from utils.logger import get_logger

logger = get_logger("health")

# Checks not updated within this window are auto-degraded to "stale".
STALENESS_THRESHOLD_SECONDS: float = 300.0  # 5 minutes

provider_health: dict[str, dict[str, Any]] = {}
_PROVIDER_HEALTH_LOCK = threading.RLock()


@dataclass
class HealthCheck:
    name: str
    status: str = "unknown"
    details: str = ""
    latency_ms: float | None = None
    recoverable: bool = True
    updated_at: float = field(default_factory=time.time)


class HealthMonitor:
    def __init__(self, service_name: str = "Dexter") -> None:
        self.service_name = service_name
        self._lock = threading.RLock()
        self._checks: dict[str, HealthCheck] = {}
        self._turn_stage_timings_ms: dict[str, deque[float]] = {}
        logger.info("health_monitor_initialized", service_name=service_name)

    def update(
        self,
        name: str,
        status: str,
        details: str = "",
        latency_ms: float | None = None,
        recoverable: bool = True,
    ) -> None:
        with self._lock:
            self._checks[name] = HealthCheck(
                name=name,
                status=status,
                details=details,
                latency_ms=latency_ms,
                recoverable=recoverable,
                updated_at=time.time(),
            )

    def healthy(self, name: str, details: str = "", latency_ms: float | None = None) -> None:
        self.update(name, "healthy", details=details, latency_ms=latency_ms)

    def degraded(self, name: str, details: str = "", latency_ms: float | None = None) -> None:
        self.update(name, "degraded", details=details, latency_ms=latency_ms)

    def unhealthy(self, name: str, details: str = "", latency_ms: float | None = None) -> None:
        self.update(name, "unhealthy", details=details, latency_ms=latency_ms, recoverable=False)

    def record_turn_stage(self, stage: str, duration_ms: float) -> None:
        with self._lock:
            timings = self._turn_stage_timings_ms.setdefault(stage, deque(maxlen=10))
            timings.append(float(duration_ms))

    def turn_stage_averages(self) -> dict[str, dict[str, float | int]]:
        with self._lock:
            return {
                stage: {
                    "average_ms": (sum(values) / len(values)) if values else 0.0,
                    "sample_count": len(values),
                }
                for stage, values in self._turn_stage_timings_ms.items()
            }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            now = time.time()
            checks: dict[str, dict[str, Any]] = {}
            for name, check in self._checks.items():
                entry = self._serialize(check)
                # Auto-degrade stale checks that haven't reported recently
                age = now - check.updated_at
                if age > STALENESS_THRESHOLD_SECONDS and check.status == "healthy":
                    entry["status"] = "stale"
                    entry["details"] = (
                        f"no update for {age:.0f}s (threshold {STALENESS_THRESHOLD_SECONDS:.0f}s)"
                    )
                    logger.debug(
                        "health_check_stale",
                        component=name,
                        age_seconds=f"{age:.0f}",
                    )
                checks[name] = entry
            healthy = (
                all(c["status"] == "healthy" for c in checks.values())
                if checks
                else True
            )
            return {
                "service_name": self.service_name,
                "healthy": healthy,
                "checks": checks,
                "turn_stage_averages_ms": self.turn_stage_averages(),
                "provider_health": get_provider_health(),
                "updated_at": now,
            }

    def render_report(self) -> str:
        snapshot = self.snapshot()
        lines = [f"Health Report: {snapshot['service_name']}"]
        if not snapshot["checks"]:
            lines.append("- no component checks recorded yet")
            return "\n".join(lines)
        for name, check in snapshot["checks"].items():
            latency = f", latency {check['latency_ms']:.0f}ms" if check["latency_ms"] is not None else ""
            detail = f", {check['details']}" if check["details"] else ""
            lines.append(f"- {name}: {check['status']}{latency}{detail}")
        stage_avgs = snapshot.get("turn_stage_averages_ms") or {}
        if stage_avgs:
            lines.append("- turn stage averages (last 10 turns):")
            for stage, values in stage_avgs.items():
                lines.append(
                    f"  - {stage}: {values['average_ms']:.0f}ms over {values['sample_count']} samples"
                )
        providers = snapshot.get("provider_health") or {}
        if providers:
            lines.append("- provider health:")
            for name, info in providers.items():
                lines.append(
                    f"  - {name}: {info.get('current_status', 'unknown')}, failures {info.get('failure_count', 0)}"
                )
        return "\n".join(lines)

    @staticmethod
    def _serialize(check: HealthCheck) -> dict[str, Any]:
        return {
            "name": check.name,
            "status": check.status,
            "details": check.details,
            "latency_ms": check.latency_ms,
            "recoverable": check.recoverable,
            "updated_at": check.updated_at,
        }


_GLOBAL_HEALTH_MONITOR: HealthMonitor | None = None


def set_global_health_monitor(monitor: HealthMonitor | None) -> None:
    global _GLOBAL_HEALTH_MONITOR
    _GLOBAL_HEALTH_MONITOR = monitor


def get_global_health_monitor() -> HealthMonitor | None:
    return _GLOBAL_HEALTH_MONITOR


def update_provider_health(
    name: str,
    current_status: str,
    *,
    success: bool = False,
    cooldown_until: float = 0.0,
    last_error: str = "",
) -> dict[str, Any]:
    now = time.time()
    key = (name or "unknown").lower()
    with _PROVIDER_HEALTH_LOCK:
        entry = provider_health.get(
            key,
            {
                "last_success_ts": 0.0,
                "last_failure_ts": 0.0,
                "failure_count": 0,
                "current_status": "unknown",
                "cooldown_until": 0.0,
                "last_error": "",
            },
        )
        entry["current_status"] = str(current_status or "unknown")
        entry["cooldown_until"] = float(cooldown_until or 0.0)
        entry["last_error"] = (last_error or "")[:120]
        if success:
            entry["last_success_ts"] = now
            entry["failure_count"] = 0
        else:
            entry["last_failure_ts"] = now
            entry["failure_count"] = int(entry.get("failure_count", 0)) + 1
        provider_health[key] = entry
        return dict(entry)


def get_provider_health() -> dict[str, dict[str, Any]]:
    with _PROVIDER_HEALTH_LOCK:
        return {name: dict(info) for name, info in provider_health.items()}
