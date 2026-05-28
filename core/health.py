from __future__ import annotations

import threading
import time
from dataclasses import asdict, is_dataclass
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

from utils.config import HealthPolicy, get_config
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


@dataclass
class GPUStatus:
    status: str = "unknown"
    device_name: str = ""
    compute_capability: str = ""
    total_vram_gb: float = 0.0
    free_vram_gb: float = 0.0
    cuda_available: bool = False
    expected_cuda: bool = False
    details: str = ""
    updated_at: float = field(default_factory=time.time)


@dataclass
class RAGStatus:
    status: str = "empty"
    doc_count: int = 0
    last_updated_ts: float = 0.0
    embedding_model_loaded: bool = False
    index_exists: bool = False
    details: str = ""
    updated_at: float = field(default_factory=time.time)


@dataclass
class HealthSummary:
    service_name: str
    overall_status: str
    gpu: dict[str, Any]
    rag: dict[str, Any]
    automation: dict[str, Any]
    checks: dict[str, dict[str, Any]]
    providers: dict[str, dict[str, Any]]
    policy: dict[str, Any]
    corrective_actions: list[dict[str, Any]]
    updated_at: float


class HealthMonitor:
    def __init__(self, service_name: str = "Dexter", automation_available: bool = False) -> None:
        self.service_name = service_name
        self._lock = threading.RLock()
        self._evaluation_lock = threading.RLock()
        self._evaluation_stop = threading.Event()
        self._evaluation_thread: threading.Thread | None = None
        self._runtime_config: Any | None = None
        self._memory_vault: Any | None = None
        self._event_bus: Any | None = None
        self._rag_reindex_inflight = False
        self._last_system_degraded_components: tuple[str, ...] | None = None
        self._degraded_since: Optional[float] = None
        self._degraded_episode_fired: bool = False
        self._automation_available = bool(automation_available)
        self._checks: dict[str, HealthCheck] = {}
        self._turn_stage_timings_ms: dict[str, deque[float]] = {}
        self.gpu = GPUStatus()
        self.rag = RAGStatus()
        self.health_policy = HealthPolicy()
        self._corrective_actions: list[dict[str, Any]] = []
        logger.info("health_monitor_initialized", service_name=service_name)

    def update(
        self,
        name: str,
        status: str,
        details: str = "",
        latency_ms: float | None = None,
        recoverable: bool = True,
    ) -> None:
        now = time.time()
        with self._lock:
            self._checks[name] = HealthCheck(
                name=name,
                status=status,
                details=details,
                latency_ms=latency_ms,
                recoverable=recoverable,
                updated_at=now,
            )
            if name == "rag":
                self.rag.status = status
                self.rag.details = details
                self.rag.updated_at = now
            elif name == "gpu":
                self.gpu.status = status
                self.gpu.details = details
                self.gpu.updated_at = now

    def healthy(self, name: str, details: str = "", latency_ms: float | None = None) -> None:
        self.update(name, "healthy", details=details, latency_ms=latency_ms)

    def degraded(self, name: str, details: str = "", latency_ms: float | None = None) -> None:
        self.update(name, "degraded", details=details, latency_ms=latency_ms)

    def unhealthy(self, name: str, details: str = "", latency_ms: float | None = None) -> None:
        self.update(name, "unhealthy", details=details, latency_ms=latency_ms, recoverable=False)

    def set_gpu_status(self, status: GPUStatus) -> None:
        with self._lock:
            self.gpu = status
            self._checks["gpu"] = HealthCheck(
                name="gpu",
                status=status.status,
                details=status.details,
                updated_at=status.updated_at,
            )

    def set_rag_status(self, status: RAGStatus) -> None:
        with self._lock:
            self.rag = status
            self._checks["rag"] = HealthCheck(
                name="rag",
                status=status.status,
                details=status.details,
                updated_at=status.updated_at,
            )

    def attach_runtime_context(self, runtime_config: Any | None = None, memory_vault: Any | None = None, event_bus: Any | None = None) -> None:
        with self._lock:
            self._runtime_config = runtime_config
            self._memory_vault = memory_vault
            self._event_bus = event_bus
            try:
                policy = getattr(runtime_config, "health_policy", None) if runtime_config is not None else None
                self.health_policy = policy if policy is not None else HealthPolicy()
            except Exception:
                self.health_policy = HealthPolicy()

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

    def start_evaluation_loop(self, interval_seconds: float = 300.0) -> None:
        if self._evaluation_thread is not None and self._evaluation_thread.is_alive():
            return

        self._evaluation_stop.clear()

        def _loop() -> None:
            while not self._evaluation_stop.wait(max(1.0, float(interval_seconds))):
                try:
                    self.evaluate()
                except Exception as exc:
                    logger.warning("health_evaluation_loop_failed", error=str(exc), exc_info=True)

        self._evaluation_thread = threading.Thread(target=_loop, daemon=True, name=f"health_eval_{self.service_name}")
        self._evaluation_thread.start()
        logger.info("health_evaluation_loop_started", service_name=self.service_name, interval_seconds=float(interval_seconds))

    def stop_evaluation_loop(self) -> None:
        self._evaluation_stop.set()

    def _current_policy(self) -> HealthPolicy:
        with self._lock:
            try:
                policy = getattr(self._runtime_config, "health_policy", None)
                if policy is not None:
                    self.health_policy = policy
            except Exception:
                pass
            return self.health_policy

    def _serialize_policy(self) -> dict[str, Any]:
        policy = self._current_policy()
        if is_dataclass(policy):
            return asdict(policy)
        if hasattr(policy, "model_dump"):
            try:
                return policy.model_dump()
            except Exception:
                pass
        if hasattr(policy, "dict"):
            try:
                return policy.dict()
            except Exception:
                pass
        try:
            return dict(policy)
        except Exception:
            return {}

    def _record_corrective_action(self, action: str, reason: str, **fields: Any) -> None:
        entry = {"action": action, "reason": reason, "ts": time.time(), **fields}
        with self._lock:
            self._corrective_actions.append(entry)
            self._corrective_actions = self._corrective_actions[-50:]

    def _trigger_background_rag_reindex(self, reason: str) -> None:
        rag_proxy = None
        with self._lock:
            if self._rag_reindex_inflight:
                logger.info("health_action_rag_reindex_skipped", reason="already_inflight")
                return
            self._rag_reindex_inflight = True
            rag_proxy = self._memory_vault
        if rag_proxy is None:
            with self._lock:
                self._rag_reindex_inflight = False
            return

        candidate = getattr(rag_proxy, "personal_rag", None)
        if candidate is None:
            return

        reindex_target = getattr(candidate, "reindex_all", None)
        if reindex_target is None:
            reindex_target = getattr(getattr(candidate, "_index", None), "reindex_all", None)
        if reindex_target is None:
            return

        def _run_reindex() -> None:
            try:
                reindex_target()
                logger.info("health_action_rag_reindex_completed", reason=reason)
            except Exception as exc:
                logger.warning("health_action_rag_reindex_failed", reason=reason, error=str(exc), exc_info=True)
            finally:
                with self._lock:
                    self._rag_reindex_inflight = False

        threading.Thread(target=_run_reindex, daemon=True, name=f"rag_reindex_{self.service_name}").start()
        logger.info("health_action_rag_reindex_scheduled", reason=reason)
        self._record_corrective_action("rag_reindex", reason)

    def _reduce_whisper_batch_size(self, reason: str) -> None:
        cfg = self._runtime_config
        if cfg is None:
            try:
                cfg = get_config()
            except Exception:
                return

        audio_cfg = getattr(cfg, "audio_settings", None)
        if audio_cfg is None:
            return

        current = int(getattr(audio_cfg, "whisper_batch_size", 16) or 16)
        if current <= 1:
            return

        new_value = max(1, current // 2)
        if new_value >= current:
            return

        audio_cfg.whisper_batch_size = new_value
        logger.info(
            "health_action_whisper_batch_reduced",
            reason=reason,
            old_batch_size=current,
            new_batch_size=new_value,
        )
        self._record_corrective_action("whisper_batch_reduced", reason, old_batch_size=current, new_batch_size=new_value)

    def _mark_provider_degraded(self, provider_name: str, failure_rate: float, threshold: float, reason: str) -> None:
        key = (provider_name or "unknown").lower()
        now = time.time()
        with _PROVIDER_HEALTH_LOCK:
            entry = provider_health.get(
                key,
                {
                    "last_success_ts": 0.0,
                    "last_failure_ts": 0.0,
                    "failure_count": 0,
                    "success_count": 0,
                    "total_attempts": 0,
                    "failure_rate": 0.0,
                    "current_status": "unknown",
                    "cooldown_until": 0.0,
                    "last_error": "",
                },
            )
            entry["current_status"] = "degraded"
            entry["last_error"] = reason[:120]
            entry["updated_at"] = now
            provider_health[key] = entry

        self.degraded(f"provider:{key}", f"failure rate {failure_rate:.2f} exceeded threshold {threshold:.2f}")
        logger.warning(
            "provider_failure_rate_exceeded",
            provider=key,
            failure_rate=round(failure_rate, 2),
            threshold=round(threshold, 2),
            reason=reason,
        )
        logger.info(
            "health_action_provider_marked_degraded",
            provider=key,
            failure_rate=round(failure_rate, 2),
            threshold=round(threshold, 2),
            reason=reason,
        )
        self._record_corrective_action(
            "provider_degraded",
            reason,
            provider=key,
            failure_rate=failure_rate,
            threshold=threshold,
        )

    def _emit_system_degraded(self, components: list[str], reason: str) -> None:
        payload = {
            "components": components,
            "reason": reason,
            "summary": self.get_health_summary(),
        }
        if self._event_bus is not None:
            try:
                self._event_bus.emit("system_degraded", payload)
            except Exception as exc:
                logger.warning("system_degraded_event_failed", error=str(exc), exc_info=True)
        logger.info("health_action_system_degraded_emitted", reason=reason, components=components)
        self._record_corrective_action("system_degraded_emitted", reason, components=components)

    def _maybe_emit_system_degraded(self, components: list[str], reason: str) -> None:
        signature = tuple(sorted(set(components)))
        if not signature:
            self._last_system_degraded_components = None
            return
        if self._last_system_degraded_components == signature:
            return
        self._last_system_degraded_components = signature
        self._emit_system_degraded(list(signature), reason)

    def evaluate(self) -> dict[str, Any]:
        with self._evaluation_lock:
            now = time.time()
            policy = self._current_policy()
            actions: list[str] = []

            # RAG staleness correction.
            rag_last_update = float(getattr(self.rag, "last_updated_ts", 0.0) or 0.0)
            rag_age_hours = ((now - rag_last_update) / 3600.0) if rag_last_update > 0 else 0.0
            if self.rag.index_exists and self.rag.doc_count > 0 and rag_age_hours > float(policy.max_rag_staleness_hours):
                reason = (
                    f"RAG age {rag_age_hours:.1f}h exceeded threshold {float(policy.max_rag_staleness_hours):.1f}h"
                )
                logger.info("health_action_rag_reindex_requested", reason=reason)
                self._trigger_background_rag_reindex(reason)
                actions.append("rag_reindex")

            # Provider degradation correction.
            for provider_name, info in get_provider_health().items():
                total_attempts = int(info.get("total_attempts", 0) or 0)
                failure_count = int(info.get("failure_count", 0) or 0)
                failure_rate = float(info.get("failure_rate", 0.0) or 0.0)
                if total_attempts > 0 and failure_rate <= 0.0:
                    failure_rate = failure_count / float(total_attempts)
                if total_attempts <= 0:
                    continue
                if failure_rate > float(policy.max_provider_failure_rate):
                    reason = f"provider {provider_name} failure rate {failure_rate:.2f} exceeded threshold {float(policy.max_provider_failure_rate):.2f}"
                    self._mark_provider_degraded(provider_name, failure_rate, float(policy.max_provider_failure_rate), reason)
                    actions.append(f"provider:{provider_name}:degraded")

            # VRAM pressure correction.
            free_vram_gb = float(getattr(self.gpu, "free_vram_gb", 0.0) or 0.0)
            if self.gpu.cuda_available and free_vram_gb > 0.0 and free_vram_gb < float(policy.min_vram_gb):
                reason = f"free VRAM {free_vram_gb:.1f}GB below threshold {float(policy.min_vram_gb):.1f}GB"
                logger.info("health_action_vram_pressure_detected", reason=reason)
                self._reduce_whisper_batch_size(reason)
                actions.append("whisper_batch_reduced")

            # Critical staleness alert.
            critical_components: list[str] = []
            for name, check in self._checks.items():
                if name not in {"startup", "stt", "vad", "memory", "brain", "tts", "rag", "gpu", "proactive"}:
                    continue
                age_seconds = now - float(check.updated_at or 0.0)
                if check.status == "stale" and age_seconds > 30 * 60:
                    critical_components.append(name)

            if self.rag.status == "stale":
                rag_age_seconds = now - float(self.rag.last_updated_ts or self.rag.updated_at or 0.0)
                if rag_age_seconds > 30 * 60:
                    critical_components.append("rag")

            if self.gpu.status == "stale":
                gpu_age_seconds = now - float(self.gpu.updated_at or 0.0)
                if gpu_age_seconds > 30 * 60:
                    critical_components.append("gpu")

            degraded_threshold_min = float(getattr(self._current_policy(), "degraded_threshold_min", 30.0) or 30.0)
            is_degraded = bool(critical_components)

            if is_degraded:
                if self._degraded_since is None:
                    self._degraded_since = time.monotonic()
                elapsed_min = (time.monotonic() - self._degraded_since) / 60.0
                if elapsed_min >= degraded_threshold_min and not self._degraded_episode_fired:
                    degraded_components = sorted(set(critical_components))
                    self._emit_system_degraded(
                        degraded_components,
                        f"system degraded for {elapsed_min:.1f} min",
                    )
                    logger.critical(f"System degraded for {elapsed_min:.1f} min — components: {degraded_components}")
                    self._degraded_episode_fired = True
                    actions.append("system_degraded_emitted")
            else:
                if self._degraded_episode_fired:
                    logger.info("System health restored — degradation episode ended")
                self._degraded_since = None
                self._degraded_episode_fired = False
                self._last_system_degraded_components = None

            summary = self.get_health_summary()
            summary["corrective_actions_last_run"] = actions
            summary["evaluated_at"] = now
            return summary

    def get_health_summary(self) -> dict[str, Any]:
        snapshot = self.snapshot()
        providers = snapshot.get("provider_health") or {}
        enriched_providers: dict[str, dict[str, Any]] = {}
        for name, info in providers.items():
            total_attempts = int(info.get("total_attempts", 0) or 0)
            failure_count = int(info.get("failure_count", 0) or 0)
            success_count = int(info.get("success_count", 0) or 0)
            failure_rate = float(info.get("failure_rate", 0.0) or 0.0)
            if total_attempts > 0 and failure_rate <= 0.0:
                failure_rate = failure_count / float(total_attempts)
            enriched_providers[name] = {
                **info,
                "failure_rate": round(failure_rate, 3),
                "total_attempts": total_attempts,
                "success_count": success_count,
            }

        overall_status = "healthy"
        for check in snapshot.get("checks", {}).values():
            if check.get("status") in {"unhealthy", "degraded"}:
                overall_status = "degraded"
                break
            if check.get("status") == "stale" and overall_status == "healthy":
                overall_status = "degraded"

        rag = dict(snapshot.get("rag") or {})
        gpu = dict(snapshot.get("gpu") or {})
        if rag.get("status") in {"stale", "warming", "empty"} or gpu.get("status") in {"unavailable", "degraded", "stale"}:
            overall_status = "degraded" if overall_status == "healthy" else overall_status

        return {
            "service_name": snapshot.get("service_name", self.service_name),
            "overall_status": overall_status,
            "gpu": gpu,
            "rag": rag,
            "automation": {
                "status": "ready" if self._automation_available else "unavailable",
            },
            "checks": snapshot.get("checks", {}),
            "providers": enriched_providers,
            "policy": self._serialize_policy(),
            "corrective_actions": list(self._corrective_actions),
            "turn_stage_averages_ms": snapshot.get("turn_stage_averages_ms", {}),
            "updated_at": snapshot.get("updated_at", time.time()),
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
                "gpu": asdict(self.gpu),
                "rag": asdict(self.rag),
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
                failure_rate = float(info.get("failure_rate", 0.0) or 0.0)
                lines.append(
                    f"  - {name}: {info.get('current_status', 'unknown')}, failures {info.get('failure_count', 0)}, rate {failure_rate:.2f}"
                )
        gpu = snapshot.get("gpu") or {}
        if gpu:
            lines.append(
                f"- gpu: {gpu.get('status', 'unknown')}, {gpu.get('device_name', '')}"
            )
        rag = snapshot.get("rag") or {}
        if rag:
            lines.append(
                f"- rag: {rag.get('status', 'unknown')}, docs {rag.get('doc_count', 0)}"
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
                "success_count": 0,
                "total_attempts": 0,
                "failure_rate": 0.0,
                "current_status": "unknown",
                "cooldown_until": 0.0,
                "last_error": "",
            },
        )
        entry["current_status"] = str(current_status or "unknown")
        entry["cooldown_until"] = float(cooldown_until or 0.0)
        entry["last_error"] = (last_error or "")[:120]
        entry["total_attempts"] = int(entry.get("total_attempts", 0)) + 1
        if success:
            entry["last_success_ts"] = now
            entry["success_count"] = int(entry.get("success_count", 0)) + 1
        else:
            entry["last_failure_ts"] = now
            entry["failure_count"] = int(entry.get("failure_count", 0)) + 1
        total_attempts = max(1, int(entry.get("total_attempts", 1)))
        entry["failure_rate"] = float(entry.get("failure_count", 0)) / float(total_attempts)
        provider_health[key] = entry
        return dict(entry)


def get_provider_health() -> dict[str, dict[str, Any]]:
    with _PROVIDER_HEALTH_LOCK:
        return {name: dict(info) for name, info in provider_health.items()}
