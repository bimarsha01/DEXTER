import time
import threading
from collections import deque


class MetricsCollector:
    def __init__(self):
        self._lock = threading.Lock()
        self._latencies = {}
        self._providers = {}

    def record_latency(self, stage: str, duration_ms: float, max_samples: int = 100) -> None:
        if duration_ms < 0:
            return
        with self._lock:
            if stage not in self._latencies:
                self._latencies[stage] = deque(maxlen=max_samples)
            self._latencies[stage].append(duration_ms)

    def update_provider_health(
        self,
        name: str,
        available: bool,
        score: float,
        cooldown_until: float = 0.0,
        last_error: str = "",
    ) -> None:
        with self._lock:
            self._providers[name] = {
                "available": available,
                "score": round(max(0.0, min(1.0, score)), 2),
                "cooldown_until": cooldown_until,
                "last_error": last_error[:120],
                "updated_at": time.time(),
            }

    def _format_latency(self, values: deque) -> str:
        if not values:
            return "n/a"
        data = sorted(values)
        avg = sum(data) / len(data)
        p95_index = max(0, int(len(data) * 0.95) - 1)
        p95 = data[p95_index]
        return f"avg {avg:.0f}ms, p95 {p95:.0f}ms"

    def get_health_report(self) -> str:
        lines = ["Dexter Health Report:"]

        with self._lock:
            if self._latencies:
                lines.append("Stage Latency:")
                for stage, values in self._latencies.items():
                    lines.append(f"- {stage}: {self._format_latency(values)}")
            else:
                lines.append("Stage Latency: no data yet")

            if self._providers:
                lines.append("Provider Health:")
                now = time.time()
                for name, info in self._providers.items():
                    cooldown = info.get("cooldown_until", 0.0)
                    if cooldown and cooldown > now:
                        cooldown_left = int(cooldown - now)
                        cooldown_text = f"cooldown {cooldown_left}s"
                    else:
                        cooldown_text = "ready"
                    status = "online" if info.get("available") else "offline"
                    score = info.get("score", 0.0)
                    err = info.get("last_error")
                    err_text = f" | last_error: {err}" if err else ""
                    lines.append(f"- {name}: {status}, score {score}, {cooldown_text}{err_text}")
            else:
                lines.append("Provider Health: no data yet")

        return "\n".join(lines)


metrics = MetricsCollector()
