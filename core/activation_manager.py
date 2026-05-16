"""
Smart activation manager.
Switches between wake-word and always-on modes based on:
- Configured active hours (work schedule)
- Explicit user commands ("stay active", "go passive")
- Recent interaction density
"""

from __future__ import annotations

import datetime
import time
from dataclasses import dataclass, field
from typing import Literal

from utils.logger import get_logger

logger = get_logger(__name__)

Mode = Literal["wake_word", "always_on", "smart"]


@dataclass
class ActivationConfig:
    mode: Mode = "smart"
    wake_word: str = "dexter"
    active_hours_start: str = "09:00"
    active_hours_end: str = "18:00"
    active_days: list[int] | None = None
    always_on_after_n_interactions: int = 3
    always_on_window_seconds: float = 120.0
    always_on_timeout_seconds: float = 300.0


class ActivationManager:
    def __init__(self, config: ActivationConfig):
        self._config = config
        self._mode_override: Mode | None = None
        self._override_until: float = 0.0
        self._recent_interactions: list[float] = []
        self._last_interaction: float = 0.0
        self._consecutive_drop_count: int = 0

    @property
    def current_mode(self) -> Mode:
        if self._mode_override and time.time() < self._override_until:
            return self._mode_override

        if self._config.mode == "always_on":
            return "always_on"
        if self._config.mode == "wake_word":
            return "wake_word"

        return self._smart_mode_decision()

    def _smart_mode_decision(self) -> Mode:
        now = time.time()

        if self._is_active_hours():
            return "always_on"

        cutoff = now - self._config.always_on_window_seconds
        recent = [t for t in self._recent_interactions if t > cutoff]
        if len(recent) >= self._config.always_on_after_n_interactions:
            return "always_on"

        if (
            self._last_interaction > 0
            and now - self._last_interaction < self._config.always_on_timeout_seconds
        ):
            return "always_on"

        return "wake_word"

    def _is_active_hours(self) -> bool:
        now = datetime.datetime.now()

        active_days = self._config.active_days
        if active_days is not None and now.weekday() not in active_days:
            return False

        try:
            start_h, start_m = map(int, self._config.active_hours_start.split(":"))
            end_h, end_m = map(int, self._config.active_hours_end.split(":"))
            start = now.replace(hour=start_h, minute=start_m, second=0, microsecond=0)
            end = now.replace(hour=end_h, minute=end_m, second=0, microsecond=0)
            if end <= start:
                return now >= start or now <= end
            return start <= now <= end
        except Exception:
            return False

    def requires_wake_word(self) -> bool:
        return self.current_mode == "wake_word"

    def record_interaction(self) -> None:
        now = time.time()
        self._last_interaction = now
        self._recent_interactions.append(now)
        self._consecutive_drop_count = 0
        cutoff = now - self._config.always_on_window_seconds
        self._recent_interactions = [t for t in self._recent_interactions if t > cutoff]
        logger.debug(
            "activation_interaction_recorded",
            mode=self.current_mode,
            recent_count=len(self._recent_interactions),
        )

    def record_drop(self) -> None:
        self._consecutive_drop_count += 1
        if self._consecutive_drop_count >= 3:
            logger.warning(
                "activation_consecutive_drops",
                count=self._consecutive_drop_count,
                mode=self.current_mode,
                hint="User may need to say the wake word or switch to smart mode",
            )

    def set_override(self, mode: Mode, duration_seconds: float) -> None:
        self._mode_override = mode
        self._override_until = time.time() + duration_seconds
        logger.info(
            "activation_override_set",
            mode=mode,
            duration_seconds=duration_seconds,
        )

    def clear_override(self) -> None:
        self._mode_override = None
        self._override_until = 0.0

    def get_status(self) -> dict:
        return {
            "effective_mode": self.current_mode,
            "config_mode": self._config.mode,
            "override": self._mode_override,
            "is_active_hours": self._is_active_hours(),
            "recent_interactions": len(self._recent_interactions),
            "last_interaction_ago_seconds": (
                round(time.time() - self._last_interaction)
                if self._last_interaction
                else None
            ),
        }
