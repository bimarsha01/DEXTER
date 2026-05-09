from __future__ import annotations

import threading
import time


class SessionActivity:
    """Track whether the assistant is actively handling a live user turn."""

    def __init__(self) -> None:
        self._is_active = False
        self._last_active = 0.0
        self._lock = threading.RLock()

    def mark_active(self) -> None:
        """Mark the current session as actively processing a turn."""
        with self._lock:
            self._is_active = True
            self._last_active = time.time()

    def mark_idle(self) -> None:
        """Mark the current session as idle after completing a turn."""
        with self._lock:
            self._is_active = False

    def is_session_idle(self, idle_threshold_seconds: float = 30.0) -> bool:
        """Return whether the session has been inactive for at least the threshold."""
        with self._lock:
            return (not self._is_active) and ((time.time() - self._last_active) > float(idle_threshold_seconds))


session_activity = SessionActivity()
