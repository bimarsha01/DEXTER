"""In-memory session-scoped state for project context.

This module provides a simple per-process, per-user session store for the
`current_project` slot used by the Project & Document Q&A path. It is
intentionally in-memory (no disk persistence) so it does not survive process
restarts and remains session-scoped.
"""
from typing import Optional, Dict, Any
import threading
import getpass

_lock = threading.RLock()
_sessions: Dict[str, Dict[str, Any]] = {}


def _user_key(user: str | None = None) -> str:
    return (user or getpass.getuser() or "default").lower()


def set_current_project(
    name: str,
    resolved_path: str,
    confidence: float,
    set_at_turn: int | None,
    user: str | None = None,
) -> None:
    """Set the current project slot for the session/user."""
    key = _user_key(user)
    with _lock:
        s = _sessions.setdefault(key, {})
        s["current_project"] = {
            "name": name,
            "resolved_path": resolved_path,
            "confidence": float(confidence),
            "set_at_turn": set_at_turn,
        }


def get_current_project(user: str | None = None) -> Optional[Dict[str, Any]]:
    """Return the current project slot for the session/user or None."""
    key = _user_key(user)
    with _lock:
        return _sessions.get(key, {}).get("current_project")


def clear_current_project(user: str | None = None) -> None:
    """Clear the current project slot for the session/user."""
    key = _user_key(user)
    with _lock:
        if key in _sessions and "current_project" in _sessions[key]:
            del _sessions[key]["current_project"]


def clear_if_stale(current_turn: int, max_turns: int = 4, user: str | None = None) -> None:
    """Clear the current project slot if it was set more than `max_turns` ago."""
    slot = get_current_project(user)
    if not slot:
        return
    set_at = slot.get("set_at_turn", 0)
    if set_at is None:
        return
    set_at = int(set_at)
    # `set_at_turn` is recorded when the project context was last "refreshed".
    # Clearing is intended to happen after a fixed number of subsequent turns
    # (counting the refresh turn as part of the window), so we subtract 1.
    threshold = max(0, int(max_turns) - 1)
    if current_turn - set_at >= threshold:
        clear_current_project(user)
