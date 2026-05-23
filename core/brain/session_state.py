"""Persistent session-scoped context storage for project and conversation state.

The module exposes compatibility helpers for the existing current-project API
while storing session state in SQLite so it survives restarts.
"""
from __future__ import annotations

import getpass
import json
import os
import sqlite3
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from utils.config import get_config
from utils.logger import get_logger

logger = get_logger("session_state")


def _user_key(user: str | None = None) -> str:
    return (user or getpass.getuser() or "default").lower()


@dataclass
class ProjectContext:
    name: str
    source_path: str
    confidence: float
    last_confirmed_ts: float
    user_scope: str

    def to_row(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source_path": self.source_path,
            "confidence": float(self.confidence),
            "last_confirmed_ts": float(self.last_confirmed_ts),
            "user_scope": self.user_scope,
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row, user_scope: str) -> "ProjectContext | None":
        if row is None:
            return None
        name = row["project_name"]
        source_path = row["project_source_path"]
        if not name and not source_path:
            return None
        return cls(
            name=str(name or ""),
            source_path=str(source_path or ""),
            confidence=float(row["project_confidence"] or 0.0),
            last_confirmed_ts=float(row["project_last_confirmed_ts"] or 0.0),
            user_scope=user_scope,
        )


@dataclass
class UserPreferences:
    verbosity: str = "normal"
    tone: str = "neutral"
    preference_change_count: int = 0
    correction_count: int = 0
    last_updated_ts: float = 0.0

    def __post_init__(self) -> None:
        verbosity = str(self.verbosity or "normal").strip().lower()
        tone = str(self.tone or "neutral").strip().lower()
        if verbosity not in {"brief", "normal", "detailed"}:
            verbosity = "normal"
        if tone not in {"casual", "neutral"}:
            tone = "neutral"
        self.verbosity = verbosity
        self.tone = tone
        self.preference_change_count = max(0, int(self.preference_change_count or 0))
        self.correction_count = max(0, int(self.correction_count or 0))
        self.last_updated_ts = float(self.last_updated_ts or 0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verbosity": self.verbosity,
            "tone": self.tone,
            "preference_change_count": int(self.preference_change_count),
            "correction_count": int(self.correction_count),
            "last_updated_ts": float(self.last_updated_ts),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "UserPreferences":
        payload = data or {}
        preference_change_count = payload.get("preference_change_count")
        if preference_change_count is None and "correction_count" in payload:
            preference_change_count = payload.get("correction_count")
        return cls(
            verbosity=str(payload.get("verbosity", "normal") or "normal"),
            tone=str(payload.get("tone", "neutral") or "neutral"),
            preference_change_count=int(preference_change_count or 0),
            correction_count=int(payload.get("correction_count", 0) or 0),
            last_updated_ts=float(payload.get("last_updated_ts", 0.0) or 0.0),
        )


@dataclass
class SessionContext:
    project: ProjectContext | None = None
    recent_turn_summaries: list[str] = field(default_factory=list)
    user_preferences: UserPreferences | dict[str, Any] = field(default_factory=UserPreferences)

    def __post_init__(self) -> None:
        self.recent_turn_summaries = [str(item) for item in self.recent_turn_summaries][-20:]
        if isinstance(self.user_preferences, UserPreferences):
            self.user_preferences = UserPreferences.from_dict(self.user_preferences.to_dict())
        else:
            self.user_preferences = UserPreferences.from_dict(dict(self.user_preferences or {}))


class ContextStore:
    """SQLite-backed persistent store for per-user session context."""

    def __init__(self, db_path: str | None = None) -> None:
        self._explicit_db_path = os.path.abspath(db_path) if db_path else None
        self._lock = threading.RLock()

    def _resolve_db_path(self) -> str:
        if self._explicit_db_path:
            return self._explicit_db_path
        cfg = get_config()
        base_dir = os.path.abspath(os.path.expandvars(os.path.expanduser(cfg.rag.persist_directory)))
        return os.path.join(base_dir, "session_context.sqlite3")

    def _ensure_parent_dir(self, db_path: str) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    def get_db_path(self) -> str:
        return self._resolve_db_path()

    @staticmethod
    def _serialize_value(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False)

    @staticmethod
    def _deserialize_value(value: str | None) -> Any:
        if value is None:
            return None
        try:
            return json.loads(value)
        except Exception:
            return value

    @staticmethod
    def _init_schema(connection: sqlite3.Connection) -> None:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS session_context (
                user_scope TEXT PRIMARY KEY,
                project_name TEXT,
                project_source_path TEXT,
                project_confidence REAL,
                project_last_confirmed_ts REAL,
                updated_at REAL NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS recent_turn_summaries (
                user_scope TEXT NOT NULL,
                position INTEGER NOT NULL,
                summary TEXT NOT NULL,
                PRIMARY KEY (user_scope, position)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS user_preferences (
                user_scope TEXT NOT NULL,
                pref_key TEXT NOT NULL,
                pref_value TEXT NOT NULL,
                PRIMARY KEY (user_scope, pref_key)
            )
            """
        )

    def save(self, session_context: SessionContext, user_scope: str | None = None) -> None:
        scope = _user_key(user_scope or (session_context.project.user_scope if session_context.project else None))
        user_preferences = session_context.user_preferences
        if not isinstance(user_preferences, UserPreferences):
            user_preferences = UserPreferences.from_dict(dict(user_preferences or {}))
        payload = SessionContext(
            project=session_context.project,
            recent_turn_summaries=list(session_context.recent_turn_summaries)[-20:],
            user_preferences=user_preferences,
        )
        if payload.project is not None and payload.project.user_scope != scope:
            payload.project.user_scope = scope

        db_path = self._resolve_db_path()
        self._ensure_parent_dir(db_path)
        try:
            connection = sqlite3.connect(db_path)
            try:
                connection.row_factory = sqlite3.Row
                connection.execute("BEGIN IMMEDIATE")
                self._init_schema(connection)
                now = time.time()
                connection.execute("DELETE FROM session_context WHERE user_scope = ?", (scope,))
                connection.execute("DELETE FROM recent_turn_summaries WHERE user_scope = ?", (scope,))
                connection.execute("DELETE FROM user_preferences WHERE user_scope = ?", (scope,))
                connection.execute(
                    """
                    INSERT INTO session_context (
                        user_scope,
                        project_name,
                        project_source_path,
                        project_confidence,
                        project_last_confirmed_ts,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        scope,
                        payload.project.name if payload.project else None,
                        payload.project.source_path if payload.project else None,
                        payload.project.confidence if payload.project else None,
                        payload.project.last_confirmed_ts if payload.project else None,
                        now,
                    ),
                )
                for position, summary in enumerate(payload.recent_turn_summaries[-20:]):
                    connection.execute(
                        "INSERT INTO recent_turn_summaries (user_scope, position, summary) VALUES (?, ?, ?)",
                        (scope, position, str(summary)),
                    )
                for pref_key, pref_value in payload.user_preferences.to_dict().items():
                    connection.execute(
                        "INSERT INTO user_preferences (user_scope, pref_key, pref_value) VALUES (?, ?, ?)",
                        (scope, str(pref_key), self._serialize_value(pref_value)),
                    )
                connection.commit()
            finally:
                connection.close()
        except Exception:
            raise

    def load(self, user_scope: str | None = None) -> SessionContext:
        scope = _user_key(user_scope)
        db_path = self._resolve_db_path()
        if not os.path.exists(db_path):
            logger.warning("session_context_db_missing", path=db_path, user_scope=scope)
            return SessionContext()

        try:
            connection = sqlite3.connect(db_path)
            try:
                connection.row_factory = sqlite3.Row
                self._init_schema(connection)
                row = connection.execute(
                    "SELECT * FROM session_context WHERE user_scope = ?",
                    (scope,),
                ).fetchone()
                project = ProjectContext.from_row(row, scope) if row is not None else None
                summaries_rows = connection.execute(
                    "SELECT summary FROM recent_turn_summaries WHERE user_scope = ? ORDER BY position ASC",
                    (scope,),
                ).fetchall()
                preference_rows = connection.execute(
                    "SELECT pref_key, pref_value FROM user_preferences WHERE user_scope = ?",
                    (scope,),
                ).fetchall()
                recent_turn_summaries = [str(item["summary"]) for item in summaries_rows]
                user_preferences_raw = {
                    str(item["pref_key"]): self._deserialize_value(item["pref_value"])
                    for item in preference_rows
                }
                # Migrate legacy rows by filling missing preference fields before rebuilding the dataclass.
                defaults = UserPreferences().to_dict()
                migrated_preferences: dict[str, Any] = {}
                for field_name, default_value in defaults.items():
                    if field_name not in user_preferences_raw:
                        logger.warning("Migrating legacy session — missing field: %s", field_name)
                        if field_name == "preference_change_count" and "correction_count" in user_preferences_raw:
                            migrated_preferences[field_name] = int(user_preferences_raw.get("correction_count") or 0)
                        else:
                            migrated_preferences[field_name] = default_value
                    else:
                        migrated_preferences[field_name] = user_preferences_raw[field_name]
                return SessionContext(
                    project=project,
                    recent_turn_summaries=recent_turn_summaries,
                    user_preferences=UserPreferences(**migrated_preferences),
                )
            finally:
                connection.close()
        except Exception as exc:
            logger.warning("session_context_load_failed", path=db_path, user_scope=scope, error=str(exc), exc_info=True)
            return SessionContext()

    def clear(self) -> None:
        db_path = self._resolve_db_path()
        with self._lock:
            try:
                if not os.path.exists(db_path):
                    return
                connection = sqlite3.connect(db_path)
                try:
                    self._init_schema(connection)
                    connection.execute("DELETE FROM session_context")
                    connection.execute("DELETE FROM recent_turn_summaries")
                    connection.execute("DELETE FROM user_preferences")
                    connection.commit()
                finally:
                    connection.close()
            except Exception as exc:
                logger.warning("session_context_clear_failed", path=db_path, error=str(exc), exc_info=True)


_STORE = ContextStore()
_PROJECT_TURN_MARKERS: dict[str, int] = {}
_CACHE_LOCK = threading.RLock()


def get_session_context(user: str | None = None) -> SessionContext:
    return _STORE.load(user_scope=user)


def set_session_context(context: SessionContext, user: str | None = None) -> None:
    _STORE.save(context, user_scope=user)


def clear() -> None:
    with _CACHE_LOCK:
        _PROJECT_TURN_MARKERS.clear()
    _STORE.clear()


def get_context_db_path() -> str:
    return _STORE.get_db_path()


def set_current_project(
    name: str,
    resolved_path: str,
    confidence: float,
    set_at_turn: int | None,
    user: str | None = None,
) -> None:
    """Set the current project slot for the session/user."""
    scope = _user_key(user)
    context = _STORE.load(scope)
    context.project = ProjectContext(
        name=name,
        source_path=resolved_path,
        confidence=float(confidence),
        last_confirmed_ts=time.time(),
        user_scope=scope,
    )
    with _CACHE_LOCK:
        if set_at_turn is not None:
            _PROJECT_TURN_MARKERS[scope] = int(set_at_turn)
    _STORE.save(context, scope)


def get_current_project(user: str | None = None) -> Optional[Dict[str, Any]]:
    """Return the current project slot for the session/user or None."""
    scope = _user_key(user)
    context = _STORE.load(scope)
    if context.project is None:
        return None
    with _CACHE_LOCK:
        set_at_turn = _PROJECT_TURN_MARKERS.get(scope)
    return {
        "name": context.project.name,
        "source_path": context.project.source_path,
        "resolved_path": context.project.source_path,
        "confidence": context.project.confidence,
        "last_confirmed_ts": context.project.last_confirmed_ts,
        "user_scope": context.project.user_scope,
        "set_at_turn": set_at_turn,
    }


def clear_current_project(user: str | None = None) -> None:
    """Clear the current project slot for the session/user."""
    scope = _user_key(user)
    context = _STORE.load(scope)
    context.project = None
    with _CACHE_LOCK:
        _PROJECT_TURN_MARKERS.pop(scope, None)
    _STORE.save(context, scope)


def clear_if_stale(current_turn: int, max_turns: int = 4, user: str | None = None) -> None:
    """Clear the current project slot if it was set more than `max_turns` ago."""
    slot = get_current_project(user)
    if not slot:
        return
    set_at = slot.get("set_at_turn", 0)
    scope = _user_key(user)
    if set_at is None:
        threshold = max(1, int(max_turns))
        if current_turn >= threshold:
            clear_current_project(user)
        return
    set_at = int(set_at)
    threshold = max(0, int(max_turns) - 1)
    if current_turn - set_at >= threshold:
        clear_current_project(user)
