from __future__ import annotations

import getpass
import os
import sqlite3
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz

from core.brain.session_state import get_context_db_path
from utils.logger import get_logger

logger = get_logger("feedback_store")

DEFAULT_QUERY_SIMILARITY_THRESHOLD = 50.0
DEFAULT_NEGATIVE_FEEDBACK_THRESHOLD = 2


def _user_key(user: str | None = None) -> str:
    return (user or getpass.getuser() or "default").lower()


def _normalize_text(value: str) -> str:
    return " ".join((value or "").strip().lower().split())


def _normalize_path(value: str) -> str:
    if not value:
        return ""
    expanded = os.path.expanduser(os.path.expandvars(value))
    try:
        return str(Path(expanded).resolve())
    except Exception:
        return os.path.abspath(expanded)


@dataclass
class RetrievalFeedback:
    turn_id: str
    query: str
    returned_path: str
    was_correct: bool
    user_note: str | None = None
    created_at: float = field(default_factory=time.time)
    user_scope: str = field(default_factory=_user_key)

    def __post_init__(self) -> None:
        self.turn_id = str(self.turn_id or "")
        self.query = str(self.query or "")
        self.returned_path = _normalize_path(self.returned_path)
        self.was_correct = bool(self.was_correct)
        self.user_note = None if self.user_note is None else str(self.user_note)
        self.user_scope = _user_key(self.user_scope)
        self.created_at = float(self.created_at or time.time())

    def to_row(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "query": self.query,
            "returned_path": self.returned_path,
            "was_correct": int(self.was_correct),
            "user_note": self.user_note,
            "created_at": float(self.created_at),
            "user_scope": self.user_scope,
        }


class FeedbackStore:
    def __init__(self, db_path: str | None = None) -> None:
        self._explicit_db_path = os.path.abspath(db_path) if db_path else None
        self._lock = threading.RLock()

    def _resolve_db_path(self) -> str:
        if self._explicit_db_path:
            return self._explicit_db_path
        return get_context_db_path()

    def _ensure_parent_dir(self, db_path: str) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _init_schema(connection: sqlite3.Connection) -> None:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS retrieval_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_scope TEXT NOT NULL,
                turn_id TEXT NOT NULL,
                query TEXT NOT NULL,
                returned_path TEXT NOT NULL,
                was_correct INTEGER NOT NULL,
                user_note TEXT,
                created_at REAL NOT NULL
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_retrieval_feedback_user_query ON retrieval_feedback(user_scope, query)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_retrieval_feedback_user_path ON retrieval_feedback(user_scope, returned_path)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_retrieval_feedback_user_correct ON retrieval_feedback(user_scope, was_correct)"
        )

    def save(self, feedback: RetrievalFeedback, user_scope: str | None = None) -> None:
        scope = _user_key(user_scope or feedback.user_scope)
        payload = RetrievalFeedback(
            turn_id=feedback.turn_id,
            query=feedback.query,
            returned_path=feedback.returned_path,
            was_correct=feedback.was_correct,
            user_note=feedback.user_note,
            created_at=feedback.created_at,
            user_scope=scope,
        )

        db_path = self._resolve_db_path()
        self._ensure_parent_dir(db_path)
        with self._lock:
            connection = sqlite3.connect(db_path)
            try:
                connection.row_factory = sqlite3.Row
                self._init_schema(connection)
                connection.execute(
                    """
                    INSERT INTO retrieval_feedback (
                        user_scope,
                        turn_id,
                        query,
                        returned_path,
                        was_correct,
                        user_note,
                        created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        payload.user_scope,
                        payload.turn_id,
                        payload.query,
                        payload.returned_path,
                        int(payload.was_correct),
                        payload.user_note,
                        payload.created_at,
                    ),
                )
                connection.commit()
            finally:
                connection.close()

    def load(self, user_scope: str | None = None, limit: int = 100) -> list[RetrievalFeedback]:
        scope = _user_key(user_scope)
        db_path = self._resolve_db_path()
        if not os.path.exists(db_path):
            return []

        with self._lock:
            connection = sqlite3.connect(db_path)
            try:
                connection.row_factory = sqlite3.Row
                self._init_schema(connection)
                rows = connection.execute(
                    """
                    SELECT turn_id, query, returned_path, was_correct, user_note, created_at, user_scope
                    FROM retrieval_feedback
                    WHERE user_scope = ?
                    ORDER BY created_at DESC, id DESC
                    LIMIT ?
                    """,
                    (scope, max(1, int(limit))),
                ).fetchall()
                return [
                    RetrievalFeedback(
                        turn_id=str(row["turn_id"] or ""),
                        query=str(row["query"] or ""),
                        returned_path=str(row["returned_path"] or ""),
                        was_correct=bool(row["was_correct"]),
                        user_note=row["user_note"],
                        created_at=float(row["created_at"] or time.time()),
                        user_scope=str(row["user_scope"] or scope),
                    )
                    for row in rows
                ]
            finally:
                connection.close()

    def record(self, feedback: RetrievalFeedback, user_scope: str | None = None) -> None:
        self.save(feedback, user_scope=user_scope)

    def _similarity(self, left: str, right: str) -> float:
        left_norm = _normalize_text(left)
        right_norm = _normalize_text(right)
        if not left_norm or not right_norm:
            return 0.0
        return max(
            float(fuzz.token_set_ratio(left_norm, right_norm)),
            float(fuzz.partial_ratio(left_norm, right_norm)),
        )

    def penalized_paths_for_query(
        self,
        query: str,
        user_scope: str | None = None,
        similarity_threshold: float = DEFAULT_QUERY_SIMILARITY_THRESHOLD,
        minimum_false_count: int = DEFAULT_NEGATIVE_FEEDBACK_THRESHOLD,
    ) -> dict[str, int]:
        scope = _user_key(user_scope)
        db_path = self._resolve_db_path()
        if not os.path.exists(db_path):
            return {}

        current_query = _normalize_text(query)
        if not current_query:
            return {}

        with self._lock:
            connection = sqlite3.connect(db_path)
            try:
                connection.row_factory = sqlite3.Row
                self._init_schema(connection)
                rows = connection.execute(
                    """
                    SELECT query, returned_path
                    FROM retrieval_feedback
                    WHERE user_scope = ? AND was_correct = 0
                    ORDER BY created_at DESC, id DESC
                    """,
                    (scope,),
                ).fetchall()
            finally:
                connection.close()

        counts: dict[str, int] = defaultdict(int)
        for row in rows:
            stored_query = str(row["query"] or "")
            if self._similarity(current_query, stored_query) < float(similarity_threshold):
                continue
            returned_path = _normalize_path(str(row["returned_path"] or ""))
            if not returned_path:
                continue
            counts[returned_path] += 1

        return {path: count for path, count in counts.items() if count >= int(minimum_false_count)}

    def get_penalized_paths(
        self,
        query: str,
        user_scope: str | None = None,
        similarity_threshold: float = DEFAULT_QUERY_SIMILARITY_THRESHOLD,
        minimum_false_count: int = DEFAULT_NEGATIVE_FEEDBACK_THRESHOLD,
    ) -> dict[str, int]:
        return self.penalized_paths_for_query(
            query,
            user_scope=user_scope,
            similarity_threshold=similarity_threshold,
            minimum_false_count=minimum_false_count,
        )

    def clear(self, user_scope: str | None = None) -> None:
        scope = _user_key(user_scope)
        db_path = self._resolve_db_path()
        if not os.path.exists(db_path):
            return

        with self._lock:
            connection = sqlite3.connect(db_path)
            try:
                self._init_schema(connection)
                if user_scope is None:
                    connection.execute("DELETE FROM retrieval_feedback")
                else:
                    connection.execute("DELETE FROM retrieval_feedback WHERE user_scope = ?", (scope,))
                connection.commit()
            finally:
                connection.close()


__all__ = ["RetrievalFeedback", "FeedbackStore"]
