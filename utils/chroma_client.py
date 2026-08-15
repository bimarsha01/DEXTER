"""Thread-safe ChromaDB client factory.

Chroma's SharedSystemClient is not safe under concurrent PersistentClient
construction. Parallel LazyLoaders (Memory + vocabulary sync + RAG) race and
raise AttributeError / KeyError on Windows. Serialize creation and cache by path.
"""

from __future__ import annotations

import os
import threading
from typing import Any

import chromadb
from chromadb.config import Settings

_LOCK = threading.RLock()
_CLIENTS: dict[str, Any] = {}


def _normalize_path(path: str | os.PathLike[str]) -> str:
    return os.path.abspath(os.path.expandvars(os.path.expanduser(str(path))))


def get_persistent_client(path: str | os.PathLike[str], **settings_kwargs: Any):
    """Return a cached PersistentClient for ``path``, created under a process lock."""
    normalized = _normalize_path(path)
    os.makedirs(normalized, exist_ok=True)

    with _LOCK:
        existing = _CLIENTS.get(normalized)
        if existing is not None:
            return existing

        settings = Settings(
            anonymized_telemetry=False,
            allow_reset=True,
            is_persistent=True,
            persist_directory=normalized,
            **settings_kwargs,
        )
        client = chromadb.PersistentClient(path=normalized, settings=settings)
        _CLIENTS[normalized] = client
        return client


def get_ephemeral_client(**settings_kwargs: Any):
    """Create an in-memory Chroma client (also serialized against PersistentClient)."""
    with _LOCK:
        settings = Settings(anonymized_telemetry=False, allow_reset=True, **settings_kwargs)
        return chromadb.Client(settings)
