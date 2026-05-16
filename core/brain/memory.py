from __future__ import annotations

import asyncio
import time
import threading
from dataclasses import dataclass

import chromadb

from core.brain.rag import MultiUserRAGManager
from core.health import get_global_health_monitor
from utils.config import get_config
import getpass
from utils.logger import get_logger

logger = get_logger("memory")


@dataclass
class RecallSection:
    title: str
    lines: list[str]


class _LazyPersonalRAG:
    """Background-loading proxy for the per-user personal RAG index.

    This keeps Dexter startup fast by deferring the expensive Chroma + embedding
    model initialization until after the main boot path has already continued.
    """

    def __init__(self, manager: MultiUserRAGManager, user_id: str):
        self._manager = manager
        self.user_id = user_id
        self._index = None
        self._ready = threading.Event()
        self._init_error: Exception | None = None
        self._async_loop: asyncio.AbstractEventLoop | None = None
        self.warm_up_complete: asyncio.Event | None = None
        try:
            self._async_loop = asyncio.get_running_loop()
            self.warm_up_complete = asyncio.Event()
        except RuntimeError:
            # Not inside an event loop; pipeline will just observe _ready via build_context().
            self._async_loop = None
            self.warm_up_complete = None
        self._thread = threading.Thread(
            target=self._warm,
            daemon=True,
            name=f"rag_warm_{user_id}",
        )
        self._thread.start()
        logger.info("personal_rag_warm_started", user=user_id, thread=self._thread.name)

    def _warm(self) -> None:
        try:
            self._index = self._manager.get_index_for_user(self.user_id)
            self._ready.set()
            if self.warm_up_complete is not None and self._async_loop is not None:
                self._async_loop.call_soon_threadsafe(self.warm_up_complete.set)
            logger.info("personal_rag_ready", user=self.user_id)
        except Exception as e:
            self._init_error = e
            logger.warning("personal_rag_warm_failed", user=self.user_id, error=str(e), exc_info=True)

    @property
    def is_ready(self) -> bool:
        return self._ready.is_set() and self._init_error is None

    def build_context(self, query: str, limit: int = 4, summary: bool = False) -> str:
        index = self._index
        if index is None:
            if self._init_error is None:
                logger.debug("personal_rag_not_ready", user=self.user_id)
            return ""
        return index.build_context(query, limit=limit, summary=summary)

    def on_voice_activity(self) -> None:
        if self._index is not None:
            self._index.on_voice_activity()

    def set_pipeline_state(self, state: str) -> None:
        if self._index is not None:
            self._index.set_pipeline_state(state)

    def format_context_for_provider(self, results, query: str, provider: str = "gemini") -> str:
        if self._index is None:
            return ""
        return self._index.format_context_for_provider(results, query, provider)

    def __getattr__(self, name: str):
        index = self._index
        if index is not None:
            return getattr(index, name)
        raise AttributeError(name)


class DexterMemory:
    def __init__(
        self,
        persist_directory: str = "./memory_db",
        max_items: int = 2000,
        max_age_days: int | None = 30,
        retention_interval_seconds: int = 300,
        disable_rag_warming: bool = False,
    ):
        logger.info("Waking up Dexter's Long-Term Memory (ChromaDB)...")
        self._chroma_available = True
        self.client = None
        self.collection = None
        try:
            self.client = chromadb.PersistentClient(path=persist_directory)
            self.collection = self.client.get_or_create_collection(name="dexter_memory")
        except Exception as e:
            # If Chroma cannot start (permissions/corrupt DB), the assistant must still boot.
            if hasattr(logger, "critical"):
                logger.critical("chroma_startup_failed", error=str(e), exc_info=True)
            else:
                logger.error("chroma_startup_failed", error=str(e), exc_info=True)
            self._chroma_available = False

        self._max_items = max_items
        self._max_age_seconds = max_age_days * 86400 if max_age_days else None
        self._retention_interval_seconds = retention_interval_seconds
        self._last_retention_check = 0.0

        cfg = get_config()
        health_monitor = get_global_health_monitor()
        # Multi-user RAG manager — passes all config fields including BGE model.
        if not self._chroma_available or disable_rag_warming:
            if disable_rag_warming and self._chroma_available:
                logger.info("rag_warming_disabled", reason="low_power_mode")
            self.personal_rag = None
        else:
            manager = MultiUserRAGManager(
                persist_directory=cfg.rag.persist_directory or persist_directory,
                default_roots=cfg.rag.personal_roots,
                cfg={
                    "chunk_size": cfg.rag.chunk_size,
                    "chunk_overlap": cfg.rag.chunk_overlap,
                    "refresh_seconds": cfg.rag.refresh_seconds,
                    "exclude_patterns": cfg.rag.exclude_patterns,
                    "roots": cfg.rag.personal_roots,
                    "embedding_model": cfg.rag.embedding_model,
                    "index_schema_version": cfg.rag.index_schema_version,
                    "max_context_chars": cfg.rag.max_context_chars,
                    "batch_size": cfg.rag.batch_size,
                    "max_embedding_threads": cfg.rag.max_embedding_threads,
                },
                health_monitor=health_monitor,
            )
            current_user = (getpass.getuser() or "default").lower()
            self.personal_rag = _LazyPersonalRAG(manager, current_user)

        logger.info(
            "memory_initialized",
            document_count=self.collection.count() if self.collection is not None else 0,
            max_items=max_items,
            max_age_days=max_age_days,
        )

    def remember(self, text: str, role: str = "user"):
        """Stores an interaction into the vector database for future recall."""
        if not self._chroma_available or self.collection is None:
            return
        try:
            doc_id = f"msg_{int(time.time() * 1000)}"
            self.collection.add(
                documents=[text],
                metadatas=[{"role": role, "timestamp": time.time(), "kind": "conversation"}],
                ids=[doc_id],
            )
            logger.debug("memory_document_saved", doc_id=doc_id, preview=text[:60])
            self._maybe_enforce_retention()
        except Exception as e:
            logger.error("memory_save_failed", error=str(e), exc_info=True)

    def recall_context(self, query: str, n_results: int = 3, include_personal_rag: bool = True) -> str:
        """Return conversational memory and, optionally, personal file context."""
        sections: list[RecallSection] = []
        try:
            if self._chroma_available and self.collection is not None and self.collection.count() > 0:
                results = self.collection.query(query_texts=[query], n_results=min(n_results, self.collection.count()))
                memories = results.get("documents", [[]])[0]
                if memories:
                    sections.append(
                        RecallSection(
                            title="PAST RELEVANT MEMORIES",
                            lines=[f"- {m}" for m in memories],
                        )
                    )
        except Exception as e:
            logger.error("memory_recall_failed", error=str(e), exc_info=True)

        if include_personal_rag:
            try:
                if self.personal_rag is not None:
                    rag_context = self.personal_rag.build_context(query, limit=max(2, n_results))
                    if rag_context:
                        sections.append(RecallSection(title="PERSONAL RAG", lines=rag_context.splitlines()))
            except Exception as e:
                logger.warning("memory_personal_rag_failed", error=str(e), exc_info=True)

        if not sections:
            return ""

        context_lines = []
        for section in sections:
            context_lines.append(f"{section.title} (Use these to understand the user's context):")
            context_lines.extend(section.lines)
        return "\n".join(context_lines)

    def get_memory_count(self) -> int:
        if not self._chroma_available or self.collection is None:
            return 0
        return self.collection.count()

    def _maybe_enforce_retention(self) -> None:
        if not self._chroma_available or self.collection is None:
            return
        if not self._max_items and not self._max_age_seconds:
            return
        now = time.time()
        if now - self._last_retention_check < self._retention_interval_seconds:
            return
        self._last_retention_check = now
        self._enforce_retention()

    def _enforce_retention(self) -> None:
        if not self._chroma_available or self.collection is None:
            return
        try:
            total = self.collection.count()
            if self._max_items and total <= self._max_items and not self._max_age_seconds:
                return

            payload = self.collection.get(include=["metadatas"])
            ids = payload.get("ids") or []
            metadatas = payload.get("metadatas") or []
            if not ids:
                return

            now = time.time()
            to_delete: set[str] = set()
            entries: list[tuple[str, float]] = []

            for idx, doc_id in enumerate(ids):
                meta = metadatas[idx] if idx < len(metadatas) else {}
                ts = meta.get("timestamp") if isinstance(meta, dict) else None
                ts_value = float(ts) if isinstance(ts, (int, float)) else 0.0
                entries.append((doc_id, ts_value))
                if self._max_age_seconds and ts_value > 0 and ts_value < now - self._max_age_seconds:
                    to_delete.add(doc_id)

            if self._max_items and total - len(to_delete) > self._max_items:
                entries.sort(key=lambda item: item[1])
                excess = total - self._max_items - len(to_delete)
                if excess > 0:
                    to_delete.update([doc_id for doc_id, _ in entries[:excess]])

            if to_delete:
                self.collection.delete(ids=list(to_delete))
                logger.info("memory_retention_pruned", removed=len(to_delete))
        except Exception as e:
            logger.warning("memory_retention_failed", error=str(e), exc_info=True)
