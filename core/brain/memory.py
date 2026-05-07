from __future__ import annotations

import time
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


class DexterMemory:
    def __init__(
        self,
        persist_directory: str = "./memory_db",
        max_items: int = 2000,
        max_age_days: int | None = 30,
        retention_interval_seconds: int = 300,
    ):
        logger.info("Waking up Dexter's Long-Term Memory (ChromaDB)...")
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name="dexter_memory")
        self._max_items = max_items
        self._max_age_seconds = max_age_days * 86400 if max_age_days else None
        self._retention_interval_seconds = retention_interval_seconds
        self._last_retention_check = 0.0

        cfg = get_config()
        health_monitor = get_global_health_monitor()
        # Multi-user RAG manager — passes all config fields including BGE model.
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
            },
            health_monitor=health_monitor,
        )
        current_user = (getpass.getuser() or "default").lower()
        self.personal_rag = manager.get_index_for_user(current_user)
        logger.info(
            "memory_initialized",
            document_count=self.collection.count(),
            max_items=max_items,
            max_age_days=max_age_days,
        )

    def remember(self, text: str, role: str = "user"):
        """Stores an interaction into the vector database for future recall."""
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

    def recall_context(self, query: str, n_results: int = 3) -> str:
        """Return both conversational memory and personal file context."""
        sections: list[RecallSection] = []
        try:
            if self.collection.count() > 0:
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

        try:
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
        return self.collection.count()

    def _maybe_enforce_retention(self) -> None:
        if not self._max_items and not self._max_age_seconds:
            return
        now = time.time()
        if now - self._last_retention_check < self._retention_interval_seconds:
            return
        self._last_retention_check = now
        self._enforce_retention()

    def _enforce_retention(self) -> None:
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
