from __future__ import annotations

import hashlib
import json
import os
import re
import time
import threading
import getpass
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Dict, List, Any

import chromadb
from rapidfuzz import fuzz

from utils.logger import get_logger
from tools.document_tools import summarize_document

logger = get_logger("personal_rag")

SUPPORTED_TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".py",
    ".json",
    ".yaml",
    ".yml",
    ".csv",
    ".ini",
    ".cfg",
    ".toml",
    ".log",
}


@dataclass
class RagChunk:
    source_path: str
    text: str
    chunk_index: int
    file_hash: str
    modified_at: float
    title: str = ""
    kind: str = "document"
    importance: int = 0


class PersonalRAGIndex:
    """Per-user RAG index with incremental polling indexer and richer metadata.

    Uses a Chromadb collection per user to ensure isolation. Provide a
    moderately capable polling file watcher (no extra dependency) and
    simple reranking that blends vector distance with lexical similarity.
    """

    def __init__(
        self,
        persist_directory: str,
        user_id: str = "default",
        roots: Optional[list[str]] = None,
        chunk_size: int = 1200,
        chunk_overlap: int = 180,
        refresh_seconds: int = 600,
        exclude_patterns: Optional[list[str]] = None,
        collection_name: str = "dexter_personal_rag",
    ) -> None:
        self.user_id = (user_id or "default").lower()
        self.persist_directory = os.path.abspath(persist_directory)
        # Create an isolated chroma path per user for stronger isolation
        user_path = os.path.join(self.persist_directory, f"rag_{self.user_id}")
        os.makedirs(user_path, exist_ok=True)
        self._client = chromadb.PersistentClient(path=user_path)
        self._collection = self._client.get_or_create_collection(name=f"{collection_name}_{self.user_id}")

        roots = roots or []
        self._roots = [os.path.abspath(os.path.expandvars(os.path.expanduser(r))) for r in roots if r]
        self._chunk_size = max(300, int(chunk_size))
        self._chunk_overlap = max(0, int(chunk_overlap))
        self._refresh_seconds = max(30, int(refresh_seconds))
        self._exclude_patterns = exclude_patterns or []
        self._last_refresh = 0.0
        self._last_snapshot: Dict[str, float] = {}
        self._index_lock = threading.RLock()
        self._poller: Optional[threading.Thread] = None
        self._stop_poll = threading.Event()
        # In-memory query cache: (query)->(timestamp, results)
        self._cache: Dict[str, tuple[float, list[dict]]] = {}

        logger.info(
            "personal_rag_initialized",
            user=self.user_id,
            roots=self._roots,
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            refresh_seconds=self._refresh_seconds,
        )

    # -------------------- Indexing / Polling --------------------
    def start_polling(self) -> None:
        if self._poller and self._poller.is_alive():
            return
        self._stop_poll.clear()
        self._poller = threading.Thread(target=self._poll_loop, daemon=True, name=f"rag_poll_{self.user_id}")
        self._poller.start()

    def stop_polling(self) -> None:
        if self._poller is None:
            return
        self._stop_poll.set()
        self._poller.join(timeout=2.0)

    def _poll_loop(self) -> None:
        while not self._stop_poll.is_set():
            try:
                self.refresh_incremental()
            except Exception as e:
                logger.warning("rag_poller_error", user=self.user_id, error=str(e), exc_info=True)
            self._stop_poll.wait(self._refresh_seconds)

    def refresh_incremental(self) -> None:
        """Walk roots, detect changed/removed files, and update the index incrementally."""
        with self._index_lock:
            files = []
            for root in self._roots:
                if not root or not os.path.isdir(root):
                    continue
                for path in self._walk_files(root):
                    try:
                        mtime = os.path.getmtime(path)
                        files.append((path, mtime))
                    except Exception:
                        continue

            # Build maps
            current_map = {p: m for p, m in files}
            previous_map = self._last_snapshot

            to_remove = [p for p in previous_map.keys() if p not in current_map]
            to_add_or_update = [p for p, m in current_map.items() if previous_map.get(p) != m]

            if to_remove:
                for p in to_remove:
                    try:
                        self._delete_path(p)
                        logger.info("rag_removed_path", user=self.user_id, path=p)
                    except Exception as e:
                        logger.debug("rag_remove_failed", user=self.user_id, path=p, error=str(e))

            if to_add_or_update:
                chunks: List[RagChunk] = []
                for p in to_add_or_update:
                    try:
                        text = self._read_file(p)
                        if not text.strip():
                            continue
                        file_hash = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
                        modified_at = os.path.getmtime(p)
                        title = os.path.basename(p)
                        for idx, chunk in enumerate(self._chunk_text(text)):
                            importance = self._estimate_importance(p)
                            chunks.append(
                                RagChunk(
                                    source_path=p,
                                    text=chunk,
                                    chunk_index=idx,
                                    file_hash=file_hash,
                                    modified_at=modified_at,
                                    title=title,
                                    kind="document",
                                    importance=importance,
                                )
                            )
                    except Exception as e:
                        logger.debug("rag_index_file_failed", user=self.user_id, path=p, error=str(e))

                if chunks:
                    self._upsert_chunks(chunks)

            # Update snapshot
            self._last_snapshot = current_map
            self._last_refresh = time.time()
            logger.info("personal_rag_incremental_refreshed", user=self.user_id, added_or_updated=len(to_add_or_update), removed=len(to_remove))

    def _estimate_importance(self, path: str) -> int:
        name = os.path.basename(path).lower()
        score = 0
        if any(k in name for k in ("todo", "notes", "project", "readme", "meeting", "minutes")):
            score += 20
        if "notes" in path.lower() or "projects" in path.lower():
            score += 10
        return score

    def _upsert_chunks(self, chunks: List[RagChunk]) -> None:
        # Upsert by chunk id
        ids = [self._chunk_id(c) for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {
                "path": c.source_path,
                "title": c.title,
                "kind": c.kind,
                "chunk_index": c.chunk_index,
                "file_hash": c.file_hash,
                "modified_at": c.modified_at,
                "importance": c.importance,
                "user": self.user_id,
            }
            for c in chunks
        ]
        try:
            # chroma add will overwrite existing ids
            self._collection.add(documents=documents, metadatas=metadatas, ids=ids)
            logger.info("rag_upsert_completed", user=self.user_id, count=len(chunks))
        except Exception as e:
            logger.warning("rag_upsert_failed", user=self.user_id, error=str(e), exc_info=True)

    def _delete_path(self, path: str) -> None:
        # Delete all chunks that reference this path
        try:
            payload = self._collection.get(include=["ids", "metadatas"]) or {}
            ids = payload.get("ids") or []
            metadatas = payload.get("metadatas") or []
            to_delete = []
            for idx, meta in enumerate(metadatas):
                if meta and meta.get("path") == path:
                    to_delete.append(ids[idx])
            if to_delete:
                self._collection.delete(ids=to_delete)
        except Exception as e:
            logger.debug("rag_delete_path_failed", user=self.user_id, path=path, error=str(e))

    # -------------------- Search & Scoring --------------------
    def search(self, query: str, limit: int = 4, use_cache: bool = True) -> List[Dict[str, Any]]:
        query_key = f"{self.user_id}:{query}:{limit}"
        now = time.time()
        if use_cache:
            cached = self._cache.get(query_key)
            if cached and now - cached[0] < 60.0:
                return cached[1]

        try:
            results = self._collection.query(query_texts=[query], n_results=max(1, int(limit)))
            documents = (results.get("documents") or [[]])[0]
            metadatas = (results.get("metadatas") or [[]])[0]
            distances = (results.get("distances") or [[]])[0]
            payload: List[Dict[str, Any]] = []
            for idx, document in enumerate(documents):
                meta = metadatas[idx] if idx < len(metadatas) else {}
                distance = distances[idx] if idx < len(distances) else None
                vector_score = self._distance_to_score(distance)
                text_sim = float(fuzz.partial_ratio(query, document or meta.get("title", "")))
                # Combine scores: weight vector more but respect lexical match and importance
                importance = float(meta.get("importance") or 0)
                final_score = (0.65 * vector_score) + (0.30 * text_sim) + (0.05 * importance)
                payload.append(
                    {
                        "text": document,
                        "path": meta.get("path", ""),
                        "title": meta.get("title", ""),
                        "kind": meta.get("kind", "document"),
                        "score": final_score,
                        "raw_vector_score": vector_score,
                    }
                )

            # Rerank by final_score desc
            payload.sort(key=lambda p: p.get("score", 0.0), reverse=True)
            self._cache[query_key] = (now, payload)
            logger.info("rag_search", user=self.user_id, query=query, results=len(payload))
            return payload[:int(limit)]
        except Exception as e:
            logger.warning("personal_rag_search_failed", user=self.user_id, error=str(e), exc_info=True)
            return []

    def build_context(self, query: str, limit: int = 4, summary: bool = True) -> str:
        matches = self.search(query, limit=limit)
        if not matches:
            return ""

        lines = ["RELEVANT PERSONAL FILE CONTEXT:"]
        for match in matches:
            path = match.get("path") or "unknown"
            score = match.get("score")
            title = match.get("title") or os.path.basename(path or "")
            lines.append(f"- {title}: {path} (score {score:.2f})")
            if summary and path and os.path.exists(path):
                try:
                    excerpt = summarize_document(path, max_bullets=2)
                    lines.append(f"  {excerpt}")
                    continue
                except Exception:
                    pass

            excerpt = (match.get("text") or "").strip().replace("\n", " ")
            if len(excerpt) > 500:
                excerpt = excerpt[:500].rstrip() + "..."
            lines.append(f"  {excerpt}")
        return "\n".join(lines)

    # -------------------- Helpers --------------------
    def _collect_chunks(self) -> List[RagChunk]:
        chunks: List[RagChunk] = []
        for root in self._roots:
            if not root or not os.path.exists(root):
                continue
            for file_path in self._walk_files(root):
                try:
                    text = self._read_file(file_path)
                    if not text.strip():
                        continue
                    file_hash = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
                    modified_at = os.path.getmtime(file_path)
                    title = os.path.basename(file_path)
                    for idx, chunk in enumerate(self._chunk_text(text)):
                        importance = self._estimate_importance(file_path)
                        chunks.append(
                            RagChunk(
                                source_path=file_path,
                                text=chunk,
                                chunk_index=idx,
                                file_hash=file_hash,
                                modified_at=modified_at,
                                title=title,
                                kind="document",
                                importance=importance,
                            )
                        )
                except Exception as e:
                    logger.debug("personal_rag_file_skip", user=self.user_id, path=file_path, error=str(e))
        return chunks

    def _walk_files(self, root: str) -> Iterable[str]:
        for dirpath, _, filenames in os.walk(root):
            for filename in filenames:
                path = os.path.join(dirpath, filename)
                if self._is_supported(path) and not self._is_excluded(path):
                    yield path

    def _is_excluded(self, path: str) -> bool:
        low = path.lower()
        for pat in self._exclude_patterns:
            if pat.lower() in low:
                return True
        return False

    def _is_supported(self, path: str) -> bool:
        ext = Path(path).suffix.lower()
        return ext in SUPPORTED_TEXT_EXTENSIONS or ext in {".docx", ".xlsx", ".pdf"}

    def _read_file(self, path: str) -> str:
        ext = Path(path).suffix.lower()
        if ext == ".docx":
            return self._read_docx(path)
        if ext == ".xlsx":
            return self._read_xlsx(path)
        if ext == ".pdf":
            return self._read_pdf(path)
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            return handle.read()

    def _read_docx(self, path: str) -> str:
        try:
            from docx import Document  # type: ignore

            doc = Document(path)
            return "\n".join(paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip())
        except Exception as e:
            logger.debug("personal_rag_docx_failed", user=self.user_id, path=path, error=str(e))
            return ""

    def _read_xlsx(self, path: str) -> str:
        try:
            from openpyxl import load_workbook  # type: ignore

            workbook = load_workbook(path, data_only=True)
            lines: list[str] = []
            for sheet in workbook.worksheets:
                lines.append(f"Sheet: {sheet.title}")
                for row in sheet.iter_rows(values_only=True):
                    values = [str(cell) for cell in row if cell is not None]
                    if values:
                        lines.append(" | ".join(values))
            return "\n".join(lines)
        except Exception as e:
            logger.debug("personal_rag_xlsx_failed", user=self.user_id, path=path, error=str(e))
            return ""

    def _read_pdf(self, path: str) -> str:
        try:
            from pypdf import PdfReader  # type: ignore

            reader = PdfReader(path)
            pages = []
            for page in reader.pages:
                try:
                    pages.append(page.extract_text() or "")
                except Exception:
                    continue
            return "\n".join(page for page in pages if page.strip())
        except Exception as e:
            logger.debug("personal_rag_pdf_failed", user=self.user_id, path=path, error=str(e))
            return ""

    def _chunk_text(self, text: str) -> list[str]:
        normalized = re.sub(r"\r\n", "\n", text)
        if len(normalized) <= self._chunk_size:
            return [normalized]

        # Try to split on sentence boundaries for better chunks
        sentences = re.split(r"(?<=[.!?])\s+", normalized)
        chunks: list[str] = []
        current = []
        current_len = 0
        for s in sentences:
            sl = len(s)
            if current_len + sl <= self._chunk_size:
                current.append(s)
                current_len += sl
            else:
                if current:
                    chunks.append(" ".join(current).strip())
                current = [s]
                current_len = sl
        if current:
            chunks.append(" ".join(current).strip())
        # Add overlap by joining trailing sentences into next chunk
        if self._chunk_overlap > 0 and len(chunks) > 1:
            merged: list[str] = []
            for i, c in enumerate(chunks):
                if i == 0:
                    merged.append(c)
                    continue
                prev = merged[-1]
                overlap = prev[-self._chunk_overlap:] if len(prev) > self._chunk_overlap else prev
                merged.append((overlap + " " + c).strip())
            chunks = merged
        return chunks

    @staticmethod
    def _chunk_id(chunk: RagChunk) -> str:
        return f"{hashlib.md5(chunk.source_path.encode('utf-8', errors='ignore')).hexdigest()}_{chunk.chunk_index}_{chunk.file_hash[:12]}"

    @staticmethod
    def _distance_to_score(distance: float | None) -> float:
        if distance is None:
            return 0.0
        return max(0.0, 100.0 - float(distance))


class MultiUserRAGManager:
    """Manager that returns per-user PersonalRAGIndex instances.

    This provides an easy extension point to add ingestion connectors
    (email, notes, outlook) that feed individual user indexes.
    """

    def __init__(self, persist_directory: str, default_roots: Optional[list[str]] = None, cfg: Optional[dict] = None) -> None:
        self.persist_directory = os.path.abspath(persist_directory)
        os.makedirs(self.persist_directory, exist_ok=True)
        self._indexes: Dict[str, PersonalRAGIndex] = {}
        self._default_roots = default_roots or []
        self._cfg = cfg or {}

    def get_index_for_user(self, user_id: Optional[str] = None) -> PersonalRAGIndex:
        uid = (user_id or getpass.getuser() or "default").lower()
        if uid in self._indexes:
            return self._indexes[uid]
        idx = PersonalRAGIndex(
            persist_directory=self.persist_directory,
            user_id=uid,
            roots=self._cfg.get("roots") or self._default_roots,
            chunk_size=int(self._cfg.get("chunk_size", 1200)),
            chunk_overlap=int(self._cfg.get("chunk_overlap", 180)),
            refresh_seconds=int(self._cfg.get("refresh_seconds", 600)),
            exclude_patterns=self._cfg.get("exclude_patterns", []),
        )
        # Start background poller by default
        idx.start_polling()
        self._indexes[uid] = idx
        return idx

    def remove_user(self, user_id: str) -> None:
        uid = user_id.lower()
        idx = self._indexes.pop(uid, None)
        if idx is not None:
            idx.stop_polling()
