from __future__ import annotations

import hashlib
import json
import os
import re
import time
import threading
import getpass
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Dict, List, Any

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from rapidfuzz import fuzz

from utils.logger import get_logger

logger = get_logger("personal_rag")

SUPPORTED_TEXT_EXTENSIONS = {
    ".txt", ".md", ".py", ".json", ".yaml", ".yml",
    ".csv", ".ini", ".cfg", ".toml", ".log",
    ".js", ".ts", ".html", ".css", ".sh", ".bat",
    ".rs", ".go", ".java", ".c", ".cpp", ".h",
}

CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".java", ".c", ".cpp", ".h",
    ".rs", ".go", ".sh", ".bat", ".css", ".html",
}

# Binary/junk extensions to always skip
SKIP_EXTENSIONS = {
    ".pyc", ".pyo", ".exe", ".dll", ".so", ".dylib",
    ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp",
    ".mp3", ".mp4", ".wav", ".avi", ".mkv", ".flac",
    ".whl", ".egg", ".db", ".sqlite", ".sqlite3",
    ".bin", ".dat", ".lock", ".map",
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
    file_extension: str = ""
    parent_folder: str = ""
    file_size_bytes: int = 0
    indexed_at: float = field(default_factory=time.time)


class _LRUCache:
    """Simple bounded LRU cache for query results."""

    def __init__(self, maxsize: int = 200):
        self._maxsize = maxsize
        self._store: OrderedDict[str, tuple[float, list]] = OrderedDict()

    def get(self, key: str) -> tuple[float, list] | None:
        if key in self._store:
            self._store.move_to_end(key)
            return self._store[key]
        return None

    def put(self, key: str, value: tuple[float, list]) -> None:
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = value
        while len(self._store) > self._maxsize:
            self._store.popitem(last=False)


INDEX_SCHEMA_VERSION = 2
MINIMUM_RELEVANCE_SCORE = 55.0


def _slugify_collection_key(value: str) -> str:
    key = re.sub(r"[^a-zA-Z0-9]+", "-", (value or "").strip().lower())
    key = re.sub(r"-+", "-", key).strip("-")
    return key or "default"


@dataclass(frozen=True)
class EmbeddingProfile:
    model_name: str
    collection_key: str
    provider: str

    @classmethod
    def from_model_name(cls, model_name: str | None) -> "EmbeddingProfile":
        normalized = (model_name or "").strip()
        if not normalized or normalized.lower() in {"default", "chromadb", "chromadb-default"}:
            return cls(
                model_name="chromadb-default",
                collection_key="chromadb-default",
                provider="chromadb-default",
            )
        return cls(
            model_name=normalized,
            collection_key=_slugify_collection_key(normalized),
            provider="sentence_transformer",
        )


_EMBEDDING_FN_CACHE: Dict[str, SentenceTransformerEmbeddingFunction] = {}
_EMBEDDING_FN_LOCK = threading.Lock()


def _get_embedding_fn(model_name: str) -> SentenceTransformerEmbeddingFunction:
    with _EMBEDDING_FN_LOCK:
        if model_name not in _EMBEDDING_FN_CACHE:
            logger.info("loading_embedding_model", model=model_name)
            _EMBEDDING_FN_CACHE[model_name] = SentenceTransformerEmbeddingFunction(model_name=model_name)
            logger.info("embedding_model_loaded", model=model_name)
        return _EMBEDDING_FN_CACHE[model_name]




class PersonalRAGIndex:
    """Per-user RAG index with versioned collections and configurable embeddings."""

    def __init__(
        self,
        persist_directory: str,
        user_id: str = "default",
        roots: Optional[list[str]] = None,
        chunk_size: int = 600,
        chunk_overlap: int = 100,
        refresh_seconds: int = 600,
        exclude_patterns: Optional[list[str]] = None,
        collection_name: str = "dexter_personal_rag",
        embedding_model: str = "BAAI/bge-base-en-v1.5",
        index_schema_version: int = INDEX_SCHEMA_VERSION,
        max_context_chars: int = 3000,
        batch_size: int = 256,
        max_embedding_threads: int = 4,
        health_monitor: Any = None,
    ) -> None:
        self.user_id = (user_id or "default").lower()
        self.persist_directory = os.path.abspath(persist_directory)
        self._embedding_profile = EmbeddingProfile.from_model_name(embedding_model)
        self._index_schema_version = max(1, int(index_schema_version))
        self._collection_base_name = collection_name
        self._collection_name = (
            f"{collection_name}_idxv{self._index_schema_version}_"
            f"{self._embedding_profile.collection_key}_{self.user_id}"
        )
        self._max_context_chars = max_context_chars
        self._batch_size = max(10, int(batch_size))
        self._max_embedding_threads = max(0, int(max_embedding_threads))
        self._health_monitor = health_monitor

        # Per-user isolated chroma path
        user_path = os.path.join(self.persist_directory, f"rag_{self.user_id}")
        os.makedirs(user_path, exist_ok=True)

        self._client = chromadb.PersistentClient(path=user_path)
        collection_metadata = {
            "index_schema_version": self._index_schema_version,
            "embedding_model": self._embedding_profile.model_name,
            "embedding_provider": self._embedding_profile.provider,
            "collection_base_name": self._collection_base_name,
            "collection_name": self._collection_name,
        }
        # Create collection without an embedding function so we control batching
        # and can compute embeddings once per batch (faster, more observable).
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata=collection_metadata,
        )
        # Preload embedding function (cached) for manual encoding
        if self._embedding_profile.provider != "chromadb-default":
            self._ef = _get_embedding_fn(self._embedding_profile.model_name)
        else:
            self._ef = None

        roots = roots or []
        self._roots = [os.path.abspath(os.path.expandvars(os.path.expanduser(r))) for r in roots if r]
        self._chunk_size = max(300, int(chunk_size))
        self._chunk_overlap = max(0, int(chunk_overlap))
        self._refresh_seconds = max(30, int(refresh_seconds))
        self._exclude_patterns = exclude_patterns or []
        self._last_refresh = 0.0
        self._last_snapshot: Dict[str, Dict[str, Any]] = {}
        self._snapshot_path = os.path.join(user_path, "snapshot.json")
        self._load_snapshot()

        self._index_lock = threading.RLock()
        self._poller: Optional[threading.Thread] = None
        self._stop_poll = threading.Event()
        self._cache = _LRUCache(maxsize=200)

        logger.info(
            "personal_rag_initialized",
            user=self.user_id,
            roots=self._roots,
            collection_name=self._collection_name,
            embedding_model=self._embedding_profile.model_name,
            index_schema_version=self._index_schema_version,
            chunk_size=self._chunk_size,
            batch_size=self._batch_size,
        )

    # ── Polling ─────────────────────────────────────────────────────
    def start_polling(self) -> None:
        if self._poller and self._poller.is_alive():
            return
        self._stop_poll.clear()
        self._poller = threading.Thread(
            target=self._poll_loop, daemon=True, name=f"rag_poll_{self.user_id}"
        )
        self._poller.start()
        logger.info("rag_poller_started", user=self.user_id, thread=self._poller.name)

    def stop_polling(self) -> None:
        if self._poller is None:
            return
        self._stop_poll.set()
        self._poller.join(timeout=2.0)
        logger.info("rag_poller_stopped", user=self.user_id)

    def _poll_loop(self) -> None:
        while not self._stop_poll.is_set():
            try:
                self.refresh_incremental()
                self._report_health("healthy", "refresh_ok")
            except Exception as e:
                logger.warning("rag_poller_error", user=self.user_id, error=str(e), exc_info=True)
                self._report_health("degraded", str(e))
            self._stop_poll.wait(self._refresh_seconds)

    def _report_health(self, status: str, details: str = "") -> None:
        hm = self._health_monitor
        if hm is None:
            return
        try:
            if status == "healthy":
                hm.healthy("rag", details=details)
            else:
                hm.degraded("rag", details=details)
        except Exception:
            pass

    def _load_snapshot(self) -> None:
        """Load the file snapshot from disk."""
        if not os.path.exists(self._snapshot_path):
            return
        try:
            with open(self._snapshot_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self._last_snapshot = data.get("files", {})
                self._last_refresh = data.get("last_refresh", 0.0)
            logger.info("rag_snapshot_loaded", path=self._snapshot_path, files=len(self._last_snapshot))
        except Exception as e:
            logger.warning("rag_snapshot_load_failed", error=str(e))

    def _save_snapshot(self) -> None:
        """Save the file snapshot to disk."""
        try:
            data = {
                "files": self._last_snapshot,
                "last_refresh": self._last_refresh,
                "indexed_at": time.time()
            }
            with open(self._snapshot_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logger.debug("rag_snapshot_saved", path=self._snapshot_path)
        except Exception as e:
            logger.warning("rag_snapshot_save_failed", error=str(e))

    # ── Indexing ────────────────────────────────────────────────────
    def refresh_incremental(self) -> None:
        """Walk roots, detect changed/removed files, update the index with progress tracking."""
        with self._index_lock:
            t0 = time.perf_counter()
            files = []
            for root in self._roots:
                if not root or not os.path.isdir(root):
                    continue
                for path in self._walk_files(root):
                    try:
                        mtime = os.path.getmtime(path)
                        size = os.path.getsize(path)
                        files.append((path, mtime, size))
                    except Exception:
                        continue

            current_map = {p: {"mtime": m, "size": s} for p, m, s in files}
            previous_map = self._last_snapshot

            logger.info("rag_refresh_scan", user=self.user_id, found_files=len(current_map))

            to_remove = [p for p in previous_map if p not in current_map]
            to_add_or_update = [
                p for p, info in current_map.items() 
                if p not in previous_map or previous_map[p].get("mtime") != info["mtime"] or previous_map[p].get("size") != info["size"]
            ]

            if to_remove:
                for p in to_remove:
                    try:
                        self._delete_path(p)
                    except Exception as e:
                        logger.debug("rag_remove_failed", path=p, error=str(e))

            if to_add_or_update:
                chunks: List[RagChunk] = []
                file_scan_start = time.perf_counter()
                for file_num, p in enumerate(to_add_or_update, 1):
                    try:
                        text = self._read_file(p)
                        if not text.strip():
                            continue
                        file_hash = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
                        modified_at = os.path.getmtime(p)
                        title = os.path.basename(p)
                        ext = Path(p).suffix.lower()
                        parent = os.path.basename(os.path.dirname(p))
                        try:
                            fsize = os.path.getsize(p)
                        except Exception:
                            fsize = 0

                        for idx, chunk in enumerate(self._chunk_text(text, ext)):
                            chunks.append(RagChunk(
                                source_path=p, text=chunk, chunk_index=idx,
                                file_hash=file_hash, modified_at=modified_at,
                                title=title, kind=self._classify_kind(ext),
                                importance=self._estimate_importance(p),
                                file_extension=ext, parent_folder=parent,
                                file_size_bytes=fsize,
                            ))

                        # Progress tracking with ETA
                        if file_num % max(1, len(to_add_or_update) // 10) == 0 or file_num == len(to_add_or_update):
                            elapsed_sec = time.perf_counter() - file_scan_start
                            progress_pct = (file_num / len(to_add_or_update)) * 100
                            rate = file_num / elapsed_sec if elapsed_sec > 0 else 0
                            remaining_files = len(to_add_or_update) - file_num
                            eta_sec = remaining_files / rate if rate > 0 else 0
                            print(f"✓ Scanned {file_num}/{len(to_add_or_update)} files ({progress_pct:.0f}%) | "
                                  f"Elapsed: {elapsed_sec/60:.1f}m | ETA: {eta_sec/60:.1f}m")

                    except Exception as e:
                        logger.debug("rag_index_file_failed", path=p, error=str(e))

                if chunks:
                    print(f"⧗ Indexing {len(chunks)} chunks in {(len(chunks) + self._batch_size - 1) // self._batch_size} batches...")
                    self._upsert_chunks(chunks)

            self._last_snapshot = current_map
            self._last_refresh = time.time()
            self._save_snapshot()
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.info(
                "personal_rag_incremental_refreshed",
                user=self.user_id,
                added_or_updated=len(to_add_or_update),
                removed=len(to_remove),
                total_files=len(current_map),
                elapsed_ms=f"{elapsed_ms:.0f}",
            )

    @staticmethod
    def _classify_kind(ext: str) -> str:
        if ext in CODE_EXTENSIONS:
            return "code"
        if ext in {".md", ".txt", ".log"}:
            return "text"
        if ext in {".json", ".yaml", ".yml", ".toml", ".ini", ".cfg"}:
            return "config"
        if ext in {".docx", ".pdf", ".xlsx"}:
            return "office"
        return "document"

    def _estimate_importance(self, path: str) -> int:
        name = os.path.basename(path).lower()
        score = 0
        if any(k in name for k in ("todo", "notes", "project", "readme", "meeting", "minutes", "architecture")):
            score += 20
        if "notes" in path.lower() or "projects" in path.lower():
            score += 10
        return score

    def _upsert_chunks(self, chunks: List[RagChunk]) -> None:
        if not chunks:
            return
        bs = self._batch_size
        total_batches = (len(chunks) + bs - 1) // bs
        logger.info("rag_upsert_start", user=self.user_id, total_chunks=len(chunks), batches=total_batches)

        # Prepare batch metadata and documents upfront
        batch_data = []
        for batch_idx in range(total_batches):
            start = batch_idx * bs
            batch = chunks[start:start + bs]
            ids = [self._chunk_id(c) for c in batch]
            documents = [c.text for c in batch]
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
                    "file_extension": c.file_extension,
                    "parent_folder": c.parent_folder,
                    "file_size_bytes": c.file_size_bytes,
                    "indexed_at": c.indexed_at,
                    "index_schema_version": self._index_schema_version,
                    "embedding_model": self._embedding_profile.model_name,
                    "embedding_provider": self._embedding_profile.provider,
                    "collection_name": self._collection_name,
                }
                for c in batch
            ]
            batch_data.append((batch_idx + 1, ids, documents, metadatas))

        # Compute embeddings (serial or parallel)
        if self._ef is None:
            # Fallback: no embeddings
            for batch_idx, ids, documents, metadatas in batch_data:
                try:
                    t0 = time.perf_counter()
                    self._collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
                    t1 = time.perf_counter()
                    logger.info("rag_upsert_batch_completed", user=self.user_id, batch=batch_idx, batch_size=len(documents), batch_ms=int((t1 - t0) * 1000))
                except Exception as e:
                    logger.warning("rag_upsert_batch_failed", batch=batch_idx, error=str(e), exc_info=True)
        elif self._max_embedding_threads > 1:
            # Concurrent batch embedding using ThreadPoolExecutor
            self._upsert_chunks_concurrent(batch_data)
        else:
            # Serial batch embedding (original approach)
            self._upsert_chunks_serial(batch_data)

        logger.info("rag_upsert_completed", user=self.user_id, count=len(chunks), batches=total_batches)

    def _upsert_chunks_serial(self, batch_data: List[tuple]) -> None:
        """Embed and upsert batches sequentially."""
        for batch_idx, ids, documents, metadatas in batch_data:
            if self._ef is None:
                continue
            embeddings = None
            try:
                t0 = time.perf_counter()
                embeddings = list(self._ef(documents))
                t1 = time.perf_counter()
                logger.debug("rag_batch_embedded", user=self.user_id, batch=batch_idx, batch_size=len(documents), embed_ms=int((t1 - t0) * 1000))
            except Exception as e:
                logger.warning("rag_batch_embedding_failed", batch=batch_idx, error=str(e), exc_info=True)

            try:
                t0 = time.perf_counter()
                if embeddings:
                    self._collection.upsert(documents=documents, metadatas=metadatas, ids=ids, embeddings=embeddings)
                else:
                    self._collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
                t1 = time.perf_counter()
                logger.info("rag_upsert_batch_completed", user=self.user_id, batch=batch_idx, batch_size=len(documents), batch_ms=int((t1 - t0) * 1000))
            except Exception as e:
                logger.warning("rag_upsert_batch_failed", batch=batch_idx, error=str(e), exc_info=True)

    def _upsert_chunks_concurrent(self, batch_data: List[tuple]) -> None:
        """Embed batches concurrently using ThreadPoolExecutor, then upsert."""
        max_workers = max(1, min(self._max_embedding_threads, len(batch_data)))
        
        # Phase 1: Embed all batches concurrently
        embeddings_map = {}  # batch_idx -> embeddings list
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for batch_idx, ids, documents, metadatas in batch_data:
                future = executor.submit(self._embed_batch_worker, batch_idx, documents)
                futures[future] = (batch_idx, ids, documents, metadatas)

            for future in as_completed(futures):
                batch_idx, embeddings, embed_time = future.result()
                embeddings_map[batch_idx] = (embeddings, embed_time)

        # Phase 2: Upsert all batches serially (to avoid Chroma threading issues)
        for batch_idx, ids, documents, metadatas in batch_data:
            embeddings, embed_time = embeddings_map.get(batch_idx, (None, 0))
            if embeddings:
                logger.debug("rag_batch_embedded", user=self.user_id, batch=batch_idx, batch_size=len(documents), embed_ms=embed_time)

            try:
                t0 = time.perf_counter()
                if embeddings:
                    self._collection.upsert(documents=documents, metadatas=metadatas, ids=ids, embeddings=embeddings)
                else:
                    self._collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
                t1 = time.perf_counter()
                logger.info("rag_upsert_batch_completed", user=self.user_id, batch=batch_idx, batch_size=len(documents), batch_ms=int((t1 - t0) * 1000))
            except Exception as e:
                logger.warning("rag_upsert_batch_failed", batch=batch_idx, error=str(e), exc_info=True)

    def _embed_batch_worker(self, batch_idx: int, documents: List[str]) -> tuple:
        """Worker function for concurrent embedding."""
        embeddings = None
        embed_time = 0
        try:
            t0 = time.perf_counter()
            embeddings = list(self._ef(documents))
            t1 = time.perf_counter()
            embed_time = int((t1 - t0) * 1000)
        except Exception as e:
            logger.warning("rag_batch_embedding_failed", batch=batch_idx, error=str(e), exc_info=True)
        return batch_idx, embeddings, embed_time

    def migrate_from_collection(self, source_collection_name: str, batch_size: int | None = None) -> int:
        """Re-embed stored chunks from an older collection into the current one."""
        try:
            source = self._client.get_collection(name=source_collection_name)
        except Exception as e:
            logger.info("rag_migration_source_missing", source_collection_name=source_collection_name, error=str(e))
            return 0

        payload = source.get(include=["ids", "documents", "metadatas"]) or {}
        ids = payload.get("ids") or []
        documents = payload.get("documents") or []
        metadatas = payload.get("metadatas") or []
        if not ids:
            return 0

        bs = max(1, int(batch_size or self._batch_size))
        migrated = 0
        for start in range(0, len(ids), bs):
            end = min(start + bs, len(ids))
            batch_ids = ids[start:end]
            batch_documents = documents[start:end]
            batch_metadatas = []
            for meta in metadatas[start:end]:
                meta_copy = dict(meta or {})
                meta_copy["index_schema_version"] = self._index_schema_version
                meta_copy["embedding_model"] = self._embedding_profile.model_name
                meta_copy["embedding_provider"] = self._embedding_profile.provider
                meta_copy["migrated_at"] = time.time()
                meta_copy["migrated_from_collection"] = source_collection_name
                batch_metadatas.append(meta_copy)

            try:
                self._collection.upsert(documents=batch_documents, metadatas=batch_metadatas, ids=batch_ids)
                migrated += len(batch_ids)
            except Exception as e:
                logger.warning("rag_migration_batch_failed", source_collection_name=source_collection_name, batch_start=start, error=str(e))

        logger.info(
            "rag_migration_completed",
            source_collection_name=source_collection_name,
            destination_collection_name=self._collection_name,
            migrated=migrated,
        )
        return migrated

    def migrate_from_model(self, source_model: str, source_index_schema_version: int | None = None) -> int:
        source_profile = EmbeddingProfile.from_model_name(source_model)
        schema_version = self._index_schema_version if source_index_schema_version is None else max(1, int(source_index_schema_version))
        source_collection_name = (
            f"{self._collection_base_name}_idxv{schema_version}_"
            f"{source_profile.collection_key}_{self.user_id}"
        )
        return self.migrate_from_collection(source_collection_name)

    def _delete_path(self, path: str) -> None:
        try:
            payload = self._collection.get(include=["metadatas"]) or {}
            ids = payload.get("ids") or []
            metadatas = payload.get("metadatas") or []
            to_delete = [ids[i] for i, m in enumerate(metadatas) if m and m.get("path") == path]
            if to_delete:
                self._collection.delete(ids=to_delete)
        except Exception as e:
            logger.debug("rag_delete_path_failed", path=path, error=str(e))

    # ── Search & Scoring ───────────────────────────────────────────
    def search(self, query: str, limit: int = 4, use_cache: bool = True) -> List[Dict[str, Any]]:
        query_key = f"{self.user_id}:{query}:{limit}"
        now = time.time()
        if use_cache:
            cached = self._cache.get(query_key)
            if cached and now - cached[0] < 60.0:
                return cached[1]

        try:
            # Compute query embedding (use cached embedding fn) and query by embeddings
            query_embeddings = None
            if self._ef is not None:
                try:
                    t0 = time.perf_counter()
                    query_embeddings = list(self._ef([query]))
                    t1 = time.perf_counter()
                    logger.debug("rag_query_embedded", user=self.user_id, embed_ms=int((t1 - t0) * 1000))
                except Exception:
                    query_embeddings = None

            if query_embeddings is not None:
                results = self._collection.query(query_embeddings=query_embeddings, n_results=max(1, int(limit) * 2))
            else:
                # Fallback to text query if collection supports it
                results = self._collection.query(query_texts=[query], n_results=max(1, int(limit) * 2))

            documents = (results.get("documents") or [[]])[0]
            metadatas = (results.get("metadatas") or [[]])[0]
            distances = (results.get("distances") or [[]])[0]

            payload: List[Dict[str, Any]] = []
            for idx, document in enumerate(documents):
                meta = metadatas[idx] if idx < len(metadatas) else {}
                distance = distances[idx] if idx < len(distances) else None
                vector_score = self._distance_to_score(distance)
                text_sim = float(fuzz.partial_ratio(query, document or meta.get("title", "")))
                importance = float(meta.get("importance") or 0)
                final_score = (0.65 * vector_score) + (0.30 * text_sim) + (0.05 * importance)
                payload.append({
                    "text": document,
                    "path": meta.get("path", ""),
                    "title": meta.get("title", ""),
                    "kind": meta.get("kind", "document"),
                    "score": final_score,
                    "raw_vector_score": vector_score,
                    "file_extension": meta.get("file_extension", ""),
                    "parent_folder": meta.get("parent_folder", ""),
                })

            payload.sort(key=lambda p: p.get("score", 0.0), reverse=True)
            
            # Filter by minimum relevance score
            filtered_payload = [p for p in payload if p.get("score", 0.0) >= MINIMUM_RELEVANCE_SCORE]
            
            payload = filtered_payload[:int(limit)]
            self._cache.put(query_key, (now, payload))
            logger.info("rag_search", user=self.user_id, query=query, results=len(payload), threshold=MINIMUM_RELEVANCE_SCORE)
            return payload
        except Exception as e:
            logger.warning("personal_rag_search_failed", user=self.user_id, error=str(e), exc_info=True)
            return []

    def build_context(self, query: str, limit: int = 4, summary: bool = False) -> str:
        """Build compressed context from RAG results — no file re-reads."""
        matches = self.search(query, limit=limit)
        if not matches:
            return ""

        lines = ["RELEVANT PERSONAL FILE CONTEXT:"]
        seen_paths: set[str] = set()
        total_chars = 0

        for match in matches:
            path = match.get("path") or "unknown"
            # Deduplicate: skip if we already have a chunk from this file
            if path in seen_paths:
                continue
            seen_paths.add(path)

            score = match.get("score", 0.0)
            title = match.get("title") or os.path.basename(path)
            kind = match.get("kind", "document")
            lines.append(f"- [{kind}] {title}: {path} (score {score:.1f})")

            excerpt = (match.get("text") or "").strip()
            if len(excerpt) > 400:
                excerpt = excerpt[:400].rstrip() + "..."
            lines.append(f"  {excerpt}")
            total_chars += len(excerpt)

            if total_chars >= self._max_context_chars:
                break

        return "\n".join(lines)

    # ── File Walking ───────────────────────────────────────────────
    def _walk_files(self, root: str) -> Iterable[str]:
        for dirpath, dirnames, filenames in os.walk(root):
            # Prune excluded dirs in-place to avoid descending
            dirnames[:] = [d for d in dirnames if not self._is_excluded(os.path.join(dirpath, d))]
            for filename in filenames:
                path = os.path.join(dirpath, filename)
                if self._is_supported(path) and not self._is_excluded(path):
                    yield path

    def _is_excluded(self, path: str) -> bool:
        low = path.replace("\\", "/").lower()
        for pat in self._exclude_patterns:
            if pat.lower() in low:
                return True
        return False

    def _is_supported(self, path: str) -> bool:
        ext = Path(path).suffix.lower()
        if ext in SKIP_EXTENSIONS:
            return False
        return ext in SUPPORTED_TEXT_EXTENSIONS or ext in {".docx", ".xlsx", ".pdf"}

    # ── File Reading ───────────────────────────────────────────────
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
            from docx import Document
            doc = Document(path)
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except Exception as e:
            logger.debug("rag_docx_failed", path=path, error=str(e))
            return ""

    def _read_xlsx(self, path: str) -> str:
        try:
            from openpyxl import load_workbook
            wb = load_workbook(path, data_only=True)
            lines: list[str] = []
            for sheet in wb.worksheets:
                lines.append(f"Sheet: {sheet.title}")
                for row in sheet.iter_rows(values_only=True):
                    vals = [str(c) for c in row if c is not None]
                    if vals:
                        lines.append(" | ".join(vals))
            return "\n".join(lines)
        except Exception as e:
            logger.debug("rag_xlsx_failed", path=path, error=str(e))
            return ""

    def _read_pdf(self, path: str) -> str:
        try:
            from pypdf import PdfReader
            reader = PdfReader(path)
            pages = []
            for page in reader.pages:
                try:
                    pages.append(page.extract_text() or "")
                except Exception:
                    continue
            return "\n".join(p for p in pages if p.strip())
        except Exception as e:
            logger.debug("rag_pdf_failed", path=path, error=str(e))
            return ""

    # ── Smart Chunking ─────────────────────────────────────────────
    def _chunk_text(self, text: str, ext: str = "") -> list[str]:
        normalized = text.replace("\r\n", "\n")
        if len(normalized) <= self._chunk_size:
            return [normalized]

        if ext in CODE_EXTENSIONS:
            return self._chunk_by_lines(normalized)
        return self._chunk_by_paragraphs(normalized)

    def _chunk_by_paragraphs(self, text: str) -> list[str]:
        """Split on double-newlines (paragraphs), then merge into chunks."""
        paragraphs = re.split(r"\n\s*\n", text)
        return self._merge_segments(paragraphs)

    def _chunk_by_lines(self, text: str) -> list[str]:
        """Split code files by lines, grouping into chunk_size blocks."""
        lines = text.split("\n")
        return self._merge_segments(lines, separator="\n")

    def _merge_segments(self, segments: list[str], separator: str = "\n\n") -> list[str]:
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0
        for seg in segments:
            sl = len(seg) + len(separator)
            if current_len + sl <= self._chunk_size:
                current.append(seg)
                current_len += sl
            else:
                if current:
                    chunks.append(separator.join(current).strip())
                current = [seg]
                current_len = sl
        if current:
            chunks.append(separator.join(current).strip())

        # Add overlap between chunks
        if self._chunk_overlap > 0 and len(chunks) > 1:
            merged: list[str] = [chunks[0]]
            for i in range(1, len(chunks)):
                prev = chunks[i - 1]
                overlap = prev[-self._chunk_overlap:] if len(prev) > self._chunk_overlap else prev
                merged.append((overlap + separator + chunks[i]).strip())
            chunks = merged

        return [c for c in chunks if c.strip()]

    # ── Helpers ─────────────────────────────────────────────────────
    @staticmethod
    def _chunk_id(chunk: RagChunk) -> str:
        return f"{hashlib.md5(chunk.source_path.encode('utf-8', errors='ignore')).hexdigest()}_{chunk.chunk_index}_{chunk.file_hash[:12]}"

    @staticmethod
    def _distance_to_score(distance: float | None) -> float:
        """Normalize ChromaDB distance to 0–100 using 1/(1+d) scaling."""
        if distance is None:
            return 0.0
        return 100.0 * (1.0 / (1.0 + float(distance)))


class MultiUserRAGManager:
    """Manager that returns per-user PersonalRAGIndex instances."""

    def __init__(
        self,
        persist_directory: str,
        default_roots: Optional[list[str]] = None,
        cfg: Optional[dict] = None,
        health_monitor: Any = None,
    ) -> None:
        self.persist_directory = os.path.abspath(persist_directory)
        os.makedirs(self.persist_directory, exist_ok=True)
        self._indexes: Dict[str, PersonalRAGIndex] = {}
        self._default_roots = default_roots or []
        self._cfg = cfg or {}
        self._health_monitor = health_monitor

    def get_index_for_user(self, user_id: Optional[str] = None) -> PersonalRAGIndex:
        uid = (user_id or getpass.getuser() or "default").lower()
        if uid in self._indexes:
            return self._indexes[uid]
        idx = PersonalRAGIndex(
            persist_directory=self.persist_directory,
            user_id=uid,
            roots=self._cfg.get("roots") or self._default_roots,
            chunk_size=int(self._cfg.get("chunk_size", 600)),
            chunk_overlap=int(self._cfg.get("chunk_overlap", 100)),
            refresh_seconds=int(self._cfg.get("refresh_seconds", 600)),
            exclude_patterns=self._cfg.get("exclude_patterns", []),
            embedding_model=self._cfg.get("embedding_model", "BAAI/bge-base-en-v1.5"),
            index_schema_version=int(self._cfg.get("index_schema_version", INDEX_SCHEMA_VERSION)),
            max_context_chars=int(self._cfg.get("max_context_chars", 3000)),
            batch_size=int(self._cfg.get("batch_size", 256)),
            max_embedding_threads=int(self._cfg.get("max_embedding_threads", 4)),
            health_monitor=self._health_monitor,
        )
        idx.start_polling()
        self._indexes[uid] = idx
        return idx

    def remove_user(self, user_id: str) -> None:
        uid = user_id.lower()
        idx = self._indexes.pop(uid, None)
        if idx is not None:
            idx.stop_polling()
