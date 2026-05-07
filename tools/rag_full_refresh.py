#!/usr/bin/env python
"""Run one incremental refresh for the current OS user.

By default this uses the repository root so accidental laptop-draining runs are
avoided. Pass --all-personal-roots to index the configured Documents/Desktop/
Projects tree.
"""
from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.brain.rag import MultiUserRAGManager
from utils.config import get_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Dexter RAG refresh.")
    parser.add_argument(
        "--all-personal-roots",
        action="store_true",
        help="Index the configured personal roots from config.yaml.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    cfg = get_config()
    user = getpass.getuser().lower()

    roots = cfg.rag.personal_roots if args.all_personal_roots else [str(ROOT)]

    manager = MultiUserRAGManager(
        persist_directory=cfg.rag.persist_directory,
        default_roots=roots,
        cfg={
            "roots": roots,
            "chunk_size": cfg.rag.chunk_size,
            "chunk_overlap": cfg.rag.chunk_overlap,
            "refresh_seconds": cfg.rag.refresh_seconds,
            "exclude_patterns": cfg.rag.exclude_patterns,
            "embedding_model": cfg.rag.embedding_model,
            "index_schema_version": cfg.rag.index_schema_version,
            "batch_size": cfg.rag.batch_size,
            "max_context_chars": cfg.rag.max_context_chars,
            "max_embedding_threads": cfg.rag.max_embedding_threads,
        },
    )

    idx = manager.get_index_for_user(user)
    # Keep this one-shot and deterministic.
    idx.stop_polling()

    print("=" * 72)
    print("DEXTER RAG FULL REFRESH")
    print("=" * 72)
    print(f"USER: {user}")
    print(f"ROOTS: {idx._roots}")
    if args.all_personal_roots:
        print("MODE: all configured personal roots")
    else:
        print("MODE: repository root only (safe default)")
    print(f"COLLECTION: {idx._collection_name}")
    print(f"EMBEDDING MODEL: {idx._embedding_profile.model_name}")
    print(f"INDEX SCHEMA VERSION: {idx._index_schema_version}")
    print(f"CHUNK SIZE: {cfg.rag.chunk_size} | OVERLAP: {cfg.rag.chunk_overlap}")
    print(f"REFRESH SECONDS (CONFIG): {cfg.rag.refresh_seconds}")
    print("Starting refresh_incremental() ...")

    idx.refresh_incremental()

    print("Refresh completed.")
    print(f"SNAPSHOT FILE COUNT: {len(idx._last_snapshot)}")

    probe_queries = [
        "project architecture",
        "pipeline async",
        "memory rag chromadb",
    ]
    for q in probe_queries:
        results = idx.search(q, limit=3)
        print(f"QUERY: {q} | RESULTS: {len(results)}")
        for i, r in enumerate(results, start=1):
            print(
                f"  {i}. {r.get('title', '')} | score={r.get('score', 0):.2f} | path={r.get('path', '')}"
            )

    print("=" * 72)
    print("FULL REFRESH DONE")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
