#!/usr/bin/env python
"""Full RAG request pipeline test — exercises search, context building,
and recall through DexterMemory, validating the end-to-end retrieval
path that the live pipeline uses when answering user questions.

Usage:
    python tools/rag_full_request.py                          # default queries
    python tools/rag_full_request.py "my custom question"     # single custom query
    python tools/rag_full_request.py --quick                  # fast test (DEXTER folder only)
    python tools/rag_full_request.py --skip-refresh           # skip indexing, search existing
    python tools/rag_full_request.py --quick --skip-refresh   # fastest: search existing DEXTER index
"""
from __future__ import annotations

import argparse
import getpass
import sys
import time
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.brain.rag import MultiUserRAGManager
from core.brain.memory import DexterMemory
from utils.config import get_config


# ── colour helpers (safe on Windows) ────────────────────────────────
def _cyan(text: str) -> str:
    return f"\033[96m{text}\033[0m"


def _green(text: str) -> str:
    return f"\033[92m{text}\033[0m"


def _yellow(text: str) -> str:
    return f"\033[93m{text}\033[0m"


def _dim(text: str) -> str:
    return f"\033[90m{text}\033[0m"


def _bold(text: str) -> str:
    return f"\033[1m{text}\033[0m"


# ── default queries ─────────────────────────────────────────────────
DEFAULT_QUERIES: List[str] = [
    "project architecture",
    "what does the pipeline do",
    "how does memory work",
    "rag chromadb search",
    "wake word detection",
    "tool executor timeout",
    "system status health monitor",
    "what files are on my desktop",
]


def _print_banner(title: str) -> None:
    width = 72
    print()
    print(_cyan("=" * width))
    print(_cyan(f"  {title}"))
    print(_cyan("=" * width))


def _print_section(title: str) -> None:
    print()
    print(_bold(f"── {title} " + "─" * max(0, 56 - len(title))))


def _run_rag_search(
    manager: MultiUserRAGManager, user: str, queries: List[str], skip_refresh: bool = False
) -> None:
    """Run raw RAG index searches and display results."""
    idx = manager.get_index_for_user(user)
    # Ensure the index is fresh (one-shot, no background polling)
    idx.stop_polling()

    _print_section("Index Stats")
    print(f"  User ID       : {idx.user_id}")
    print(f"  Roots         : {idx._roots}")
    print(f"  Snapshot files: {len(idx._last_snapshot)}")
    print(f"  Chunk size    : {idx._chunk_size}  |  Overlap: {idx._chunk_overlap}")

    # Incremental refresh to guarantee latest files are indexed
    _print_section("Incremental Refresh")
    if skip_refresh:
        print(f"  {_dim('(skipped via --skip-refresh)')}")  
    else:
        t0 = time.perf_counter()
        idx.refresh_incremental()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"  Refresh completed in {elapsed_ms:.0f} ms")
    print(f"  Snapshot files after refresh: {len(idx._last_snapshot)}")

    # Run search queries
    _print_section("RAG Search Results")
    for query in queries:
        print()
        print(f"  {_bold('QUERY')}: {_yellow(query)}")
        t0 = time.perf_counter()
        results = idx.search(query, limit=3, use_cache=False)
        search_ms = (time.perf_counter() - t0) * 1000
        print(f"  {_dim(f'({len(results)} results in {search_ms:.0f} ms)')}")

        if not results:
            print(f"    {_dim('No matches found.')}")
            continue

        for i, r in enumerate(results, start=1):
            title = r.get("title", "")
            score = r.get("score", 0.0)
            raw_vs = r.get("raw_vector_score", 0.0)
            path = r.get("path", "")
            excerpt = (r.get("text") or "")[:120].replace("\n", " ").strip()

            score_color = _green if score >= 50 else (_yellow if score >= 30 else _dim)
            print(f"    {i}. {title}")
            print(f"       Score: {score_color(f'{score:.2f}')}  |  Vector: {raw_vs:.2f}  |  Path: {_dim(path)}")
            if excerpt:
                print(f"       {_dim(excerpt)}{'...' if len(r.get('text', '')) > 120 else ''}")


def _run_context_build(manager: MultiUserRAGManager, user: str, queries: List[str]) -> None:
    """Test the build_context method (what the LLM actually receives)."""
    idx = manager.get_index_for_user(user)
    idx.stop_polling()

    _print_section("Context Building (LLM Input)")
    for query in queries[:4]:  # Limit to avoid excessive output
        print()
        print(f"  {_bold('QUERY')}: {_yellow(query)}")
        t0 = time.perf_counter()
        context = idx.build_context(query, limit=3, summary=False)
        ctx_ms = (time.perf_counter() - t0) * 1000
        if context:
            lines = context.splitlines()
            print(f"  {_dim(f'({len(lines)} lines, {len(context)} chars, {ctx_ms:.0f} ms)')}")
            for line in lines[:8]:
                print(f"    {line}")
            if len(lines) > 8:
                print(f"    {_dim(f'... +{len(lines) - 8} more lines')}")
        else:
            print(f"    {_dim('(empty context)')}")


def _run_memory_recall(queries: List[str]) -> None:
    """Test the full DexterMemory recall_context path (conversation + RAG)."""
    cfg = get_config()

    _print_section("DexterMemory Recall (Conversation + RAG)")
    dm = DexterMemory(persist_directory=cfg.rag.persist_directory)
    print(f"  Memory documents : {dm.get_memory_count()}")
    print(f"  RAG user         : {dm.personal_rag.user_id}")

    for query in queries[:4]:
        print()
        print(f"  {_bold('QUERY')}: {_yellow(query)}")
        t0 = time.perf_counter()
        recall = dm.recall_context(query, n_results=3)
        recall_ms = (time.perf_counter() - t0) * 1000

        if recall:
            lines = recall.splitlines()
            print(f"  {_dim(f'({len(lines)} lines, {len(recall)} chars, {recall_ms:.0f} ms)')}")
            for line in lines[:10]:
                print(f"    {line}")
            if len(lines) > 10:
                print(f"    {_dim(f'... +{len(lines) - 10} more lines')}")
        else:
            print(f"    {_dim('(no recall context)')}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dexter RAG full request pipeline test",
    )
    parser.add_argument(
        "query", nargs="*", default=None,
        help="Custom query string(s). If omitted, uses built-in defaults.",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Index only the DEXTER project folder for a fast test.",
    )
    parser.add_argument(
        "--skip-refresh", action="store_true", dest="skip_refresh",
        help="Skip the incremental refresh and search the existing index.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    cfg = get_config()
    user = getpass.getuser().lower()

    queries = [" ".join(args.query)] if args.query else DEFAULT_QUERIES

    # In quick mode, scope indexing to just the DEXTER project root
    if args.quick:
        roots = [str(ROOT)]
        exclude = [".venv", "__pycache__", ".git", "memory_db", ".pytest_cache"]
    else:
        roots = cfg.rag.personal_roots
        exclude = cfg.rag.exclude_patterns

    mode_label = "QUICK (DEXTER only)" if args.quick else "FULL"
    _print_banner(f"DEXTER RAG FULL REQUEST TEST — {mode_label}")
    print(f"  User   : {user}")
    print(f"  Roots  : {roots}")
    print(f"  Queries: {len(queries)}")
    print(f"  Refresh: {'skip' if args.skip_refresh else 'enabled'}")

    manager = MultiUserRAGManager(
        persist_directory=cfg.rag.persist_directory,
        default_roots=roots,
        cfg={
            "roots": roots,
            "chunk_size": cfg.rag.chunk_size,
            "chunk_overlap": cfg.rag.chunk_overlap,
            "refresh_seconds": cfg.rag.refresh_seconds,
            "exclude_patterns": exclude,
            "embedding_model": cfg.rag.embedding_model,
            "index_schema_version": cfg.rag.index_schema_version,
            "batch_size": cfg.rag.batch_size,
            "max_context_chars": cfg.rag.max_context_chars,
        },
    )

    # Phase 1: Raw RAG search
    _run_rag_search(manager, user, queries, skip_refresh=args.skip_refresh)

    # Phase 2: Context building (what gets injected into LLM prompts)
    _run_context_build(manager, user, queries)

    # Phase 3: Full DexterMemory recall (conversation history + RAG combined)
    _run_memory_recall(queries)

    _print_banner("RAG FULL REQUEST TEST COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
