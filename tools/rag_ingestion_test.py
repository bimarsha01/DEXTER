import sys
from pathlib import Path

# Ensure project root on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.config import get_config
from core.brain.rag import MultiUserRAGManager

cfg = get_config()

# Test with a small targeted folder: the DEXTER project root itself
test_roots = [str(ROOT)]  # Index DEXTER project folder only (small, quick)

print("=" * 60)
print("TARGETED RAG INGESTION TEST")
print("=" * 60)

# Create a manager with test config (short refresh, small roots)
manager = MultiUserRAGManager(
    persist_directory=cfg.rag.persist_directory,
    default_roots=test_roots,
    cfg={
        "chunk_size": cfg.rag.chunk_size,
        "chunk_overlap": cfg.rag.chunk_overlap,
        "refresh_seconds": 30,  # Short for testing
        "exclude_patterns": cfg.rag.exclude_patterns,
        "roots": test_roots,
        "embedding_model": cfg.rag.embedding_model,
        "index_schema_version": cfg.rag.index_schema_version,
        "batch_size": cfg.rag.batch_size,
        "max_context_chars": cfg.rag.max_context_chars,
    },
)

user = "loq"
idx = manager.get_index_for_user(user)

print(f"USER: {idx.user_id}")
print(f"TEST_ROOTS: {idx._roots}")
print(f"COLLECTION: {idx._collection_name}")
print(f"EMBEDDING MODEL: {idx._embedding_profile.model_name}")
print()

# Stop the background poller temporarily (to manually control refresh)
idx.stop_polling()

# Force a full incremental refresh on small folder
print("Running incremental refresh on small test folder...")
try:
    idx.refresh_incremental()
    print(f"✓ Refresh completed")
    print(f"  Snapshot paths: {len(idx._last_snapshot)}")
except Exception as e:
    print(f"✗ Refresh failed: {e}")
    import traceback
    traceback.print_exc()

print()

# Try some searches
test_queries = [
    "project architecture",
    "pipeline async",
    "memory rag chromadb",
    "health monitor",
    "main.py",
]

print("Testing search queries:")
for q in test_queries:
    try:
        results = idx.search(q, limit=2)
        print(f"\nQuery: '{q}'")
        print(f"  Found: {len(results)} results")
        for i, r in enumerate(results, 1):
            score = r.get("score", 0)
            title = r.get("title", "unknown")
            print(f"    {i}. {title} (score: {score:.2f})")
            excerpt = r.get("text", "")[:100].replace("\n", " ")
            print(f"       {excerpt}...")
    except Exception as e:
        print(f"  Error: {e}")

print()
print("=" * 60)
print("TEST COMPLETE")
print("=" * 60)

# Restart background poller for production
idx.start_polling()
print("\nBackground poller restarted.")
