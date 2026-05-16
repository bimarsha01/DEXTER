import json
import time
import sys
from pathlib import Path

# Ensure project root is on sys.path so local packages import correctly
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.config import get_config
from core.brain.memory import DexterMemory

cfg = get_config()
print("CONFIG_RAG_PERSIST:", cfg.rag.persist_directory)

# Initialize DexterMemory (starts per-user RAG poller)
dm = DexterMemory(persist_directory=cfg.rag.persist_directory)
idx = dm.personal_rag

print("USER_ID:", getattr(idx, 'user_id', 'unknown'))
print("ROOTS:", idx._roots)
print("INITIAL_SNAPSHOT_COUNT:", len(idx._last_snapshot))

# Force an immediate incremental refresh to pick up files now
try:
    idx.refresh_incremental()
    print("REFRESHED_SNAPSHOT_COUNT:", len(idx._last_snapshot))
except Exception as e:
    print("REFRESH_FAILED:", str(e))

queries = ["project", "notes", "meeting", "Dexter"]
results = {}
for q in queries:
    try:
        res = idx.search(q, limit=3)
        ctx = idx.build_context(q, limit=2)
        results[q] = {"matches": res, "context_preview": ctx[:1000]}
        print(f"QUERY={q} -> {len(res)} matches")
    except Exception as e:
        results[q] = {"error": str(e)}
        print(f"QUERY={q} -> ERROR: {e}")

# Test recall_context via DexterMemory
try:
    recall = dm.recall_context("project", n_results=3)
    print("RECALL_CONTEXT_PREVIEW:", recall[:1000])
except Exception as e:
    print("RECALL_CONTEXT_FAILED:", str(e))

# Print a small diagnostic of indexed paths
print("SNAPSHOT_PATHS_SAMPLE:")
for i, p in enumerate(sorted(idx._last_snapshot.keys())[:10]):
    print(f" - {p}")

print("DONE")
