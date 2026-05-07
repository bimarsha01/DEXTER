import sys
from pathlib import Path

# Ensure project root on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.config import get_config
from core.brain.memory import DexterMemory

cfg = get_config()
dm = DexterMemory(persist_directory=cfg.rag.persist_directory)
idx = dm.personal_rag

print("USER:", idx.user_id)
print("ROOTS:", idx._roots)

# Count supported files (no heavy parsing)
count = 0
sample = []
for root in idx._roots:
    p = Path(root)
    if not p.exists():
        continue
    for f in p.rglob("*"):
        try:
            if f.is_file() and idx._is_supported(str(f)) and not idx._is_excluded(str(f)):
                count += 1
                if len(sample) < 10:
                    sample.append(str(f))
        except Exception:
            continue

print("SUPPORTED_FILE_COUNT:", count)
print("SUPPORTED_SAMPLE:")
for s in sample:
    print(" -", s)

# Quick search sample
q = "project"
try:
    results = idx.search(q, limit=3)
    print("SEARCH_RESULTS_COUNT:", len(results))
    for r in results:
        print(" -", r.get('title') or r.get('path'), "score=", round(r.get('score',0),2))
except Exception as e:
    print("SEARCH_ERROR:", e)

# Can RAG answer simple question about a local file? Try build_context
try:
    ctx = idx.build_context('what is in my projects folder', limit=2)
    print("CONTEXT_PREVIEW:\n", ctx[:800])
except Exception as e:
    print("BUILD_CONTEXT_ERROR:", e)

print("DONE")
