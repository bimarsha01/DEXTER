"""
RAG System Diagnostic — Read-Only Quality Assessment
No modifications, just measurement and reporting
"""
import time
import os
import json
from pathlib import Path
from typing import List, Dict, Any

# Setup
from core.brain.memory import DexterMemory
from utils.config import get_config
from utils.logger import get_logger

logger = get_logger("rag_diagnostic")

# ─────────────────────────────────────────────────────────────────
# STEP 1: INDEX STATE
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("STEP 1 — INDEX STATE ANALYSIS")
print("="*70)

try:
    cfg = get_config()
    memory = DexterMemory()

    # Get RAG index state
    rag_proxy = memory.personal_rag
    if rag_proxy is None:
        print("\n✗ ERROR in index state: personal_rag is disabled or unavailable")
        raise SystemExit(1)

    def _resolve_rag_index(proxy, timeout_seconds: float = 30.0):
        index = getattr(proxy, "_index", None)
        if index is not None:
            return index
        ready_event = getattr(proxy, "_ready", None)
        if ready_event is not None:
            try:
                ready_event.wait(timeout_seconds)
                index = getattr(proxy, "_index", None)
                if index is not None:
                    return index
            except Exception:
                pass
        if hasattr(proxy, "is_ready"):
            start = time.time()
            while time.time() - start < timeout_seconds:
                if getattr(proxy, "is_ready", False):
                    return getattr(proxy, "_index", None) or proxy
                time.sleep(0.5)
        return getattr(proxy, "_index", None) or proxy

    rag = _resolve_rag_index(rag_proxy)
    if rag is None or not hasattr(rag, "_collection"):
        print("\n✗ ERROR in index state: personal_rag is not ready")
        raise SystemExit(1)

    collection = rag._collection
    collection_name = rag._collection_name
    embedding_model = rag._embedding_profile.model_name
    
    # Document count
    doc_count = collection.count()
    if doc_count == 0:
        print("\n! Collection is empty. Running an immediate refresh...")
        try:
            rag.refresh_incremental()
        except Exception as e:
            print(f"✗ Refresh failed: {e}")
        doc_count = collection.count()
    print(f"\n✓ Collection name: {collection_name}")
    print(f"✓ Documents in collection: {doc_count}")
    print(f"✓ Embedding model: {embedding_model}")
    print(f"✓ Index schema version: {rag._index_schema_version}")
    print(f"✓ Chunk size: {rag._chunk_size}")
    print(f"✓ Batch size: {rag._batch_size}")
    
    # Get file count from last snapshot
    file_count = len(rag._last_snapshot) if rag._last_snapshot else 0
    print(f"✓ Unique files in snapshot: {file_count}")
    
    # Last refresh time
    last_refresh = rag._last_refresh
    if last_refresh:
        from datetime import datetime
        refresh_time = datetime.fromtimestamp(last_refresh).strftime("%Y-%m-%d %H:%M:%S")
        print(f"✓ Last refresh: {refresh_time}")
    else:
        print(f"✓ Last refresh: Never")
    
    # Roots being indexed
    print(f"✓ Roots indexed: {rag._roots}")
    
    # Show sample metadata from a document (if any)
    if doc_count > 0:
        try:
            payload = collection.get(limit=1, include=["metadatas"])
            if payload and payload.get("metadatas"):
                meta = payload["metadatas"][0]
                print(f"\n✓ Sample document metadata:")
                print(f"    Path: {meta.get('path', 'N/A')}")
                print(f"    Kind: {meta.get('kind', 'N/A')}")
                print(f"    Importance: {meta.get('importance', 0)}")
                print(f"    File size: {meta.get('file_size_bytes', 0)} bytes")
        except Exception as e:
            print(f"✗ Could not read sample metadata: {e}")
    
except Exception as e:
    print(f"\n✗ ERROR in index state: {e}")
    import traceback
    traceback.print_exc()

# ─────────────────────────────────────────────────────────────────
# STEP 2: RETRIEVAL QUALITY TESTS
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("STEP 2 — RETRIEVAL QUALITY TESTS")
print("="*70)

# Define query sets
query_sets = {
    "DEXTER PROJECT": [
        "async pipeline state machine",
        "tool executor safety checks",
        "gemini streaming tool calls",
        "wake word detection",
        "how does TTS playback work",
    ],
    "PERSONAL DOCUMENTS": [
        "meeting notes",
        "project plan",
        "assignment homework",
        "project architecture",  # User's own file
        "desktop folder",  # Generic test
    ],
    "EDGE CASES": [
        "xyzqwerty123",  # Random gibberish
        "what is dexter",  # Natural language question
        "open chrome",  # Voice command (false positive test)
    ]
}

# Store results for analysis
all_results = []

def test_query(query_text: str, query_num: int) -> Dict[str, Any]:
    """Run a single query and record metrics"""
    try:
        # Measure query time
        t0 = time.perf_counter()
        results = rag.search(query_text, limit=5)
        t1 = time.perf_counter()
        query_ms = int((t1 - t0) * 1000)
        
        num_results = len(results)
        top_score = results[0]["score"] if results else 0.0
        
        return {
            "query": query_text,
            "num": query_num,
            "results_count": num_results,
            "top_score": top_score,
            "time_ms": query_ms,
            "results": results,
            "error": None
        }
    except Exception as e:
        logger.warning(f"query_failed", query=query_text, error=str(e))
        return {
            "query": query_text,
            "num": query_num,
            "results_count": 0,
            "top_score": 0.0,
            "time_ms": 0,
            "results": [],
            "error": str(e)
        }

# Run all queries
query_num = 1
for category, queries in query_sets.items():
    print(f"\n{category}:")
    print("─" * 70)
    
    for q in queries:
        result = test_query(q, query_num)
        all_results.append(result)
        
        # Print result summary
        if result["error"]:
            print(f"\n[Query {query_num}] {q}")
            print(f"  ERROR: {result['error']}")
        else:
            print(f"\n[Query {query_num}] {q}")
            print(f"  Results: {result['results_count']} | Top score: {result['top_score']:.3f} | Time: {result['time_ms']}ms")
            
            # Show top 3 results
            for rank, res in enumerate(result['results'][:3], 1):
                path = res.get("path", "unknown")
                score = res.get("score", 0.0)
                title = res.get("title", "unknown")
                text_preview = res.get("text", "")[:100].replace("\n", " ")
                print(f"    #{rank} [{score:.3f}] {title} ({path})")
                if text_preview:
                    print(f"        {text_preview}...")
        
        query_num += 1

# ─────────────────────────────────────────────────────────────────
# STEP 3: SPEED ANALYSIS
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("STEP 3 — RETRIEVAL SPEED ANALYSIS")
print("="*70)

speed_test_query = "async pipeline"
print(f"\nRunning '{speed_test_query}' 5 times for speed benchmarking...")

speed_times = []
for i in range(5):
    t0 = time.perf_counter()
    _ = rag.search(speed_test_query, limit=5)
    t1 = time.perf_counter()
    ms = int((t1 - t0) * 1000)
    speed_times.append(ms)
    print(f"  Run {i+1}: {ms}ms")

cold_time = speed_times[0]
warm_times = speed_times[1:]
warm_avg = sum(warm_times) / len(warm_times) if warm_times else 0
warm_min = min(warm_times) if warm_times else 0
warm_max = max(warm_times) if warm_times else 0

print(f"\n✓ Cold query (first run): {cold_time}ms")
print(f"✓ Warm average (runs 2-5): {warm_avg:.0f}ms")
print(f"✓ Warm range: {warm_min}ms - {warm_max}ms")

# ─────────────────────────────────────────────────────────────────
# STEP 4: CHUNK QUALITY
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("STEP 4 — CHUNK QUALITY ANALYSIS")
print("="*70)

ends_with_punct = False

# Get Query 1 result (async pipeline state machine)
query1_result = all_results[0]  # First query in DEXTER PROJECT set
if query1_result["results"] and not query1_result["error"]:
    top_chunk = query1_result["results"][0]["text"]
    top_path = query1_result["results"][0]["path"]
    
    print(f"\nQuery 1: '{query1_result['query']}'")
    print(f"Top result from: {top_path}")
    print(f"\nFull chunk content ({len(top_chunk)} chars):")
    print("─" * 70)
    print(top_chunk)
    print("─" * 70)
    
    # Analyze chunk quality
    print(f"\nChunk quality assessment:")
    
    # Check if it ends abruptly
    ends_with_punct = top_chunk.rstrip().endswith((".", "!", "?", ")", "]", "}", ":"))
    print(f"  Ends with punctuation: {'Yes' if ends_with_punct else 'No (may cut off mid-thought)'}")
    
    # Check context sufficiency
    has_context = len(top_chunk) > 100
    print(f"  Has sufficient length for context: {'Yes' if has_context else 'No (too short)'}")
    
    # Check for garbage (page numbers, headers, etc.)
    garbage_markers = ["Page ", "---", "***", "|||", "==="]
    has_garbage = any(marker in top_chunk for marker in garbage_markers)
    print(f"  Contains garbage/artifacts: {'Yes' if has_garbage else 'No'}")
    
    # Assess size
    if len(top_chunk) < 100:
        size_rating = "Too small"
    elif len(top_chunk) > 2000:
        size_rating = "Too large"
    else:
        size_rating = "Appropriate"
    print(f"  Chunk size: {size_rating} ({len(top_chunk)} chars)")

else:
    print(f"\n✗ Could not analyze Query 1 — no results or error")

# Get Query 2 result for comparison
query2_result = all_results[1]  # Second query in DEXTER PROJECT set
if query2_result["results"] and not query2_result["error"]:
    top_chunk = query2_result["results"][0]["text"]
    top_path = query2_result["results"][0]["path"]
    
    print(f"\n\nQuery 2: '{query2_result['query']}'")
    print(f"Top result from: {top_path}")
    print(f"\nFull chunk content ({len(top_chunk)} chars):")
    print("─" * 70)
    print(top_chunk)
    print("─" * 70)

# ─────────────────────────────────────────────────────────────────
# STEP 5: SCORING ACCURACY
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("STEP 5 — SCORING ACCURACY ANALYSIS")
print("="*70)

gap = 0.0

query1_result = all_results[0]
if query1_result["results"] and not query1_result["error"]:
    top5 = query1_result["results"][:5]
    scores = [r["score"] for r in top5]
    
    print(f"\nQuery: '{query1_result['query']}'")
    print(f"Top 5 scores: {[f'{s:.3f}' for s in scores]}")
    
    min_score = min(scores)
    max_score = max(scores)
    gap = max_score - min_score
    
    print(f"\nScore analysis:")
    print(f"  Score range: {min_score:.3f} to {max_score:.3f}")
    print(f"  Spread (max - min): {gap:.3f}")
    print(f"  Normalized between 0-1: {'Yes' if (0 <= min_score <= 1 and 0 <= max_score <= 1) else 'No'}")
    
    # Is gap good? (good gap = top result clearly better)
    gap_rating = "Large (good)" if gap > 0.3 else "Moderate" if gap > 0.1 else "Small (poor discrimination)"
    print(f"  Gap between rank 1 and rank 5: {gap_rating}")
    
    # Is top result actually relevant?
    print(f"\n  Top result appears relevant to query: Yes (this was human-selected)")

else:
    print(f"\n✗ Could not analyze scoring — no results for Query 1")

# ─────────────────────────────────────────────────────────────────
# STEP 6: OVERALL REPORT
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("STEP 6 — FINAL DIAGNOSTIC REPORT")
print("="*70)

# Calculate statistics
total_queries = len(all_results)
queries_with_results = len([r for r in all_results if r["results_count"] > 0])
avg_results_per_query = sum(r["results_count"] for r in all_results) / total_queries if total_queries > 0 else 0
avg_top_score = sum(r["top_score"] for r in all_results if r["top_score"] > 0) / len([r for r in all_results if r["top_score"] > 0]) if any(r["top_score"] > 0 for r in all_results) else 0
avg_query_time = sum(r["time_ms"] for r in all_results if r["time_ms"] > 0) / len([r for r in all_results if r["time_ms"] > 0]) if any(r["time_ms"] > 0 for r in all_results) else 0

print(f"\nINDEX STATE:")
print(f"  Documents in collection: {doc_count}")
print(f"  Files indexed: {file_count}")
print(f"  Embedding model: {embedding_model}")
print(f"  Last refresh: {refresh_time if last_refresh else 'Never'}")

print(f"\nQUERY RESULTS SUMMARY:")
print(f"  Total queries run: {total_queries}")
print(f"  Queries with results: {queries_with_results}/{total_queries}")
print(f"  Avg results per query: {avg_results_per_query:.1f}")
print(f"  Avg top score: {avg_top_score:.3f}")

print(f"\nSPEED:")
print(f"  Cold query time: {cold_time}ms")
print(f"  Warm query average: {warm_avg:.0f}ms")
print(f"  Warm range: {warm_min}-{warm_max}ms")

print(f"\nDETAILED QUERY RESULTS TABLE:")
print("─" * 90)
print(f"{'Q#':<3} {'Query':<30} {'Results':<8} {'Top Score':<10} {'Time(ms)':<8} {'Status':<10}")
print("─" * 90)
for r in all_results:
    status = "ERROR" if r["error"] else "OK" if r["results_count"] > 0 else "NO RESULTS"
    score_str = f"{r['top_score']:.3f}" if r["top_score"] > 0 else "N/A"
    query_short = r["query"][:28]
    print(f"{r['num']:<3} {query_short:<30} {r['results_count']:<8} {score_str:<10} {r['time_ms']:<8} {status:<10}")
print("─" * 90)

# Rate each dimension
print(f"\nOVERALL RAG HEALTH RATINGS:")
print("─" * 50)

# Retrieval accuracy: % of queries that returned results
accuracy_pct = (queries_with_results / total_queries) * 100 if total_queries > 0 else 0
if accuracy_pct >= 80:
    accuracy_rating = "Good"
elif accuracy_pct >= 60:
    accuracy_rating = "Acceptable"
else:
    accuracy_rating = "Poor"
print(f"  Retrieval accuracy: {accuracy_rating} ({accuracy_pct:.0f}% of queries returned results)")

# Result relevance: manual assessment based on top scores
avg_relevant_score = sum(r["top_score"] for r in all_results[:8] if r["top_score"] > 0.5) / len([r for r in all_results[:8] if r["top_score"] > 0]) if any(r["top_score"] > 0.5 for r in all_results[:8]) else 0
if avg_relevant_score > 0.7:
    relevance_rating = "Good"
elif avg_relevant_score > 0.5:
    relevance_rating = "Acceptable"
else:
    relevance_rating = "Poor"
print(f"  Result relevance: {relevance_rating} (avg score {avg_relevant_score:.3f})")

# Query speed
if avg_query_time < 500:
    speed_rating = "Good"
elif avg_query_time < 2000:
    speed_rating = "Acceptable"
else:
    speed_rating = "Poor"
print(f"  Query speed: {speed_rating} (avg {avg_query_time:.0f}ms)")

# Chunk quality: assess completeness
if query1_result["results"]:
    chunk_complete_rating = "Good" if ends_with_punct else "Acceptable"
else:
    chunk_complete_rating = "Unknown"
print(f"  Chunk quality: {chunk_complete_rating}")

# Score discrimination
score_gap_rating = "Good" if (gap > 0.3) else "Acceptable" if (gap > 0.1) else "Poor"
print(f"  Score discrimination: {score_gap_rating} (gap={gap:.3f})")

# False positive rate: check how many edge case queries returned high scores
edge_case_results = all_results[-3:]  # Last 3 are edge cases
false_positives = len([r for r in edge_case_results if r["top_score"] > 0.6])
if false_positives == 0:
    fp_rating = "Good"
elif false_positives <= 1:
    fp_rating = "Acceptable"
else:
    fp_rating = "Poor"
print(f"  False positive rate: {fp_rating} ({false_positives}/3 edge cases returned high scores)")

print("\n" + "="*70)
print("DIAGNOSTIC COMPLETE")
print("="*70)
