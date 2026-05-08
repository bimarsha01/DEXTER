# DEXTER Project — Comprehensive Technical Report

Generated: 2026-05-08

This document is an exhaustive technical report for the DEXTER project in this workspace. It covers project purpose, features, architecture, components, file-level descriptions, design decisions, recent changes, test results, runtime instructions, and recommended next steps. Use this as a single reference for understanding the entire codebase and recent work performed.

---

**Contents**

- Project Summary
- Key Features
- Architecture Overview
- Components & File Map (detailed)
- Technology Stack
- RAG (Retrieval-Augmented Generation) Design
- LLM Routing & Provider Logic
- Audio Pipeline (STT, VAD, TTS)
- Tools & Tooling
- Configuration and Hardware Detection
- Testing, CI, and Audits
- Recent Changes & Rationale
- Known Issues & Limitations
- How to Run & Useful Commands
- Next Steps and Recommendations
- Appendix: Modified files list (recent)

---

**Project Summary**

DEXTER is an offline-first, extensible voice assistant / agent framework. It orchestrates audio input, speech-to-text (STT), voice activity detection (VAD), large language model (LLM) routing, retrieval-augmented generation (RAG) personal context, and tool invocation. The system is designed to support multiple LLM providers and local tools while remaining configurable for low-power or constrained environments.

Primary goals:

- Provide reliable, context-aware assistant responses using personal RAG context.
- Minimize noisy or irrelevant RAG results and ensure manifest persistence.
- Reduce startup time by lazy-loading heavy models and optionally enabling low-power runtime modes.
- Improve audio robustness (avoid feedback loops / static noise) and concise verbal responses.
- Provide a tools framework enabling function-calling style behaviors (document readers, open resolvers, system tools).

**Key Features**

- Multi-provider LLM routing: Google Gemini (cloud), Groq (cloud), and Ollama (local) with provider-specific prompt handling and graceful fallbacks.
- Personal RAG using a persistent Chroma DB index for per-user document retrieval and injection into LLM prompts.
- Atomic snapshot persistence for RAG index manifests to avoid corruption (tmp file + fsync + os.replace pattern).
- Relevance filtering for RAG search results (configurable MINIMUM_RELEVANCE_SCORE) and combined vector+text scoring.
- Audio stack:
  - STT via faster-whisper (lazy-initialized, GPU if available).
  - VAD via Silero with suppress_for(seconds) to avoid VAD triggering during TTS playback.
  - TTS via Edge-TTS writing into in-memory buffers and safe playback (pygame-based fallback), with improved interrupt handling.
- Tools registry and executor model for reusable capabilities (document reading, web browsing, system tools, vision tools, YouTube, etc.).
- Config-driven runtime modes and hardware detection for low-power operation.
- Summarization of tool outputs to avoid verbose spoken feedback.
- Tests and audit: pytest test suite and a pre-MCP audit script.

**Architecture Overview**

High-level flow:

1. System boot: configuration and hardware detection; optional safe-mode and low-power flag overrides.
2. Background warming (optional) for RAG and other heavy components; STT model is lazy-loaded on first use.
3. Main loop: VAD detects speech, audio is captured and passed to the transcriber.
4. Transcription: faster-whisper produces text; text is sanitized (e.g., ASR alias maps such as "Cut Mondo" → "San Mondo" pattern) and passed to pipeline.
5. Pipeline composes memory context and fetches personal RAG context asynchronously (with small timeout) to produce indexed_context.
6. LLM Router composes provider-specific prompts combining system instruction, long-term memory, and indexed_context, then calls provider streaming or batch endpoints. Provider-specific truncation is applied (Groq gets tighter RAG truncation).
7. Function-calling / tool invocation: if LLM selects a tool, the router invokes tools via the executor and summarizes results for the LLM.
8. Response streaming: streaming responses are voiced using TTS while VAD is suppressed to avoid feedback.

Concurrency model:

- Asyncio for overall pipeline and streaming LLM calls.
- Background threads (ThreadPoolExecutor) for blocking I/O (Chromadb, Whisper model loading, disk fsync) and for warming operations.

**Components & File Map (detailed)**

This section maps directories and important files with explanations of responsibilities and notable functions.

- config.yaml — Global configuration values used across the app.
- requirements.txt — Python dependencies for the project runtime; keep updated for heavy libraries (whisper, chromadb, transformers, google-genai, groq client, etc.).

- main.py
  - Boot orchestration: loads runtime config, detects hardware, instantiates memory, pipeline, audio, and starts the assistant main loop.
  - New features: safe-mode gating (`DEXTER_SAFE_MODE`), runtime flags to disable RAG warming and proactive modes for low-power.

- core/
  - __init__.py
  - event_bus.py — internal pub/sub for events between components.
  - pipeline.py — central orchestration of listen→transcribe→process→speak. Key functions:
    - _get_rag_context(query): fetches indexed RAG snippet with a small timeout (2s) via run_in_executor.
    - _stream_response(...) now accepts indexed_context and ensures VAD suppression around TTS.
    - Memory and rag contexts are composed separately to avoid duplication.

  - state_machine.py — conversational state machine.

  - audio/
    - __init__.py
    - transcriber.py — faster-whisper based STT with lazy model initialization (`_ensure_model`), GPU float16 when available, CPU safe fallbacks.
    - speaker.py — TTS playback using Edge-TTS to produce mp3 data; uses in-memory BytesIO or safe temporary files and pygame playback. TTSManager supports interrupts.
    - vad.py — Silero VAD wrapper with `suppress_for(seconds)` to temporarily ignore voice detection (used during and after TTS playback to avoid feedback loops).

  - brain/
    - __init__.py
    - intent_rag.py — helpers to build RAG contexts for intents (topic-specific behaviors).
    - intent_router.py — matches intents from NLU and routes to the correct action or tool; includes ASR sanitization heuristics (alias maps for misrecognized city names such as "Cut Mondo").
    - llm_router.py — core of LLM orchestration and function calling. Notable additions:
      - Rewritten system instruction to a human-friendly Dexter persona.
      - _compose_prompt(...) to build prompts from `user_command`, `long_term_memory`, and `indexed_context`.
      - _truncate_rag_for_provider(...) to aggressively trim RAG for Groq to avoid token blowups.
      - _summarize_tool_output(...) to produce short, human-friendly summaries of tool outputs before sending back to LLM or TTS.
      - process_command_stream(...) accepts an `indexed_context` parameter and routes responses to providers.

    - memory.py — conversational memory and personal RAG integration.
      - _LazyPersonalRAG: background warm loader for personal RAG index.
      - DexterMemory: `recall_context(query, include_personal_rag=True)` and support for disabling warm-up via runtime flags.

- tools/
  - __init__.py
  - executor.py — tool execution orchestration; manages tool input/output and error handling.
  - registry.py — tool registry mapping names to implementations.
  - open_resolver.py — resolves targets to actions (open app or ask for clarification). Important for function-calling flows.
  - document_tools.py — read_document, summarize_document, answer_document_question; updated to allow resolving a query to a document path using the RAG index when a path is not provided.
  - youtube_tool.py, vision_tools.py, web_browser.py, etc. — domain-specific tool implementations.
  - schemas/tool_schemas.json — JSON schema used to register tool signatures; updated to reflect answer_document_question accepting partial queries.

- utils/
  - __init__.py
  - config.py — pydantic-based configuration loader; added `RuntimeConfig` and flags `disable_rag_warming` and `disable_proactive_mode`.
  - hardware_detect.py — NEW. Detects system profile (RAM, CPU, GPU availability) and returns low-power overrides (recommended whisper model, embedding thread counts, and flags to disable RAG warming/proactive modes).
  - logger.py — project logging wrapper.
  - metrics.py — basic telemetry and metrics hooks.
  - transcript_correction.py — heuristics for cleaning ASR output.

- tests/
  - test_dexter_smoke.py — top-level smoke checks and ASR alias regression tests (e.g., "Cut Mondo").
  - test_rag_document_resolution.py — tests for `answer_document_question` path resolution via RAG.
  - test_gemini_streaming.py, test_vision_tools.py, test_youtube_tool.py — provider- and tool-specific tests.

**Technology Stack**

- Python (3.10+ recommended)
- Asyncio for concurrency
- Chromadb for persistent RAG index
- Embeddings: SentenceTransformer or BGE (BAAI/bge-base-en-v1.5 recommended)
- STT: faster-whisper (supports GPU float16 and CPU fallbacks)
- VAD: Silero VAD
- TTS: Edge-TTS (async or subprocess) with playback via pygame or platform-specific audio
- LLM Providers:
  - Google Gemini via google-genai SDK (streaming + function-calling)
  - Groq (AsyncGroq client) — used as fallback; careful with token usage
  - Ollama (local model) — optional local-only failover
- Utilities: rapidfuzz (text similarity), psutil (hardware detection), pydantic (config), pytest (testing)

**RAG Design & Scoring**

Key design points:

- PersonalRAGIndex uses vector search (Chroma) plus text similarity via rapidfuzz to compute a composite score.
- Final score composition (example used): final_score = 0.65 * vector_score + 0.30 * text_sim + 0.05 * importance
- MINIMUM_RELEVANCE_SCORE gate (configurable; default applied ~55.0) filters out irrelevant or low-value results before building the indexed_context block.
- build_context(...) returns a human-readable context snippet prefixed with a header like "RELEVANT PERSONAL FILE CONTEXT:" with numbered, truncated excerpts.
- Snapshot persistence uses atomic write pattern to ensure manifest integrity: write to tmp file, flush, os.fsync, os.replace.

Why composite scoring?

- Vector search finds semantic matches, but short filenames or exact matches benefit from text-sim boosting which catches partial-named projects and precise filename hits. Rapidfuzz partial ratio helps prioritize exact textual matches.

**LLM Routing & Provider Logic**

Overview:

- The system composes prompts in three parts: system_instruction (personality + safety), long_term_memory (conversation/context), and indexed_context (personal RAG results) + user command.
- Provider-specific logic:
  - Gemini: supports richer streaming and function-calling; preserves more RAG context.
  - Groq: lower token envelope — RAG is aggressively truncated to a single best result and shortened to ~300 characters to avoid token blowups and unexpected costs; Groq used as fallback when Gemini fails or rate-limits.
  - Ollama: local model with limited capacity; used for private/local-only responses.
- Function-calling/tool flows: when an LLM returns a function call, the router invokes the tool via tools/executor, summarizes the output with _summarize_tool_output, then returns a short summarized payload back into the LLM stream as the tool result.

Design rationale:

- Summarization prevents the TTS from speaking giant raw JSON or long tool dumps and keeps spoken responses human-friendly.
- Passing RAG in an explicit `indexed_context` argument avoids duplication in `memory.recall_context` and makes it easier to control provider-specific truncation.

**Audio Pipeline (STT, VAD, TTS)**

STT (Speech-to-Text):

- Implemented using faster-whisper; the model is lazy-initialized on first use to reduce startup time.
- When a GPU is available, the model uses float16 for speed; otherwise CPU with safe quantized or int8 fallback is used.

VAD (Voice Activity Detection):

- Silero VAD wrapper provides start/stop voice detection.
- `suppress_for(seconds)` API is used by the pipeline to disable VAD during TTS playback and for a short window after TTS completes. This prevents the assistant's own speech from re-triggering STT and causing feedback / the "krrrr" noise.

TTS (Text-to-Speech):

- Edge-TTS is used to generate mp3 audio bytes which are played back using pygame or another audio sink.
- Playback reads audio into BytesIO where possible to avoid repeatedly writing to disk; fallback temporary file handling is robust and safely deleted.
- TTSManager supports interruption: pipeline can cancel ongoing speech to reply sooner if needed.

**Tools & Tooling**

- Tools are registered in `tools/registry.py` and executed via `tools/executor.py`.
- Tools include:
  - Document tools: `answer_document_question`, `read_document`, `summarize_document`.
  - System tools: clipboard read/write, open app, file operations.
  - Web tools: `web_browser.py`, `youtube_tool.py`.
  - Vision helpers: `vision_tools.py` for image analysis.
- Tool schemas live in `tools/schemas/tool_schemas.json` and are used to generate function signatures and enforce contract for LLM function-calls.

Special behaviors:

- `answer_document_question` now accepts either a path or a freeform query. If a path is not found on disk, it queries the PersonalRAGIndex for candidate documents and resolves to the best match.
- `open_resolver` returns a status of `open` or `ask` (pending_action). The llm_router handles `ask` by creating a pending action for the user to confirm.

**Configuration & Hardware Detection**

Runtime configuration is provided by `utils/config.py` (pydantic) and by `config.yaml`.

Key runtime flags:

- `disable_rag_warming` — prevents background RAG index warm-up for low-power scenarios.
- `disable_proactive_mode` — disables the assistant’s proactive background behaviors.

Hardware detection (`utils/hardware_detect.py`):

- Uses psutil and optional torch GPU checks to determine system capabilities.
- Produces recommended overrides such as `whisper_model` size, embedding_thread_count, and flags to disable heavy background tasks.

**Testing, CI, and Pre-MCP Audit**

- Test suite: pytest. Tests are in `tests/` covering smoke tests, provider-specific flows, tool behaviors, and regression checks.
- Audit: `tools/pre_mcp_audit.py` verifies tools and schemas coverage before MCP packaging.
- Recent results (at time of report): full pytest run produced 26 passed, 2 warnings. pre_mcp_audit reported ok: true and tool/schema counts matched.

**Recent Changes & Rationale (detailed)**

This project has recently had a set of changes aimed at improving RAG usefulness, audio robustness, startup performance, and user-facing quality. Highlights:

Brief summary: Over the last changes we hardened the Project & Document Q&A path by adding a universal file reader, code-aware extraction, and a safer, configurable RAG ranking and boosting system. We added an in-memory session-scoped `current_project` slot to preserve follow-up context across turns and injected that into RAG queries when present. The LLM router gained provider-specific guards (Groq excerpt caps, Gemini fallback handling) to avoid token blowups and encourage concise spoken responses. These edits are narrowly scoped to document/RAG/LLM routing and avoid touching audio/VAD/TTS subsystems.

- Personal RAG changes (`core/brain/rag.py`):
  - Implemented atomic snapshot persistence to prevent manifest corruption.
  - Added MINIMUM_RELEVANCE_SCORE and composite scoring (vector+text_sim+importance) to filter irrelevant RAG results.
  - `build_context` now returns numbered truncated excerpts to limit token usage and focus relevance.

- Pipeline changes (`core/pipeline.py`):
  - Added `_get_rag_context(query)` which runs the RAG search with a 2s timeout and returns formatted indexed_context to LLM router.
  - Ensures `memory.recall_context(..., include_personal_rag=False)` is used to avoid double-including personal RAG content.
  - VAD suppression is applied around TTS playback to avoid feedback.

- LLM Router changes (`core/brain/llm_router.py`):
  - Rewrote system prompt to a more natural Dexter persona.
  - _compose_prompt and provider-focused truncation (_truncate_rag_for_provider) implemented — Groq receives aggressively truncated RAG.
  - Summarization of tool outputs before returning them to LLM to keep spoken responses concise.

- Audio improvements:
  - Lazy-load STT model (faster-whisper) to remove long startup delays.
  - VAD `suppress_for` used to avoid VAD triggering during TTS.
  - TTS playback adjusted to prefer in-memory audio to reduce disk churn and glitches.

- Document tools:
  - `answer_document_question` now resolves project names via RAG when a direct path is not provided — improves UX for asking about project docs by title.

- Hardware detection & runtime flags:
  - New `utils/hardware_detect.py` determines a safe low-power runtime profile and sets `disable_rag_warming` for low-end devices.

Why these changes?

- The main driver was user feedback about noisy/irrelevant RAG results, a manifest persistence bug, startup slowness (heavy model initialization), audio feedback during TTS playback, misrecognized city names, and overly verbose spoken outputs.

**Known Issues & Limitations**

- Provider limits: Google Gemini quota errors (429) were observed in logs; the system falls back to Groq but Groq requires aggressive truncation to avoid token costs.
- RAG tuning is heuristic: MINIMUM_RELEVANCE_SCORE and composite weights may need real-world tuning for different users and domains.
- Summarization heuristics may sometimes omit critical details — tuning summary length per tool type is recommended.
- Some heavy operations still rely on global threadpools and blocking I/O (Chromadb calls, disk fsync). Consider async-friendly Chroma or background workers for high-load cases.

**How to Run & Useful Commands**

Assuming a Python virtual environment is set up (use the provided .venv in this workspace):

1. Activate virtualenv (Windows PowerShell):

```powershell
(Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned) ; (& .venv\\Scripts\\Activate.ps1)
```

2. Install dependencies (if not already):

```bash
python -m pip install -r requirements.txt
```

3. Run test suite:

```bash
pytest tests -v
```

4. Run pre-MCP audit:

```bash
python tools/pre_mcp_audit.py
```

5. Start the assistant (development run):

```bash
python main.py
```

Environment flags:

- `DEXTER_SAFE_MODE=1` — start in safe diagnostics mode (disables TTS/VAD). Set via environment before running.

**Next Steps & Recommendations**

Immediate improvements to consider:

1. Fine-tune RAG scoring weights and MINIMUM_RELEVANCE_SCORE using a sample of real user queries and documents.
2. Add automated integration tests for RAG→LLM prompt composition to ensure the indexed_context is actually used in responses.
3. Extend summarization to vision and long document tool outputs with configurable summary lengths per tool type.
4. Add telemetry to capture RAG relevance acceptance (e.g., Was this result helpful?) to gather supervised feedback for reranking.
5. Consider replacing blocking Chromadb calls with an async-compatible adapter or dedicated worker process for scalability.

Security & privacy considerations:

- Personal RAG content may contain sensitive data. Ensure proper encryption-at-rest for the Chroma DB and access controls.
- When using cloud LLM providers, be mindful of PII leaving the local environment. Offer a privacy-safe mode that disables cloud providers and uses local models only.

**Appendix: Recent Modified Files**

Files changed during the recent work (high-level):

- core/brain/rag.py
- core/pipeline.py
- core/brain/llm_router.py
- core/brain/memory.py
- core/audio/transcriber.py
- core/audio/vad.py
- core/audio/speaker.py
- tools/document_tools.py
- utils/config.py
- utils/hardware_detect.py (new)
- main.py
- tools/schemas/tool_schemas.json
- tests/test_rag_document_resolution.py (new)
- tests/test_dexter_smoke.py (updated)

Refer to the Git diff or commit history for exact changed lines.

---

If you'd like, I can:

-+- Expand this report with code snippets and the exact diffs for each modified file.
-+- Create a summary slide deck or generate an architecture diagram (Mermaid) from this content.
-+- Run live audio acceptance tests and collect logs to further tune RAG thresholds and audio suppression windows.

Tell me which next action you prefer.
