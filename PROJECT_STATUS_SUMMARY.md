# Dexter Project Status Summary

As of May 2, 2026.

## Executive Summary
Dexter is no longer just a proof-of-concept voice loop. The codebase now has a real async control plane, a safer tool execution path, explicit tool schemas, a more robust wake-word detector, lower-latency TTS, typed config validation, structured logging foundations, a pre-MCP audit step, and a new smoke-test suite. The current focus is not “make the assistant speak at all”; it is “make the assistant stable, observable, and safe enough to support MCP/FastMCP and a GUI later.”

The most recent work completed in this session was:
- config cleanup around typed settings usage
- logging hardening and correlation-ID support
- a safe streamed Groq tool-call path that buffers tool-call fragments before execution
- a new minimal `tests/` package with smoke tests
- a successful `tools.pre_mcp_audit` run and a passing unittest smoke run

## What Has Happened So Far

### 1. Core Assistant Architecture Was Rebuilt Around an Async Pipeline
The old blocking main loop was replaced by an async pipeline in [core/pipeline.py](core/pipeline.py).

What that changed in practice:
- startup now initializes assistant components once, then hands control to a pipeline loop
- the assistant has explicit state transitions instead of ad hoc flags
- wake detection, response streaming, and speech playback are coordinated rather than mixed together in one loop
- the pipeline supports per-utterance tracing and event emission

Supporting pieces that were added around that core:
- [core/state_machine.py](core/state_machine.py) for assistant state tracking
- [core/event_bus.py](core/event_bus.py) for lightweight event dispatch
- [core/wake_word/__init__.py](core/wake_word/__init__.py) and [core/wake_word/detector.py](core/wake_word/detector.py) for wake detection logic

### 2. Wake-Word Handling Became More Realistic
Wake detection was moved away from naive substring matching.

Current behavior:
- the wake phrase is simplified to `hey`
- the detector is token-aware instead of just scanning raw text
- matching uses normalization and confidence-aware logic
- false positives are reduced compared to substring matching
- once wake is detected, the assistant stays active for a 30-second wake window

Practical result:
- the user does not need to repeat the wake word for every follow-up command within the active window
- the assistant is much less likely to trigger accidentally on unrelated phrases

### 3. Audio Capture and TTS Were Tightened Up
Audio handling was improved in two key directions: capture reliability and playback latency.

Capture-side changes:
- [core/audio/vad.py](core/audio/vad.py) now uses unique temporary filenames rather than a shared `temp_mic.wav`
- that avoids collisions when the assistant records repeatedly or when multiple flows overlap

Playback-side changes:
- TTS now prefers `pygame-ce` for lower-latency playback
- fallback playback paths still exist for compatibility
- streaming sentence-level output is supported so the assistant can start speaking earlier instead of waiting for a full response

### 4. LLM Routing Was Expanded Into a Multi-Provider System
The router in [core/brain/llm_router.py](core/brain/llm_router.py) now routes across multiple providers:
- Gemini as the primary provider
- Groq as fallback
- Ollama as offline/local fallback

What changed architecturally:
- conversation history is token-aware instead of relying on a fixed message count
- intent routing still exists
- tool execution still exists
- the router now has streaming helpers for text output
- the Groq path was just extended with safer streamed tool-call handling

Current streaming state:
- text streaming exists
- Groq streaming now buffers streamed tool-call fragments before executing them
- tool calls are not executed from partial payloads
- after buffered tool execution, the follow-up response is streamed back out

What this means right now:
- streamed output is more practical and safer than before
- the assistant can move toward streamed tool use without immediately risking half-parsed arguments

### 5. Tool Execution Was Hardened
Tool execution was moved behind a safer runtime in [tools/executor.py](tools/executor.py).

Current safety checks include:
- tool existence checks
- JSON schema validation
- execution timeout enforcement
- path validation
- unsafe string rejection
- non-throwing result handling

Tool registry and schema work:
- [tools/registry.py](tools/registry.py) routes tool execution through the executor
- [tools/schemas/tool_schemas.json](tools/schemas/tool_schemas.json) contains explicit schemas
- [tools/schema_registry.py](tools/schema_registry.py) loads schema definitions
- [tools/audit_tool_schemas.py](tools/audit_tool_schemas.py) verifies that registered tools and schemas stay in sync

Current audit status:
- schema coverage is complete
- the audit reported 27 tools matched by 27 schemas

### 6. Config Handling Was Moved Toward Typed Validation
The config layer in [utils/config.py](utils/config.py) now has a typed `DexterConfig` model and runtime validation.

Current config shape:
- environment variables and `.env` are the source of secret values such as API keys
- typed config is available for runtime use
- compatibility helpers still exist so older code paths continue to work while the migration finishes

What was recently cleaned up:
- runtime code paths were moved away from direct raw YAML access where practical
- the app now uses typed config first in the startup path
- legacy dict conversion remains available for code that still expects it

### 7. Logging Got a Real Foundation
[utils/logger.py](utils/logger.py) was improved so the project can move beyond plain console prints.

What is already in place:
- correlation-ID support
- structlog-based foundation
- JSON-oriented logging direction
- file rotation support in the logger layer

What this gives the project now:
- better traceability for a single user command across the assistant lifecycle
- a path toward log files that are actually useful for debugging and later GUI display

### 8. The Project Now Has a Pre-MCP Validation Step
A dedicated pre-MCP audit script exists in [tools/pre_mcp_audit.py](tools/pre_mcp_audit.py).

That audit currently checks:
- config loads successfully
- tool schemas are in sync
- core imports resolve

Current result:
- the audit passed
- `ok` is true
- config, schema, and import checks are all green

### 9. A Smoke-Test Suite Now Exists
The project now has a `tests/` directory with a minimal smoke suite in [tests/test_dexter_smoke.py](tests/test_dexter_smoke.py).

The smoke tests currently cover:
- schema audit cleanliness
- wake-word detector prefix behavior
- config load and validation

Current test status:
- the unittest suite passes
- 3 tests ran successfully

## What Is Done Right Now

These items are effectively done or close enough to count as stable foundations:
- async pipeline core
- assistant state machine
- event bus foundation
- wake-word detector upgrade
- 30-second wake window
- low-latency TTS path
- unique temp audio filenames
- token-aware history pruning
- explicit tool schemas
- safe tool execution layer
- typed config foundation
- structured logging foundation
- pre-MCP audit tooling
- smoke-test baseline
- safe Groq streamed tool-call buffering

## What Is Partially Done

### 1. Structured Logging
This is no longer “missing”; it is started and useful, but not finished.

Already done:
- correlation IDs exist
- logger infrastructure exists
- log rotation support exists

Still incomplete:
- consistent structured logging across every module
- complete per-utterance trace propagation through all layers
- GUI log sink, since the GUI does not exist yet
- consistent log formatting rules across the entire codebase

### 2. Typed Config Migration
Typed config exists and is being used, but compatibility paths still remain.

Already done:
- `DexterConfig` exists
- validation exists
- startup and several runtime paths use typed config

Still incomplete:
- complete migration away from all legacy config access patterns
- elimination of compatibility workarounds once everything is updated
- deeper typed coverage for all config sections

### 3. Streaming Tool Calls
This is now partially implemented, not absent.

Already done:
- text streaming exists
- Groq streamed tool-call buffering exists
- follow-up streamed response handling exists after tool execution

Still incomplete:
- Gemini parity for streamed tool calls
- broader backend-specific handling if other providers expose different chunk shapes
- a dedicated set of tests for streamed tool-call parsing and replay

### 4. Security Hardening
The assistant is much safer than it was, but it is not fully hardened.

Already done:
- tool schema validation
- tool allow-listing through the registry
- timeout handling
- unsafe string rejection
- path validation

Still incomplete:
- broader permission scoping for every filesystem-related operation
- pre-commit or repo-level secret scanning
- deeper validation and normalization for all model-generated arguments
- a dedicated security layer for future MCP tools

### 5. Testing
Testing exists now, but coverage is still thin.

Already done:
- smoke tests for core facts
- audit verification for schema coverage
- startup/config validation checks

Still incomplete:
- unit tests for the pipeline internals
- regression tests for wake-word behavior
- regression tests for TTS playback behavior
- integration tests for tool execution
- tests for the streaming tool-call path

## What Is Still Left Completely

### 1. MCP / FastMCP Integration
This has not started yet.

What is still missing:
- `mcp_server/` package or equivalent server layer
- FastMCP server implementation
- stdio client or transport wrapper
- routing between native tools and MCP-exposed tools
- permission and path handling designed specifically for MCP

Why it is still left:
- the core assistant is being stabilized first
- the current priority is to avoid building MCP on top of unstable primitives

### 2. GUI
There is no desktop GUI yet.

What is missing:
- PyQt6 or other desktop UI layer
- tray icon or background control surface
- settings panel
- conversation display
- event-bus driven UI updates
- headless/GUI mode switching

### 3. Full Production-Grade Logging Pipeline
The logging foundation is in place, but the full production experience is not.

What is missing:
- complete structured file logging across every module
- a polished rotation strategy for long-term use
- trace-friendly logs that can be correlated across startup, wake, tool use, and response playback
- later GUI integration for visible logs

### 4. Final Architecture Glue
The application still needs the last layer that ties everything together cleanly.

What is missing:
- formal lifecycle management across startup, runtime, and shutdown
- a watchdog or launcher wrapper if needed
- consistent event emission from all major assistant phases
- a clean boundary between local native tools and future MCP tools

## Current Project Plan

### Immediate Next Work
1. Finish the remaining streaming-tool-call parity work, especially Gemini.
2. Add more tests around the new streaming and assistant control paths.
3. Keep the typed config/logging migration consistent instead of reintroducing raw access patterns.

### Next Major Milestone
1. Start MCP/FastMCP only after the core assistant path stays stable under the new tests and audit checks.
2. Keep MCP small at first: server layer, tool exposure, and transport plumbing before any UI.

### Later Milestones
1. Build the GUI once the assistant core and MCP boundary are stable.
2. Add richer observability and log display once the GUI exists.
3. Expand integration coverage and runtime hardening as the project approaches a real desktop product.

## Current Practical Status
Dexter is now in the “core stable, platform still unfinished” phase.

That means:
- the assistant can already do the main voice-loop job in a structured way
- tool execution is much safer than before
- config and logging are no longer ad hoc
- basic validation is now present
- the project is ready for the next layer of work, but MCP and GUI should still wait until the remaining core gaps are closed

The biggest remaining risk is not the voice loop itself. It is finishing the production layer cleanly enough that MCP, GUI, and long-term maintenance do not inherit unstable behavior.
