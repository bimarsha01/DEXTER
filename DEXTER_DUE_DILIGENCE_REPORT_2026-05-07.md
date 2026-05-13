# DEXTER Product + Engineering Due Diligence Report

Date: 2026-05-07
Scope: Full workspace scan + runtime verification + external market context

## 1) Executive Snapshot
Dexter is a serious pre-product prototype with a strong architecture direction and real implementation depth. It is already beyond a demo: async pipeline, multi-LLM fallback, tool schemas, guardrails, memory, wake handling, metrics, and test coverage exist.

Current maturity (honest view):
- Technical maturity: late prototype / early 
- Product maturity: pre-commercial MVP
- Commercial maturity: not yet sale-ready without packaging, compliance, licensing, and support hardening

## 2) What Is Implemented Right Now

### Core architecture (working)
- Async orchestration pipeline: `core/pipeline.py`
- State machine: `core/state_machine.py`
- Event bus: `core/event_bus.py`
- Wake detection + active window logic: `core/wake_word/detector.py`, `core/pipeline.py`
- VAD + recording: `core/audio/vad.py`
- STT (faster-whisper): `core/audio/transcriber.py`
- TTS (edge-tts + pygame playback): `core/audio/speaker.py`
- Multi-LLM brain/router: `core/brain/llm_router.py`
- Memory via ChromaDB: `core/brain/memory.py`
- Tool registry + safe executor + schema validation:
  - `tools/registry.py`
  - `tools/executor.py`
  - `tools/schema_registry.py`
  - `tools/schemas/tool_schemas.json`

### Safety + reliability controls (implemented)
- Tool JSON schema validation
- Tool timeout enforcement
- Path root restrictions
- Unsafe shell pattern rejection in tool arguments
- Provider health scoring + cooldown logic
- Correlation-ID logging + rotating file logs
- Pre-MCP baseline audit script

### Current tool surface (implemented)
- PC controls, app open/close, volume: `tools/pc_controls.py`
- Browser/search/media routing: `tools/web_browser.py`, `tools/youtube_tool.py`
- File notes operations: `tools/file_tools.py`
- Input automation: `tools/input_tools.py`
- System utilities: `tools/system_tools.py`
- Vision/screen capture/read workspace file: `tools/vision_tools.py`

## 3) Models, Dependencies, and Runtime Stack

### Model strategy (configured)
From `config.yaml`:
- Primary LLM: `gemini-2.0-flash`
- Fallback LLM: `llama-3.3-70b-versatile` (Groq)
- Local fallback: `qwen3-coder:480b-cloud` (Ollama setting; naming should be re-validated)
- STT: Whisper `small.en`
- TTS voice: `en-GB-RyanNeural`

### Installed dependency surface (declared)
From `requirements.txt`:
- LLM SDKs: `google-genai`, `groq`, `ollama`
- STT/audio: `faster-whisper`, `torch`, `torchaudio`, `sounddevice`, `soundfile`, `numpy`
- TTS/playback: `edge-tts`, `pygame-ce`
- Memory/vector: `chromadb`, `sentence-transformers`
- Automation/vision: `pyautogui`, `pillow`
- Validation/safety: `pydantic`, `jsonschema`, `structlog`, `rapidfuzz`

### Hardware assumptions
- CUDA-first STT setup is explicitly supported in `main.py` and `core/audio/transcriber.py`
- CPU fallback path exists

## 4) Verification Results (not just static review)

### Verified green
- Pre-MCP audit passes: config + imports + schema sync all pass
- Schema coverage: 32/32 tools covered
- Test suite under `tests/` passes: 23 passed

### Verified breakage
- Running `pytest` at repo root fails due to accidental collection of `test_results.txt` (non-UTF8/binary-like content)
- This is CI fragility, not a core runtime failure

## 5) What Works, What Is Partially Complete, What Is Broken

### Working well now
- End-to-end voice pipeline structure
- Multi-provider fallback routing
- Tool safety layer (better than most hobby assistants)
- Typed config loading with runtime validation
- Logging, metrics, and health reporting foundation
- Recent interrupt/barge-in and screen-capture behavior improvements are integrated in code

### Partially complete (needs more work before production)
- Testing depth: good smoke/integration slices, still missing broader failure-mode and load tests
- Security hardening: baseline present, but still lacks stronger policy/permissions model for enterprise-grade trust
- Product packaging: no installer, updater, service mode, or crash-recovery supervisor yet
- GUI/app shell: not implemented yet (currently headless/script-driven)
- MCP: config flag exists, but server/client integration is not implemented
- Observability: metrics exist, but no centralized dashboard/alerting/log shipping

### Broken/right-now issues
- `pytest` root collection issue with text artifacts (`test_results.txt`)
- Documentation drift exists: some markdown architecture notes are older than actual code evolution

## 6) Project Structure Assessment
Overall structure is good and understandable:
- `core/` for runtime engine
- `tools/` for capabilities
- `utils/` for cross-cutting concerns
- `tests/` for validation
- `assets/`, `logs/`, `memory_db/` for runtime artifacts

What to improve for professional scale:
- Add `app/` or `desktop/` package for UI and app lifecycle
- Add `packaging/` for installer/build scripts
- Add `infra/` for CI/CD, linting, type checks, release automation
- Add `security/` docs and policy gates

## 7) Can You Sell This (with CUDA/local hardware assumptions)?
Short answer: Yes, possible — but not yet in current form.

### Commercially possible path
You can sell this as a Windows desktop AI assistant if you do all of:
1. Productization: signed installer, auto-update, telemetry controls, crash handling, settings UX
2. Reliability: robust device handling, retries, offline degraded mode, support logs
3. Legal/license: dependency and model license compliance, EULA, privacy policy
4. Security: permission model, explicit high-risk action confirmations, secrets management
5. Supportability: reproducible builds, diagnostics bundle, versioned migrations

### CUDA/local hardware concern
- You can ship CPU baseline + optional GPU acceleration profile.
- Do not assume customer CUDA stacks match yours.
- Production strategy should detect capability at runtime and choose profiles:
  - Profile A: CPU-safe default
  - Profile B: CUDA-accelerated
- Provide compatibility matrix and fallback guarantees.

## 8) Commercial/Legal Readiness Checklist (Critical)
Before selling:
- Build a Software Bill of Materials (SBOM)
- Verify licenses for every dependency and model/provider terms
- Check `open-interpreter` style AGPL risks are avoided if borrowing patterns/code
- Ensure API provider terms allow your resale model (Gemini/Groq/Ollama usage model)
- Add explicit user consent and data-handling policy for mic/screen/clipboard access

## 9) Market/Competition Reality Check (Web scan)
This space is active; your idea is valid and not unique, but differentiation is very possible.

Observed references:
- OpenVoiceOS: open, community voice-assistant ecosystem
- Rhasspy: well-known offline voice assistant stack (repo now archived)
- Open Interpreter: strong local computer-control assistant momentum
- MCP ecosystem: now broadly supported standard for tool/data integration

Implication:
- Yes, people have built similar categories.
- Your edge must be: better Windows UX, reliability, safety, latency, and clear “assistant that actually works daily” focus.

## 10) Production-Grade Upgrade Plan

### Immediate (1-2 weeks)
- Fix root test collection (`pytest.ini` + ignore txt artifacts)
- Add strict lint/type/test CI gates
- Add crash watchdog + graceful restart
- Add structured error taxonomy and user-facing recoveries

### Near term (2-6 weeks)
- Implement desktop app shell (tray + settings + conversation UI)
- Build installer (MSIX or Inno Setup) + signed binaries
- Add update mechanism + semantic versioning
- Add telemetry opt-in + privacy controls

### Mid term (6-12 weeks)
- Implement MCP server/client bridge for extensible tools
- Add plugin model + permission scoping per tool category
- Harden security posture with red-team style prompt-injection tests
- Add load/performance testing and latency SLOs

## 11) Recommended Product Positioning
Do not market as “another AI chatbot.”
Market as:
- Windows productivity copilot with voice + action reliability
- Local-first privacy-conscious assistant with optional cloud intelligence
- Professional automation assistant (browser/system/files/workflows)

## 12) Final Verdict
- Engineering direction: strong
- Current code quality: good for serious prototype
- Current sale readiness: not yet
- Feasibility to become sellable app: high, if productization/compliance/reliability tracks are executed

If executed well, this can become a real product, not just a side project.

---

## 13) Detailed Technical Truth Report (Evidence-Based)

This section is intentionally direct. No marketing language, no optimism bias.

### A) What Is Definitely Working (Verified)

1. Core startup and wiring work.
- Entry bootstrap in `main.py` initializes config, STT, VAD, TTS, memory, brain, and pipeline.
- CUDA-first and CPU fallback are both implemented in `core/audio/transcriber.py`.

2. Core pipeline loop works structurally.
- State transitions and event emission are implemented in `core/pipeline.py`.
- Wake-word mode and active window behavior are implemented.
- Transcript correction and memory recall integration exist in live flow.

3. Tool system is real, not mock.
- Tool registry and dynamic dispatch are active in `tools/registry.py`.
- Schema validation + timeout + path controls are active in `tools/executor.py`.
- Tool/schema parity is currently complete (32/32).

4. LLM fallback logic is implemented.
- Gemini, Groq, and Ollama initializers and routing logic are present in `core/brain/llm_router.py`.
- Provider cooldown and health score logic is present.

5. Observability foundation is functional.
- Structured logging with correlation IDs and log rotation is implemented in `utils/logger.py`.
- Latency/provider metrics exist and are queryable via `utils/metrics.py`.

6. Test baseline is healthy for existing test suite.
- `pytest -q tests` passed (23 passed).
- Includes routing behavior, streaming behavior, and selected tool behavior checks.

### B) What Is Partially Working (Implemented but Incomplete)

1. Interrupt/barge-in behavior.
- Recent improvements exist in `core/audio/speaker.py`, `core/audio/vad.py`, `core/pipeline.py`.
- This is improved but not yet validated under noisy real-world stress (TV/music/background audio).

2. Screen/vision capture behavior.
- IDE-hide and full-screen capture logic exists in `tools/vision_tools.py`.
- Works at code level and has tests, but still depends on Windows window manager edge cases and optional win32 modules.

3. Memory quality.
- Chroma persistence/recall works in `core/brain/memory.py`.
- Retrieval is simple and unranked beyond vector search; no advanced memory policy or factuality guard.

4. Safety posture.
- Good baseline controls exist (schema, timeout, path restrictions, unsafe patterns).
- Still missing stronger human approval boundaries for high-risk classes beyond power actions.

5. Configuration maturity.
- Typed config model is in place and practical.
- Still lacks environment profile strategy for dev/stage/prod and deployment-time config contracts.

### C) What Is Not Working / Missing Right Now

1. Root-level pytest execution is broken.
- `pytest` from repo root fails because `test_results.txt` is collected and decoded as UTF-8.
- This is a real CI blocker until test discovery is constrained.

2. No desktop application shell yet.
- No GUI module, no tray app, no settings UI.
- Current product is CLI/background code, not distributable app UX.

3. No installer/distribution pipeline.
- No MSIX/Inno/NSIS/WiX files found.
- No signed binary pipeline, no auto-updater, no release channel management.

4. No CI/CD automation found.
- No `.github` workflows, no project build/test pipeline config.

5. MCP is not implemented.
- `mcp.enabled` exists in config but no `mcp_server` or client bridge module exists.

6. Production operations are missing.
- No service wrapper, watchdog supervisor, uptime monitor, or remote diagnostics package workflow.

### D) Architecture Standing (Detailed)

Current architecture is above-average for an indie assistant project:
- Clear separation: audio, brain, tools, config, metrics, logging.
- Async orchestration is already in place.
- Safety primitives are integrated near execution boundary.

Main architectural risks:
- Single-process app means one catastrophic component failure can still degrade whole runtime.
- No plugin isolation boundary yet.
- No strict contracts for tool side effects and rollback behavior.

### E) Product Standing (Detailed)

User-value strength today:
- Strong local assistant foundation with practical Windows automation.
- Multi-LLM fallback improves resilience.

Product weaknesses today:
- Not packaged for normal users.
- No polished UI and onboarding.
- No formal support/recovery workflow for non-technical customers.

### F) Commercial Readiness Reality Check

Can this be sold eventually? Yes.
Can this be sold safely now as a mainstream product? No.

What prevents immediate sale:
- No installer/update/signing pipeline.
- No legal readiness pack (EULA/privacy/data policy/SBOM/license review evidence).
- No customer support instrumentation and crash triage flow.
- No compatibility matrix and deployment profiles for mixed hardware environments.

### G) CUDA/Hardware Commercialization Truth

You can commercialize with your current CUDA-first approach only if you ship two runtime tracks:
- Track 1: CPU-safe default profile.
- Track 2: CUDA-accelerated profile with capability detection and fallback.

Without this dual-track packaging, customer environments will fail unpredictably and support costs will be high.

### H) Security and Trust Standing

Current status: moderate safety, not enterprise-grade.

Strong points:
- Tool schema validation
- Timeout controls
- Path constraints
- Unsafe token filtering

Gaps to close before selling:
- Threat model document
- Prompt-injection simulation testing
- Expanded user confirmation policy for destructive or sensitive actions
- Secrets and telemetry governance documentation

### I) Test Quality Standing

Good:
- Real tests exist and pass.
- Includes critical logic and routing assertions.

Needs improvement:
- Root discovery config is missing (causing root pytest breakage).
- Missing long-run endurance tests, stress tests, audio-noise regression tests, and fault-injection tests.

### J) Professionalism Assessment of Codebase

Professional indicators already present:
- Modular design
- Typed config
- Structured logs
- Schema-driven tools
- Health reporting and preflight audit

Non-professional gaps still present:
- Missing delivery pipeline
- Missing formal release engineering
- Missing app lifecycle and updater mechanisms

### K) Ratings Out of 10 (Truth-First)

1. Core architecture: 8.0/10
2. Codebase organization: 8.2/10
3. Reliability (runtime): 6.8/10
4. Safety/security controls: 6.5/10
5. Testing maturity: 6.9/10
6. Observability/diagnostics: 6.7/10
7. Product UX readiness: 3.8/10
8. Deployment/release readiness: 3.2/10
9. Commercial/legal readiness: 3.5/10

### L) Final Scores

- Engineering prototype score: 7.1/10
- Productization score: 4.0/10
- Sell-now readiness score: 3.6/10

Bottom line:
- This is a strong engineering foundation.
- It is not yet a sellable end-user product.
- With focused productization, compliance, packaging, and ops work, it can realistically become one.
