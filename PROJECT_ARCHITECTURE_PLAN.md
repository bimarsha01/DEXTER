Dexter: Expert Architecture Review & Strategic Roadmap
Let me work through this systematically, giving you the honest, senior-architect perspective this project deserves.

1. Deep Project Analysis
What's Done Well
The foundation shows solid architectural instincts. The VAD → STT → LLM → Tool → TTS pipeline is correctly decomposed into discrete, swappable modules rather than a monolithic script — that's the right call. Specific strengths:
Multi-LLM fallback chain is genuinely production-thinking. Using Gemini → Groq → Ollama with health tracking and cooldown is better than 90% of hobby projects. The idea of a shared conversation history cap at 20 messages shows you've thought about context window costs.
Tool registry pattern in tools/registry.py is the right abstraction. Centralized registration with metrics means you can introspect what's being called, which is essential for debugging voice AI.
GPU-first with CPU fallback in faster-whisper is correct. Failing gracefully to int8 CPU is what separates a demo from something usable.
ChromaDB for long-term memory is a reasonable choice at this scale — it's lightweight and local, which fits the "mostly local" design philosophy.
Config-driven architecture via config.yaml is the right instinct. Everything tunable without code edits.

Major Flaws, Risks & Technical Debt
1. Single-threaded pipeline bottleneck (Critical)
The described flow is sequential: listen → transcribe → LLM → tool → speak. If any step blocks (e.g., a slow LLM call or a tool that hangs), the entire pipeline stalls. There's no indication of async task queuing. In practice, this means Dexter freezes during a cloud LLM call while the mic is dead. You need an async event loop with a proper pipeline manager.
2. Tool execution has no sandboxing or timeout (Security/Reliability)
tools/pc_controls.py, input_tools.py, and especially anything with file or power actions have zero mention of timeouts, permission scoping, or execution isolation. A misbehaving tool (e.g., a hung subprocess in web_browser.py) will block the whole assistant indefinitely. Every tool call needs a timeout wrapper and result validation.
3. PowerShell MediaPlayer for TTS playback (Fragility)
Using PowerShell as the primary audio playback mechanism with ffplay and start as fallbacks is brittle. PowerShell startup latency adds 200-800ms to every response. This will make Dexter feel sluggish on every single utterance. pygame.mixer or sounddevice direct playback would be ~10x more responsive.
4. temp_mic.wav as a single shared file (Race condition)
If anything triggers two simultaneous listen cycles (which can happen during testing or if VAD misfires), both writes go to the same file. This is a latent data corruption bug. Audio buffers should use UUIDs or a proper ring buffer.
5. Conversation history is a flat list with no structural memory (Architectural debt)
Capping at 20 messages by count is a blunt instrument. A 20-message history could be 200 tokens or 20,000 tokens depending on what was said. You need token-aware truncation, not message-count truncation. More importantly, ChromaDB memory and in-context history are operating as two separate, uncoordinated systems right now — there's no logic described for how retrieved memory is merged with recent context.
6. inspect.signature for Groq tool schema (Fragility)
Auto-building tool schemas from Python function signatures is clever but dangerous. It relies on your function signatures being perfectly descriptive, type-annotated, and docstring-complete. One poorly typed function will produce a broken schema that causes Groq to fail silently or hallucinate tool calls. Tool schemas should be explicitly declared, not inferred.
7. Wake word stripping is naive string matching (Accuracy risk)
Stripping wake words with simple string replacement on the transcript will fail on variations ("hey dex", "yo dexter"), false positives mid-sentence ("I told dexter about it"), and will mangle commands that legitimately contain the wake word string. This needs a proper wake word confidence threshold, not substring removal.
8. No input validation on LLM outputs before tool execution (Security)
The LLM decides which tool to call and with what parameters. If the LLM is compromised (prompt injection via a web search result, a malicious file read, etc.), it could instruct Dexter to execute power_action("shutdown") or type arbitrary text. There is no described validation layer between LLM intent and tool execution.
9. Incomplete modules with no enforcement (vision_tools, intent_router, metrics)
Three referenced modules are explicitly noted as unverified. In a production system, importing an undefined module causes silent failures at runtime. These need either proper implementation or explicit stubs with NotImplementedError.
10. config.yaml contains API keys directly (Assumed security risk)
The config drives "model selection" and presumably API keys. If config.yaml is not in .gitignore and the project is ever pushed anywhere, keys get leaked. All secrets must live in environment variables or a secrets manager, with config.yaml holding only non-secret configuration.

Incomplete or Poorly Structured Parts

No state machine for the assistant lifecycle. Dexter has states: idle, listening, processing, speaking, error. Without explicit state management, edge cases (user speaks while Dexter is speaking, two VAD triggers, STT failure) produce undefined behavior.
No health check or self-recovery loop. If the audio stream crashes or ChromaDB becomes unavailable, there's no described recovery mechanism.
clap_wake is dead config. Configuration for an unimplemented feature creates confusion and false documentation.
TTS cancellation exists but interrupt handling is unclear. Can Dexter be interrupted mid-speech by a new wake word? The flow doesn't describe this.
No test infrastructure mentioned anywhere.


2. MCP Integration Strategy
Should You Add MCP? — Honest Assessment
Yes, but with clear eyes about the tradeoff. MCP (Model Context Protocol) is the right long-term architecture for extensibility, but it adds complexity and latency. Here's the breakdown:
Pros of adding MCP:

Dexter's tools become accessible to any MCP-compatible client, not just your voice loop — meaning you could later drive Dexter from Claude Desktop, a web UI, or a mobile app using the same tool definitions.
Standard protocol means you get ecosystem tools for free (filesystem, browser, Office, etc.) without writing them yourself.
Clean separation: the LLM reasons about tools via a standard interface; the implementations can be swapped independently.
Future-proof: as the MCP ecosystem matures, you can plug in community servers.

Cons:

Network round-trip overhead. Local MCP over stdio is fast (~1-5ms), but over HTTP/SSE it's 10-50ms per tool call. For a voice assistant where latency is perceptible, this matters.
Complexity: you're now running a server process in addition to the assistant process. Process management, restart logic, and IPC error handling become your problem.
Your existing tools need to be rewrapped or replaced — it's not a zero-effort migration.

Verdict: Implement MCP as the external extensibility layer (for filesystem, Office, web, third-party integrations), while keeping latency-critical native tools (volume control, app launching, screenshots) as direct Python calls. Don't MCP-ify everything.

Recommended MCP Architecture
Use FastMCP (the Python library). It's the cleanest implementation with decorator-based tool definition, built-in schema generation, and both stdio and HTTP transport support.
Two-layer tool architecture:
Layer 1: Native Tools (direct Python, <5ms)
  - Audio control, volume, app launch/close
  - Keyboard/mouse input
  - System info, clipboard
  - Screenshot capture

Layer 2: MCP Tools (via FastMCP server, 5-50ms)
  - Filesystem (read/write/search files)
  - Office documents (Word, Excel via python-docx/openpyxl)
  - Browser automation
  - Email/calendar (Outlook via win32com)
  - Notes and knowledge base
  - External APIs (weather, news)
How the brain decides which layer to use:
The LLM router should present all tools in a unified schema to the LLM. Behind the scenes, the tool registry checks whether the tool ID maps to a native handler or an MCP server call. The LLM doesn't know or care — it just calls a tool name. The routing is transparent. This is the clean approach: one tool call interface, two execution backends.
MCP Server capabilities to expose:

filesystem — read_file, write_file, list_directory, search_files, create_directory
documents — read_word_doc, create_word_doc, read_excel, write_excel, read_pdf
email_calendar — list_emails, send_email, list_events, create_event (via Outlook COM)
browser — navigate, screenshot_page, extract_text, fill_form
knowledge_base — add_note, search_notes, summarize_topic (wrapping your ChromaDB)
system_extended — process_list, network_info, disk_usage, startup_apps


Cursor Prompt for MCP Server Implementation
Here is a ready-to-use prompt:
You are implementing an MCP (Model Context Protocol) server for a voice AI assistant called Dexter, running on Windows.

TASK: Create `mcp_server/dexter_mcp_server.py` using the FastMCP library.

REQUIREMENTS:
1. Use FastMCP with stdio transport (not HTTP) for low-latency local use.
2. Implement the following tool groups. Each tool must have:
   - A clear docstring (used as the tool description in the schema)
   - Typed parameters with descriptions
   - Proper error handling that returns structured error strings, never raises
   - A max execution timeout of 10 seconds

TOOL GROUPS TO IMPLEMENT:

Group A — Filesystem:
- read_file(path: str) -> str: Read text file contents. Handle encoding errors gracefully.
- write_file(path: str, content: str, mode: str = "overwrite") -> str: Write or append to file.
- list_directory(path: str, pattern: str = "*") -> list[dict]: List files with metadata (name, size, modified).
- search_files(root: str, query: str, file_types: list[str] = None) -> list[str]: Search filenames and optionally content.
- create_directory(path: str) -> str: Create directory and parents.

Group B — Office Documents (use python-docx and openpyxl):
- read_word_doc(path: str) -> str: Extract full text from .docx, preserving paragraph structure.
- create_word_doc(path: str, title: str, content: str) -> str: Create formatted .docx.
- read_excel(path: str, sheet: str = None) -> dict: Return sheet names and data as JSON-serializable dict.
- write_excel_cell(path: str, sheet: str, cell: str, value) -> str: Update a cell value.

Group C — Email/Calendar (via win32com Outlook integration):
- list_emails(folder: str = "Inbox", count: int = 10, unread_only: bool = False) -> list[dict]
- send_email(to: str, subject: str, body: str) -> str
- list_calendar_events(days_ahead: int = 7) -> list[dict]
- create_calendar_event(title: str, start: str, end: str, description: str = "") -> str

STRUCTURE:
- All tools registered with @mcp.tool() decorator
- A `__main__` block that runs mcp.run(transport="stdio")
- A companion `mcp_server/client.py` that wraps the stdio subprocess connection and exposes a clean async `call_tool(name, args)` coroutine for Dexter's tool registry to import
- Error responses should always return a dict with {"success": false, "error": "message"} structure

CONSTRAINTS:
- Do not use HTTP transport — stdio only for now
- All file paths must be validated against an allowed_roots list loaded from config.yaml
- Log all tool calls and their duration to a rotating file log
- The server must handle being killed and restarted cleanly (no orphan processes)

3. GUI Strategy
Library Recommendation: PyQt6 (with Qt Designer optional)
Here's the honest comparison at your use case:
LibraryVerdictCustomTkinterGood for quick UIs but limited layout control, slow rendering, poor for real-time dataDear PyGuiExcellent for dashboards/debug UIs, immediate mode means easy real-time updates, but looks "developer tool" not "butler"PyQt6Most powerful, native OS integration, system tray support built-in, excellent threading model, steeper curveTkinterToo primitive for a production-grade assistant UIElectron/TauriOverkill and adds Node.js/Rust dependency
Recommendation: PyQt6. Reasons specific to Dexter:

System tray icon with context menu is a first-class Qt feature — one of your key requirements.
QThread and signals/slots integrate cleanly with your async pipeline.
QWebEngineView lets you embed rich HTML/JS visualizations if you want an animated orb or waveform later.
Qt's native Windows integration means proper DPI scaling, Windows 11 theming, and taskbar behavior.


GUI Features Specification
Core Views:

Main Window (Floating HUD) — Compact, semi-transparent overlay (like Jarvis arc reactor aesthetic). Always-on-top optional. Shows: current state (Listening/Processing/Speaking), last transcript, last response, active tool indicator.
Conversation Log — Scrollable chat-style history with timestamps, tool calls shown as expandable cards (e.g., "Launched Chrome"), error states highlighted.
Settings Panel — Live config editing: wake words, LLM selection, voice selection, microphone device selector, toggle switches for each feature.
Tool Dashboard — Heatmap or list of most-used tools with success/failure rates (feeding from your metrics system).
Memory Viewer — Browse ChromaDB stored memories, delete entries, force-save context.
MCP Server Status — Which MCP tools are connected, latency, last call time.

Headless Mode:
Pass --no-gui CLI flag. In main.py, the GUI import is conditional:
pythonif not args.headless:
    from gui.main_window import DexterGUI
    app = DexterGUI(core)
All core functionality must work without the GUI module being imported at all. Use an event bus (described below) so the GUI just subscribes to events rather than being in the critical path.

Cursor Prompt for GUI
You are building a PyQt6 GUI for Dexter, a voice AI assistant. The GUI is a SECONDARY layer — the core assistant runs without it.

ARCHITECTURE REQUIREMENT:
- The GUI must communicate with the core via an event bus (use Python's built-in queue.Queue or a simple EventEmitter pattern), never by directly importing core modules.
- The core emits events: state_changed, transcript_received, response_generated, tool_called, error_occurred, memory_stored.
- The GUI subscribes to these events via a QTimer polling the queue every 50ms (not blocking the Qt event loop).

CREATE the following files:

1. `gui/main_window.py` — QMainWindow subclass:
   - Frameless window with custom titlebar (drag to move)
   - Semi-transparent dark glass aesthetic (#1a1a2e background, #00d4ff accent color)
   - State indicator: animated pulsing circle (idle=grey, listening=blue pulse, processing=amber spin, speaking=green pulse)
   - Transcript area: last user input in white, in smaller italic text
   - Response area: last Dexter response, white, with markdown rendering (use QTextBrowser)
   - Compact mode: 300x150px floating widget
   - Full mode: 800x600px with tabbed panels

2. `gui/tray_icon.py` — QSystemTrayIcon:
   - Icon changes with assistant state
   - Right-click context menu: Show/Hide window, Mute/Unmute, Settings, Quit
   - Double-click to toggle window visibility
   - Balloon notifications for important events

3. `gui/settings_panel.py` — QWidget panel:
   - Loaded from config.yaml, writes changes back on save
   - Microphone device dropdown (enumerate via sounddevice)
   - Wake word list (editable QListWidget)
   - LLM provider selector with API key fields (masked)
   - Voice selector with preview button
   - All changes validated before saving

4. `gui/conversation_log.py` — QWidget:
   - Chat-bubble style layout (user right, Dexter left)
   - Tool call events shown as grey cards between messages
   - Timestamp on hover
   - Export to text button

CONSTRAINTS:
- Zero blocking calls on the Qt main thread. Any core interaction goes through the event queue.
- Support Windows 11 dark mode (use QApplication.setStyle("Fusion") with dark palette)
- The window should respect --always-on-top flag from config
- Include a --headless CLI argument handler in a companion `gui/__init__.py` that returns None for all GUI objects when headless=True

4. Production-Grade Recommendations
Logging
Replace the current single-logger approach with structured logging:

Use structlog library instead of standard logging. It outputs JSON lines in production and human-readable in dev, with automatic context fields (timestamp, component, level, duration).
Each component (LLM router, tool executor, audio pipeline) gets its own logger with component name as a bound field.
Rotate logs daily, keep 14 days. Log to both file and (in GUI mode) the GUI log panel.
Add a correlation ID per user utterance that flows through the entire pipeline — so you can trace "what happened when I said X" across all log lines.

Config Management
Move to Pydantic Settings (pydantic-settings library):

Define a typed DexterConfig model. Every field is typed, validated, and documented.
Load order: defaults → config.yaml → environment variables → CLI flags (each overrides the previous).
API keys and secrets: always from environment variables, never from YAML. Use python-dotenv to load a .env file that is gitignored.
Validate the entire config on startup and fail fast with a clear error if required fields are missing.

Error Handling
Implement a circuit breaker pattern for external services:

Each LLM provider and MCP server gets a circuit breaker with: failure threshold (e.g., 3 failures), cooldown period (e.g., 60s), and half-open retry logic.
All tool calls wrapped in a ToolExecutor class that: enforces a timeout, catches all exceptions, returns a standardized ToolResult(success, data, error, duration_ms) object.
Never let an exception from a tool propagate to the LLM router. The router should always get a result object, even if it's a failure result.

Security

Tool permission scoping: Define a permissions section in config for each tool category. File tools default to ~/Documents and ~/Desktop only. Any path outside allowed roots is rejected before execution.
LLM output validation: Before executing any tool call from LLM output, validate: tool name exists in registry, all required parameters are present and correctly typed, no parameter contains shell injection patterns.
Rate limiting: Even for local LLMs, add a minimum time between requests to prevent runaway loops.
Secret scanning: Add a pre-commit hook using detect-secrets to prevent accidental API key commits.

Performance & Resilience

Pipeline concurrency: Use asyncio throughout. The VAD → STT path and the memory retrieval path can run concurrently with early LLM priming.
STT streaming: faster-whisper supports streaming transcription. Use it — start processing words as they come in rather than waiting for end-of-speech.
LLM response streaming: Stream token-by-token from the LLM and begin TTS on the first complete sentence. This cuts perceived latency from 3-5s to under 1s on typical responses.
TTS pre-generation: For common responses ("Got it", "Working on it", "Done"), pre-generate audio files and play them instantly while the real response is being generated.
Audio playback: Replace PowerShell with pygame.mixer or direct sounddevice playback for <10ms audio start latency.

System Integration (Windows)

Tray icon: Use Qt's QSystemTrayIcon (not a third-party library). Register it on startup.
Global hotkeys: Use keyboard library for a global push-to-talk hotkey (e.g., Ctrl+Space) that works even when Dexter's window is not focused.
Startup: Create a startup shortcut in %APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup pointing to a dexter.bat that activates the venv and runs main.py --minimized.
Auto-restart: Wrap main.py in a launcher.py watchdog that restarts the main process if it exits with a non-zero code.


5. Feature Roadmap (Prioritized)
Tier 1 — High Impact, Achievable (Next 4-8 weeks)
1. Streaming Pipeline (Latency Fix)
Convert the entire pipeline to async with LLM streaming + sentence-level TTS. This single change makes Dexter feel 3x more responsive. Every other feature is more valuable on top of a fast foundation. Priority: Do this first.
2. RAG on Personal Files
Index ~/Documents, ~/Desktop, and user-specified folders with llama-index or LangChain. Let Dexter answer "What did I write in my Q3 report?" or "Find my notes on Project X." This is the feature that transitions Dexter from a novelty to a genuine productivity tool.
3. Proactive Reminders & Calendar Awareness
Connect to Windows Calendar/Outlook. At startup and on a background timer, check for upcoming events and proactively surface them: "You have a meeting in 20 minutes." Combine with a gentle audio chime.
4. Multi-turn Task Execution
Allow Dexter to break complex commands into steps and ask clarifying questions: "Book a meeting with John tomorrow at 3pm" → checks calendar → "John has a conflict at 3pm, would 4pm work?" This requires a task state machine, not just single-shot tool calls.
Tier 2 — High Value, Medium Complexity (Weeks 8-16)
5. Screen Understanding (Vision)
Actually implement vision_tools.py properly. Integrate a vision-capable model (GPT-4o vision or Gemini 1.5) for screen context: "What's on my screen?" or "Summarize this document" (screenshot → OCR → LLM). This dramatically expands what Dexter can help with.
6. Automation Recording & Replay
Record a sequence of Dexter actions ("every morning, open email, read top 5 unread, summarize to me") and save as named routines. "Dexter, run my morning briefing." This is the Jarvis moment users will show their friends.
7. Personalization & Learning
Track which tools are called most, which commands succeed vs fail, user corrections ("no, I meant the Downloads folder"). Build a lightweight preference model that biases tool selection and response style over time.
8. Multi-Modal Input
Add camera support for gesture-based activation (raise hand = start listening) and visual context injection (point camera at document = Dexter describes it). Uses your existing vision infrastructure.
Tier 3 — Advanced (Months 4+)
9. Multi-Agent Orchestration
For complex long-running tasks ("research this topic and write a report"), spawn sub-agents using Claude/Gemini's API with specific tool subsets. The main Dexter brain acts as orchestrator. Uses LangGraph or a custom state machine.
10. Emotional/Situational Awareness
Detect stress in voice (pitch, pace, energy) and adapt tone: slower, calmer responses when user seems stressed. Subtle but powerful for an AI butler persona.
11. Local Fine-Tuning Pipeline
Periodically fine-tune a small local model (Qwen or Phi-3) on your own Dexter interaction logs to personalize the offline fallback model to your vocabulary and habits.
12. Multi-Profile Support
Support multiple user profiles with separate memory, preferences, and permissions. Useful if family members or colleagues share the machine.

6. Refactoring & Project Structure
Proposed Clean Structure
dexter/
├── main.py                     # Entry point, CLI args, process orchestration
├── launcher.py                 # Watchdog/auto-restart wrapper
├── config/
│   ├── settings.py             # Pydantic Settings model (typed config)
│   ├── config.yaml             # Non-secret config values
│   └── .env.example            # Template for secrets (gitignored .env)
├── core/
│   ├── pipeline.py             # Async pipeline orchestrator (the main event loop)
│   ├── state_machine.py        # AssistantState enum + transitions
│   ├── event_bus.py            # Simple pub/sub for decoupling components
│   ├── audio/
│   │   ├── vad.py
│   │   ├── transcriber.py      # Streaming transcription
│   │   ├── speaker.py          # TTS with pygame playback
│   │   └── audio_manager.py    # Device management, stream health
│   ├── brain/
│   │   ├── llm_router.py       # Multi-provider with circuit breakers
│   │   ├── memory.py           # ChromaDB + context merge logic
│   │   ├── prompt_builder.py   # System prompt assembly (context + memory + tools)
│   │   └── response_parser.py  # Validate & extract tool calls from LLM output
│   └── wake_word/
│       └── detector.py         # Proper wake word handling (not string match)
├── tools/
│   ├── registry.py             # Unified registry, routes to native or MCP
│   ├── executor.py             # Timeout wrapper, ToolResult model, error handling
│   ├── native/                 # Direct Python tools (low latency)
│   │   ├── pc_controls.py
│   │   ├── system_tools.py
│   │   ├── input_tools.py
│   │   └── vision_tools.py
│   └── schemas/                # Explicit JSON schemas for all tools
│       └── *.json
├── mcp_server/
│   ├── dexter_mcp_server.py    # FastMCP server (filesystem, docs, email)
│   ├── client.py               # Async MCP client wrapper for tool registry
│   └── permissions.py          # Path validation, security rules
├── gui/
│   ├── __init__.py             # Headless guard
│   ├── main_window.py
│   ├── tray_icon.py
│   ├── settings_panel.py
│   ├── conversation_log.py
│   └── assets/                 # Icons, sounds
├── rag/
│   ├── indexer.py              # File watcher + llama-index ingestion
│   ├── retriever.py            # Query interface for RAG
│   └── pipeline.py             # Background indexing scheduler
├── automation/
│   ├── routine_recorder.py     # Record/replay action sequences
│   └── routines/               # Saved YAML routines
├── utils/
│   ├── logger.py               # structlog configuration
│   ├── metrics.py              # Tool/LLM call metrics
│   ├── circuit_breaker.py      # Generic circuit breaker
│   └── security.py             # Input validation, path sanitization
└── tests/
    ├── unit/
    ├── integration/
    └── fixtures/
Key Refactoring Principles
1. Event-driven over call-stack-driven. The pipeline should be a series of event emitters and subscribers, not a linear function call chain. This enables the GUI, logging, metrics, and memory to all react to events without being in the critical path.
2. Explicit over implicit. Replace inspect.signature tool schema generation with explicit JSON schema files. Replace string-match wake word stripping with a proper detection module. Prefer boring, readable code over clever one-liners in infrastructure code.
3. Fail fast, recover gracefully. Validate config at startup. Use circuit breakers for all external services. Every component should have a health_check() method and the pipeline manager should poll them.
4. Separate concerns ruthlessly. The LLM router should not know about audio. The GUI should not know about LLMs. The tool executor should not know about the conversation. Enforce this with the event bus — if two modules need to share state, they should do it through events, not direct imports.
5. Test the pipeline, not just functions. Unit tests for utilities are fine, but the highest-value tests for a voice AI are integration tests that run the full pipeline with mocked audio input and assert on tool calls and responses.

7. Actionable Next Steps — Copy-Paste Prompts
Prompt 1: Async Pipeline Core
Refactor Dexter's main pipeline in `core/pipeline.py` to be fully async using Python asyncio.

CURRENT STATE: Sequential blocking loop in main.py.

REQUIREMENTS:
1. Create an `AsyncPipeline` class that manages the full utterance lifecycle as an async state machine.
2. States: IDLE, LISTENING, TRANSCRIBING, PROCESSING, EXECUTING_TOOL, SPEAKING, ERROR
3. The pipeline must support interruption: if a new wake word is detected while SPEAKING, cancel the TTS and restart from LISTENING.
4. Use asyncio.Queue for audio chunks from VAD.
5. LLM calls use httpx async client (not sync requests).
6. TTS generation and the first sentence playback should pipeline: start speaking sentence 1 while sentence 2 is being generated.
7. Emit events via an EventBus (simple asyncio.Queue per subscriber) for: state_changed(new_state), transcript(text), response_chunk(text), tool_called(name, args, result), error(component, exception).
8. Include a watchdog coroutine that checks pipeline health every 30s and logs a warning if stuck in any state.

Do not modify audio/ or brain/ modules yet. Use dependency injection for all sub-components.
Prompt 2: Pydantic Config System
Replace config.yaml loading in Dexter with a Pydantic Settings system.

CREATE `config/settings.py`:
1. Define `DexterConfig` using pydantic-settings BaseSettings.
2. Include typed, documented fields for: audio (device, sample_rate, vad_threshold), stt (model, device, beam_size), llm (primary provider/model, fallback provider/model, local model, max_history_tokens NOT message count), tts (voice, rate, volume), memory (collection_name, max_results), tools (allowed_file_roots as list of paths), gui (enabled, always_on_top, theme), wake_words (list of strings).
3. All API keys (gemini_api_key, groq_api_key) come ONLY from environment variables, never YAML. Use Field(default=None, env="GEMINI_API_KEY").
4. Load order: defaults -> config.yaml (using yaml_settings_source) -> .env file -> actual environment variables.
5. On startup, call `config.validate_runtime()` which checks: all API keys are present for enabled providers, audio device index exists, allowed_file_roots all exist.
6. Provide a `get_config()` singleton function with lazy initialization.

Do not change any other files yet. Show the complete settings.py and an updated config.yaml template.
Prompt 3: Tool Executor with Safety
Create `tools/executor.py` — a safe, production-grade tool execution layer for Dexter.

REQUIREMENTS:
1. Define `ToolResult` dataclass: success: bool, data: Any, error: str | None, tool_name: str, duration_ms: float, timestamp: datetime.
2. Create `ToolExecutor` class with:
   - `async def execute(tool_name: str, args: dict) -> ToolResult`
   - Validates tool_name exists in registry before calling
   - Validates all args against the tool's declared JSON schema (use jsonschema library)
   - Wraps execution in asyncio.wait_for with configurable timeout (default 10s from config)
   - Catches ALL exceptions, never propagates — always returns a ToolResult
   - Checks file path args against config.allowed_file_roots (reject traversal attempts)
   - Logs every call with structlog: tool name, args (redacted if sensitive), duration, success
   - Records metrics: call count, success rate, p95 latency per tool
3. Add a `sanitize_args(args: dict, schema: dict) -> dict` function that strips unexpected keys and validates types.
4. Security: reject any string arg that matches shell injection patterns: semicolons, backticks, $(), &&, ||, pipes.

This executor should be the ONLY way tools are called. The LLM router calls executor.execute(), never tool functions directly.
Prompt 4: Streaming LLM + TTS Pipeline
Implement streaming response pipeline for Dexter — this is the most important latency improvement.

MODIFY `core/brain/llm_router.py` and `core/audio/speaker.py`:

LLM Router changes:
1. Add `async def stream_response(messages, tools) -> AsyncGenerator[str, None]` to each provider.
2. For Gemini: use generate_content with stream=True, yield text chunks.
3. For Groq: use chat.completions.create with stream=True, yield delta.content.
4. In the router, accumulate streamed tokens and detect sentence boundaries (end with . ? ! followed by space or end).
5. When a complete sentence is detected, emit it immediately to the TTS queue — don't wait for full response.

Speaker changes:
1. Accept sentences via an asyncio.Queue.
2. Pre-generate edge-tts audio for each sentence as it arrives.
3. Play sentences sequentially with no gap between them (pre-buffer next while playing current).
4. Use sounddevice for playback (not PowerShell). Load audio with soundfile, play with sounddevice.playrec or sounddevice.play.
5. Support cancellation: a cancel() method that stops current playback and clears the queue.

Target: First words spoken within 800ms of LLM start generating (on fast network). Measure and log actual latency.