"""
Dexter Brain — Multi-LLM Router with Automatic Fallback Chain
Priority: Gemini (Primary) → Groq (Fallback) → Ollama (Local Offline)

Handles tool/function calling for Gemini and Groq backends.
Ollama serves as a text-only emergency fallback.

MIGRATED: Uses the new `google-genai` SDK (replaces deprecated `google-generativeai`).
"""
import re
import json
import asyncio
import time
import random
import groq
from typing import Any, Optional
from rapidfuzz import fuzz

from utils.logger import get_logger, get_correlation_id
from utils.metrics import metrics

logger = get_logger("llm_router")
from tools.registry import load_tools, execute_tool, EXECUTOR
from tools.schema_registry import get_tool_schema
from core.brain.intent_router import IntentRouter, PendingAction
from tools.vision_tools import describe_screen, describe_screen_without_vision
from utils.config import DexterConfig, get_config
from core.event_bus import EventBus


class _StreamingFallback(Exception):
    pass


TOOL_GROUPS = {
    "time": ["get_current_time", "get_current_datetime"],
    "weather": ["get_weather"],
    "app": ["open_application", "close_application"],
    "volume": ["set_system_volume"],
    "search": ["search_google", "search_youtube", "search_content_platform"],
    "document": ["read_document", "summarize_document", "answer_document_question"],
    "system": ["get_system_status", "lock_workstation", "shutdown_pc", "restart_pc", "sleep_pc", "cancel_shutdown"],
    "clipboard": ["read_clipboard", "copy_to_clipboard"],
    "screen": ["take_screenshot", "capture_screen"],
    "note": ["create_note", "read_note", "list_notes"],
    "browser": ["open_url", "open_url_in_browser"],
    "youtube": ["play_youtube"],
    "media": ["play_media", "play_music"],
    "routine": ["save_automation_routine", "list_automation_routines", "run_automation_routine", "delete_automation_routine"],
    "keyboard": ["type_text", "press_shortcut", "enter_key", "minimize_all_windows"],
    "health": ["get_health_report"],
}

ALWAYS_INCLUDE = ["get_current_datetime", "get_weather", "open_application"]

SCREEN_AWARENESS_TRIGGERS = [
    "what am i looking at",
    "what is on my screen",
    "what do you see",
    "describe my screen",
    "what is on screen",
    "what am i watching",
    "what is playing",
]


class ProviderHealth:
    """Persisted in-memory provider cooldown and failure state."""

    gemini_disabled_until: float = 0.0

    def __init__(self):
        self.gemini_disabled_until: float = 0.0
        self.groq_failure_count: int = 0
        self.groq_last_failure: float = 0.0

    @classmethod
    def disable_gemini_daily(cls) -> None:
        cls.gemini_disabled_until = time.time() + 86400
        logger.warning(
            "gemini_disabled_24h",
            reason="daily_quota_exhausted",
            retry_after="24 hours",
        )

    def disable_gemini(self, seconds: float) -> None:
        self.gemini_disabled_until = time.time() + max(0.0, float(seconds))

    @classmethod
    def is_gemini_available(cls) -> bool:
        return time.time() > cls.gemini_disabled_until

    def gemini_is_available(self) -> bool:
        return time.time() > max(self.gemini_disabled_until, self.__class__.gemini_disabled_until)

    def record_groq_failure(self) -> None:
        self.groq_failure_count += 1
        self.groq_last_failure = time.time()


class Brain:
    """Dexter's cognitive center — routes commands to the best available LLM."""

    RATE_LIMIT_BACKOFF_BASE = 2
    RATE_LIMIT_BACKOFF_MAX = 60
    GEMINI_QUOTA_DISABLE_SECONDS = 86400.0
    _gemini_disabled_until_global: float = 0.0

    def __init__(self, event_bus: Optional[EventBus] = None, asr_engine=None):
        logger.info("brain_initializing")

        self._cfg: DexterConfig = get_config()
        self._event_bus: Optional[EventBus] = event_bus

        self.tools_list = load_tools()
        self.intent_router = IntentRouter(self._cfg, asr_engine=asr_engine)
        self.shared_history = []
        self.pending_action = None
        self.max_history_tokens = int(self._cfg.history.max_tokens)
        self.last_provider: Optional[str] = None
        self._provider_health = ProviderHealth()
        self.provider_state = {
            "gemini": {"failures": 0, "score": 1.0, "cooldown_until": 0.0, "last_error": ""},
            "groq": {"failures": 0, "score": 1.0, "cooldown_until": 0.0, "last_error": ""},
            "ollama": {"failures": 0, "score": 1.0, "cooldown_until": 0.0, "last_error": ""},
        }

        self.system_instruction = """You are Dexter, a personal AI assistant running on this Windows PC.

You are smart, direct, and warm. You talk like a capable friend — not a butler, not a support agent, not a robot.

Rules:
- Never say: sir, ma'am, Certainly, Of course, Sure thing, Great question, Absolutely, Right away, Understood.
- Never repeat the question back.
- Never say what you're about to do before doing it. Just do it.
- Use contractions: I'll, it's, that's, you're, I've. They sound natural spoken aloud.
- Keep answers to 1-3 sentences unless the user asked for detail.
- For PC actions, confirm in one word or phrase: "Done." "Opened." "Locked." "Volume's at 50."
- When wrong: "You're right, let me fix that." Never a paragraph of apology.
- When unsure: "Not sure about that one." That's enough.

Examples:
  "What time is it?" → "It's 2:30."
  "Open Chrome." → "Done."
  "What's the weather?" → "22 degrees, light rain this afternoon."
  "Play Hotel California on Spotify." → "On it." [plays it]
  "You got that wrong." → "You're right, let me fix that."
  "Tell me about UserAuth." → [reads files] "That's a movie ticket booking system. Main service handles reservations and payment. Want more detail on any part?"

When you have PERSONAL FILE CONTEXT: read it, understand it, explain it like a smart person talking to a friend. Specific, natural, no hedging phrases like "based on the retrieved context."

When you receive code in the file context, explain the logic in plain English as if talking to the developer who wrote it. Describe what each section does, what problem it solves, and how the pieces fit together. Do not just restate the code — explain it.

For PC control: one short confirmation after the action. No narration before.
When asked about weather: call get_weather immediately with the city.
When user asks to play any music, song, video, or podcast — always use play_media.
If they name a platform (Spotify, YouTube, Apple Music, etc.) include it.
If they don't name a platform, omit platform and the tool uses their default.
Never use open_application just to open a music app — use play_media so content gets searched.
When asked to open an app: use open_application.
When the user says to write or type something in a document that was just opened, always call type_text with the exact text they want.
If the user refers to a previously opened document with words like "that document", "in Word", or "add to it", type into the currently active window without opening anything new.
After opening any application always wait 2 seconds before typing into it.
Never open an application that was not explicitly requested in the current message. If the user's message does not mention Notepad do not open Notepad. Only execute actions explicitly requested right now.

Spoken response length depends on the request type:
- Simple commands (open, close, volume, time, weather): 1 sentence maximum. Short and done.
- General questions: 2-3 sentences.
- When user asks to summarize, explain, describe, tell me about, or give details: speak naturally for as long as the topic needs. Do not cut yourself short. Cover the key points properly. Aim for 4-8 sentences for summaries.
- Never pad with filler. Never cut off important information to stay short.

When user asks to play music use play_media with song/artist and platform. Never use open_application for music — use play_media.

You have access to filesystem and document tools prefixed with mcp_. Use them when the user asks about files, documents, emails, or calendar events.

When to use MCP tools:
- mcp_read_file: when user asks to read any file
- mcp_write_file: when user asks to save or create a file
- mcp_list_directory: when user asks what is in a folder
- mcp_search_files: when user asks to find a file
- mcp_read_word_doc: when user asks about a Word document
- mcp_read_excel: when user asks about a spreadsheet
- mcp_read_pdf: when user asks about a PDF
- mcp_list_emails: when user asks about emails
- mcp_create_email_draft: when user asks to send or draft an email (always creates draft, never auto-sends)
- mcp_list_calendar_events: when user asks about meetings, appointments, or schedule

When using file tools you do not need the full path. Describe the file or folder naturally and combine with mcp_search_files to find it first if needed.

Never use mcp_write_file to overwrite important system files. Always confirm with the user before writing to existing files."""

        # Initialize all three LLM backends
        self._init_gemini()
        self._init_groq()
        self._init_ollama()

        # Report status
        available = []
        if self.gemini_available:
            available.append(f"Gemini ({self._cfg.models.primary_llm})")
        if self.groq_available:
            available.append(f"Groq ({self._cfg.models.fallback_llm})")
        if self.ollama_available:
            available.append(f"Ollama ({self._cfg.models.local_llm})")

        if available:
            logger.info("brain_online", providers=available)
        else:
            logger.error("brain_no_llm_available")

        metrics.update_provider_health("gemini", self.gemini_available, 1.0 if self.gemini_available else 0.0)
        metrics.update_provider_health("groq", self.groq_available, 1.0 if self.groq_available else 0.0)
        metrics.update_provider_health("ollama", self.ollama_available, 1.0 if self.ollama_available else 0.0)

    def _gemini_cooldown_seconds(self, error: Exception) -> float:
        msg = str(error)
        lower = msg.lower()
        if "perday" in lower or "limit: 0" in lower:
            return self.GEMINI_QUOTA_DISABLE_SECONDS
        if "perminute" in lower:
            return 65.0
        if self._is_rate_limit_error(error):
            return 65.0
        return 0.0

    @classmethod
    def _disable_gemini_globally(cls, seconds: float) -> None:
        cls._gemini_disabled_until_global = max(
            cls._gemini_disabled_until_global,
            time.time() + max(0.0, float(seconds)),
        )

    @classmethod
    def _is_gemini_globally_available(cls) -> bool:
        return time.time() >= cls._gemini_disabled_until_global

    def _can_use_gemini(self) -> bool:
        return self.gemini_available and self._is_gemini_globally_available() and self._can_use_provider("gemini", True)

    def _select_tools_for_provider(self, query: str, provider: str) -> list[dict]:
        if provider != "groq":
            return list(self.groq_tools)

        def _is_knowledge_query(text: str) -> bool:
            q = (text or "").lower()
            triggers = (
                "tell me about",
                "summary",
                "summarize",
                "summarise",
                "describe",
                "explain",
                "overview",
                "documentation",
                "readme",
                "project",
                "codebase",
                "repository",
            )
            return any(t in q for t in triggers)

        query_text = (query or "").lower()
        is_knowledge_query = _is_knowledge_query(query_text)
        media_query = any(keyword in query_text for keyword in ("play ", "spotify", "youtube music", "music", "song", "album", "playlist", "artist"))
        available = {t.get("function", {}).get("name"): t for t in (self.groq_tools or [])}
        if not available:
            return []

        group_scores: list[tuple[float, str]] = []
        allowed_groups = {"document"} if is_knowledge_query else set(TOOL_GROUPS.keys())
        for group_name in TOOL_GROUPS:
            if group_name not in allowed_groups:
                continue
            score = float(fuzz.partial_ratio(group_name, query_text))
            group_scores.append((score, group_name))

        selected_names: set[str] = set()
        for _, group_name in sorted(group_scores, key=lambda item: item[0], reverse=True)[:2]:
            for tool_name in TOOL_GROUPS.get(group_name, []):
                if tool_name in available:
                    selected_names.add(tool_name)

        if media_query:
            if "play_media" in available:
                selected_names.add("play_media")
            elif "play_music" in available:
                selected_names.add("play_music")

        if is_knowledge_query:
            for tool_name in TOOL_GROUPS.get("document", []):
                if tool_name in available:
                    selected_names.add(tool_name)

        for tool_name in ALWAYS_INCLUDE:
            if is_knowledge_query:
                continue
            # For media queries avoid including time-related tools which can confuse intent
            if media_query and tool_name in {"get_current_time", "get_current_datetime"}:
                continue
            if media_query and tool_name == "open_application":
                continue
            if tool_name in available:
                selected_names.add(tool_name)

        # Ensure time tools are not present when this is clearly a media playback request
        if media_query:
            for t in TOOL_GROUPS.get("time", []):
                if t in selected_names:
                    selected_names.discard(t)

        if is_knowledge_query:
            for t in TOOL_GROUPS.get("search", []) + TOOL_GROUPS.get("youtube", []):
                if t in selected_names:
                    selected_names.discard(t)

        max_tools = min(10, max(1, int(self._cfg.providers.groq_max_tools)))
        ordered_names = sorted(selected_names)
        trimmed = ordered_names[:max_tools]
        selected = [available[name] for name in trimmed if name in available]

        logger.debug(
            "groq_tools_selected",
            query=query,
            selected_count=len(selected),
            max_tools=max_tools,
            selected=[tool.get("function", {}).get("name") for tool in selected],
        )
        return selected

    # ─── LLM Initialization ──────────────────────────────────────────────────

    def _init_gemini(self):
        """Initialize Google Gemini using the NEW google-genai SDK with automatic function calling."""
        self.gemini_available = False
        try:
            from google import genai
            from google.genai import types

            gemini_key = self._cfg.gemini_api_key
            if not gemini_key or "YOUR" in gemini_key.upper():
                logger.info("gemini_skipped", reason="no_api_key")
                return

            # New SDK: create a Client with the API key
            self.gemini_client = genai.Client(api_key=gemini_key)
            self._genai_types = types

            # Store model name for later use
            self.gemini_model_name = self._cfg.models.primary_llm

            self.gemini_available = True
            logger.info("gemini_ready", model=self.gemini_model_name, sdk="google-genai")

        except ImportError:
            logger.warning("gemini_import_failed", hint="pip install google-genai")
        except Exception as e:
            logger.warning("gemini_init_failed", error=str(e), exc_info=True)

    def _init_groq(self):
        """Initialize Groq as the fallback LLM with manual function calling."""
        self.groq_available = False
        try:
            from groq import AsyncGroq

            groq_key = self._cfg.groq_api_key
            if not groq_key or "YOUR" in groq_key.upper():
                logger.info("groq_skipped", reason="no_api_key")
                return

            self.groq_client = AsyncGroq(api_key=groq_key)
            self.groq_tools = self._build_groq_tool_schemas()
            self.groq_available = True
            logger.info("groq_ready", model=self._cfg.models.fallback_llm)

        except ImportError:
            logger.warning("groq_import_failed", hint="pip install groq")
        except Exception as e:
            logger.warning("groq_init_failed", error=str(e), exc_info=True)

    def _init_ollama(self):
        """Initialize local Ollama as the offline emergency fallback (text-only, no tools)."""
        self.ollama_available = False
        try:
            import ollama as ollama_lib

            self.ollama = ollama_lib
            # Quick connection test — if Ollama server isn't running, this fails fast
            ollama_lib.list()
            self.ollama_available = True
            logger.info("ollama_ready", model=self._cfg.models.local_llm)

        except ImportError:
            logger.info("ollama_optional_missing", hint="pip install ollama")
        except Exception as e:
            logger.info("ollama_unavailable", error=str(e))

    # ─── Tool Schema Generation ──────────────────────────────────────────────

    def _build_groq_tool_schemas(self):
        """
        Builds OpenAI-compatible tool schemas from explicit JSON schema files.
        """
        schemas = []
        for func in self.tools_list:
            schema = get_tool_schema(func.__name__)
            tool_schema = {
                "type": "function",
                "function": {
                    "name": func.__name__,
                    "description": (func.__doc__ or "").strip() or f"Executes {func.__name__}",
                },
            }

            if schema and schema.get("properties"):
                tool_schema["function"]["parameters"] = schema

            schemas.append(tool_schema)

        logger.debug("groq_tool_schemas_built", count=len(schemas))
        return schemas

    # ─── History Management ──────────────────────────────────────────────────

    def _estimate_tokens(self, text: str) -> int:
        if not text:
            return 0
        return max(1, int(len(text) / 4))

    def _prune_history_by_tokens(self) -> None:
        if self.max_history_tokens <= 0:
            return

        total = 0
        pruned = []
        for msg in reversed(self.shared_history):
            total += self._estimate_tokens(msg.get("content", "")) + 4
            if total > self.max_history_tokens and pruned:
                break
            pruned.append(msg)

        self.shared_history = list(reversed(pruned))

    def _add_history(self, role: str, content: str) -> None:
        self.shared_history.append({"role": role, "content": content})
        self._prune_history_by_tokens()

    def _sanitize_conversation_history(self, messages: list) -> list:
        """
        Remove malformed tool call messages from conversation history.
        These appear when a tool call fails to parse and gets stored as raw text.
        Keeping them confuses the LLM into repeating or hallucinating tool calls.
        """
        sanitized = []
        for msg in messages or []:
            content = msg.get("content", "") or ""
            if isinstance(content, str) and (
                "<function(" in content
                or ("function_call" in content.lower() and "{" in content)
            ):
                logger.warning(
                    "conversation_message_sanitized",
                    role=msg.get("role"),
                    preview=content[:80],
                )
                continue
            sanitized.append(msg)

        return sanitized

    @staticmethod
    def _truncate_rag_for_provider(rag_context: str, provider: str) -> str:
        if not rag_context:
            return ""
        if provider != "groq":
            return rag_context
        # For Groq, cap individual excerpts to avoid token blowup.
        # Supports both:
        # 1) Old format: Source/Path/Content
        # 2) Numbered format: [1] ... <excerpt line>
        lines = rag_context.splitlines()
        truncated: list[str] = []
        per_excerpt_cap = 800

        # Old Source/Path/Content format
        if any(l.startswith("Source: ") for l in lines) and any(l.startswith("Content: ") for l in lines):
            source_seen = 0
            in_first_source = False
            for line in lines:
                if line.startswith("Source: "):
                    source_seen += 1
                    if source_seen > 1:
                        break
                    in_first_source = True
                    truncated.append(line)
                    continue
                if not in_first_source:
                    truncated.append(line)
                    continue
                if line.startswith("Path: "):
                    truncated.append(line)
                    continue
                if line.startswith("Content: "):
                    content = line[len("Content: ") :]
                    if len(content) > per_excerpt_cap:
                        if per_excerpt_cap <= 3:
                            content = content[:per_excerpt_cap].rstrip()
                        else:
                            content = content[: per_excerpt_cap - 3].rstrip() + "..."
                    truncated.append(f"Content: {content}")
                    continue
                if line.startswith("("):
                    truncated.append(line)

            return "\n".join(truncated).strip()

        # Numbered [n] format
        source_seen = 0
        in_first_source = False
        for line in lines:
            stripped = line.strip()
            # Reconstruct header lines to ensure they begin with plain ASCII `[n]`
            # even if they contain odd leading whitespace/zero-width characters.
            header_match = re.search(r"\[(\d+)\]", stripped)
            # The numbered context format should have headers near the beginning
            # of the line; keep this permissive so hidden/unprintable prefix
            # characters do not prevent the header from being detected.
            if header_match and header_match.start() <= 20:
                num = header_match.group(1)
                source_seen += 1
                if source_seen > 1:
                    break
                in_first_source = True
                rest = stripped[header_match.end() :].lstrip()
                truncated.append(f"[{num}]{(' ' + rest) if rest else ''}")
                continue

            if not in_first_source:
                truncated.append(line)
                continue

            # Within the first excerpt block, cap the excerpt line(s).
            if stripped:
                if len(stripped) > per_excerpt_cap:
                    if per_excerpt_cap <= 3:
                        truncated.append(stripped[:per_excerpt_cap].rstrip())
                    else:
                        truncated.append(stripped[: per_excerpt_cap - 3].rstrip() + "...")
                else:
                    truncated.append(stripped)
            else:
                truncated.append(line)

        return "\n".join(truncated).strip()

    def _compose_prompt(self, user_command: str, long_term_memory: str = "", indexed_context: str = "", provider: str = "gemini") -> str:
        sections: list[str] = []
        if long_term_memory:
            sections.append(long_term_memory.strip())

        rag_context = self._truncate_rag_for_provider(indexed_context.strip(), provider)
        if rag_context:
            sections.append(rag_context)
            sections.append("Answer questions about files in maximum 4 sentences. User is listening not reading.")
            sections.append(
                "IMPORTANT ABOUT FILE CONTEXT:\n"
                "When you receive file content from indexed files read it carefully and answer from what is actually written there. Do not answer from the project name or folder name alone.\n\n"
                "If the files show a movie ticket booking system say it is a movie ticket booking system.\n"
                "If the files show payment processing say it handles payments.\n"
                "If the files show theater and seat management describe theaters and seats.\n\n"
                "Never assume what a project does based on its name. Always base your answer on the actual file contents provided to you."
            )

            # If the truncated RAG context indicates a direct file read, explicitly forbid tool calling
            try:
                if "[Direct File Read:" in rag_context:
                    sections.append(
                        "CRITICAL: The prompt already contains the exact file contents. Do NOT call any external tools or functions. Answer directly and specifically from the provided file content."
                    )
            except Exception:
                pass

        # If indexed_context appears to contain at least one substantial excerpt,
        # instruct the model to answer assertively and specifically.
        try:
            if indexed_context and any(len(line.strip()) >= 100 for line in indexed_context.splitlines() if line.strip()):
                sections.append(
                    "You have strong file context. Answer specifically and confidently from it. Do not hedge unless the files are genuinely ambiguous."
                )
                # If the injected context is a direct file read, explicitly forbid tool calling
                try:
                    if "[Direct File Read:" in indexed_context:
                        sections.append(
                            "CRITICAL: The prompt already contains the exact file contents. Do NOT call any external tools or functions. Answer directly and specifically from the provided file content."
                        )
                except Exception:
                    pass
        except Exception:
            pass

        if rag_context:
            source_count = rag_context.count("\n[")
            if source_count > 2:
                sections.append(
                    "NOTE: Multiple files are shown above. "
                    "Synthesize across them — explain how they relate "
                    "rather than summarising each one separately."
                )

        sections.append(f"User question: {user_command}")
        return "\n\n".join(section for section in sections if section)

    def _build_shared_messages(self):
        sanitized_history = self._sanitize_conversation_history(self.shared_history)
        return [
            {"role": msg["role"], "content": msg["content"]}
            for msg in sanitized_history
            if msg.get("content")
        ]

    def _build_gemini_contents(self, types):
        contents = []
        for msg in self._sanitize_conversation_history(self.shared_history):
            role = "user" if msg["role"] == "user" else "model"
            contents.append(types.Content(
                role=role,
                parts=[types.Part(text=msg["content"])],
            ))
        return contents

    def _can_use_provider(self, name: str, available_flag: bool) -> bool:
        if not available_flag:
            return False
        state = self.provider_state.get(name, {})
        cooldown_until = state.get("cooldown_until", 0.0)
        return time.time() >= cooldown_until

    def _record_provider_success(self, name: str) -> None:
        state = self.provider_state.get(name, {})
        state["failures"] = 0
        state["score"] = min(1.0, state.get("score", 1.0) + 0.1)
        state["cooldown_until"] = 0.0
        state["last_error"] = ""
        self.provider_state[name] = state
        metrics.update_provider_health(name, True, state["score"], 0.0, "")

    def _record_provider_failure(self, name: str, error: Exception, rate_limited: bool = False) -> None:
        state = self.provider_state.get(name, {})
        failures = state.get("failures", 0) + 1
        state["failures"] = failures
        state["score"] = max(0.0, state.get("score", 1.0) - 0.2)
        state["last_error"] = str(error)

        if rate_limited:
            backoff = min(self.RATE_LIMIT_BACKOFF_MAX, self.RATE_LIMIT_BACKOFF_BASE ** min(failures, 5))
            jitter = random.uniform(0.0, 1.0)
            state["cooldown_until"] = time.time() + backoff + jitter

        self.provider_state[name] = state
        metrics.update_provider_health(name, True, state["score"], state.get("cooldown_until", 0.0), str(error))

    def _emit_llm_event(self, event_type: str, **fields: Any) -> None:
        provider = fields.get("provider")
        if provider and event_type in {
            "llm_stream_started",
            "llm_stream_completed",
            "llm_call_started",
            "llm_call_completed",
        }:
            self.last_provider = str(provider)
        if self._event_bus:
            self._event_bus.emit(event_type, fields)

    @staticmethod
    def _is_time_query(text: str) -> bool:
        q = (text or "").lower()
        if re.search(r"\b(time|date|timezone|clock)\b", q):
            return True
        if "what day" in q or "day is it" in q or "today" in q:
            return True
        return False

    @staticmethod
    def _is_screen_awareness_query(command: str) -> bool:
        cmd_lower = (command or "").lower()
        return any(trigger in cmd_lower for trigger in SCREEN_AWARENESS_TRIGGERS)

    async def _describe_screen_or_unavailable(self, user_question: str) -> str:
        if not self._can_use_gemini():
            logger.info("screen_awareness_unavailable", reason="vision_quota_exhausted")
            return describe_screen_without_vision()

        try:
            description = await describe_screen(user_question, self.gemini_client)
            if description and description.strip():
                return description.strip()
        except Exception as e:
            logger.error("screen_awareness_failed", error=str(e), exc_info=True)

        return describe_screen_without_vision()

    def _is_rate_limit_error(self, error: Exception) -> bool:
        # Be defensive: provider SDKs use different exception classes / messages.
        msg = str(error).lower()
        try:
            from google.api_core import exceptions as google_exceptions  # type: ignore

            if isinstance(error, getattr(google_exceptions, "ResourceExhausted", ())):
                return True
            if hasattr(error, "status_code") and getattr(error, "status_code", None) == 429:
                return True
        except Exception:
            pass

        if "rate limit" in msg or "resource_exhausted" in msg or "quota" in msg:
            return True
        if "429" in msg:
            return True
        status = getattr(error, "status_code", None)
        return status == 429

    # ─── Main Command Processing ─────────────────────────────────────────────

    async def process_command(self, user_command: str, long_term_memory: str = "", indexed_context: str = "") -> str:
        """
        Routes a user command through the LLM fallback chain:
        Gemini → Groq → Ollama
        """
        prompt = self._compose_prompt(user_command, long_term_memory, indexed_context, provider="gemini")

        logger.info("command_processing_started")

        # Clear expired pending action
        if self.pending_action and time.time() > self.pending_action.expires_at:
            self.pending_action = None

        # Resolve pending confirmations / slots
        if self.pending_action:
            decision = self.intent_router.resolve_pending(user_command, self.pending_action)
            if decision.action == "cancel":
                self.pending_action = None
                self._add_history("user", user_command)
                self._add_history("assistant", decision.prompt)
                return decision.prompt
            if decision.action == "ask":
                self._add_history("user", user_command)
                self._add_history("assistant", decision.prompt)
                return decision.prompt
            if decision.action == "vision":
                self.pending_action = None
                prompt = self._compose_prompt(user_command, long_term_memory, indexed_context, provider="gemini")
                response_text = await self._handle_vision(decision, prompt, user_question=user_command)
                self._add_history("user", user_command)
                self._add_history("assistant", response_text)
                return response_text
            if decision.action == "tool":
                args = dict(decision.args)
                if self.pending_action.kind == "confirm":
                    args["confirm"] = True
                self.pending_action = None
                tool_result = await execute_tool(decision.tool_name, args, event_bus=self._event_bus)
                response_text = self._handle_tool_response(decision.tool_name, tool_result)
                self._add_history("user", user_command)
                self._add_history("assistant", response_text)
                return response_text

        if self._is_screen_awareness_query(user_command):
            response_text = await self._describe_screen_or_unavailable(user_command)
            self._add_history("user", user_command)
            self._add_history("assistant", response_text)
            return response_text

        # Intent routing for high-value tools / vision
        decision = self.intent_router.detect_intent(user_command)
        if decision.action == "ask":
            if decision.vision_mode == "file":
                self.pending_action = PendingAction(
                    kind="slot",
                    tool_name="vision_file",
                    args={},
                    prompt=decision.prompt,
                    expires_at=time.time() + 45,
                )
            elif decision.tool_name:
                self.pending_action = self.intent_router.build_pending_slot(decision)
            self._add_history("user", user_command)
            self._add_history("assistant", decision.prompt)
            return decision.prompt

        if decision.action == "tool":
            if self._requires_confirmation(decision.tool_name):
                self.pending_action = self.intent_router.build_pending_confirm(
                    decision.tool_name,
                    decision.args,
                    f"Please confirm: should I proceed with {decision.tool_name}?",
                )
                self._add_history("user", user_command)
                self._add_history("assistant", self.pending_action.prompt)
                return self.pending_action.prompt

            tool_result = await execute_tool(decision.tool_name, decision.args, event_bus=self._event_bus)
            response_text = self._handle_tool_response(decision.tool_name, tool_result)
            self._add_history("user", user_command)
            self._add_history("assistant", response_text)
            return response_text

        if decision.action == "vision":
            prompt = self._compose_prompt(user_command, long_term_memory, indexed_context, provider="gemini")
            response_text = await self._handle_vision(decision, prompt, user_question=user_command)
            self._add_history("user", user_command)
            self._add_history("assistant", response_text)
            return response_text

        fallback_note = ""
        # ── Try Gemini (Primary) ──
        if self.gemini_available and not self._provider_health.is_gemini_available():
            logger.info(
                "router_gemini_unavailable",
                disabled_until=self._provider_health.gemini_disabled_until,
                target_provider="groq",
            )
        elif self._can_use_gemini():
            try:
                self._emit_llm_event("llm_call_started", provider="gemini")
                _t0 = time.perf_counter()
                prompt = self._compose_prompt(user_command, long_term_memory, indexed_context, provider="gemini")
                response_text = await self._process_gemini(prompt)
                _ms = (time.perf_counter() - _t0) * 1000
                logger.info("llm_call_completed", provider="gemini", duration_ms=_ms)
                self._emit_llm_event("llm_call_completed", provider="gemini", duration_ms=_ms)
                self._record_provider_success("gemini")
                self._add_history("user", user_command)
                self._add_history("assistant", response_text)
                return response_text
            except Exception as e:
                rate_limited = self._is_rate_limit_error(e)
                self._record_provider_failure("gemini", e, rate_limited)
                rate_msg = str(e)
                cooldown = self._gemini_cooldown_seconds(e)
                if cooldown > 0:
                    self._provider_health.disable_gemini(cooldown)
                    if "perday" in rate_msg.lower() or "limit: 0" in rate_msg.lower():
                        self._disable_gemini_globally(cooldown)
                logger.warning("gemini_request_failed", error=rate_msg, exc_info=True)
                logger.info("llm_fallback", from_provider="gemini", to_provider="groq")
                self._emit_llm_event("llm_call_failed", provider="gemini", error=rate_msg)
                self._emit_llm_event("llm_fallback", from_provider="gemini", to_provider="groq")
                # On quota/rate-limit, wait briefly and annotate the next prompt to encourage brevity
                if rate_limited:
                    try:
                        await asyncio.sleep(1.5)
                    except Exception:
                        pass
                    logger.warning("gemini_rate_limited", error=rate_msg)
                    fallback_note = "[Note: fallback provider — keep response concise]\n"

        # ── Try Groq (Fallback) ──
        if self._can_use_provider("groq", self.groq_available):
            try:
                self._emit_llm_event("llm_call_started", provider="groq")
                _t0 = time.perf_counter()
                prompt = self._compose_prompt(user_command, long_term_memory, indexed_context, provider="groq")
                if fallback_note:
                    prompt = fallback_note + prompt
                response_text = await self._process_groq(prompt, query_hint=user_command)
                _ms = (time.perf_counter() - _t0) * 1000
                logger.info("llm_call_completed", provider="groq", duration_ms=_ms)
                self._emit_llm_event("llm_call_completed", provider="groq", duration_ms=_ms)
                self._record_provider_success("groq")
                self._add_history("user", user_command)
                self._add_history("assistant", response_text)
                return response_text
            except Exception as e:
                rate_limited = self._is_rate_limit_error(e)
                self._record_provider_failure("groq", e, rate_limited)
                self._provider_health.record_groq_failure()
                logger.warning("groq_request_failed", error=str(e), exc_info=True)
                logger.info("llm_fallback", from_provider="groq", to_provider="ollama")
                self._emit_llm_event("llm_call_failed", provider="groq", error=str(e))
                self._emit_llm_event("llm_fallback", from_provider="groq", to_provider="ollama")

        # ── Try Ollama (Local Offline) ──
        if self._can_use_provider("ollama", self.ollama_available):
            try:
                self._emit_llm_event("llm_call_started", provider="ollama")
                _t0 = time.perf_counter()
                prompt = self._compose_prompt(user_command, long_term_memory, indexed_context, provider="ollama")
                response_text = await self._process_ollama(prompt)
                _ms = (time.perf_counter() - _t0) * 1000
                logger.info("llm_call_completed", provider="ollama", duration_ms=_ms)
                self._emit_llm_event("llm_call_completed", provider="ollama", duration_ms=_ms)
                self._record_provider_success("ollama")
                self._add_history("user", user_command)
                self._add_history("assistant", response_text)
                return response_text
            except Exception as e:
                self._record_provider_failure("ollama", e, False)
                logger.error("ollama_request_failed", error=str(e), exc_info=True)
                self._emit_llm_event("llm_call_failed", provider="ollama", error=str(e))

        return (
            "All my providers are unreachable right now. "
            "Check your API keys in config.yaml and your internet connection."
        )

    async def process_command_stream(self, user_command: str, long_term_memory: str = "", indexed_context: str = ""):
        if self.pending_action:
            response_text = await self.process_command(user_command, long_term_memory, indexed_context)
            yield response_text
            return

        if self._is_screen_awareness_query(user_command):
            response_text = await self._describe_screen_or_unavailable(user_command)
            self._add_history("user", user_command)
            self._add_history("assistant", response_text)
            yield response_text
            return

        if not ProviderHealth.is_gemini_available():
            logger.info("gemini_skipped", reason="quota_exhausted")

        decision = self.intent_router.detect_intent(user_command)
        if decision.action != "none":
            response_text = await self.process_command(user_command, long_term_memory, indexed_context)
            yield response_text
            return

        if self.gemini_available and not self._provider_health.is_gemini_available():
            logger.info(
                "router_gemini_unavailable",
                disabled_until=self._provider_health.gemini_disabled_until,
                target_provider="groq",
            )
        elif self._can_use_gemini():
            fallback_note = ""
            try:
                response_text = ""
                self._emit_llm_event("llm_stream_started", provider="gemini")
                _t0 = time.perf_counter()
                prompt = self._compose_prompt(user_command, long_term_memory, indexed_context, provider="gemini")
                async for chunk in self._stream_gemini(prompt):
                    response_text += chunk
                    yield chunk
                if response_text:
                    _ms = (time.perf_counter() - _t0) * 1000
                    logger.info("llm_stream_completed", provider="gemini", duration_ms=_ms)
                    self._emit_llm_event("llm_stream_completed", provider="gemini", duration_ms=_ms)
                    self._record_provider_success("gemini")
                    self._add_history("user", user_command)
                    self._add_history("assistant", response_text)
                    return
            except Exception as e:
                rate_limited = self._is_rate_limit_error(e)
                self._record_provider_failure("gemini", e, rate_limited)
                cooldown = self._gemini_cooldown_seconds(e)
                if cooldown > 0:
                    self._provider_health.disable_gemini(cooldown)
                    if "perday" in str(e).lower() or "limit: 0" in str(e).lower():
                        ProviderHealth.disable_gemini_daily()
                        self._disable_gemini_globally(cooldown)
                logger.warning("gemini_stream_failed", error=str(e), exc_info=True)
                self._emit_llm_event("llm_stream_failed", provider="gemini", error=str(e))
                if rate_limited:
                    rate_msg = str(e)
                    try:
                        await asyncio.sleep(1.5)
                    except Exception:
                        pass
                    logger.warning("gemini_rate_limited", error=rate_msg)
                    fallback_note = "[Note: fallback provider — keep response concise]\n"

        if self._can_use_provider("groq", self.groq_available):
            try:
                response_text = ""
                self._emit_llm_event("llm_stream_started", provider="groq")
                _t0 = time.perf_counter()
                prompt = self._compose_prompt(user_command, long_term_memory, indexed_context, provider="groq")
                if "fallback_note" in locals() and fallback_note:
                    prompt = fallback_note + prompt
                async for chunk in self._stream_groq_with_tools(prompt, query_hint=user_command, allow_tools=True):
                    response_text += chunk
                    yield chunk
                if response_text:
                    _ms = (time.perf_counter() - _t0) * 1000
                    logger.info("llm_stream_completed", provider="groq", duration_ms=_ms)
                    self._emit_llm_event("llm_stream_completed", provider="groq", duration_ms=_ms)
                    self._record_provider_success("groq")
                    self._add_history("user", user_command)
                    self._add_history("assistant", response_text)
                    return
            except Exception as e:
                rate_limited = self._is_rate_limit_error(e)
                self._record_provider_failure("groq", e, rate_limited)
                self._provider_health.record_groq_failure()
                logger.warning("groq_stream_failed", error=str(e), exc_info=True)
                self._emit_llm_event("llm_stream_failed", provider="groq", error=str(e))

                decision = self.intent_router.detect_intent(user_command)
                if decision.action in {"tool", "ask", "vision"}:
                    response_text = await self.process_command(user_command, long_term_memory, indexed_context)
                    yield response_text
                    return

        if self._can_use_provider("ollama", self.ollama_available):
            try:
                self._emit_llm_event("llm_stream_started", provider="ollama")
                _t0 = time.perf_counter()
                prompt = self._compose_prompt(user_command, long_term_memory, indexed_context, provider="ollama")
                response_text = await self._process_ollama(prompt)
                if response_text:
                    _ms = (time.perf_counter() - _t0) * 1000
                    logger.info("llm_stream_completed", provider="ollama", duration_ms=_ms)
                    self._emit_llm_event("llm_stream_completed", provider="ollama", duration_ms=_ms)
                    self._record_provider_success("ollama")
                    self._add_history("user", user_command)
                    self._add_history("assistant", response_text)
                    yield response_text
                    return
            except Exception as e:
                self._record_provider_failure("ollama", e, False)
                logger.warning("ollama_stream_failed", error=str(e), exc_info=True)
                self._emit_llm_event("llm_stream_failed", provider="ollama", error=str(e))

        response_text = await self.process_command(user_command, long_term_memory, indexed_context)
        yield response_text

    def _requires_confirmation(self, tool_name: str) -> bool:
        return tool_name in {"shutdown_pc", "restart_pc", "sleep_pc"}

    def _handle_tool_response(self, tool_name: str, tool_result: Any) -> str:
        if tool_name != "resolve_open_target":
            return self._summarize_tool_output(tool_name, tool_result)

        payload = self._parse_tool_payload(tool_result)
        if not isinstance(payload, dict):
            return str(tool_result)

        status = payload.get("status")
        if status == "ask":
            prompt = payload.get("message") or "Which one?"
            match_id = payload.get("match_id")
            if match_id:
                self.pending_action = PendingAction(
                    kind="open_choice",
                    tool_name="resolve_open_target",
                    args={"match_id": match_id},
                    prompt=prompt,
                    expires_at=time.time() + 45,
                )
            return prompt

        message = payload.get("message")
        return message or str(tool_result)

    def _parse_tool_payload(self, tool_result: Any) -> Any:
        if isinstance(tool_result, dict):
            return tool_result
        if isinstance(tool_result, str):
            try:
                return json.loads(tool_result)
            except Exception as e:
                logger.debug("tool_payload_parse_failed", error=str(e), exc_info=True)
                return tool_result
        return tool_result

    def _summarize_tool_output(self, tool_name: str, data: Any, max_preview: int = 400) -> str:
        """Create a short, human-friendly summary of a tool's output.

        Keeps the preview concise to avoid long TTS or LLM dumps.
        """
        try:
            if data is None:
                return f"[{tool_name}] No result."

            # Dictionaries: show key summary
            if isinstance(data, dict):
                keys = list(data.keys())
                if not keys:
                    return f"[{tool_name}] Empty object."
                preview_keys = keys[:8]
                return f"[{tool_name}] Object with keys: {', '.join(map(str, preview_keys))}{'...' if len(keys) > len(preview_keys) else ''}."

            # Lists: show length and small preview
            if isinstance(data, list):
                length = len(data)
                preview = json.dumps(data[:3], default=str)
                if len(preview) > max_preview:
                    preview = preview[: max_preview - 3] + "..."
                return f"[{tool_name}] List with {length} items. Preview: {preview}"

            # Bytes / binary
            if isinstance(data, (bytes, bytearray)):
                return f"[{tool_name}] Binary output ({len(data)} bytes)."

            # Strings: truncate long text
            if isinstance(data, str):
                text = data.strip()
                if len(text) <= max_preview:
                    return text
                preview = text[: max_preview].rsplit("\n", 1)[0]
                return f"[{tool_name}] Long text ({len(text)} chars). Preview: {preview}..."

            # Fallback to string repr
            s = str(data)
            if len(s) > max_preview:
                return f"[{tool_name}] {s[: max_preview]}..."
            return s
        except Exception as e:
            logger.debug("tool_summary_failed", tool=tool_name, error=str(e), exc_info=True)
            return f"[{tool_name}] (unavailable)"

    async def _handle_vision(self, decision, prompt: str, user_question: str | None = None) -> str:
        question_text = user_question or prompt
        if decision.vision_mode == "screen":
            return await self._describe_screen_or_unavailable(question_text)

        if decision.vision_mode == "file":
            if not decision.file_path:
                return "Which file should I look at?"
            file_text = await execute_tool("read_workspace_file", {"relative_path": decision.file_path}, event_bus=self._event_bus)
            file_prompt = f"{prompt}\n\nFile: {decision.file_path}\n\n{file_text}"
            return await self._process_text_fallback(file_prompt)

        return "I need either a screen capture or a file path to proceed."

    async def _process_text_fallback(self, prompt: str) -> str:
        if self.gemini_available and not self._provider_health.is_gemini_available():
            logger.info(
                "router_gemini_unavailable",
                disabled_until=self._provider_health.gemini_disabled_until,
                target_provider="groq",
            )
        elif self._can_use_gemini():
            try:
                response_text = await self._process_gemini(prompt)
                self._record_provider_success("gemini")
                return response_text
            except Exception as e:
                rate_limited = self._is_rate_limit_error(e)
                self._record_provider_failure("gemini", e, rate_limited)
                cooldown = self._gemini_cooldown_seconds(e)
                if cooldown > 0:
                    self._provider_health.disable_gemini(cooldown)
                    if "perday" in str(e).lower() or "limit: 0" in str(e).lower():
                        self._disable_gemini_globally(cooldown)
                logger.warning("text_fallback_gemini_failed", error=str(e), exc_info=True)

        if self._can_use_provider("groq", self.groq_available):
            try:
                response_text = await self._process_groq(prompt)
                self._record_provider_success("groq")
                return response_text
            except Exception as e:
                rate_limited = self._is_rate_limit_error(e)
                self._record_provider_failure("groq", e, rate_limited)
                logger.warning("text_fallback_groq_failed", error=str(e), exc_info=True)

        if self._can_use_provider("ollama", self.ollama_available):
            try:
                response_text = await self._process_ollama(prompt)
                self._record_provider_success("ollama")
                return response_text
            except Exception as e:
                self._record_provider_failure("ollama", e, False)
                logger.warning("text_fallback_ollama_failed", error=str(e), exc_info=True)

        return "Can't reach any LLM providers right now."

    async def check_provider_status(self) -> tuple[dict[str, str], str]:
        """Ping providers at startup and return status map and primary provider."""
        status = {
            "Gemini": "UNAVAILABLE",
            "Groq": "UNAVAILABLE",
            "Ollama": "UNAVAILABLE",
        }

        if self.gemini_available:
            try:
                types = self._genai_types
                await asyncio.to_thread(
                    self.gemini_client.models.generate_content,
                    model=self.gemini_model_name,
                    contents="ping",
                    config=types.GenerateContentConfig(max_output_tokens=1),
                )
                status["Gemini"] = "OK"
            except Exception as e:
                msg = str(e).lower()
                if "perday" in msg or "limit: 0" in msg:
                    status["Gemini"] = "QUOTA_EXHAUSTED"
                    self._provider_health.disable_gemini(
                        float(self._cfg.providers.gemini_daily_quota_cooldown_hours) * 3600.0
                    )
                elif "perminute" in msg or "429" in msg:
                    status["Gemini"] = "RATE_LIMITED"
                    self._provider_health.disable_gemini(65.0)
                else:
                    status["Gemini"] = "ERROR"
        elif self._cfg.gemini_api_key:
            status["Gemini"] = "INIT_FAILED"
        else:
            status["Gemini"] = "NO_KEY"

        if self.groq_available:
            try:
                await self.groq_client.chat.completions.create(
                    model=self._cfg.models.fallback_llm,
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=1,
                )
                status["Groq"] = "OK"
            except Exception:
                status["Groq"] = "ERROR"
        elif self._cfg.groq_api_key:
            status["Groq"] = "INIT_FAILED"
        else:
            status["Groq"] = "NO_KEY"

        if self.ollama_available:
            try:
                await asyncio.wait_for(asyncio.to_thread(self.ollama.list), timeout=3.0)
                status["Ollama"] = "REACHABLE"
            except Exception:
                status["Ollama"] = "UNREACHABLE"
        else:
            status["Ollama"] = "UNAVAILABLE"

        primary = "None"
        if status["Gemini"] == "OK" and self._provider_health.is_gemini_available():
            primary = "Gemini"
        elif status["Groq"] == "OK":
            primary = "Groq"
        elif status["Ollama"] == "REACHABLE":
            primary = "Ollama"

        return status, primary

    # ─── Gemini Processing (NEW google-genai SDK) ────────────────────────────

    async def _process_gemini(self, prompt: str) -> str:
        """
        Process via Google Gemini using the NEW google-genai SDK.
        Automatic function calling is enabled by default — the SDK handles:
        1. Gemini decides to call a function
        2. SDK calls our Python function automatically
        3. SDK sends the result back to Gemini
        4. Returns final text response
        """
        types = self._genai_types

        # Build the contents list (shared history + new message)
        contents = self._build_gemini_contents(types)
        contents.append(types.Content(
            role="user",
            parts=[types.Part(text=prompt)]
        ))

        llm_start = time.perf_counter()
        response = await asyncio.to_thread(
            self.gemini_client.models.generate_content,
            model=self.gemini_model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=self.system_instruction,
                tools=self.tools_list,
            ),
        )
        metrics.record_latency("llm_gemini_ms", (time.perf_counter() - llm_start) * 1000)

        # Extract text from the response
        if response.text:
            return response.text

        return "Done."

    @staticmethod
    def _split_sentences_for_stream(text: str) -> tuple[list[str], str]:
        parts = re.split(r"(?<=[.!?])\s+", text)
        if len(parts) <= 1:
            return [], text
        if parts and not parts[-1].strip():
            return [p for p in parts[:-1] if p.strip()], ""
        return [p for p in parts[:-1] if p.strip()], parts[-1]

    def _emit_tts_sentences(self, fragment: str, sentence_carry: list[str]) -> None:
        if not self._event_bus or not fragment:
            return
        buf = sentence_carry[0] + fragment
        sentences, rest = self._split_sentences_for_stream(buf)
        for s in sentences:
            self._event_bus.emit("tts_sentence", {"text": s})
        sentence_carry[0] = rest

    @staticmethod
    def _normalize_fc_args(raw: Any) -> Optional[dict[str, Any]]:
        if raw is None:
            return None
        if isinstance(raw, dict):
            return dict(raw)
        try:
            return dict(raw)
        except Exception as e:
            logger.debug("function_call_args_normalize_failed", error=str(e), exc_info=True)
            return None

    @staticmethod
    def _merge_function_call_chunk(
        buffered_tool_calls: dict[str, dict[str, Any]],
        fc,
    ) -> None:
        key = fc.id if getattr(fc, "id", None) else "0"
        slot = buffered_tool_calls.setdefault(
            key, {"name": "", "args": None, "args_nonempty": False}
        )
        if fc.name:
            slot["name"] = fc.name
        merged = Brain._normalize_fc_args(getattr(fc, "args", None))
        if merged is not None:
            if slot["args"] is None:
                slot["args"] = {}
            slot["args"].update(merged)
            if len(slot["args"]) > 0:
                slot["args_nonempty"] = True

    @staticmethod
    def _gemini_text_delta(chunk, last_cumulative: str) -> tuple[str, str]:
        cur = chunk.text or ""
        if cur.startswith(last_cumulative):
            delta = cur[len(last_cumulative) :]
        else:
            delta = cur
        return delta, cur

    @staticmethod
    def _slot_waiting_args(slot: dict[str, Any]) -> bool:
        name = (slot.get("name") or "").strip()
        if not name:
            return False
        args = slot.get("args")
        if args is None:
            return True
        if len(args) == 0 and not slot.get("args_nonempty"):
            return True
        return False

    def _finalize_executable_tool_calls(
        self, buffered_tool_calls: dict[str, dict[str, Any]]
    ) -> list[tuple[str, dict[str, Any]]]:
        ready: list[tuple[str, dict[str, Any]]] = []
        for key in sorted(buffered_tool_calls.keys()):
            slot = buffered_tool_calls[key]
            if self._slot_waiting_args(slot):
                continue
            name = (slot.get("name") or "").strip()
            args = slot.get("args")
            if not name or args is None:
                continue
            ready.append((name, dict(args)))
        return ready

    async def _stream_gemini(self, prompt: str):
        types = self._genai_types
        contents: list = list(self._build_gemini_contents(types))
        contents.append(
            types.Content(
                role="user",
                parts=[types.Part(text=prompt)],
            )
        )

        afc_off = types.AutomaticFunctionCallingConfig(disable=True)
        stream_config = types.GenerateContentConfig(
            system_instruction=self.system_instruction,
            tools=self.tools_list,
            automatic_function_calling=afc_off,
        )

        llm_start = time.perf_counter()
        buffered_tool_calls: dict[str, dict[str, Any]] = {}
        last_cumulative = ""
        sentence_carry = [""]
        model_text_parts: list = []
        current_text_buf = ""

        try:
            stream = await self.gemini_client.aio.models.generate_content_stream(
                model=self.gemini_model_name,
                contents=contents,
                config=stream_config,
            )
            async for chunk in stream:
                delta, last_cumulative = self._gemini_text_delta(chunk, last_cumulative)
                if delta:
                    incomplete_fc = any(
                        self._slot_waiting_args(s) for s in buffered_tool_calls.values()
                    )
                    if not incomplete_fc:
                        yield delta
                        self._emit_tts_sentences(delta, sentence_carry)
                        current_text_buf += delta

                cand = chunk.candidates[0] if chunk.candidates else None
                parts = (
                    cand.content.parts
                    if cand and cand.content and cand.content.parts
                    else []
                )
                for part in parts:
                    fc = getattr(part, "function_call", None)
                    if fc is None:
                        continue
                    if current_text_buf.strip():
                        model_text_parts.append(types.Part(text=current_text_buf))
                        current_text_buf = ""
                    self._merge_function_call_chunk(buffered_tool_calls, fc)

            if current_text_buf.strip():
                model_text_parts.append(types.Part(text=current_text_buf))

            ready_calls = self._deduplicate_tool_calls(self._finalize_executable_tool_calls(buffered_tool_calls))
            if not ready_calls:
                metrics.record_latency(
                    "llm_gemini_ms", (time.perf_counter() - llm_start) * 1000
                )
                if self._event_bus and sentence_carry[0].strip():
                    self._event_bus.emit("tts_sentence", {"text": sentence_carry[0].strip()})
                return

            fc_parts = [
                types.Part(
                    function_call=types.FunctionCall(name=name, args=args)
                )
                for name, args in ready_calls
            ]
            model_text_parts.extend(fc_parts)
            contents.append(types.Content(role="model", parts=model_text_parts))

            response_parts = []
            for tool_name, args in ready_calls:
                tr = await EXECUTOR.execute(tool_name, args, event_bus=self._event_bus)
                if tr.success:
                    data = tr.data
                    # Send a concise preview back to the LLM instead of full dumps
                    body = {"result": self._summarize_tool_output(tool_name, data)}
                else:
                    body = {"error": str(tr.error or "execution failed")}
                response_parts.append(
                    types.Part.from_function_response(name=tool_name, response=body)
                )
            contents.append(types.Content(role="user", parts=response_parts))

            follow_config = types.GenerateContentConfig(
                system_instruction=self.system_instruction,
                automatic_function_calling=afc_off,
            )
            follow_stream = await self.gemini_client.aio.models.generate_content_stream(
                model=self.gemini_model_name,
                contents=contents,
                config=follow_config,
            )
            last2 = ""
            async for chunk in follow_stream:
                delta, last2 = self._gemini_text_delta(chunk, last2)
                if delta:
                    yield delta
                    self._emit_tts_sentences(delta, sentence_carry)

            metrics.record_latency(
                "llm_gemini_ms", (time.perf_counter() - llm_start) * 1000
            )
            if self._event_bus and sentence_carry[0].strip():
                self._event_bus.emit("tts_sentence", {"text": sentence_carry[0].strip()})

        except Exception as e:
            metrics.record_latency(
                "llm_gemini_ms", (time.perf_counter() - llm_start) * 1000
            )
            logger.error(
                "gemini_stream_exception",
                correlation_id=get_correlation_id(),
                error=str(e),
                exc_info=True,
            )
            buffered_tool_calls.clear()
            raise

    async def _process_gemini_vision(self, prompt: str, image_bytes: bytes) -> str:
        types = self._genai_types
        contents = self._build_gemini_contents(types)
        contents.append(types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                types.Part(text=prompt),
            ],
        ))

        llm_start = time.perf_counter()
        response = await asyncio.to_thread(
            self.gemini_client.models.generate_content,
            model=self.gemini_model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=self.system_instruction,
            ),
        )
        metrics.record_latency("llm_gemini_vision_ms", (time.perf_counter() - llm_start) * 1000)

        if response.text:
            return response.text

        return "Done."

    # ─── Groq Processing ─────────────────────────────────────────────────────

    async def _process_groq(self, prompt: str, query_hint: str = "", allow_tools: bool = True) -> str:
        """
        Process via Groq with manual function calling.
        If the LLM wants to call a tool, we execute it and send the result back.
        """
        query_text = query_hint or prompt
        base_messages = [{"role": "system", "content": self.system_instruction}]
        base_messages += self._build_shared_messages()
        base_messages.append({"role": "user", "content": prompt})

        llm_start = time.perf_counter()
        selected_tools = self._select_tools_for_provider(query_hint or prompt, "groq") if allow_tools else []
        try:
            response = await self.groq_client.chat.completions.create(
                model=self._cfg.models.fallback_llm,
                messages=base_messages,
                tools=selected_tools if selected_tools else None,
                tool_choice="auto" if selected_tools else "none",
                max_tokens=1024,
            )
        except groq.APIError as e:
            if "Failed to call a function" in str(e):
                logger.warning(
                    "groq_tool_call_failed_retrying",
                    error=str(e)
                )
                return await self._process_groq(prompt, query_hint=query_hint, allow_tools=False)
            raise
        elapsed_ms = (time.perf_counter() - llm_start) * 1000

        msg = response.choices[0].message
        tool_calls = msg.tool_calls

        # ── Handle Tool Calls ──
        if tool_calls and allow_tools:
            tool_calls = self._deduplicate_tool_calls(tool_calls)
            tool_messages = list(base_messages)
            tool_messages.append(
                {
                    "role": msg.role,
                    "content": msg.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls
                    ],
                }
            )

            tool_summaries = []
            for tc in tool_calls:
                func_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                logger.info("groq_tool_call", tool_name=func_name)

                if func_name in {"get_current_time", "get_current_datetime"} and not self._is_time_query(query_text):
                    logger.warning("tool_call_blocked", tool_name=func_name, reason="non_time_query")
                    return await self._process_groq(prompt, query_hint=query_hint, allow_tools=False)

                tool_result = await execute_tool(func_name, args, event_bus=self._event_bus)
                summary = self._summarize_tool_output(func_name, tool_result)
                tool_summaries.append(f"[tool:{func_name}] {summary}")

                tool_messages.append(
                    {
                        "tool_call_id": tc.id,
                        "role": "tool",
                        "name": func_name,
                        "content": summary,
                    }
                )

            followup_start = time.perf_counter()
            try:
                followup = await self.groq_client.chat.completions.create(
                    model=self._cfg.models.fallback_llm,
                    messages=tool_messages,
                )
            except groq.APIError as e:
                if "Failed to call a function" in str(e):
                    logger.warning(
                        "groq_tool_call_failed_retrying",
                        error=str(e)
                    )
                    return await self._process_groq(prompt, query_hint=query_hint, allow_tools=False)
                raise
            elapsed_ms += (time.perf_counter() - followup_start) * 1000
            metrics.record_latency("llm_groq_ms", elapsed_ms)

            final_text = followup.choices[0].message.content
            for summary in tool_summaries:
                self._add_history("assistant", summary)
            return final_text

        # ── No tool calls — direct text response ──
        metrics.record_latency("llm_groq_ms", elapsed_ms)
        if msg.content:
            return msg.content

        return "Done."

    async def _stream_groq_with_tools(self, prompt: str, query_hint: str = "", allow_tools: bool = True):
        query_text = query_hint or prompt
        base_messages = [{"role": "system", "content": self.system_instruction}]
        base_messages += self._build_shared_messages()
        base_messages.append({"role": "user", "content": prompt})

        llm_start = time.perf_counter()
        selected_tools = self._select_tools_for_provider(query_hint or prompt, "groq") if allow_tools else []
        try:
            stream = await self.groq_client.chat.completions.create(
                model=self._cfg.models.fallback_llm,
                messages=base_messages,
                tools=selected_tools if selected_tools else None,
                tool_choice="auto" if selected_tools else "none",
                stream=True,
            )
        except groq.APIError as e:
            if "Failed to call a function" in str(e):
                logger.warning(
                    "groq_tool_call_failed_retrying",
                    error=str(e)
                )
                # Retry without tools
                async for chunk in self._stream_groq_with_tools(
                    prompt,
                    query_hint=query_hint,
                    allow_tools=False  # plain text only
                ):
                    yield chunk
                return
            raise

        assistant_text = ""
        tool_buffers: dict[int, dict] = {}

        async for chunk in stream:
            choice = chunk.choices[0]
            delta = getattr(choice, "delta", None)
            if not delta:
                continue
            text = getattr(delta, "content", None)
            if text:
                assistant_text += text
                yield text

            streamed_tool_calls = getattr(delta, "tool_calls", None)
            if streamed_tool_calls:
                for tool_delta in streamed_tool_calls:
                    index = getattr(tool_delta, "index", 0)
                    buffer = tool_buffers.setdefault(
                        index,
                        {
                            "id": None,
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        },
                    )

                    if getattr(tool_delta, "id", None):
                        buffer["id"] = tool_delta.id
                    if getattr(tool_delta, "type", None):
                        buffer["type"] = tool_delta.type

                    function_delta = getattr(tool_delta, "function", None)
                    if function_delta:
                        if getattr(function_delta, "name", None):
                            buffer["function"]["name"] = function_delta.name
                        if getattr(function_delta, "arguments", None):
                            buffer["function"]["arguments"] += function_delta.arguments

        if tool_buffers and allow_tools:
            tool_calls = self._deduplicate_tool_calls([tool_buffers[idx] for idx in sorted(tool_buffers)])
            if any(
                call.get("function", {}).get("name") in {"get_current_time", "get_current_datetime"}
                for call in tool_calls
            ) and not self._is_time_query(query_text):
                logger.warning("tool_call_blocked", tool_name="get_current_time", reason="non_time_query")
                async for chunk in self._stream_groq_with_tools(prompt, query_hint=query_hint, allow_tools=False):
                    yield chunk
                return
            tool_messages = list(base_messages)
            tool_messages.append(
                {
                    "role": "assistant",
                    "content": assistant_text or None,
                    "tool_calls": tool_calls,
                }
            )

            for tool_call in tool_calls:
                func_name = tool_call["function"]["name"]
                raw_args = tool_call["function"].get("arguments") or "{}"
                try:
                    args = json.loads(raw_args)
                except json.JSONDecodeError:
                    args = {}

                logger.info("groq_tool_call", tool_name=func_name)
                tool_result = await execute_tool(func_name, args, event_bus=self._event_bus)
                tool_messages.append(
                    {
                        "tool_call_id": tool_call.get("id"),
                        "role": "tool",
                        "name": func_name,
                        "content": self._summarize_tool_output(func_name, tool_result),
                    }
                )

            try:
                followup = await self.groq_client.chat.completions.create(
                    model=self._cfg.models.fallback_llm,
                    messages=tool_messages,
                    stream=True,
                )
            except groq.APIError as e:
                if "Failed to call a function" in str(e):
                    logger.warning(
                        "groq_tool_call_failed_retrying",
                        error=str(e)
                    )
                    # Retry without tools
                    async for chunk in self._stream_groq_with_tools(
                        prompt,
                        query_hint=query_hint,
                        allow_tools=False  # plain text only
                    ):
                        yield chunk
                    return
                raise

            async for follow_chunk in followup:
                choice = follow_chunk.choices[0]
                delta = getattr(choice, "delta", None)
                if not delta:
                    continue
                text = getattr(delta, "content", None)
                if text:
                    yield text

        metrics.record_latency("llm_groq_ms", (time.perf_counter() - llm_start) * 1000)

    # ─── Ollama Processing ───────────────────────────────────────────────────

    async def _process_ollama(self, prompt: str) -> str:
        """
        Process via local Ollama. Text-only fallback (no tool calling).
        Useful when internet is down or API keys are exhausted.
        """
        messages = [
            {"role": "system", "content": self.system_instruction},
            {"role": "user", "content": prompt},
        ]

        llm_start = time.perf_counter()
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self.ollama.chat,
                    model=self._cfg.models.local_llm,
                    messages=messages,
                ),
                timeout=25.0,
            )
        except asyncio.TimeoutError:
            logger.warning("ollama_timeout", timeout_seconds=25.0)
            return "Having trouble with the local model, try again."
        metrics.record_latency("llm_ollama_ms", (time.perf_counter() - llm_start) * 1000)

        return response["message"]["content"]

    def _deduplicate_tool_calls(self, tool_calls):
        seen = set()
        unique = []
        for call in tool_calls or []:
            if isinstance(call, tuple) and len(call) == 2:
                name, args = call
            elif isinstance(call, dict):
                name = call.get("name") or call.get("function", {}).get("name") or ""
                args = call.get("args")
                if args is None:
                    args = call.get("arguments", {})
            else:
                name = getattr(call, "name", None) or getattr(getattr(call, "function", None), "name", "")
                args = getattr(call, "args", None)
                if args is None:
                    function = getattr(call, "function", None)
                    args = getattr(function, "arguments", {}) if function else {}

            key = (name, json.dumps(args if args is not None else {}, sort_keys=True, default=str))
            if key not in seen:
                seen.add(key)
                unique.append(call)
            else:
                logger.warning("duplicate_tool_call_removed", tool_name=name)
        return unique