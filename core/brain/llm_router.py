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
from typing import Any, Optional

from utils.logger import get_logger, get_correlation_id
from utils.metrics import metrics

logger = get_logger("llm_router")
from tools.registry import load_tools, execute_tool, EXECUTOR
from tools.schema_registry import get_tool_schema
from core.brain.intent_router import IntentRouter, PendingAction
from tools.vision_tools import capture_screen_for_vision
from utils.config import DexterConfig, get_config
from core.event_bus import EventBus


class _StreamingFallback(Exception):
    pass


class Brain:
    """Dexter's cognitive center — routes commands to the best available LLM."""

    RATE_LIMIT_BACKOFF_BASE = 2
    RATE_LIMIT_BACKOFF_MAX = 60

    def __init__(self, event_bus: Optional[EventBus] = None):
        logger.info("brain_initializing")

        self._cfg: DexterConfig = get_config()
        self._event_bus: Optional[EventBus] = event_bus

        self.tools_list = load_tools()
        self.intent_router = IntentRouter(self._cfg)
        self.shared_history = []
        self.pending_action = None
        self.max_history_tokens = int(self._cfg.history.max_tokens)
        self.provider_state = {
            "gemini": {"failures": 0, "score": 1.0, "cooldown_until": 0.0, "last_error": ""},
            "groq": {"failures": 0, "score": 1.0, "cooldown_until": 0.0, "last_error": ""},
            "ollama": {"failures": 0, "score": 1.0, "cooldown_until": 0.0, "last_error": ""},
        }

        self.system_instruction = (
            "You are Dexter, a highly capable, professional, and sophisticated AI assistant "
            "running locally on a Windows PC. You act like a digital butler similar to Jarvis from Iron Man. "
            "Your tone is polite, calm, slightly formal, and extremely concise. "
            "You can control the user's PC, search the web, take screenshots, manage notes, "
            "check weather, system status, clipboard, and more using your tools. "
            "When the user asks about weather, extract the city from the request and use get_weather with that city. "
            "If no city is mentioned, use the configured default city before asking a follow-up. "
            "When the user asks for the current time in a city, use get_current_time with that city. "
            "When the user says open X in Y where X is a website and Y is a browser, always use the open_url_in_browser tool with both parameters. "
            "When the user asks to open an application by name (e.g., 'open Spotify', 'open Discord', 'open Word'), ALWAYS use the open_application tool first. Never use a web browser to open a desktop application. "
            "When the user asks to play, watch, find, search, or open content on a platform such as Spotify, YouTube Music, Netflix, ESPN, SoundCloud, Apple Music, Prime Video, or another site, use search_content_platform with both the content query and platform. "
            "If no platform is mentioned, infer a sensible default from the content type, such as music, video, podcast, sports, or movie. "
            "Use search_google only for general web searches, and use open_url or open_url_in_browser only when the user provides a direct URL. "
            "Never confuse this with opening an application. "
            "If you do not have a tool to perform an action, tell the user politely. "
            "Speech transcription may contain minor errors; infer likely intended app names or commands "
            "from context and ask for confirmation if ambiguous. "
            "Never use emojis. Start confirmations with 'Yes sir', 'Right away sir', or 'Understood'. "
            "If a request is ambiguous, politely ask for clarification. "
            "Keep responses short — 1 to 3 sentences maximum unless the user asks for detail."
        )

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

    def _build_shared_messages(self):
        return [
            {"role": msg["role"], "content": msg["content"]}
            for msg in self.shared_history
            if msg.get("content")
        ]

    def _build_gemini_contents(self, types):
        contents = []
        for msg in self.shared_history:
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

    def _is_rate_limit_error(self, error: Exception) -> bool:
        msg = str(error).lower()
        if "rate limit" in msg or "resource_exhausted" in msg or "quota" in msg:
            return True
        if "429" in msg:
            return True
        status = getattr(error, "status_code", None)
        if status == 429:
            return True
        return False

    # ─── Main Command Processing ─────────────────────────────────────────────

    async def process_command(self, user_command: str, long_term_memory: str = "") -> str:
        """
        Routes a user command through the LLM fallback chain:
        Gemini → Groq → Ollama
        """
        prompt = user_command
        if long_term_memory:
            prompt = f"{long_term_memory}\n\nCurrent User Command: {user_command}"

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
                response_text = await self._handle_vision(decision, prompt)
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
            response_text = await self._handle_vision(decision, prompt)
            self._add_history("user", user_command)
            self._add_history("assistant", response_text)
            return response_text

        # ── Try Gemini (Primary) ──
        if self._can_use_provider("gemini", self.gemini_available):
            try:
                _t0 = time.perf_counter()
                response_text = await self._process_gemini(prompt)
                _ms = (time.perf_counter() - _t0) * 1000
                logger.info("llm_call_completed", provider="gemini", duration_ms=_ms)
                self._record_provider_success("gemini")
                self._add_history("user", user_command)
                self._add_history("assistant", response_text)
                return response_text
            except Exception as e:
                rate_limited = self._is_rate_limit_error(e)
                self._record_provider_failure("gemini", e, rate_limited)
                logger.warning("gemini_request_failed", error=str(e), exc_info=True)
                logger.info("llm_fallback", from_provider="gemini", to_provider="groq")

        # ── Try Groq (Fallback) ──
        if self._can_use_provider("groq", self.groq_available):
            try:
                _t0 = time.perf_counter()
                response_text = await self._process_groq(prompt)
                _ms = (time.perf_counter() - _t0) * 1000
                logger.info("llm_call_completed", provider="groq", duration_ms=_ms)
                self._record_provider_success("groq")
                self._add_history("user", user_command)
                self._add_history("assistant", response_text)
                return response_text
            except Exception as e:
                rate_limited = self._is_rate_limit_error(e)
                self._record_provider_failure("groq", e, rate_limited)
                logger.warning("groq_request_failed", error=str(e), exc_info=True)
                logger.info("llm_fallback", from_provider="groq", to_provider="ollama")

        # ── Try Ollama (Local Offline) ──
        if self._can_use_provider("ollama", self.ollama_available):
            try:
                _t0 = time.perf_counter()
                response_text = await self._process_ollama(prompt)
                _ms = (time.perf_counter() - _t0) * 1000
                logger.info("llm_call_completed", provider="ollama", duration_ms=_ms)
                self._record_provider_success("ollama")
                self._add_history("user", user_command)
                self._add_history("assistant", response_text)
                return response_text
            except Exception as e:
                self._record_provider_failure("ollama", e, False)
                logger.error("ollama_request_failed", error=str(e), exc_info=True)

        return (
            "I apologize, sir. All of my neural networks are currently unreachable. "
            "Please verify your API keys in config.yaml and check your internet connection."
        )

    async def process_command_stream(self, user_command: str, long_term_memory: str = ""):
        if self.pending_action:
            response_text = await self.process_command(user_command, long_term_memory)
            yield response_text
            return

        decision = self.intent_router.detect_intent(user_command)
        if decision.action != "none":
            response_text = await self.process_command(user_command, long_term_memory)
            yield response_text
            return

        prompt = user_command
        if long_term_memory:
            prompt = f"{long_term_memory}\n\nCurrent User Command: {user_command}"

        if self._can_use_provider("gemini", self.gemini_available):
            try:
                response_text = ""
                _t0 = time.perf_counter()
                async for chunk in self._stream_gemini(prompt):
                    response_text += chunk
                    yield chunk
                if response_text:
                    _ms = (time.perf_counter() - _t0) * 1000
                    logger.info("llm_stream_completed", provider="gemini", duration_ms=_ms)
                    self._record_provider_success("gemini")
                    self._add_history("user", user_command)
                    self._add_history("assistant", response_text)
                    return
            except Exception as e:
                rate_limited = self._is_rate_limit_error(e)
                self._record_provider_failure("gemini", e, rate_limited)
                logger.warning("gemini_stream_failed", error=str(e), exc_info=True)

        if self._can_use_provider("groq", self.groq_available):
            try:
                response_text = ""
                _t0 = time.perf_counter()
                async for chunk in self._stream_groq_with_tools(prompt, allow_tools=True):
                    response_text += chunk
                    yield chunk
                if response_text:
                    _ms = (time.perf_counter() - _t0) * 1000
                    logger.info("llm_stream_completed", provider="groq", duration_ms=_ms)
                    self._record_provider_success("groq")
                    self._add_history("user", user_command)
                    self._add_history("assistant", response_text)
                    return
            except Exception as e:
                rate_limited = self._is_rate_limit_error(e)
                self._record_provider_failure("groq", e, rate_limited)
                logger.warning("groq_stream_failed", error=str(e), exc_info=True)

        if self._can_use_provider("ollama", self.ollama_available):
            try:
                _t0 = time.perf_counter()
                response_text = await self._process_ollama(prompt)
                if response_text:
                    _ms = (time.perf_counter() - _t0) * 1000
                    logger.info("llm_stream_completed", provider="ollama", duration_ms=_ms)
                    self._record_provider_success("ollama")
                    self._add_history("user", user_command)
                    self._add_history("assistant", response_text)
                    yield response_text
                    return
            except Exception as e:
                self._record_provider_failure("ollama", e, False)
                logger.warning("ollama_stream_failed", error=str(e), exc_info=True)

        response_text = await self.process_command(user_command, long_term_memory)
        yield response_text

    def _requires_confirmation(self, tool_name: str) -> bool:
        return tool_name in {"shutdown_pc", "restart_pc", "sleep_pc"}

    def _handle_tool_response(self, tool_name: str, tool_result: Any) -> str:
        if tool_name != "resolve_open_target":
            return str(tool_result)

        payload = self._parse_tool_payload(tool_result)
        if not isinstance(payload, dict):
            return str(tool_result)

        status = payload.get("status")
        if status == "ask":
            prompt = payload.get("message") or "Please choose an option, sir."
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

    async def _handle_vision(self, decision, prompt: str) -> str:
        if decision.vision_mode == "screen":
            try:
                capture = await asyncio.to_thread(capture_screen_for_vision)
            except Exception as e:
                logger.error("vision_capture_failed", error=str(e), exc_info=True)
                return "I was unable to capture the screen for analysis."
            if self._can_use_provider("gemini", self.gemini_available):
                vision_prompt = f"{prompt}\n\nCapture mode: {capture.capture_mode}\n"
                if capture.foreground_window:
                    vision_prompt += f"Foreground window (Active App): {capture.foreground_window}\n"
                vision_prompt += (
                    "Describe what is visible naturally and directly. Do not mention screenshots or files. "
                    "If a Foreground window is provided, focus your description primarily on that application's content, as it is what the user is currently interacting with. "
                    "CRITICAL: If you see a terminal, IDE, or code editor (e.g., VS Code, PyCharm, Command Prompt) running this assistant, COMPLETELY IGNORE IT. Describe the other applications visible on the screen (like web browsers, video players, etc.) instead."
                )
                return await self._process_gemini_vision(vision_prompt, capture.image_bytes)
            return "Vision analysis is only available with Gemini at the moment."

        if decision.vision_mode == "file":
            if not decision.file_path:
                return "Which file should I inspect, sir?"
            file_text = await execute_tool("read_workspace_file", {"relative_path": decision.file_path}, event_bus=self._event_bus)
            file_prompt = f"{prompt}\n\nFile: {decision.file_path}\n\n{file_text}"
            return await self._process_text_fallback(file_prompt)

        return "I need either a screen capture or a file path to proceed."

    async def _process_text_fallback(self, prompt: str) -> str:
        if self._can_use_provider("gemini", self.gemini_available):
            try:
                response_text = await self._process_gemini(prompt)
                self._record_provider_success("gemini")
                return response_text
            except Exception as e:
                rate_limited = self._is_rate_limit_error(e)
                self._record_provider_failure("gemini", e, rate_limited)
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

        return "I cannot access any LLM providers at the moment, sir."

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

        return "Command executed, sir."

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

            ready_calls = self._finalize_executable_tool_calls(buffered_tool_calls)
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
                    if isinstance(data, (dict, list)):
                        body = {"result": json.dumps(data)}
                    else:
                        body = {"result": str(data)}
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

        return "Vision task completed, sir."

    # ─── Groq Processing ─────────────────────────────────────────────────────

    async def _process_groq(self, prompt: str) -> str:
        """
        Process via Groq with manual function calling.
        If the LLM wants to call a tool, we execute it and send the result back.
        """
        base_messages = [{"role": "system", "content": self.system_instruction}]
        base_messages += self._build_shared_messages()
        base_messages.append({"role": "user", "content": prompt})

        llm_start = time.perf_counter()
        response = await self.groq_client.chat.completions.create(
            model=self._cfg.models.fallback_llm,
            messages=base_messages,
            tools=self.groq_tools if self.groq_tools else None,
            tool_choice="auto",
            max_tokens=1024,
        )
        elapsed_ms = (time.perf_counter() - llm_start) * 1000

        msg = response.choices[0].message
        tool_calls = msg.tool_calls

        # ── Handle Tool Calls ──
        if tool_calls:
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

                tool_result = await execute_tool(func_name, args, event_bus=self._event_bus)
                tool_summaries.append(f"[tool:{func_name}] {tool_result}")

                tool_messages.append(
                    {
                        "tool_call_id": tc.id,
                        "role": "tool",
                        "name": func_name,
                        "content": str(tool_result),
                    }
                )

            followup_start = time.perf_counter()
            followup = await self.groq_client.chat.completions.create(
                model=self._cfg.models.fallback_llm,
                messages=tool_messages,
            )
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

        return "Command executed, sir."

    async def _stream_groq(self, prompt: str):
        async for chunk in self._stream_groq_with_tools(prompt, allow_tools=False):
            yield chunk

    async def _stream_groq_with_tools(self, prompt: str, allow_tools: bool = True):
        base_messages = [{"role": "system", "content": self.system_instruction}]
        base_messages += self._build_shared_messages()
        base_messages.append({"role": "user", "content": prompt})

        llm_start = time.perf_counter()
        stream = await self.groq_client.chat.completions.create(
            model=self._cfg.models.fallback_llm,
            messages=base_messages,
            tools=self.groq_tools if allow_tools and self.groq_tools else None,
            tool_choice="auto" if allow_tools else "none",
            stream=True,
        )

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
            tool_calls = [tool_buffers[idx] for idx in sorted(tool_buffers)]
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
                        "content": str(tool_result),
                    }
                )

            followup = await self.groq_client.chat.completions.create(
                model=self._cfg.models.fallback_llm,
                messages=tool_messages,
                stream=True,
            )

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
        response = await asyncio.to_thread(
            self.ollama.chat,
            model=self._cfg.models.local_llm,
            messages=messages,
        )
        metrics.record_latency("llm_ollama_ms", (time.perf_counter() - llm_start) * 1000)

        return response["message"]["content"]