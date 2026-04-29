"""
Dexter Brain — Multi-LLM Router with Automatic Fallback Chain
Priority: Gemini (Primary) → Groq (Fallback) → Ollama (Local Offline)

Handles tool/function calling for Gemini and Groq backends.
Ollama serves as a text-only emergency fallback.

MIGRATED: Uses the new `google-genai` SDK (replaces deprecated `google-generativeai`).
"""
import yaml
import json
import asyncio
import inspect
import time
import random
import base64
from utils.logger import logger
from utils.metrics import metrics
from tools.registry import load_tools, execute_tool
from core.brain.intent_router import IntentRouter, PendingAction


class Brain:
    """Dexter's cognitive center — routes commands to the best available LLM."""

    MAX_SHARED_HISTORY = 20  # Keep conversation manageable for token limits
    RATE_LIMIT_BACKOFF_BASE = 2
    RATE_LIMIT_BACKOFF_MAX = 60

    def __init__(self):
        logger.info("Initializing Dexter's Brain (Multi-LLM Router)...")

        with open("config.yaml", "r") as file:
            self.config = yaml.safe_load(file)

        self.tools_list = load_tools()
        self.intent_router = IntentRouter(self.config)
        self.shared_history = []
        self.pending_action = None
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
            "If you do not have a tool to perform an action, tell the user politely. "
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
            available.append(f"Gemini ({self.config['models']['primary_llm']})")
        if self.groq_available:
            available.append(f"Groq ({self.config['models']['fallback_llm']})")
        if self.ollama_available:
            available.append(f"Ollama ({self.config['models']['local_llm']})")

        if available:
            logger.info(f"Brain ONLINE → {' → '.join(available)}")
        else:
            logger.error("CRITICAL: No LLM backend is available. Dexter cannot think!")

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

            gemini_key = self.config.get("api_keys", {}).get("gemini", "")
            if not gemini_key or "YOUR" in gemini_key.upper():
                logger.info("Gemini: No valid API key configured. Skipping.")
                return

            # New SDK: create a Client with the API key
            self.gemini_client = genai.Client(api_key=gemini_key)
            self._genai_types = types

            # Store model name for later use
            self.gemini_model_name = self.config["models"]["primary_llm"]

            self.gemini_available = True
            logger.info(f"PRIMARY: Gemini ({self.gemini_model_name}) ✓ [google-genai SDK]")

        except ImportError:
            logger.warning("Gemini: 'google-genai' package not installed. pip install google-genai")
        except Exception as e:
            logger.warning(f"Gemini initialization failed: {e}")

    def _init_groq(self):
        """Initialize Groq as the fallback LLM with manual function calling."""
        self.groq_available = False
        try:
            from groq import AsyncGroq

            groq_key = self.config.get("api_keys", {}).get("groq", "")
            if not groq_key or "YOUR" in groq_key.upper():
                logger.info("Groq: No valid API key configured. Skipping.")
                return

            self.groq_client = AsyncGroq(api_key=groq_key)
            self.groq_tools = self._build_groq_tool_schemas()
            self.groq_available = True
            logger.info(f"FALLBACK: Groq ({self.config['models']['fallback_llm']}) ✓")

        except ImportError:
            logger.warning("Groq: 'groq' package not installed. pip install groq")
        except Exception as e:
            logger.warning(f"Groq initialization failed: {e}")

    def _init_ollama(self):
        """Initialize local Ollama as the offline emergency fallback (text-only, no tools)."""
        self.ollama_available = False
        try:
            import ollama as ollama_lib

            self.ollama = ollama_lib
            # Quick connection test — if Ollama server isn't running, this fails fast
            ollama_lib.list()
            self.ollama_available = True
            logger.info(f"LOCAL: Ollama ({self.config['models']['local_llm']}) ✓")

        except ImportError:
            logger.info("Ollama: package not installed (optional). pip install ollama")
        except Exception:
            logger.info("Ollama: server not running (optional offline fallback).")

    # ─── Tool Schema Generation ──────────────────────────────────────────────

    def _build_groq_tool_schemas(self):
        """
        Auto-generates OpenAI-compatible tool schemas from Python functions
        using inspect.signature(). No more brittle hardcoded parameter checks.
        """
        schemas = []
        for func in self.tools_list:
            sig = inspect.signature(func)
            properties = {}
            required = []

            for param_name, param in sig.parameters.items():
                # Map Python type annotations to JSON Schema types
                ptype = "string"
                if param.annotation == int:
                    ptype = "integer"
                elif param.annotation == float:
                    ptype = "number"
                elif param.annotation == bool:
                    ptype = "boolean"

                properties[param_name] = {
                    "type": ptype,
                    "description": f"The {param_name} to provide.",
                }

                # If no default value, it's required
                if param.default == inspect.Parameter.empty:
                    required.append(param_name)

            tool_schema = {
                "type": "function",
                "function": {
                    "name": func.__name__,
                    "description": (func.__doc__ or "").strip() or f"Executes {func.__name__}",
                },
            }

            # Only add parameters block if the function actually takes arguments
            if properties:
                tool_schema["function"]["parameters"] = {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                }

            schemas.append(tool_schema)

        logger.debug(f"Built {len(schemas)} Groq tool schemas via inspect.")
        return schemas

    # ─── History Management ──────────────────────────────────────────────────

    def _add_history(self, role: str, content: str) -> None:
        self.shared_history.append({"role": role, "content": content})
        if len(self.shared_history) > self.MAX_SHARED_HISTORY:
            self.shared_history = self.shared_history[-self.MAX_SHARED_HISTORY :]

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

        logger.info("Thinking...")

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
                tool_result = await execute_tool(decision.tool_name, args)
                response_text = str(tool_result)
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

            tool_result = await execute_tool(decision.tool_name, decision.args)
            response_text = str(tool_result)
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
                response_text = await self._process_gemini(prompt)
                self._record_provider_success("gemini")
                self._add_history("user", user_command)
                self._add_history("assistant", response_text)
                return response_text
            except Exception as e:
                rate_limited = self._is_rate_limit_error(e)
                self._record_provider_failure("gemini", e, rate_limited)
                logger.warning(f"Gemini failed: {e}")
                logger.info("Falling back to Groq...")

        # ── Try Groq (Fallback) ──
        if self._can_use_provider("groq", self.groq_available):
            try:
                response_text = await self._process_groq(prompt)
                self._record_provider_success("groq")
                self._add_history("user", user_command)
                self._add_history("assistant", response_text)
                return response_text
            except Exception as e:
                rate_limited = self._is_rate_limit_error(e)
                self._record_provider_failure("groq", e, rate_limited)
                logger.warning(f"Groq failed: {e}")
                logger.info("Falling back to Ollama...")

        # ── Try Ollama (Local Offline) ──
        if self._can_use_provider("ollama", self.ollama_available):
            try:
                response_text = await self._process_ollama(prompt)
                self._record_provider_success("ollama")
                self._add_history("user", user_command)
                self._add_history("assistant", response_text)
                return response_text
            except Exception as e:
                self._record_provider_failure("ollama", e, False)
                logger.error(f"Ollama also failed: {e}")

        return (
            "I apologize, sir. All of my neural networks are currently unreachable. "
            "Please verify your API keys in config.yaml and check your internet connection."
        )

    def _requires_confirmation(self, tool_name: str) -> bool:
        return tool_name in {"shutdown_pc", "restart_pc", "sleep_pc"}

    async def _handle_vision(self, decision, prompt: str) -> str:
        if decision.vision_mode == "screen":
            image_b64 = await execute_tool("capture_screen", {})
            if not isinstance(image_b64, str) or "base64," not in image_b64:
                return str(image_b64)
            try:
                encoded = image_b64.split("base64,", 1)[1]
                image_bytes = base64.b64decode(encoded)
            except Exception:
                return "I could not decode the captured image."
            if self._can_use_provider("gemini", self.gemini_available):
                return await self._process_gemini_vision(prompt, image_bytes)
            return "Vision analysis is only available with Gemini at the moment."

        if decision.vision_mode == "file":
            if not decision.file_path:
                return "Which file should I inspect, sir?"
            file_text = await execute_tool("read_workspace_file", {"relative_path": decision.file_path})
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

        if self._can_use_provider("groq", self.groq_available):
            try:
                response_text = await self._process_groq(prompt)
                self._record_provider_success("groq")
                return response_text
            except Exception as e:
                rate_limited = self._is_rate_limit_error(e)
                self._record_provider_failure("groq", e, rate_limited)

        if self._can_use_provider("ollama", self.ollama_available):
            try:
                response_text = await self._process_ollama(prompt)
                self._record_provider_success("ollama")
                return response_text
            except Exception as e:
                self._record_provider_failure("ollama", e, False)

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
            model=self.config["models"]["fallback_llm"],
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
                logger.info(f"Groq requested tool: {func_name}")

                tool_result = await execute_tool(func_name, args)
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
                model=self.config["models"]["fallback_llm"],
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
            model=self.config["models"]["local_llm"],
            messages=messages,
        )
        metrics.record_latency("llm_ollama_ms", (time.perf_counter() - llm_start) * 1000)

        return response["message"]["content"]