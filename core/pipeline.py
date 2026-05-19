import asyncio
import os
import re
import time
from functools import partial
from pathlib import Path
from rapidfuzz import fuzz
from uuid import uuid4
from typing import Optional

from core.activation_manager import ActivationConfig, ActivationManager
from core.event_bus import EventBus, DexterEvents
from core.health import HealthMonitor
from core.state_machine import AssistantState
from core.wake_word.detector import WakeWordDetector
from core.session_activity import session_activity
from utils.transcript_correction import TranscriptCorrector, apply_wake_word_corrections
from utils.logger import get_logger, bind_correlation_id, clear_correlation_id
from utils.metrics import metrics
from utils.config import DexterConfig
from core.brain import session_state

logger = get_logger("pipeline")


class AsyncPipeline:
    def __init__(
        self,
        config: DexterConfig,
        transcriber,
        vad_listener,
        tts_manager,
        memory_vault,
        brain,
        event_bus: Optional[EventBus] = None,
        health_monitor: Optional[HealthMonitor] = None,
        asr_engine=None,
    ) -> None:
        self.config = config
        self.transcriber = transcriber
        self.vad = vad_listener
        self.tts = tts_manager
        self.memory = memory_vault
        self.brain = brain
        self.event_bus = event_bus or EventBus()
        self.health_monitor = health_monitor
        self.asr_engine = asr_engine
        self._last_transcript = ""

        self.state = AssistantState.IDLE
        self._state_changed_at = time.time()

        wb = config.wake_behavior
        activation = config.activation
        self.activation_mode = (activation.mode or "wake_word").strip().lower()
        self.command_window_seconds = (
            activation.active_window_seconds
            if self.activation_mode == "clap"
            else wb.active_seconds
        )
        self.clap_sensitivity = float(activation.clap_sensitivity)
        self.start_active = bool(activation.start_active)
        self.min_command_words = max(1, int(activation.min_command_words or 1))
        self.wake_words = list(activation.wake_words or config.wake_words)
        configured_primary_wake_word = (activation.wake_word or "").strip().lower()
        if configured_primary_wake_word and configured_primary_wake_word not in [w.lower() for w in self.wake_words]:
            self.wake_words.insert(0, configured_primary_wake_word)
        self.wake_detector = None
        if self.activation_mode == "wake_word":
            self.wake_detector = WakeWordDetector(
                wake_phrases=self.wake_words,
                match_mode=wb.match_mode,
                min_confidence=wb.min_confidence,
                max_prefix_tokens=wb.max_prefix_tokens,
            )
        self.awake_until = 0.0
        self.corrector = TranscriptCorrector()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._turn_count = 0
        self._consecutive_activation_drops = 0
        self._always_on_until = 0.0
        self._diag_enabled = os.environ.get("DEXTER_DIAGNOSTIC", "0") == "1"

        act = config.activation
        smart_mode = (act.mode or "smart").strip().lower()
        if smart_mode not in {"smart", "wake_word", "always_on"}:
            smart_mode = "wake_word" if self.activation_mode == "wake_word" else "smart"
        self._activation = ActivationManager(
            ActivationConfig(
                mode=smart_mode,  # type: ignore[arg-type]
                wake_word=(act.wake_word or "dexter").strip().lower(),
                active_hours_start=getattr(act, "active_hours_start", "09:00"),
                active_hours_end=getattr(act, "active_hours_end", "18:00"),
                active_days=getattr(act, "active_days", None),
                always_on_after_n_interactions=int(
                    getattr(act, "always_on_after_n_interactions", 3) or 3
                ),
                always_on_window_seconds=float(
                    getattr(act, "always_on_window_seconds", 120) or 120
                ),
                always_on_timeout_seconds=float(
                    getattr(act, "always_on_timeout_seconds", 300) or 300
                ),
            )
        )

    def _contains_wake_word(self, text: str) -> bool:
        if self.wake_detector is None:
            return False
        return bool(self.wake_detector.detect(text).triggered)

    def _strip_wake_word(self, text: str) -> str:
        if self.wake_detector is None:
            return text
        detection = self.wake_detector.detect(text)
        if detection.triggered and detection.cleaned_text.strip():
            return detection.cleaned_text
        return text

    def _effective_activation_mode(self) -> str:
        if self.activation_mode == "clap":
            return "clap"
        if self._always_on_until > time.time():
            return "always_on"
        if self._activation.current_mode == "always_on":
            return "always_on"
        return "wake_word"

    def _record_activation_drop(self, transcript_text: str, reason: str) -> None:
        self._consecutive_activation_drops += 1
        logger.info(
            "activation_failed",
            transcript=transcript_text,
            wake_word_found=False,
            reason=reason,
        )
        threshold = max(1, int(self.config.activation.fallback_to_always_on_after_failures or 3))
        if self._consecutive_activation_drops >= threshold:
            self._always_on_until = time.time() + 60.0
            self._consecutive_activation_drops = 0
            logger.warning("[ACTIVATION] 3 consecutive wake word failures - switching to always_on for 60s")

    def _reset_activation_drop_counter(self) -> None:
        self._consecutive_activation_drops = 0

    def _set_state(self, state: AssistantState) -> None:
        if self.state != state:
            old = self.state
            self.state = state
            self._state_changed_at = time.time()
            logger.info(
                "state_transition",
                from_state=old.name,
                to_state=state.name,
            )
            self.event_bus.emit("state_changed", {"state": state.name})
            rag_index = getattr(self.memory, "personal_rag", None)
            if rag_index is not None and hasattr(rag_index, "set_pipeline_state"):
                try:
                    rag_index.set_pipeline_state(state.name)
                except Exception:
                    pass
            if state == AssistantState.PROCESSING:
                session_activity.mark_active()
            elif state == AssistantState.IDLE:
                session_activity.mark_idle()
            if self.health_monitor is not None:
                self.health_monitor.healthy("pipeline", f"state={state.name}")

    def _log_activation_failure(self, transcript_text: str, reason: str) -> None:
        logger.warning(
            "command_dropped_activation_failed",
            transcript=transcript_text[:50],
            reason=reason,
            activation_mode=self.config.activation.mode,
        )

    def _is_awake(self) -> bool:
        return time.time() < self.awake_until

    def _open_wake_window(self) -> None:
        self.awake_until = time.time() + self.command_window_seconds

    def _diag(self, event: str, **fields) -> None:
        if self._diag_enabled:
            logger.info(f"diagnostic_{event}", **fields)

    def _handle_clap_activation(self) -> None:
        self._open_wake_window()
        try:
            asyncio.create_task(self.tts.play_chime())
        except RuntimeError:
            pass

    def _on_clap_detected(self) -> None:
        if self._loop:
            self._loop.call_soon_threadsafe(self._handle_clap_activation)

    @staticmethod
    def _split_sentences(text: str) -> tuple[list[str], str]:
        parts = re.split(r"(?<=[.!?])\s+", text)
        if len(parts) <= 1:
            return [], text
        if parts and not parts[-1].strip():
            return [p for p in parts[:-1] if p.strip()], ""
        return [p for p in parts[:-1] if p.strip()], parts[-1]

    @staticmethod
    def _strip_leading_fillers(text: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", (text or "").strip().lower())
        tokens = [t for t in cleaned.split() if t]
        fillers = {"hey", "hi", "hello", "ok", "okay", "dexter"}
        while tokens and tokens[0] in fillers:
            tokens.pop(0)
        return " ".join(tokens)

    @staticmethod
    def _looks_actionable_utterance(text: str) -> bool:
        normalized = AsyncPipeline._strip_leading_fillers(text)
        if not normalized:
            return False

        command_starts = (
            "open ",
            "launch ",
            "start ",
            "close ",
            "play ",
            "watch ",
            "find ",
            "search ",
            "set ",
            "increase ",
            "decrease ",
            "turn ",
            "lock ",
            "shutdown",
            "restart",
            "sleep",
            "take ",
            "capture ",
            "read ",
            "copy ",
            "type ",
            "press ",
            "describe ",
            "analyze ",
        )
        if normalized.startswith(command_starts):
            return True

        query_hints = ("what is", "what's", "whats", "tell me", "how is", "how's", "what am", "what do")
        tool_keywords = (
            "weather",
            "temperature",
            "forecast",
            "time",
            "date",
            "clipboard",
            "screenshot",
            "screen",
            "volume",
            "system status",
            "battery",
            "cpu",
            "ram",
            "looking at",
            "look at",
            "see",
        )
        if any(keyword in normalized for keyword in tool_keywords):
            if normalized.endswith("?") or normalized.startswith(query_hints):
                return True

        return False

    @staticmethod
    def _detect_activation_command(text: str) -> tuple[str, float, str] | None:
        """Voice commands to switch activation mode. Returns (mode, duration_sec, spoken)."""
        normalized = re.sub(r"\s+", " ", (text or "").lower().strip())
        commands = {
            "stay active": ("always_on", 3600.0, "I'll stay active for the next hour."),
            "go passive": ("wake_word", 3600.0, "Going passive — say Dexter when you need me."),
            "always listen": ("always_on", 86400.0, "Always listening."),
            "go quiet": ("wake_word", 86400.0, "Going quiet."),
            "active mode": ("always_on", 3600.0, "Active mode on."),
            "passive mode": ("wake_word", 3600.0, "Passive mode on."),
        }
        for phrase, payload in commands.items():
            if normalized == phrase or normalized.endswith(phrase):
                return payload
        return None

    def _detect_correction_intent(self, text: str) -> str | None:
        """Detect if the user is explicitly correcting the previous misheard command."""
        text = text.lower().strip()
        patterns = [
            r"^no i said\s+(.+)$",
            r"^i meant\s+(.+)$",
            r"^no,? i meant\s+(.+)$",
            r"^not that,? i meant\s+(.+)$"
        ]
        for p in patterns:
            m = re.match(p, text)
            if m:
                return m.group(1).strip()
        return None

    @staticmethod
    def _transcript_is_usable(text: str) -> bool:
        """
        Check if a transcript is worth processing.
        Filters out noise, empty results, and transcription artifacts.
        """
        if not text or not text.strip():
            return False

        cleaned = re.sub(r"\s+", " ", text.strip().lower())

        # Too short to be a real command
        if len(cleaned.split()) < 2:
            return False

        noise_patterns = [
            "thank you",
            "thanks for watching",
            "please subscribe",
            "[music]",
            "[applause]",
            "...",
        ]
        if any(p in cleaned for p in noise_patterns):
            return False

        return True

    def _should_use_rag(self, command: str) -> bool:
        """
        Determine if this command benefits from personal file context.
        Prefer injecting context when uncertain — missing context hurts more
        than an extra retrieval.
        """
        words = (command or "").lower().split()
        word_count = len(words)

        if word_count < 3:
            return False

        hard_skip_starters = {
            "open", "close", "launch", "start",
            "shutdown", "restart", "sleep", "lock",
            "volume", "mute", "unmute", "screenshot",
            "type", "press", "click", "minimize",
            "maximize", "play", "pause", "stop",
        }
        if words[0] in hard_skip_starters and word_count <= 5:
            return False

        knowledge_triggers = {
            "what", "tell", "explain", "describe",
            "summarize", "summarise", "how", "why",
            "show", "give", "find", "search", "look",
            "read", "check", "review", "analyse",
            "analyze", "about", "regarding",
        }
        if words[0] in knowledge_triggers:
            return True

        project_indicators = [
            "project", "folder", "file", "code",
            "function", "class", "module", "script",
            "lab", "assignment", "document", "report",
            "userauth", "bimarsha", "practical",
            "office", "reporting", "dexter",
        ]
        command_lower = command.lower()
        if any(p in command_lower for p in project_indicators):
            return True

        return word_count > 6

    def _extract_file_reference(self, command: str) -> tuple[str | None, str | None, str | None]:
        """
        Detect whether the command references a specific file and try to extract
        (folder_hint, subfolder_hint, filename_hint). Returns (None, None, None)
        when no obvious file reference is present.
        """
        cmd = (command or "").lower()
        # Look for a filename with common extensions
        file_exts = r"(?:\.php|\.java|\.py|\.js|\.html|\.css|\.txt|\.md|\.ts|\.json|\.yaml|\.yml|\.xml)"
        m = re.search(r"([\w\-\s]+?\S+)" + file_exts, cmd)
        filename = None
        if m:
            filename = m.group(0).strip()

        # Look for lab/subfolder hints like 'lab 20' or 'lab-20'
        subfolder = None
        labm = re.search(r"lab[\s_-]?(\d{1,3})", cmd)
        if labm:
            subfolder = f"lab {labm.group(1)}"

        # Folder hint: phrases after 'in' or 'of' up to 6 tokens
        folder = None
        fm = re.search(r"(?:in|of)\s+([\w\-\s]{3,60})", cmd)
        if fm:
            candidate = fm.group(1).strip()
            # trim to first reasonable phrase
            folder = " ".join(candidate.split()[:6])

        if not filename and not subfolder and not folder:
            return (None, None, None)
        return (folder, subfolder, filename)

    def _resolve_file_by_description(self, folder_hint: str | None, subfolder_hint: str | None, filename_hint: str | None, rag_index) -> str:
        """
        Attempt to resolve a file path described by hints using the RAG index's
        known filenames. Returns the file content if a confident match is found,
        otherwise empty string.
        """
        try:
            candidates = rag_index.get_all_indexed_filenames()
        except Exception:
            candidates = []

        if not candidates:
            return ""

        q = " ".join(filter(None, [folder_hint or "", subfolder_hint or "", filename_hint or ""]))
        q_compact = re.sub(r"[^a-z0-9]+", "", (q or "").lower())
        best = None
        best_score = 0
        for p in candidates:
            name_compact = re.sub(r"[^a-z0-9]+", "", p.lower())
            try:
                score = fuzz.partial_ratio(q_compact, name_compact)
            except Exception:
                score = 0
            if score > best_score:
                best_score = score
                best = p

        if best and best_score >= 60:
            try:
                # Limit read size to avoid huge contexts
                with open(best, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read(64 * 1024)  # read first 64KB
                logger.info("file_read_injected", query=q, matched_path=best, match_score=best_score)
                return f"[Direct File Read: {os.path.basename(best)} ({os.path.dirname(best)})]\n" + content
            except Exception as e:
                logger.debug("file_read_failed", path=best, error=str(e))
        return ""

    def _extract_key_nouns(self, query: str) -> str:
        """Extract the most important nouns from a query for fallback retrieval."""
        stop_words = {
            "tell",
            "me",
            "about",
            "the",
            "a",
            "an",
            "what",
            "is",
            "are",
            "how",
            "does",
            "do",
            "summarise",
            "summarize",
            "describe",
            "explain",
            "show",
            "give",
            "please",
            "can",
            "you",
            "and",
            "or",
            "in",
            "of",
            "for",
            "to",
            "with",
        }
        words = (query or "").lower().split()
        key_words = [w for w in words if w not in stop_words and len(w) > 2]
        return " ".join(key_words)

    def _active_llm_provider(self) -> str:
        """Best-effort guess of which provider will handle this turn."""
        try:
            if self.brain.gemini_available and self.brain._can_use_gemini():
                return "gemini"
            if self.brain.groq_available and self.brain._can_use_provider("groq", True):
                return "groq"
            if self.brain.ollama_available:
                return "ollama"
        except Exception:
            pass
        return "gemini"

    def _last_llm_provider(self) -> str:
        provider = getattr(self.brain, "last_provider", None)
        return provider or self._active_llm_provider()

    def _expand_project_query(self, query: str) -> str:
        """Expand project knowledge questions with likely code/file terms."""
        query_lower = (query or "").lower()
        knowledge_triggers = [
            "tell me about",
            "what is",
            "summarise",
            "summarize",
            "describe",
            "explain",
            "what does",
            "how does",
            "what are",
            "show me",
        ]
        is_knowledge_query = any(trigger in query_lower for trigger in knowledge_triggers)
        if not is_knowledge_query:
            return query

        expanded = query
        if "userauth" in query_lower or "user auth" in query_lower:
            expanded += (
                " authentication login register "
                "user session JWT token controller "
                "service repository movie ticket theatre theater screen seat booking payment"
            )

        if "project" in query_lower:
            expanded += " main service controller repository model class method"

        return expanded

    async def _get_rag_context(self, query: str, provider: str = "gemini") -> str:
        rag_index = getattr(self.memory, "personal_rag", None)
        if rag_index is None:
            return ""

        try:
            if hasattr(rag_index, "is_ready") and not rag_index.is_ready:
                logger.debug("[RAG] index still warming up, context will be empty this turn")
                return ""
        except Exception:
            pass

        timeout = 1.5 if getattr(rag_index, "_indexing_active", False) else 3.0

        try:
            loop = asyncio.get_running_loop()
            folder_hint, subfolder_hint, filename_hint = self._extract_file_reference(query)
            if any((folder_hint, subfolder_hint, filename_hint)):
                try:
                    direct = await asyncio.wait_for(
                        loop.run_in_executor(
                            None,
                            partial(
                                self._resolve_file_by_description,
                                folder_hint,
                                subfolder_hint,
                                filename_hint,
                                rag_index,
                            ),
                        ),
                        timeout=timeout,
                    )
                    if direct:
                        logger.info("rag_direct_file_context", query=query, method="direct_read")
                        return direct
                except Exception:
                    pass

            search_query = self._expand_project_query(query)

            def _search_matches(q: str):
                return rag_index.search(q, limit=5, use_cache=True)

            results = await asyncio.wait_for(
                loop.run_in_executor(None, partial(_search_matches, search_query)),
                timeout=timeout,
            )

            top_score = float(results[0].get("score", 0.0)) if results else 0.0
            if not results or top_score < 58.0:
                key_nouns = self._extract_key_nouns(query)
                if key_nouns and key_nouns != search_query:
                    fallback_results = await asyncio.wait_for(
                        loop.run_in_executor(None, partial(_search_matches, key_nouns)),
                        timeout=timeout,
                    )
                    if fallback_results and float(fallback_results[0].get("score", 0.0)) >= top_score:
                        results = fallback_results
                        search_query = key_nouns
                        top_score = float(results[0].get("score", 0.0))

            if not results:
                self.event_bus.emit(
                    DexterEvents.RAG_CONTEXT_EMPTY,
                    {"query": query, "reason": "no_matches"},
                )
                return ""

            def _format_context():
                if hasattr(rag_index, "format_context_for_provider"):
                    return rag_index.format_context_for_provider(
                        results, search_query, provider
                    )
                return rag_index.build_context(search_query, limit=3)

            context = await asyncio.wait_for(
                loop.run_in_executor(None, _format_context),
                timeout=timeout,
            )
            if not context:
                return ""

            logger.info(
                "rag_context_injected",
                query=query[:80],
                search_query=search_query[:80],
                provider=provider,
                results_count=len(results),
                top_score=round(top_score, 2),
                sources=[
                    os.path.basename((r.get("path") or "").replace("\\", "/"))
                    for r in results
                ],
            )
            self.event_bus.emit(
                DexterEvents.RAG_CONTEXT_USED,
                {
                    "query": query,
                    "provider": provider,
                    "sources": [
                        {
                            "path": r.get("path"),
                            "chunk_label": (r.get("metadata") or {}).get("chunk_label"),
                            "chunk_type": (r.get("metadata") or {}).get("chunk_type"),
                            "score": r.get("score"),
                            "rerank_score": r.get("rerank_score"),
                        }
                        for r in results
                    ],
                },
            )
            return context

        except asyncio.TimeoutError:
            logger.warning("rag_context_timeout", query=query[:80], provider=provider)
            return ""
        except Exception as e:
            logger.error("rag_context_failed", error=str(e), exc_info=True)
            return ""

    async def _stream_response(self, command: str, memory_context: str, indexed_context: str = "") -> str:
        response_text = ""
        sentence_buffer = ""
        sentences_queue: list[tuple[str, bool]] = []
        speaking_started = False

        async for chunk in self.brain.process_command_stream(command, long_term_memory=memory_context, indexed_context=indexed_context):
            if not chunk:
                continue
            response_text += chunk
            self.event_bus.emit("response_chunk", {"text": chunk})
            sentence_buffer += chunk

            sentences, sentence_buffer = self._split_sentences(sentence_buffer)
            for sentence in sentences:
                if not speaking_started:
                    self._set_state(AssistantState.SPEAKING)
                    speaking_started = True
                    interrupt = True
                else:
                    interrupt = False
                sentences_queue.append((sentence, interrupt))

        if sentence_buffer.strip():
            if not speaking_started:
                self._set_state(AssistantState.SPEAKING)
                speaking_started = True
                interrupt = True
            else:
                interrupt = False
            sentences_queue.append((sentence_buffer.strip(), interrupt))

        # Accumulate sentences into natural-length chunks before TTS to avoid choppy playback
        chunk_buffer = ""
        first_chunk = True
        for sentence, interrupt in sentences_queue:
            try:
                # Decide whether to flush the buffer with this new sentence
                flush = False
                try:
                    if hasattr(self.tts, "should_flush_sentence_buffer"):
                        flush = self.tts.should_flush_sentence_buffer(chunk_buffer, sentence)
                except Exception:
                    # Default: flush after each sentence if helper is not available
                    flush = True

                if flush and chunk_buffer.strip():
                    # Play the accumulated chunk
                    try:
                        words = len(chunk_buffer.split())
                        est_seconds = max(0.5, (words / 2.5) + 0.5)
                    except Exception:
                        est_seconds = 2.0
                    try:
                        if hasattr(self.vad, "suppress_for"):
                            await asyncio.to_thread(self.vad.suppress_for, est_seconds + 1.5)
                    except Exception:
                        pass
                    try:
                        await self.tts.speak(chunk_buffer.strip(), interrupt=first_chunk)
                    except Exception as e:
                        logger.error("tts_speak_failed", error=str(e), exc_info=True)
                        self.event_bus.emit("error_occurred", {"component": "tts", "error": str(e)})
                    chunk_buffer = ""
                    first_chunk = False

                # Append the current sentence to the buffer
                if chunk_buffer:
                    chunk_buffer += " " + sentence
                else:
                    chunk_buffer = sentence

            except Exception as e:
                logger.error("tts_chunking_failed", error=str(e), exc_info=True)

        # Flush any remaining buffer
        if chunk_buffer.strip():
            try:
                words = len(chunk_buffer.split())
                est_seconds = max(0.5, (words / 2.5) + 0.5)
            except Exception:
                est_seconds = 2.0
            try:
                if hasattr(self.vad, "suppress_for"):
                    await asyncio.to_thread(self.vad.suppress_for, est_seconds + 1.5)
            except Exception:
                pass
            try:
                await self.tts.speak(chunk_buffer.strip(), interrupt=first_chunk)
            except Exception as e:
                logger.error("tts_speak_failed", error=str(e), exc_info=True)
                self.event_bus.emit("error_occurred", {"component": "tts", "error": str(e)})

        self.event_bus.emit("response_completed", {"text": response_text.strip()})
        return response_text.strip()

    async def _watchdog(self) -> None:
        while True:
            await asyncio.sleep(30)
            stuck_for = time.time() - self._state_changed_at
            if stuck_for > 60:
                logger.warning(
                    "pipeline_stuck",
                    state=self.state.name,
                    stuck_seconds=int(stuck_for),
                )

    async def run(self) -> None:
        logger.info("pipeline_online")
        if self.health_monitor is not None:
            self.health_monitor.healthy("pipeline", "online")
        self._loop = asyncio.get_running_loop()
        if self.start_active:
            self._always_on_until = time.time() + 60.0
            self._open_wake_window()
            logger.info("activation_window_started", seconds=60)
        watchdog = asyncio.create_task(self._watchdog())
        try:
            while True:
                try:
                    await self._handle_once()
                except Exception as e:
                    logger.error("pipeline_loop_error", error=str(e), exc_info=True)
                    if self.health_monitor is not None:
                        self.health_monitor.degraded("pipeline", f"loop error: {e}")
                    try:
                        # Ensure we return to IDLE to avoid stuck states
                        self._set_state(AssistantState.IDLE)
                    except Exception as reset_error:
                        logger.error("pipeline_state_reset_failed", error=str(reset_error), exc_info=True)
                    # small delay before retrying to avoid busy-looping on persistent errors
                    await asyncio.sleep(1)
        finally:
            watchdog.cancel()

    async def _handle_once(self) -> None:
        cid = bind_correlation_id(uuid4().hex)
        turn_start = time.perf_counter()
        self._set_state(AssistantState.LISTENING)
        logger.debug("pipeline_listening_started", cid=cid)
        
        try:
            vad_start = time.perf_counter()
            
            # Define a safe wrapper for the interrupt callback
            def _interrupt_handler():
                """Called when user starts speaking during TTS playback."""
                logger.debug("interrupt_detected_stopping_tts", cid=cid)
                self.tts.stop()
            
            if self.activation_mode == "clap":
                audio_path = await asyncio.to_thread(
                    self.vad.listen,
                    on_speech_start=_interrupt_handler,
                    on_clap=self._on_clap_detected,
                    clap_sensitivity=self.clap_sensitivity,
                )
            else:
                audio_path = await asyncio.to_thread(
                    self.vad.listen, on_speech_start=_interrupt_handler
                )
            metrics.record_latency("vad_ms", (time.perf_counter() - vad_start) * 1000)

            if not audio_path:
                logger.debug("vad_no_audio_captured", cid=cid)
                self._set_state(AssistantState.IDLE)
                await asyncio.sleep(1)
                return

            self._set_state(AssistantState.TRANSCRIBING)
            stt_start = time.perf_counter()

            def _on_partial(text: str) -> None:
                if not text:
                    return
                payload = {"text": text}
                # Callback is invoked from a worker thread (transcription runs in to_thread).
                # Marshal back to the asyncio event loop for thread safety.
                if self._loop is not None:
                    self._loop.call_soon_threadsafe(self.event_bus.emit, "transcript_partial", payload)
                else:
                    self.event_bus.emit("transcript_partial", payload)

            try:
                identified_text = await asyncio.wait_for(
                    asyncio.to_thread(self.transcriber.transcribe, audio_path, on_partial=_on_partial),
                    timeout=10.0,
                )
                metrics.record_latency("stt_ms", (time.perf_counter() - stt_start) * 1000)
            except asyncio.TimeoutError:
                logger.warning("transcription_timeout", cid=cid)
                self.event_bus.emit(
                    "error_occurred",
                    {"component": "stt", "error": "transcription_timeout"},
                )
                self._set_state(AssistantState.IDLE)
                return
            except Exception as e:
                logger.error("transcription_failed", error=str(e), exc_info=True)
                self.event_bus.emit(
                    "error_occurred",
                    {"component": "stt", "error": str(e)},
                )
                self._set_state(AssistantState.IDLE)
                return

            if not identified_text:
                logger.debug("transcription_empty", cid=cid)
                self._set_state(AssistantState.IDLE)
                return

            privacy_cfg = getattr(self.config, "privacy", None)
            if privacy_cfg is not None and bool(getattr(privacy_cfg, "debug_log_transcripts", False)):
                logger.info(
                    "utterance_started",
                    cid=cid,
                    transcript=identified_text,
                )
            else:
                logger.debug(
                    "utterance_started",
                    cid=cid,
                    transcript_length=len(identified_text),
                )
            self.event_bus.emit("transcript_received", {"text": identified_text, "correlation_id": cid})
            logger.debug("transcript_final", text=identified_text)

            if not self._transcript_is_usable(identified_text):
                logger.info(
                    "transcript_rejected_low_quality",
                    transcript=identified_text,
                )
                self._set_state(AssistantState.IDLE)
                return

            effective_mode = self._effective_activation_mode()
            preprocessed_text = apply_wake_word_corrections(identified_text)
            self._diag(
                "transcript_received",
                transcript=identified_text,
                activation_mode=effective_mode,
            )

            activation_cmd = self._detect_activation_command(preprocessed_text)
            if activation_cmd:
                mode, duration, spoken = activation_cmd
                self._activation.set_override(mode, duration)  # type: ignore[arg-type]
                self.event_bus.emit(
                    DexterEvents.ACTIVATION_MODE_CHANGED,
                    {"mode": mode, "reason": "voice_command", "duration": duration},
                )
                await self.tts.speak(spoken)
                self._set_state(AssistantState.IDLE)
                return

            # Detect explicit correction ("I meant X") and train ASR engine
            if self.asr_engine and self._last_transcript:
                intended = self._detect_correction_intent(preprocessed_text)
                if intended:
                    self.asr_engine.confirm_correction(self._last_transcript, intended)
                    preprocessed_text = intended
                    logger.info("user_corrected_asr", wrong=self._last_transcript, right=intended)

            self._last_transcript = preprocessed_text

            if effective_mode == "wake_word":
                detection = self.wake_detector.detect(preprocessed_text) if self.wake_detector else None
                bypass_activation = False

                if detection and detection.triggered:
                    self.event_bus.emit(
                        DexterEvents.WAKE_WORD_DETECTED,
                        {"transcript": preprocessed_text[:80]},
                    )
                    self._open_wake_window()
                    clean_command = detection.cleaned_text
                    if not clean_command.strip():
                        logger.info(
                            "wake_word_detected",
                            wake_window_seconds=self.command_window_seconds,
                        )
                        self._set_state(AssistantState.IDLE)
                        return
                elif self._is_awake():
                    clean_command = preprocessed_text
                else:
                    if self._looks_actionable_utterance(preprocessed_text):
                        clean_command = preprocessed_text
                        bypass_activation = True
                        self._open_wake_window()
                        logger.info("activation_bypassed", mode="wake_word", reason="actionable_utterance")
                    else:
                        self._activation.record_drop()
                        self._log_activation_failure(preprocessed_text, "wake_word_not_found")
                        self._record_activation_drop(preprocessed_text, "wake_word_not_detected")
                        self.event_bus.emit(
                            DexterEvents.COMMAND_DROPPED,
                            {"reason": "wake_word_required", "transcript": preprocessed_text[:50]},
                        )
                        self._set_state(AssistantState.IDLE)
                        return

                correction = self.corrector.correct(clean_command)
                clean_command = correction.corrected
            else:
                bypass_activation = False
                if not self._is_awake() and not self.brain.pending_action:
                    if effective_mode == "always_on" or self._looks_actionable_utterance(preprocessed_text):
                        bypass_activation = True
                        self._open_wake_window()
                        logger.info("activation_bypassed", mode=effective_mode, reason="actionable_utterance")
                    else:
                        self._log_activation_failure(preprocessed_text, "activation_not_awake")
                        self._set_state(AssistantState.IDLE)
                        return

                correction = self.corrector.correct(preprocessed_text)
                clean_command = correction.corrected
                if (
                    not bypass_activation
                    and not self.brain.pending_action
                    and len(clean_command.split()) < self.min_command_words
                ):
                    self._set_state(AssistantState.IDLE)
                    return

            self._reset_activation_drop_counter()
            self._activation.record_interaction()
            prev_mode = self._activation.current_mode
            self.event_bus.emit(
                DexterEvents.ACTIVATION_MODE_CHANGED,
                {"mode": prev_mode, "reason": "interaction"},
            )
            logger.info("command_accepted", command=clean_command)
            self._diag(
                "command_accepted",
                command=clean_command,
                activation_mode=effective_mode,
                bypass_activation=bypass_activation,
            )
            # Advance turn counter for this accepted command
            self._turn_count += 1

            # Clear stale project slot if needed
            try:
                session_state.clear_if_stale(self._turn_count)
            except Exception:
                pass
            if self.activation_mode == "clap":
                self._open_wake_window()
                logger.info("activation_window_extended", seconds=self.command_window_seconds)
            self._set_state(AssistantState.PROCESSING)
            memory_context = await asyncio.to_thread(self.memory.recall_context, clean_command, 3, False)

            # If a current_project session slot exists, prepend its name to the RAG query
            try:
                proj = session_state.get_current_project()
            except Exception:
                proj = None

            if proj:
                # Tools may set a sentinel (None) for "just set"; record the real turn now.
                set_at_turn = proj.get("set_at_turn")
                if set_at_turn is None or int(set_at_turn or 0) == 0:
                    try:
                        session_state.set_current_project(proj.get("name"), proj.get("resolved_path"), proj.get("confidence", 0.0), self._turn_count)
                    except Exception:
                        pass
                rag_query = f"{proj.get('name')} {clean_command}"

                # If this is the earliest project context and the RAG index is still warming,
                # optionally wait briefly so we don't miss the first project turn.
                rag_index = getattr(self.memory, "personal_rag", None)
                warm_evt = getattr(rag_index, "warm_up_complete", None) if rag_index is not None else None
                try:
                    if (
                        rag_index is not None
                        and warm_evt is not None
                        and hasattr(rag_index, "is_ready")
                        and not rag_index.is_ready
                        and self._turn_count == 1
                    ):
                        await asyncio.wait_for(warm_evt.wait(), timeout=1.5)
                except asyncio.TimeoutError:
                    pass

                rag_context = (
                    await self._get_rag_context(rag_query, provider=self._active_llm_provider())
                    if self._should_use_rag(clean_command)
                    else ""
                )
                if rag_context:
                    rag_context = f"[Context: user is currently asking about {proj.get('name')}]\n" + rag_context
            else:
                rag_context = (
                    await self._get_rag_context(clean_command, provider=self._active_llm_provider())
                    if self._should_use_rag(clean_command)
                    else ""
                )
            logger.debug(
                "rag_context_result",
                has_context=bool(rag_context),
                context_length=len(rag_context) if rag_context else 0,
                preview=rag_context[:100] if rag_context else "EMPTY",
            )
            rag_source_count = 0
            if rag_context:
                rag_source_count = rag_context.count("\n[")
            self._diag(
                "rag_context",
                sources=rag_source_count,
                context_chars=len(rag_context) if rag_context else 0,
            )
            augmented_command = clean_command
            if rag_context:
                augmented_command = f"{rag_context}\nUser question: {clean_command}"
                augmented_command += "\nAnswer questions about files in maximum 4 sentences. User is listening not reading."

            configured_timeout = float(getattr(self.config.providers, "overall_turn_timeout_seconds", 30.0))
            default_timeout = 45.0 if (rag_context and len(rag_context) > 100) else 20.0
            turn_timeout_seconds = min(configured_timeout, default_timeout)
            try:
                response_text = await asyncio.wait_for(
                    self._stream_response(augmented_command, memory_context, rag_context),
                    timeout=turn_timeout_seconds,
                )
            except asyncio.TimeoutError:
                logger.error("pipeline_llm_timeout", turn_id=cid)
                self.event_bus.emit(
                    "error_occurred",
                    {"component": "llm", "error": "pipeline_llm_timeout"},
                )
                await self.tts.speak("I didn't get a response in time. Please try again.")
                self._set_state(AssistantState.IDLE)
                return

            if self.activation_mode == "clap" and self.brain.pending_action:
                self._open_wake_window()
                logger.info("activation_window_extended", seconds=self.command_window_seconds)

            self.event_bus.emit("response_generated", {"text": response_text, "correlation_id": cid})
            logger.info("response_complete", response_preview=response_text[:500])
            self._diag(
                "turn_complete",
                command=clean_command,
                provider_hint=self._last_llm_provider(),
                response_preview=response_text[:200],
                duration_ms=int((time.perf_counter() - turn_start) * 1000),
            )

            try:
                await asyncio.to_thread(
                    self.memory.remember,
                    f"User: {clean_command} | Dexter: {response_text}",
                )
            except Exception as e:
                logger.error("memory_save_failed", error=str(e), exc_info=True)

            rag_proxy = getattr(self.memory, "personal_rag", None)
            if rag_proxy is not None and hasattr(rag_proxy, "on_voice_activity"):
                try:
                    rag_proxy.on_voice_activity()
                except Exception:
                    pass

            self._set_state(AssistantState.IDLE)
        finally:
            try:
                duration_ms = (time.perf_counter() - turn_start) * 1000
                metrics.record_latency("turn_ms", duration_ms)
                self.event_bus.emit("turn_completed", {"duration_ms": duration_ms})
            except Exception:
                pass
            clear_correlation_id()
