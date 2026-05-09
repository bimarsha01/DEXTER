import asyncio
import os
import re
import time
from uuid import uuid4
from typing import Optional

from core.event_bus import EventBus
from core.health import HealthMonitor
from core.state_machine import AssistantState
from core.wake_word.detector import WakeWordDetector
from core.session_activity import session_activity
from utils.transcript_correction import TranscriptCorrector, apply_wake_word_aliases
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
    ) -> None:
        self.config = config
        self.transcriber = transcriber
        self.vad = vad_listener
        self.tts = tts_manager
        self.memory = memory_vault
        self.brain = brain
        self.event_bus = event_bus or EventBus()
        self.health_monitor = health_monitor

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

    def _effective_activation_mode(self) -> str:
        if self._always_on_until > time.time():
            return "always_on"
        return self.activation_mode

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
            if state == AssistantState.PROCESSING:
                session_activity.mark_active()
            elif state == AssistantState.IDLE:
                session_activity.mark_idle()
            if self.health_monitor is not None:
                self.health_monitor.healthy("pipeline", f"state={state.name}")

    def _is_awake(self) -> bool:
        return time.time() < self.awake_until

    def _open_wake_window(self) -> None:
        self.awake_until = time.time() + self.command_window_seconds

    def _handle_clap_activation(self) -> None:
        self._open_wake_window()
        try:
            asyncio.create_task(self.tts.play_chime())
        except RuntimeError:
            pass

    def _on_clap_detected(self) -> None:
        if self._loop:
            self._loop.call_soon_threadsafe(self._handle_clap_activation)

    def _split_sentences(self, text: str) -> tuple[list[str], str]:
        parts = re.split(r"(?<=[.!?])\s+", text)
        if len(parts) <= 1:
            return [], text
        if parts and not parts[-1].strip():
            return [p for p in parts[:-1] if p.strip()], ""
        return [p for p in parts[:-1] if p.strip()], parts[-1]

    @staticmethod
    def _looks_actionable_utterance(text: str) -> bool:
        normalized = (text or "").strip().lower()
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

    async def _get_rag_context(self, query: str) -> str:
        rag_index = getattr(self.memory, "personal_rag", None)
        if rag_index is None:
            return ""

        try:
            if hasattr(rag_index, "is_ready") and not rag_index.is_ready:
                logger.debug("[RAG] index still warming up, context will be empty this turn")
                return ""
        except Exception:
            # If readiness probing fails, fall back to best-effort context building.
            pass

        try:
            loop = asyncio.get_running_loop()
            context = await asyncio.wait_for(
                loop.run_in_executor(None, rag_index.build_context, query),
                timeout=2.0,
            )
            return context or ""

        except asyncio.TimeoutError:
            logger.warning("rag_context_timeout", query=query)
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

        # Play sentences sequentially to avoid overlapping audio
        for sentence, interrupt in sentences_queue:
            try:
                # Estimate playback duration (approx words / 2.5 words-per-second) and suppress VAD
                try:
                    words = len(sentence.split())
                    est_seconds = max(0.5, (words / 2.5) + 0.5)
                except Exception:
                    est_seconds = 2.0

                # If the VAD supports suppression, request it for the estimated duration
                try:
                    if hasattr(self.vad, "suppress_for"):
                        await asyncio.to_thread(self.vad.suppress_for, est_seconds + 1.5)
                except Exception:
                    pass

                await self.tts.speak(sentence, interrupt=interrupt)
            except Exception as e:
                logger.error("tts_speak_failed", error=str(e), exc_info=True)
                self.event_bus.emit(
                    "error_occurred",
                    {"component": "tts", "error": str(e)},
                )

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
            self._open_wake_window()
            logger.info("activation_window_started", seconds=self.command_window_seconds)
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
                self._set_state(AssistantState.IDLE)
                return
            except Exception as e:
                logger.error("transcription_failed", error=str(e), exc_info=True)
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

            effective_mode = self._effective_activation_mode()
            preprocessed_text = apply_wake_word_aliases(identified_text)

            if effective_mode == "wake_word":
                detection = self.wake_detector.detect(preprocessed_text) if self.wake_detector else None
                bypass_activation = False

                if detection and detection.triggered:
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
                        self._record_activation_drop(preprocessed_text, "wake_word_not_detected")
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
            logger.info("command_accepted", command=clean_command)
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

            # Skip RAG retrieval for non-document intents to avoid polluting prompts
            # when the minimum relevance threshold is permissive.
            skip_rag = False
            try:
                decision = self.brain.intent_router.detect_intent(clean_command)
                non_document_tools = {
                    "get_weather",
                    "get_current_time",
                    "get_current_datetime",
                    "get_system_status",
                    "read_clipboard",
                    "take_screenshot",
                    "get_health_report",
                }
                if decision.action == "vision":
                    skip_rag = True
                elif decision.action == "tool" and decision.tool_name in non_document_tools:
                    skip_rag = True
                elif decision.action == "ask" and decision.tool_name in non_document_tools:
                    skip_rag = True
            except Exception:
                pass

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

                rag_context = "" if skip_rag else await self._get_rag_context(rag_query)
                if rag_context:
                    rag_context = f"[Context: user is currently asking about {proj.get('name')}]\n" + rag_context
            else:
                rag_context = "" if skip_rag else await self._get_rag_context(clean_command)
            augmented_command = clean_command
            if rag_context:
                augmented_command = f"{rag_context}\nUser question: {clean_command}"

            turn_timeout_seconds = float(getattr(self.config.providers, "overall_turn_timeout_seconds", 30.0))
            try:
                response_text = await asyncio.wait_for(
                    self._stream_response(augmented_command, memory_context, rag_context),
                    timeout=turn_timeout_seconds,
                )
            except asyncio.TimeoutError:
                logger.error("pipeline_llm_timeout", turn_id=cid)
                await self.tts.speak("I didn't get a response in time. Please try again.")
                self._set_state(AssistantState.IDLE)
                return

            if self.activation_mode == "clap" and self.brain.pending_action:
                self._open_wake_window()
                logger.info("activation_window_extended", seconds=self.command_window_seconds)

            self.event_bus.emit("response_generated", {"text": response_text, "correlation_id": cid})
            logger.info("response_complete", response_preview=response_text[:500])

            try:
                await asyncio.to_thread(
                    self.memory.remember,
                    f"User: {clean_command} | Dexter: {response_text}",
                )
            except Exception as e:
                logger.error("memory_save_failed", error=str(e), exc_info=True)

            self._set_state(AssistantState.IDLE)
        finally:
            clear_correlation_id()
