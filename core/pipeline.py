import asyncio
import os
import re
import time
import threading
from dataclasses import dataclass, field
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
from core.brain.session_state import ContextStore, SessionContext, UserPreferences
from core.feedback import FeedbackStore, RetrievalFeedback
from tools.input_tools import AutomationFocusError

logger = get_logger("pipeline")


@dataclass(eq=True)
class TurnContext:
    cid: str
    turn_start: float
    audio_path: str | None = None
    identified_text: str = ""
    preprocessed_text: str = ""
    effective_mode: str = ""
    bypass_activation: bool = False
    clean_command: str = ""
    memory_context: str = ""
    rag_context: str = ""
    augmented_command: str = ""
    response_text: str = ""
    provider_hint: str = ""
    turn_timeout_seconds: float = 0.0
    stop_turn: bool = False
    stop_reason: str = ""

    def __repr__(self) -> str:
        return (
            f"TurnContext(cid={self.cid!r}, turn_start={self.turn_start!r}, clean_command={self.clean_command!r}, "
            f"response_text={self.response_text!r}, provider_hint={self.provider_hint!r})"
        )


@dataclass(eq=True)
class PreferenceDetection:
    updates: dict[str, str] = field(default_factory=dict)
    confidence: float = 0.0
    matched_phrases: list[str] = field(default_factory=list)
    ambiguous: bool = False

    def __repr__(self) -> str:
        return (
            f"PreferenceDetection(updates={self.updates!r}, confidence={self.confidence!r}, "
            f"matched_phrases={self.matched_phrases!r}, ambiguous={self.ambiguous!r})"
        )


@dataclass
class ToolError:
    message: str


class TurnStageError(RuntimeError):
    def __init__(self, stage: str, message: str, *, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.stage = stage
        self.cause = cause


class TurnController:
    STAGE_TIMEOUTS = {
        "transcribe": 10.0,
        "activate": 5.0,
        "retrieve_context": 5.0,
        "execute_tools": 30.0,
        "generate_response": 45.0,
        "speak": 45.0,
    }

    def __init__(self, pipeline: "AsyncPipeline", watchdog_stop_event: threading.Event | None = None) -> None:
        self.pipeline = pipeline
        self._current_turn_context: TurnContext | None = None
        self._watchdog_stop_event = watchdog_stop_event

    def _effective_stage_timeout(self, stage: str) -> float:
        configured = float(getattr(self.pipeline.config.providers, "overall_turn_timeout_seconds", 0.0) or 0.0)
        base = float(self.STAGE_TIMEOUTS.get(stage, 30.0))
        if configured > 0:
            return min(base, configured)
        return base

    def _emit_stage_event(self, stage: str, status: str, ctx: TurnContext, **fields) -> None:
        payload = {
            "stage": stage,
            "status": status,
            "turn_id": ctx.cid,
            "ts": time.time(),
            **fields,
        }

    def _record_stage_timing(self, stage: str, duration_ms: float) -> None:
        if self.pipeline.health_monitor is None:
            return
        try:
            self.pipeline.health_monitor.record_turn_stage(stage, duration_ms)
        except Exception:
            pass

    def _stage_entry(self, stage: str, ctx: TurnContext) -> None:
        logger.info(
            "turn_stage_enter",
            stage=stage,
            turn_id=ctx.cid,
            state=self.pipeline.state.name,
        )
        self._emit_stage_event(stage, "start", ctx)

    def _stage_exit(self, stage: str, ctx: TurnContext, duration_ms: float) -> None:
        logger.info(
            "turn_stage_exit",
            stage=stage,
            turn_id=ctx.cid,
            state=self.pipeline.state.name,
            stop_turn=ctx.stop_turn,
            duration_ms=round(duration_ms, 2),
        )
        self._emit_stage_event(stage, "done", ctx, duration_ms=round(duration_ms, 2))

    async def _run_stage(self, stage: str, ctx: TurnContext, handler, timeout: float) -> TurnContext:
        self._stage_entry(stage, ctx)
        stage_start = time.perf_counter()
        try:
            result = await asyncio.wait_for(handler(ctx), timeout=timeout)
        except asyncio.TimeoutError as e:
            duration_ms = (time.perf_counter() - stage_start) * 1000
            self._record_stage_timing(stage, duration_ms)
            logger.warning(
                "turn_stage_time_budget_exceeded",
                stage=stage,
                turn_id=ctx.cid,
                duration_ms=round(duration_ms, 2),
                budget_ms=round(timeout * 1000, 2),
            )
            self._emit_stage_event(
                stage,
                "error",
                ctx,
                error=f"{stage} stage timed out after {timeout:.1f}s",
                duration_ms=round(duration_ms, 2),
            )
            try:
                self.pipeline.event_bus.emit("turn_stage_error", {"stage": stage, "error": f"{stage} stage timed out after {timeout:.1f}s", "turn_id": ctx.cid, "duration_ms": round(duration_ms, 2)})
            except Exception:
                pass
            raise TurnStageError(stage, f"{stage} stage timed out after {timeout:.1f}s", cause=e) from e
        except TurnStageError as e:
            duration_ms = (time.perf_counter() - stage_start) * 1000
            self._record_stage_timing(stage, duration_ms)
            if duration_ms > timeout * 1000:
                logger.warning(
                    "turn_stage_time_budget_exceeded",
                    stage=stage,
                    turn_id=ctx.cid,
                    duration_ms=round(duration_ms, 2),
                    budget_ms=round(timeout * 1000, 2),
                )
            self._emit_stage_event(
                stage,
                "error",
                ctx,
                error=str(e),
                duration_ms=round(duration_ms, 2),
            )
            try:
                self.pipeline.event_bus.emit("turn_stage_error", {"stage": stage, "error": str(e), "turn_id": ctx.cid, "duration_ms": round(duration_ms, 2)})
            except Exception:
                pass
            raise
        except Exception as e:
            duration_ms = (time.perf_counter() - stage_start) * 1000
            self._record_stage_timing(stage, duration_ms)
            if duration_ms > timeout * 1000:
                logger.warning(
                    "turn_stage_time_budget_exceeded",
                    stage=stage,
                    turn_id=ctx.cid,
                    duration_ms=round(duration_ms, 2),
                    budget_ms=round(timeout * 1000, 2),
                )
            self._emit_stage_event(
                stage,
                "error",
                ctx,
                error=f"{stage} stage failed",
                duration_ms=round(duration_ms, 2),
            )
            try:
                self.pipeline.event_bus.emit("turn_stage_error", {"stage": stage, "error": f"{stage} stage failed", "turn_id": ctx.cid, "duration_ms": round(duration_ms, 2)})
            except Exception:
                pass
            raise TurnStageError(stage, f"{stage} stage failed", cause=e) from e
        duration_ms = (time.perf_counter() - stage_start) * 1000
        self._record_stage_timing(stage, duration_ms)
        if duration_ms > timeout * 1000:
            logger.warning(
                "turn_stage_time_budget_exceeded",
                stage=stage,
                turn_id=ctx.cid,
                duration_ms=round(duration_ms, 2),
                budget_ms=round(timeout * 1000, 2),
            )
        self._stage_exit(stage, result, duration_ms)
        return result

    async def run_turn(self) -> None:
        cid = bind_correlation_id(uuid4().hex)
        turn_start = time.perf_counter()
        ctx = TurnContext(cid=cid, turn_start=turn_start)

        def _watchdog_active() -> bool:
            return bool(self._watchdog_stop_event is not None and self._watchdog_stop_event.is_set())

        def _abort_for_watchdog() -> bool:
            if not _watchdog_active():
                return False
            logger.critical("Watchdog stop active — DEXTER paused for hardware safety")
            self.pipeline._set_state(AssistantState.IDLE)
            ctx.stop_turn = True
            ctx.stop_reason = "hardware_safety"
            return True

        # Leak-guard: ensure previous turn context cleaned up
        try:
            if getattr(self, "_current_turn_context", None) is not None:
                logger.warning("leaked_turn_context_cleared", previous_turn_id=getattr(self._current_turn_context, "cid", None))
                self._current_turn_context = None
        except Exception:
            self._current_turn_context = None

        try:
            initial_user_scope = getattr(self.pipeline.session_context, "_user_scope", None)
        except Exception:
            initial_user_scope = None
        loaded_context = self.pipeline.context_store.load(user_scope=initial_user_scope)
        self.pipeline._sync_session_context(loaded_context)
        logger.info(
            "session_context_loaded",
            source="turn_start",
            turn_id=ctx.cid,
            has_project=bool(self.pipeline.session_context.project),
            turn_summaries=len(self.pipeline.session_context.recent_turn_summaries),
        )

        try:
            # mark active turn context for leak detection
            self._current_turn_context = ctx
            if _abort_for_watchdog():
                return
            self.pipeline._set_state(AssistantState.LISTENING)
            try:
                logger.debug("response_interrupted_flag_at_turn_start", value=self.pipeline._response_interrupted)
            except Exception:
                pass
            ctx = await self._run_stage("transcribe", ctx, self._stage_transcribe, self._effective_stage_timeout("transcribe"))
            if ctx.stop_turn:
                return
            if _abort_for_watchdog():
                return

            ctx = await self._run_stage("activate", ctx, self._stage_activate, self._effective_stage_timeout("activate"))
            if ctx.stop_turn:
                return
            if _abort_for_watchdog():
                return

            ctx = await self._run_stage("retrieve_context", ctx, self._stage_retrieve_context, self._effective_stage_timeout("retrieve_context"))
            if ctx.stop_turn:
                return
            if _abort_for_watchdog():
                return

            ctx = await self._run_stage("execute_tools", ctx, self._stage_execute_tools, self._effective_stage_timeout("execute_tools"))
            if ctx.stop_turn:
                return
            if _abort_for_watchdog():
                return

            ctx = await self._run_stage("generate_response", ctx, self._stage_generate_response, self._effective_stage_timeout("generate_response"))
            if ctx.stop_turn:
                return
            if _abort_for_watchdog():
                return

            ctx = await self._run_stage("speak", ctx, self._stage_speak, self._effective_stage_timeout("speak"))

            self.pipeline.event_bus.emit("response_generated", {"text": ctx.response_text, "correlation_id": ctx.cid})
            logger.info("response_complete", response_preview=ctx.response_text[:500])
            self.pipeline._diag(
                "turn_complete",
                command=ctx.clean_command,
                provider_hint=ctx.provider_hint or self.pipeline._last_llm_provider(),
                response_preview=ctx.response_text[:200],
                duration_ms=int((time.perf_counter() - ctx.turn_start) * 1000),
            )

            try:
                await asyncio.to_thread(
                    self.pipeline.memory.remember,
                    f"User: {ctx.clean_command} | Dexter: {ctx.response_text}",
                )
            except Exception as e:
                logger.error("memory_save_failed", error=str(e), exc_info=True)

            rag_proxy = getattr(self.pipeline.memory, "personal_rag", None)
            if rag_proxy is not None and hasattr(rag_proxy, "on_voice_activity"):
                try:
                    rag_proxy.on_voice_activity()
                except Exception:
                    pass

            summary = f"{ctx.clean_command} -> {ctx.response_text[:240]}".strip()
            if summary:
                try:
                    with getattr(self.pipeline.session_context, "_write_lock", threading.RLock()):
                        self.pipeline.session_context.recent_turn_summaries.append(summary)
                        self.pipeline.session_context.recent_turn_summaries = self.pipeline.session_context.recent_turn_summaries[-20:]
                except Exception:
                    # best-effort
                    self.pipeline.session_context.recent_turn_summaries.append(summary)
                    self.pipeline.session_context.recent_turn_summaries = self.pipeline.session_context.recent_turn_summaries[-20:]
            if self.pipeline.session_context.project is not None:
                try:
                    with getattr(self.pipeline.session_context, "_write_lock", threading.RLock()):
                        self.pipeline.session_context.project.last_confirmed_ts = time.time()
                except Exception:
                    self.pipeline.session_context.project.last_confirmed_ts = time.time()
            try:
                if getattr(self.pipeline.session_context, "_just_saved", False):
                    try:
                        delattr(self.pipeline.session_context, "_just_saved")
                    except Exception:
                        pass
                else:
                    with getattr(self.pipeline.session_context, "_write_lock", threading.RLock()):
                        self.pipeline.context_store.save(self.pipeline.session_context)
            except Exception:
                # preserve behavior: still log and continue
                try:
                    self.pipeline.context_store.save(self.pipeline.session_context)
                except Exception:
                    pass
            logger.info(
                "session_context_saved",
                turn_id=ctx.cid,
                summaries=len(self.pipeline.session_context.recent_turn_summaries),
                has_project=bool(self.pipeline.session_context.project),
            )

            self.pipeline._set_state(AssistantState.IDLE)
        except TurnStageError as e:
            logger.error(
                "turn_stage_failed",
                stage=e.stage,
                error=str(e),
                exc_info=True,
            )
            self.pipeline.event_bus.emit(
                "error_occurred",
                {"component": e.stage, "error": str(e)},
            )
            # Speak a short error for non-response stage failures so the turn does not end silently.
            if e.stage == "generate_response":
                try:
                    await self.pipeline.tts.speak("I didn't get a response in time. Please try again.")
                except Exception:
                    pass
            else:
                try:
                    await self.pipeline.tts.speak("I hit a problem handling that request. Please try again.")
                except Exception:
                    pass
            self.pipeline._set_state(AssistantState.IDLE)
            await asyncio.sleep(1)
        finally:
            try:
                duration_ms = (time.perf_counter() - turn_start) * 1000
                metrics.record_latency("turn_ms", duration_ms)
                self.pipeline.event_bus.emit("turn_completed", {"duration_ms": duration_ms})
                if duration_ms > 60000:
                    logger.error(
                        "turn_total_time_budget_exceeded",
                        turn_id=ctx.cid,
                        duration_ms=round(duration_ms, 2),
                        budget_ms=60000,
                    )
            except Exception:
                pass
            try:
                latest_retrieval_event = self._drain_retrieval_events()
                self._latest_retrieval_event = latest_retrieval_event
            except Exception:
                self._latest_retrieval_event = None
            clear_correlation_id()
            # Clear active turn context to avoid leaks
            try:
                self._current_turn_context = None
            except Exception:
                pass

    async def _stage_transcribe(self, ctx: TurnContext) -> TurnContext:
        pipeline = self.pipeline
        vad_start = time.perf_counter()

        def _interrupt_handler():
            logger.debug("interrupt_detected_stopping_tts", cid=ctx.cid)
            if pipeline.state == AssistantState.SPEAKING:
                pipeline._mark_response_interrupted()
            pipeline.tts.stop()

        try:
            if pipeline.activation_mode == "clap":
                audio_path = await asyncio.to_thread(
                    pipeline.vad.listen,
                    on_speech_start=_interrupt_handler,
                    on_clap=pipeline._on_clap_detected,
                    clap_sensitivity=pipeline.clap_sensitivity,
                )
            else:
                audio_path = await asyncio.to_thread(
                    pipeline.vad.listen,
                    on_speech_start=_interrupt_handler,
                )
            ctx.audio_path = audio_path
            metrics.record_latency("vad_ms", (time.perf_counter() - vad_start) * 1000)

            if not audio_path:
                logger.debug("vad_no_audio_captured", cid=ctx.cid)
                pipeline._set_state(AssistantState.IDLE)
                ctx.stop_turn = True
                ctx.stop_reason = "no_audio"
                return ctx

            pipeline._set_state(AssistantState.TRANSCRIBING)
            stt_start = time.perf_counter()

            def _on_partial(text: str) -> None:
                if not text:
                    return
                payload = {"text": text}
                if pipeline._loop is not None:
                    pipeline._loop.call_soon_threadsafe(pipeline.event_bus.emit, "transcript_partial", payload)
                else:
                    pipeline.event_bus.emit("transcript_partial", payload)

            try:
                identified_text = await asyncio.wait_for(
                    asyncio.to_thread(pipeline.transcriber.transcribe, audio_path, on_partial=_on_partial),
                    timeout=self.STAGE_TIMEOUTS["transcribe"],
                )
                metrics.record_latency("stt_ms", (time.perf_counter() - stt_start) * 1000)
            except asyncio.TimeoutError as e:
                logger.warning("transcription_timeout", cid=ctx.cid)
                pipeline.event_bus.emit("error_occurred", {"component": "stt", "error": "transcription_timeout"})
                raise TurnStageError("transcribe", "transcription timed out", cause=e) from e
            except Exception as e:
                logger.error("transcription_failed", error=str(e), exc_info=True)
                pipeline.event_bus.emit("error_occurred", {"component": "stt", "error": str(e)})
                raise TurnStageError("transcribe", "transcription failed", cause=e) from e

            if not identified_text:
                logger.debug("transcription_empty", cid=ctx.cid)
                pipeline._set_state(AssistantState.IDLE)
                ctx.stop_turn = True
                ctx.stop_reason = "empty_transcript"
                return ctx

            privacy_cfg = getattr(pipeline.config, "privacy", None)
            if privacy_cfg is not None and bool(getattr(privacy_cfg, "debug_log_transcripts", False)):
                logger.info("utterance_started", cid=ctx.cid, transcript=identified_text)
            else:
                logger.debug("utterance_started", cid=ctx.cid, transcript_length=len(identified_text))
            pipeline.event_bus.emit("transcript_received", {"text": identified_text, "correlation_id": ctx.cid})
            logger.debug("transcript_final", text=identified_text)

            if not pipeline._transcript_is_usable(identified_text):
                logger.info("transcript_rejected_low_quality", transcript=identified_text)
                pipeline._set_state(AssistantState.IDLE)
                ctx.stop_turn = True
                ctx.stop_reason = "low_quality_transcript"
                return ctx

            ctx.identified_text = identified_text
            ctx.preprocessed_text = apply_wake_word_corrections(identified_text)
            if pipeline._consume_response_interrupted():
                # Barge-in implies the user wanted less verbosity — treat as a brief preference signal.
                try:
                    updated = pipeline._apply_preference_updates(
                        {"verbosity": "brief"},
                        source="barge_in",
                        reason="user interrupted Dexter mid-response",
                        turn_id=ctx.cid,
                        trigger_text=identified_text,
                    )
                    if updated:
                        try:
                            with getattr(pipeline.session_context, "_write_lock", threading.RLock()):
                                pipeline.context_store.save(pipeline.session_context)
                                try:
                                    setattr(pipeline.session_context, "_just_saved", True)
                                except Exception:
                                    pass
                        except Exception:
                            try:
                                pipeline.context_store.save(pipeline.session_context)
                            except Exception:
                                pass
                except Exception:
                    pass
            pipeline._diag("transcript_received", transcript=identified_text, activation_mode=pipeline._effective_activation_mode())
            return ctx
        except TurnStageError:
            raise
        except Exception as e:
            raise TurnStageError("transcribe", "unexpected transcription stage failure", cause=e) from e

    async def _stage_activate(self, ctx: TurnContext) -> TurnContext:
        pipeline = self.pipeline
        if ctx.stop_turn:
            return ctx

        effective_mode = pipeline._effective_activation_mode()
        ctx.effective_mode = effective_mode

        activation_cmd = pipeline._detect_activation_command(ctx.preprocessed_text)
        if activation_cmd:
            mode, duration, spoken = activation_cmd
            pipeline._activation.set_override(mode, duration)  # type: ignore[arg-type]
            pipeline.event_bus.emit(DexterEvents.ACTIVATION_MODE_CHANGED, {"mode": mode, "reason": "voice_command", "duration": duration})
            await pipeline.tts.speak(spoken)
            pipeline._set_state(AssistantState.IDLE)
            ctx.stop_turn = True
            ctx.stop_reason = "activation_command"
            return ctx

        pipeline._drain_retrieval_events()
        if pipeline._looks_like_retrieval_correction(ctx.preprocessed_text):
            pipeline._record_retrieval_feedback(ctx.cid, ctx.preprocessed_text)
            pipeline._set_state(AssistantState.IDLE)
            ctx.stop_turn = True
            ctx.stop_reason = "retrieval_feedback"
            return ctx

        if pipeline.asr_engine and pipeline._last_transcript:
            intended = pipeline._detect_correction_intent(ctx.preprocessed_text)
            if intended:
                pipeline.asr_engine.confirm_correction(pipeline._last_transcript, intended)
                ctx.preprocessed_text = intended
                logger.info("user_corrected_asr", wrong=pipeline._last_transcript, right=intended)

        pipeline._last_transcript = ctx.preprocessed_text

        if effective_mode == "wake_word":
            detection = pipeline.wake_detector.detect(ctx.preprocessed_text) if pipeline.wake_detector else None
            bypass_activation = False

            if detection and detection.triggered:
                pipeline.event_bus.emit(DexterEvents.WAKE_WORD_DETECTED, {"transcript": ctx.preprocessed_text[:80]})
                pipeline._open_wake_window()
                clean_command = detection.cleaned_text
                if not clean_command.strip():
                    logger.info("wake_word_detected", wake_window_seconds=pipeline.command_window_seconds)
                    pipeline._set_state(AssistantState.IDLE)
                    ctx.stop_turn = True
                    ctx.stop_reason = "wake_word_only"
                    return ctx
            elif pipeline._is_awake():
                clean_command = ctx.preprocessed_text
            else:
                if pipeline._looks_actionable_utterance(ctx.preprocessed_text):
                    clean_command = ctx.preprocessed_text
                    bypass_activation = True
                    pipeline._open_wake_window()
                    logger.info("activation_bypassed", mode="wake_word", reason="actionable_utterance")
                else:
                    pipeline._activation.record_drop()
                    pipeline._log_activation_failure(ctx.preprocessed_text, "wake_word_not_found")
                    pipeline._record_activation_drop(ctx.preprocessed_text, "wake_word_not_detected")
                    pipeline.event_bus.emit(DexterEvents.COMMAND_DROPPED, {"reason": "wake_word_required", "transcript": ctx.preprocessed_text[:50]})
                    pipeline._set_state(AssistantState.IDLE)
                    ctx.stop_turn = True
                    ctx.stop_reason = "activation_required"
                    return ctx

            correction = pipeline.corrector.correct(clean_command)
            ctx.clean_command = correction.corrected
            ctx.bypass_activation = bypass_activation
        else:
            bypass_activation = False
            if not pipeline._is_awake() and not pipeline.brain.pending_action:
                if effective_mode == "always_on" or pipeline._looks_actionable_utterance(ctx.preprocessed_text):
                    bypass_activation = True
                    pipeline._open_wake_window()
                    logger.info("activation_bypassed", mode=effective_mode, reason="actionable_utterance")
                else:
                    pipeline._log_activation_failure(ctx.preprocessed_text, "activation_not_awake")
                    pipeline._set_state(AssistantState.IDLE)
                    ctx.stop_turn = True
                    ctx.stop_reason = "activation_not_awake"
                    return ctx

            correction = pipeline.corrector.correct(ctx.preprocessed_text)
            ctx.clean_command = correction.corrected
            ctx.bypass_activation = bypass_activation
            pipeline._apply_preference_signals(
                ctx.clean_command,
                source="explicit_command",
                reason="explicit preference request",
                turn_id=ctx.cid,
            )
            if not bypass_activation and not pipeline.brain.pending_action and len(ctx.clean_command.split()) < pipeline.min_command_words:
                pipeline._set_state(AssistantState.IDLE)
                ctx.stop_turn = True
                ctx.stop_reason = "too_short"
                return ctx

        pipeline._record_explicit_correction(ctx.clean_command, ctx.cid)
        pipeline._reset_activation_drop_counter()
        pipeline._activation.record_interaction()
        prev_mode = pipeline._activation.current_mode
        pipeline.event_bus.emit(DexterEvents.ACTIVATION_MODE_CHANGED, {"mode": prev_mode, "reason": "interaction"})
        logger.info("command_accepted", command=ctx.clean_command)
        pipeline._diag("command_accepted", command=ctx.clean_command, activation_mode=effective_mode, bypass_activation=ctx.bypass_activation)
        pipeline._turn_count += 1

        try:
            session_state.clear_if_stale(pipeline._turn_count)
        except Exception:
            pass
        # Reload the same user scope we started the turn with (if available)
        try:
            user_scope = getattr(pipeline.session_context, "_user_scope", None)
        except Exception:
            user_scope = None
        pipeline._sync_session_context(pipeline.context_store.load(user_scope=user_scope))

        if pipeline.activation_mode == "clap":
            pipeline._open_wake_window()
            logger.info("activation_window_extended", seconds=pipeline.command_window_seconds)

        pipeline._set_state(AssistantState.PROCESSING)
        return ctx

    async def _stage_retrieve_context(self, ctx: TurnContext) -> TurnContext:
        pipeline = self.pipeline
        if ctx.stop_turn:
            return ctx

        ctx.memory_context = await asyncio.to_thread(pipeline.memory.recall_context, ctx.clean_command, 3, False)

        proj_ctx = pipeline.session_context.project
        proj = None
        if proj_ctx is not None:
            proj = {
                "name": proj_ctx.name,
                "resolved_path": proj_ctx.source_path,
                "confidence": proj_ctx.confidence,
                "last_confirmed_ts": proj_ctx.last_confirmed_ts,
                "user_scope": proj_ctx.user_scope,
            }

        if proj:
            rag_query = f"{proj.get('name')} {ctx.clean_command}"

            rag_index = getattr(pipeline.memory, "personal_rag", None)
            warm_evt = getattr(rag_index, "warm_up_complete", None) if rag_index is not None else None
            try:
                if (
                    rag_index is not None
                    and warm_evt is not None
                    and hasattr(rag_index, "is_ready")
                    and not rag_index.is_ready
                    and pipeline._turn_count == 1
                ):
                    await asyncio.wait_for(warm_evt.wait(), timeout=1.5)
            except asyncio.TimeoutError:
                pass

            ctx.rag_context = (
                await pipeline._get_rag_context(rag_query, provider=pipeline._active_llm_provider())
                if pipeline._should_use_rag(ctx.clean_command)
                else ""
            )
            if ctx.rag_context:
                ctx.rag_context = f"[Context: user is currently asking about {proj.get('name')}]\n" + ctx.rag_context
        else:
            ctx.rag_context = (
                await pipeline._get_rag_context(ctx.clean_command, provider=pipeline._active_llm_provider())
                if pipeline._should_use_rag(ctx.clean_command)
                else ""
            )

        logger.debug("rag_context_result", has_context=bool(ctx.rag_context), context_length=len(ctx.rag_context) if ctx.rag_context else 0, preview=ctx.rag_context[:100] if ctx.rag_context else "EMPTY")
        rag_source_count = ctx.rag_context.count("\n[") if ctx.rag_context else 0
        pipeline._diag("rag_context", sources=rag_source_count, context_chars=len(ctx.rag_context) if ctx.rag_context else 0)
        return ctx

    async def _stage_execute_tools(self, ctx: TurnContext) -> TurnContext:
        pipeline = self.pipeline
        if ctx.stop_turn:
            return ctx
        try:
            ctx.provider_hint = pipeline._active_llm_provider()
            ctx.augmented_command = ctx.clean_command
            if ctx.rag_context:
                ctx.augmented_command = f"{ctx.rag_context}\nUser question: {ctx.clean_command}"
                ctx.augmented_command += "\nAnswer questions about files in maximum 4 sentences. User is listening not reading."

            configured_timeout = float(getattr(pipeline.config.providers, "overall_turn_timeout_seconds", 30.0))
            default_timeout = 45.0 if (ctx.rag_context and len(ctx.rag_context) > 100) else 20.0
            ctx.turn_timeout_seconds = min(configured_timeout, default_timeout)
            return ctx
        except AutomationFocusError as e:
            # HARDENING RULE 4 — surface focus failures clearly to telemetry + user
            msg = "I couldn't focus the right window to complete that action. Is the target window open?"
            logger.error("automation_focus_failed", error=str(e))
            try:
                pipeline.event_bus.emit("automation_focus_failed", {"expected": getattr(e, "expected", None), "actual": getattr(e, "actual", None), "retries": getattr(e, "retries", 0)})
            except Exception:
                pass
            try:
                await pipeline.tts.speak(msg)
            except Exception:
                pass
            ctx.tool_result = ToolError(message=msg)
            ctx.stop_turn = True
            ctx.stop_reason = "automation_focus_failed"
            return ctx

    async def _stage_generate_response(self, ctx: TurnContext) -> TurnContext:
        pipeline = self.pipeline
        if ctx.stop_turn:
            return ctx

        response_text = ""
        async for chunk in pipeline.brain.process_command_stream(
            ctx.augmented_command,
            long_term_memory=ctx.memory_context,
            indexed_context=ctx.rag_context,
        ):
            if not chunk:
                continue
            response_text += chunk
            pipeline.event_bus.emit("response_chunk", {"text": chunk})

        ctx.response_text = response_text.strip()
        return ctx

    async def _stage_speak(self, ctx: TurnContext) -> TurnContext:
        pipeline = self.pipeline
        if ctx.stop_turn:
            return ctx

        sentence_buffer = ""
        speaking_started = False
        sentences_queue: list[tuple[str, bool]] = []
        sentence_buffer += ctx.response_text

        sentences, sentence_buffer = pipeline._split_sentences(sentence_buffer)
        for sentence in sentences:
            if not speaking_started:
                pipeline._set_state(AssistantState.SPEAKING)
                speaking_started = True
                interrupt = True
            else:
                interrupt = False
            sentences_queue.append((sentence, interrupt))

        if sentence_buffer.strip():
            if not speaking_started:
                pipeline._set_state(AssistantState.SPEAKING)
                speaking_started = True
                interrupt = True
            else:
                interrupt = False
            sentences_queue.append((sentence_buffer.strip(), interrupt))

        chunk_buffer = ""
        first_chunk = True
        for sentence, _interrupt in sentences_queue:
            try:
                flush = True
                try:
                    if hasattr(pipeline.tts, "should_flush_sentence_buffer"):
                        flush = pipeline.tts.should_flush_sentence_buffer(chunk_buffer, sentence)
                except Exception:
                    flush = True

                if flush and chunk_buffer.strip():
                    try:
                        words = len(chunk_buffer.split())
                        est_seconds = max(0.5, (words / 2.5) + 0.5)
                    except Exception:
                        est_seconds = 2.0
                    try:
                        if hasattr(pipeline.vad, "suppress_for"):
                            await asyncio.to_thread(pipeline.vad.suppress_for, est_seconds + 1.5)
                    except Exception:
                        pass
                    try:
                        await pipeline.tts.speak(chunk_buffer.strip(), interrupt=first_chunk)
                    except Exception as e:
                        logger.error("tts_speak_failed", error=str(e), exc_info=True)
                        pipeline.event_bus.emit("error_occurred", {"component": "tts", "error": str(e)})
                    chunk_buffer = ""
                    first_chunk = False

                if chunk_buffer:
                    chunk_buffer += " " + sentence
                else:
                    chunk_buffer = sentence
            except Exception as e:
                logger.error("tts_chunking_failed", error=str(e), exc_info=True)

        if chunk_buffer.strip():
            try:
                words = len(chunk_buffer.split())
                est_seconds = max(0.5, (words / 2.5) + 0.5)
            except Exception:
                est_seconds = 2.0
            try:
                if hasattr(pipeline.vad, "suppress_for"):
                    await asyncio.to_thread(pipeline.vad.suppress_for, est_seconds + 1.5)
            except Exception:
                pass
            try:
                await pipeline.tts.speak(chunk_buffer.strip(), interrupt=first_chunk)
            except Exception as e:
                logger.error("tts_speak_failed", error=str(e), exc_info=True)
                pipeline.event_bus.emit("error_occurred", {"component": "tts", "error": str(e)})

        pipeline.event_bus.emit("response_completed", {"text": ctx.response_text})
        return ctx


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
        context_store: ContextStore | None = None,
        session_context: SessionContext | None = None,
        feedback_store: FeedbackStore | None = None,
        watchdog_stop_event: threading.Event | None = None,
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
        self.context_store = context_store or ContextStore()
        self.session_context = session_context or SessionContext()
        self.feedback_store = feedback_store or FeedbackStore()
        self._shutdown_requested = threading.Event()
        self._watchdog_stop_event = watchdog_stop_event or threading.Event()
        self._last_transcript = ""
        self._retrieval_event_queue = self.event_bus.subscribe(maxsize=0) if hasattr(self.event_bus, "subscribe") else None
        self._latest_retrieval_event: dict | None = None
        self._response_interrupted = False
        self._turn_lock: asyncio.Lock | None = None

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
        self.turn_controller = TurnController(self, watchdog_stop_event=self._watchdog_stop_event)

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

    def stop(self) -> None:
        self._shutdown_requested.set()

    def _current_user_preferences(self) -> UserPreferences:
        prefs = self.session_context.user_preferences
        if isinstance(prefs, UserPreferences):
            return prefs
        coerced = UserPreferences.from_dict(dict(prefs or {}))
        self.session_context.user_preferences = coerced
        return coerced

    @staticmethod
    def _preference_phrase_table() -> dict[str, dict[str, tuple[str, ...]]]:
        return {
            "verbosity": {
                "brief": (
                    "keep it short",
                    "be brief",
                    "shorter please",
                    "shorter answers",
                    "don't need all that",
                    "just the short version",
                    "quick answer",
                    "less detail",
                    "no need to explain",
                    "got it move on",
                    "i know just tell me",
                    "skip the explanation",
                ),
                "detailed": (
                    "explain more",
                    "tell me more",
                    "go into detail",
                    "more detail please",
                    "can you elaborate",
                    "walk me through it",
                    "break it down",
                    "give me the full picture",
                    "i want to understand why",
                ),
            },
            "tone": {
                "casual": (
                    "stop being so formal",
                    "be more casual",
                    "relax a bit",
                    "you don't have to be so stiff",
                    "talk normally",
                    "talk like a person",
                ),
                "neutral": (
                    "be more professional",
                    "keep it professional",
                    "formal please",
                ),
            },
        }

    @staticmethod
    def _normalize_preference_text(text: str) -> str:
        return re.sub(r"[^a-z0-9']+", " ", (text or "").lower()).strip()

    @staticmethod
    def _preference_word_count(text: str) -> int:
        return len([word for word in (text or "").split() if word])

    @staticmethod
    def _phrase_in_text(phrase: str, normalized_text: str) -> bool:
        return bool(re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", normalized_text))

    @staticmethod
    def _has_both_brevity_and_detail_clauses(normalized_text: str) -> bool:
        brevity_keywords = ("short", "shorter", "brief", "concise", "skip", "less detail")
        detail_keywords = ("detail", "details", "explain", "elaborate", "why", "full picture", "walk me through")
        # Mixed brevity and detail cues are treated as contradictory and ignored.
        return any(keyword in normalized_text for keyword in brevity_keywords) and any(
            keyword in normalized_text for keyword in detail_keywords
        )

    def _detect_preference_updates(self, text: str) -> PreferenceDetection:
        normalized = self._normalize_preference_text(text)
        if not normalized:
            return PreferenceDetection()

        words = self._preference_word_count(normalized)
        if self._has_both_brevity_and_detail_clauses(normalized):
            # Ambiguous mixed signals should not mutate preferences.
            logger.debug("Ambiguous preference signal, ignoring", text_preview=normalized[:120])
            return PreferenceDetection(confidence=0.4, ambiguous=True)

        updates: dict[str, str] = {}
        matched_phrases: list[str] = []
        confidence = 0.0

        for field_name, value_map in self._preference_phrase_table().items():
            for target_value, phrases in value_map.items():
                for phrase in phrases:
                    if not self._phrase_in_text(phrase, normalized):
                        continue
                    matched_phrases.append(phrase)
                    updates[field_name] = target_value
                    phrase_confidence = 1.0 if words <= 8 and normalized == phrase else 0.7
                    confidence = max(confidence, phrase_confidence)

        if confidence < 0.7 and matched_phrases:
            # Low-confidence matches are logged but deliberately ignored.
            logger.debug(
                "Low-confidence preference signal ignored",
                matched_phrases=matched_phrases,
                word_count=words,
                confidence=confidence,
            )

        return PreferenceDetection(updates=updates, confidence=confidence, matched_phrases=matched_phrases)

    def _apply_preference_updates(
        self,
        detection: PreferenceDetection,
        *,
        source: str,
        reason: str,
        turn_id: str | None,
        trigger_text: str,
    ) -> bool:
        # Accept plain dicts for quick updates (barge-in shortcuts)
        if isinstance(detection, dict):
            detection = PreferenceDetection(updates=detection, confidence=1.0, matched_phrases=list(detection.values()))
        if not detection.updates or detection.confidence < 0.7:
            return False

        prefs = self._current_user_preferences()
        before = prefs.to_dict()
        changed = False

        verbosity = detection.updates.get("verbosity")
        if verbosity and verbosity != prefs.verbosity:
            prefs.verbosity = verbosity
            changed = True

        tone = detection.updates.get("tone")
        if tone and tone != prefs.tone:
            prefs.tone = tone
            changed = True

        if not changed:
            return False

        prefs.preference_change_count = int(prefs.preference_change_count or 0) + 1
        prefs.last_updated_ts = time.time()
        try:
            with getattr(self.session_context, "_write_lock", threading.RLock()):
                self.session_context.user_preferences = prefs
        except Exception:
            self.session_context.user_preferences = prefs
        logger.info(
            "preference_update",
            source=source,
            reason=reason,
            turn_id=turn_id,
            trigger_text=trigger_text[:120],
            confidence=detection.confidence,
            previous=before,
            updated=prefs.to_dict(),
        )
        # Emit an event for listeners (tests and telemetry)
        try:
            payload = {
                "source": source,
                "reason": reason,
                "turn_id": turn_id,
                "trigger_text": trigger_text[:120],
                "confidence": detection.confidence,
                "previous": before,
                "updated": prefs.to_dict(),
            }
            try:
                self.event_bus.emit("preference_update", payload)
            except Exception:
                # best-effort: don't fail the turn for event emission issues
                pass
        except Exception:
            pass
        return True

    def _apply_preference_signals(self, text: str, *, source: str, reason: str, turn_id: str | None) -> bool:
        detection = self._detect_preference_updates(text)
        if not detection.updates:
            return False
        updated = self._apply_preference_updates(
            detection,
            source=source,
            reason=reason,
            turn_id=turn_id,
            trigger_text=text,
        )
        if not updated:
            return False
        # Persist preferences immediately so mid-turn reloads keep the update.
        try:
            with getattr(self.session_context, "_write_lock", threading.RLock()):
                self.context_store.save(self.session_context)
                try:
                    # Mark that we've just saved this session to avoid a duplicate save at end-of-turn
                    setattr(self.session_context, "_just_saved", True)
                except Exception:
                    pass
        except Exception as exc:
            logger.warning("Preference save failed", error=str(exc))
        return True

    def _looks_like_explicit_correction(self, text: str) -> bool:
        normalized = self._normalize_preference_text(text)
        correction_phrases = (
            "that's not what i meant",
            "that is not what i meant",
            "wrong answer",
            "no that's incorrect",
            "no thats incorrect",
            "that's incorrect",
            "that is incorrect",
            "that's wrong",
            "that is wrong",
        )
        return any(self._phrase_in_text(phrase, normalized) for phrase in correction_phrases)

    def _record_explicit_correction(self, text: str, turn_id: str | None) -> bool:
        if not self._looks_like_explicit_correction(text):
            return False
        prefs = self._current_user_preferences()
        prefs.correction_count = int(prefs.correction_count or 0) + 1
        prefs.last_updated_ts = time.time()
        try:
            with getattr(self.session_context, "_write_lock", threading.RLock()):
                self.session_context.user_preferences = prefs
                self.context_store.save(self.session_context)
        except Exception as exc:
            logger.warning("Preference save failed", error=str(exc))
        logger.info("correction_count_incremented", turn_id=turn_id, text_preview=text[:120])
        return True

    def _mark_response_interrupted(self) -> None:
        self._response_interrupted = True

    def _consume_response_interrupted(self) -> bool:
        interrupted = self._response_interrupted
        try:
            logger.debug("response_interrupted_consumed", interrupted=interrupted)
        except Exception:
            pass
        self._response_interrupted = False
        return interrupted

    def _sync_session_context(self, loaded_context: SessionContext) -> None:
        self.session_context.project = loaded_context.project
        self.session_context.recent_turn_summaries = list(loaded_context.recent_turn_summaries)[-20:]
        self.session_context.user_preferences = UserPreferences.from_dict(loaded_context.user_preferences.to_dict())

    def _drain_retrieval_events(self) -> dict | None:
        if self._retrieval_event_queue is None:
            return None
        latest_event = None
        while True:
            try:
                event = self._retrieval_event_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            except Exception:
                break

            if not isinstance(event, dict):
                continue
            if event.get("type") != DexterEvents.RETRIEVAL_EVENT:
                continue
            payload = event.get("payload") or {}
            if isinstance(payload, dict) and payload.get("returned_path"):
                latest_event = dict(payload)

        return latest_event

    @staticmethod
    def _looks_like_retrieval_correction(text: str) -> bool:
        normalized = re.sub(r"\s+", " ", (text or "").lower().strip())
        if not normalized:
            return False
        patterns = (
            r"\bthat's the wrong file\b",
            r"\bthat is the wrong file\b",
            r"\bwrong file\b",
            r"\bwrong document\b",
            r"\buse the other document\b",
            r"\buse the other file\b",
            r"\bnot that one\b",
            r"\bnot this one\b",
            r"\bnot the right file\b",
            r"\bnot the right document\b",
            r"\bthat's not the right file\b",
            r"\bthat's not the right document\b",
        )
        return any(re.search(pattern, normalized) for pattern in patterns)

    def _record_retrieval_feedback(self, turn_id: str, user_note: str) -> bool:
        latest = self._latest_retrieval_event or {}
        returned_path = str(latest.get("returned_path") or "")
        query = str(latest.get("query") or "")
        if not returned_path or not query:
            return False

        feedback = RetrievalFeedback(
            turn_id=turn_id,
            query=query,
            returned_path=returned_path,
            was_correct=False,
            user_note=user_note,
        )
        try:
            self.feedback_store.record(feedback)
            logger.info(
                "retrieval_feedback_recorded",
                turn_id=turn_id,
                query=query,
                returned_path=returned_path,
                user_note=user_note[:120],
            )
            self.event_bus.emit(
                "retrieval_feedback_recorded",
                {
                    "turn_id": turn_id,
                    "query": query,
                    "returned_path": returned_path,
                    "user_note": user_note,
                },
            )
            return True
        except Exception as exc:
            logger.warning(
                "retrieval_feedback_record_failed",
                turn_id=turn_id,
                query=query,
                returned_path=returned_path,
                error=str(exc),
                exc_info=True,
            )
            return False

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
        watchdog_hold_logged = False
        try:
            while not self._shutdown_requested.is_set():
                if self._watchdog_stop_event.is_set():
                    self._set_state(AssistantState.IDLE)
                    if not watchdog_hold_logged:
                        logger.critical("Watchdog stop active — DEXTER paused for hardware safety")
                        watchdog_hold_logged = True
                    await asyncio.sleep(10)
                    continue
                watchdog_hold_logged = False
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
        if self._turn_lock is None:
            self._turn_lock = asyncio.Lock()
        async with self._turn_lock:
            await self.turn_controller.run_turn()
