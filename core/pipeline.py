import asyncio
import re
import time
from uuid import uuid4
from typing import Optional

from core.event_bus import EventBus
from core.state_machine import AssistantState
from core.wake_word.detector import WakeWordDetector
from utils.transcript_correction import TranscriptCorrector
from utils.logger import get_logger, bind_correlation_id, clear_correlation_id
from utils.metrics import metrics
from utils.config import DexterConfig

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
    ) -> None:
        self.config = config
        self.transcriber = transcriber
        self.vad = vad_listener
        self.tts = tts_manager
        self.memory = memory_vault
        self.brain = brain
        self.event_bus = event_bus or EventBus()

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

    async def _stream_response(self, command: str, memory_context: str) -> str:
        response_text = ""
        sentence_buffer = ""
        tts_tasks: list[asyncio.Task] = []
        speaking_started = False

        async for chunk in self.brain.process_command_stream(command, long_term_memory=memory_context):
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
                tts_tasks.append(
                    asyncio.create_task(self.tts.speak(sentence, interrupt=interrupt))
                )

        if sentence_buffer.strip():
            if not speaking_started:
                self._set_state(AssistantState.SPEAKING)
                speaking_started = True
                interrupt = True
            else:
                interrupt = False
            tts_tasks.append(
                asyncio.create_task(self.tts.speak(sentence_buffer.strip(), interrupt=interrupt))
            )

        if tts_tasks:
            results = await asyncio.gather(*tts_tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logger.error("tts_task_failed", error=str(result), exc_info=True)
                    self.event_bus.emit(
                        "error_occurred",
                        {"component": "tts", "error": str(result)},
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
                    try:
                        # Ensure we return to IDLE to avoid stuck states
                        self._set_state(AssistantState.IDLE)
                    except Exception:
                        pass
                    # small delay before retrying to avoid busy-looping on persistent errors
                    await asyncio.sleep(1)
        finally:
            watchdog.cancel()

    async def _handle_once(self) -> None:
        cid = bind_correlation_id(uuid4().hex)
        self._set_state(AssistantState.LISTENING)
        try:
            vad_start = time.perf_counter()
            if self.activation_mode == "clap":
                audio_path = await asyncio.to_thread(
                    self.vad.listen,
                    on_speech_start=self.tts.stop,
                    on_clap=self._on_clap_detected,
                    clap_sensitivity=self.clap_sensitivity,
                )
            else:
                audio_path = await asyncio.to_thread(
                    self.vad.listen, on_speech_start=self.tts.stop
                )
            metrics.record_latency("vad_ms", (time.perf_counter() - vad_start) * 1000)

            if not audio_path:
                self._set_state(AssistantState.IDLE)
                return

            self._set_state(AssistantState.TRANSCRIBING)
            stt_start = time.perf_counter()

            def _on_partial(text: str) -> None:
                if text:
                    self.event_bus.emit("transcript_partial", {"text": text})

            identified_text = self.transcriber.transcribe(audio_path, on_partial=_on_partial)
            metrics.record_latency("stt_ms", (time.perf_counter() - stt_start) * 1000)

            if not identified_text:
                self._set_state(AssistantState.IDLE)
                return

            logger.info(
                "utterance_started",
                cid=cid,
                transcript=identified_text,
            )
            self.event_bus.emit("transcript_received", {"text": identified_text, "correlation_id": cid})
            logger.debug("transcript_final", text=identified_text)

            if self.activation_mode == "wake_word":
                detection = self.wake_detector.detect(identified_text) if self.wake_detector else None

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
                    clean_command = identified_text
                else:
                    self._set_state(AssistantState.IDLE)
                    return

                correction = self.corrector.correct(clean_command)
                clean_command = correction.corrected
            else:
                if not self._is_awake() and not self.brain.pending_action:
                    self._set_state(AssistantState.IDLE)
                    return

                correction = self.corrector.correct(identified_text)
                clean_command = correction.corrected
                if not self.brain.pending_action and len(clean_command.split()) < self.min_command_words:
                    self._set_state(AssistantState.IDLE)
                    return

            logger.info("command_accepted", command=clean_command)
            if self.activation_mode == "clap":
                self._open_wake_window()
                logger.info("activation_window_extended", seconds=self.command_window_seconds)
            self._set_state(AssistantState.PROCESSING)

            memory_context = self.memory.recall_context(clean_command)
            response_text = await self._stream_response(clean_command, memory_context)

            if self.activation_mode == "clap" and self.brain.pending_action:
                self._open_wake_window()
                logger.info("activation_window_extended", seconds=self.command_window_seconds)

            self.event_bus.emit("response_generated", {"text": response_text, "correlation_id": cid})
            logger.info("response_complete", response_preview=response_text[:500])

            try:
                self.memory.remember(f"User: {clean_command} | Dexter: {response_text}")
            except Exception as e:
                logger.error("memory_save_failed", error=str(e), exc_info=True)

            self._set_state(AssistantState.IDLE)
        finally:
            clear_correlation_id()
