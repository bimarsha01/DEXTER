from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from core.brain.session_state import SessionContext
from core.event_bus import DexterEvents, EventBus
from core.pipeline import AsyncPipeline, AssistantState, TurnContext, TurnStageError
from utils.config import ActivationConfig, DexterConfig, ProvidersConfig


@dataclass
class _FallbackResponse:
    text: str
    provider: str


class FakeTranscriber:
    def __init__(self, transcripts: list[str]):
        self._transcripts = list(transcripts)

    def transcribe(self, _audio_file, on_partial=None):
        text = self._transcripts.pop(0) if self._transcripts else ""
        if on_partial:
            on_partial(text)
        return text


class BlockingTranscriber:
    def __init__(self, transcripts: list[str]):
        self._transcripts = list(transcripts)
        self._lock = threading.Lock()
        self.active_count = 0
        self.max_active = 0
        self.overlap_detected = False
        self.started = threading.Event()
        self.release = threading.Event()

    def transcribe(self, _audio_file, on_partial=None):
        with self._lock:
            self.active_count += 1
            self.max_active = max(self.max_active, self.active_count)
            if self.active_count > 1:
                self.overlap_detected = True
        self.started.set()
        self.release.wait(timeout=5.0)
        text = self._transcripts.pop(0) if self._transcripts else ""
        if on_partial:
            on_partial(text)
        with self._lock:
            self.active_count -= 1
        return text


class FakeVAD:
    def __init__(self, audio_path: str = "audio.wav"):
        self.audio_path = audio_path

    def listen(self, output_file=None, on_speech_start=None, on_clap=None, clap_sensitivity=None):
        return self.audio_path


class FakeTTS:
    def __init__(self):
        self.spoken: list[str] = []

    async def speak(self, sentence: str, interrupt: bool = True):
        self.spoken.append(sentence)

    async def play_chime(self):
        return None

    def stop(self):
        return None


class FakeMemory:
    personal_rag = None

    def recall_context(self, query, n_results=3, include_personal_rag=True):
        return ""

    def remember(self, text, role="user"):
        return None


class FakeIntentRouter:
    def detect_intent(self, _command):
        return SimpleNamespace(action="none", tool_name=None)


class FakeBrain:
    pending_action = None

    def __init__(self, response: str = "Done."):
        self.intent_router = FakeIntentRouter()
        self.last_provider = "gemini"
        self.response = response
        self.gemini_available = True
        self.groq_available = False
        self.ollama_available = False

    def _can_use_gemini(self) -> bool:
        return True

    def _can_use_provider(self, _name: str, available_flag: bool) -> bool:
        return available_flag

    async def process_command_stream(self, command, long_term_memory="", indexed_context=""):
        yield self.response


class ExhaustedBrain(FakeBrain):
    def __init__(self):
        super().__init__(response="I can't complete that right now, but I can try again later.")
        self.event_bus = None

    async def process_command_stream(self, command, long_term_memory="", indexed_context=""):
        self.gemini_available = False
        self.groq_available = False
        self.ollama_available = False
        self.last_provider = "fallback"
        if self.event_bus is not None:
            self.event_bus.emit(DexterEvents.PROVIDER_FALLBACK, {"provider": "gemini", "reason": "quota", "fallback_to": "groq"})
            self.event_bus.emit(DexterEvents.PROVIDER_FALLBACK, {"provider": "groq", "reason": "quota", "fallback_to": "ollama"})
            self.event_bus.emit(DexterEvents.PROVIDER_FALLBACK, {"provider": "ollama", "reason": "unavailable", "fallback_to": "none"})
            self.event_bus.emit("all_providers_exhausted", {"providers": ["gemini", "groq", "ollama"], "prompt_preview": command[:120]})
        yield self.response


class FakeContextStore:
    def __init__(self, session_context: SessionContext | None = None):
        self.session_context = session_context or SessionContext()
        self.load_calls: list[str | None] = []
        self.save_calls: list[SessionContext] = []

    def load(self, user_scope: str | None = None) -> SessionContext:
        self.load_calls.append(user_scope)
        return self.session_context

    def save(self, session_context: SessionContext, user_scope: str | None = None) -> None:
        self.session_context = session_context
        self.save_calls.append(session_context)


class StageEventCollector:
    def __init__(self) -> None:
        self.bus = EventBus(maxsize=0)
        self.queue = self.bus.subscribe(maxsize=0)

    def drain(self) -> list[dict]:
        events: list[dict] = []
        while True:
            try:
                events.append(self.queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return events


@pytest.fixture
def collector() -> StageEventCollector:
    return StageEventCollector()


@pytest.fixture
def pipeline_factory(tmp_path):
    def _factory(*, transcript: str, brain=None, event_bus: EventBus | None = None, transcriber=None):
        config = DexterConfig(
            activation=ActivationConfig(
                mode="always_on",
                wake_word="dexter",
                wake_words=["dexter"],
                min_command_words=1,
                start_active=True,
                fallback_to_always_on_after_failures=3,
            ),
            providers=ProvidersConfig(overall_turn_timeout_seconds=2.0),
        )
        bus = event_bus or EventBus(maxsize=0)
        store = FakeContextStore()
        pipeline = AsyncPipeline(
            config=config,
            transcriber=transcriber or FakeTranscriber([transcript]),
            vad_listener=FakeVAD(),
            tts_manager=FakeTTS(),
            memory_vault=FakeMemory(),
            brain=brain or FakeBrain(response="Here is the answer."),
            event_bus=bus,
            health_monitor=None,
            context_store=store,
            session_context=store.load(user_scope="turn_user"),
        )
        pipeline._loop = asyncio.get_running_loop()
        return pipeline, store

    return _factory


@pytest.mark.asyncio
async def test_normal_turn_completes_all_stages_in_order_and_emits_events(collector, pipeline_factory):
    pipeline, store = pipeline_factory(transcript="dexter what time is it", event_bus=collector.bus, brain=FakeBrain(response="It is 2 PM."))

    await pipeline._handle_once()

    events = collector.drain()
    stage_events = [event for event in events if event.get("type") == DexterEvents.TURN_STAGE]
    stage_pairs = [(event["payload"]["stage"], event["payload"]["status"]) for event in stage_events]
    assert stage_pairs == [
        ("transcribe", "start"),
        ("transcribe", "done"),
        ("activate", "start"),
        ("activate", "done"),
        ("retrieve_context", "start"),
        ("retrieve_context", "done"),
        ("execute_tools", "start"),
        ("execute_tools", "done"),
        ("generate_response", "start"),
        ("generate_response", "done"),
        ("speak", "start"),
        ("speak", "done"),
    ]
    assert any(event.get("type") == "response_generated" for event in events)
    assert any(event.get("type") == "response_completed" for event in events)
    assert any(event.get("type") == "turn_completed" for event in events)
    assert not any(event.get("type") == "turn_stage_error" for event in events)
    assert not any(event.get("type") == "error_occurred" for event in events)
    assert pipeline.tts.spoken == ["It is 2 PM."]
    assert pipeline.state == AssistantState.IDLE
    assert pipeline.turn_controller._current_turn_context is None
    assert store.save_calls


@pytest.mark.asyncio
async def test_stt_stage_times_out_raises_turn_stage_error_and_stops_after_error(collector, pipeline_factory, monkeypatch):
    class SlowTranscriber:
        def transcribe(self, _audio_file, on_partial=None):
            time.sleep(0.05)
            return "dexter what time is it"

    pipeline, _store = pipeline_factory(
        transcript="dexter what time is it",
        event_bus=collector.bus,
        transcriber=SlowTranscriber(),
    )
    monkeypatch.setitem(pipeline.turn_controller.STAGE_TIMEOUTS, "transcribe", 0.01)
    ctx = TurnContext(cid="turn-stt", turn_start=time.perf_counter())

    with pytest.raises(TurnStageError) as excinfo:
        await pipeline.turn_controller._run_stage(
            "transcribe",
            ctx,
            pipeline.turn_controller._stage_transcribe,
            pipeline.turn_controller._effective_stage_timeout("transcribe"),
        )

    assert excinfo.value.stage == "transcribe"
    events = collector.drain()
    stage_events = [event for event in events if event.get("type") == DexterEvents.TURN_STAGE]
    assert [(event["payload"]["stage"], event["payload"]["status"]) for event in stage_events] == [
        ("transcribe", "start"),
        ("transcribe", "error"),
    ]
    assert any(event.get("type") == "turn_stage_error" for event in events)
    assert not any(event.get("payload", {}).get("stage") == "activate" for event in stage_events)


@pytest.mark.asyncio
async def test_tool_execution_exception_fails_cleanly_and_speaks_error(collector, pipeline_factory, monkeypatch):
    pipeline, _store = pipeline_factory(transcript="dexter what time is it", event_bus=collector.bus)

    async def broken_stage(_ctx):
        raise RuntimeError("tool execution exploded")

    pipeline.turn_controller._stage_execute_tools = broken_stage

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(asyncio, "sleep", no_sleep)

    await pipeline._handle_once()

    events = collector.drain()
    error_events = [event for event in events if event.get("type") == "turn_stage_error"]
    assert error_events and error_events[-1]["payload"]["stage"] == "execute_tools"
    assert any(event.get("type") == "error_occurred" for event in events)
    assert pipeline.tts.spoken == ["I hit a problem handling that request. Please try again."]
    assert pipeline.state == AssistantState.IDLE
    assert pipeline.turn_controller._current_turn_context is None


@pytest.mark.asyncio
async def test_all_llm_providers_exhausted_spoke_fallback_text(collector, pipeline_factory):
    pipeline, _store = pipeline_factory(
        transcript="dexter tell me about the project",
        event_bus=collector.bus,
        brain=ExhaustedBrain(),
    )
    pipeline.brain.event_bus = collector.bus

    await pipeline._handle_once()

    events = collector.drain()
    assert any(event.get("type") == "all_providers_exhausted" for event in events)
    assert any(event.get("type") == DexterEvents.PROVIDER_FALLBACK for event in events)
    assert pipeline.tts.spoken
    assert pipeline.tts.spoken[-1].strip()


@pytest.mark.asyncio
async def test_concurrent_turns_do_not_interleave(collector, pipeline_factory):
    transcriber = BlockingTranscriber([
        "dexter what time is it",
        "dexter what time is it again",
    ])
    pipeline, _store = pipeline_factory(
        transcript="dexter what time is it",
        event_bus=collector.bus,
        transcriber=transcriber,
    )

    first = asyncio.create_task(pipeline._handle_once())
    assert await asyncio.to_thread(transcriber.started.wait, 2.0) is True

    second = asyncio.create_task(pipeline._handle_once())
    await asyncio.sleep(0.05)
    assert transcriber.max_active == 1
    assert transcriber.overlap_detected is False

    transcriber.release.set()
    await asyncio.gather(first, second)

    assert transcriber.max_active == 1
    assert pipeline.turn_controller._current_turn_context is None
    assert pipeline.state == AssistantState.IDLE


@pytest.mark.asyncio
async def test_failed_turn_does_not_leak_context_into_next_turn(collector, pipeline_factory, monkeypatch):
    pipeline, _store = pipeline_factory(transcript="dexter what time is it", event_bus=collector.bus)
    original_stage = pipeline.turn_controller._stage_execute_tools

    async def broken_stage(_ctx):
        raise RuntimeError("tool execution exploded")

    pipeline.turn_controller._stage_execute_tools = broken_stage

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(asyncio, "sleep", no_sleep)

    await pipeline._handle_once()
    assert pipeline.state == AssistantState.IDLE
    assert pipeline.turn_controller._current_turn_context is None
    first_events = collector.drain()
    assert any(event.get("type") == "turn_stage_error" for event in first_events)

    pipeline.turn_controller._stage_execute_tools = original_stage
    pipeline.transcriber = FakeTranscriber(["dexter what time is it again"])
    pipeline.brain = FakeBrain(response="It is 2 PM.")

    await pipeline._handle_once()

    second_events = collector.drain()
    second_stage_pairs = [
        (event["payload"]["stage"], event["payload"]["status"])
        for event in second_events
        if event.get("type") == DexterEvents.TURN_STAGE
    ]
    assert second_stage_pairs[:2] == [("transcribe", "start"), ("transcribe", "done")]
    assert pipeline.state == AssistantState.IDLE
    assert pipeline.turn_controller._current_turn_context is None
    assert pipeline.tts.spoken[-1] == "It is 2 PM."
