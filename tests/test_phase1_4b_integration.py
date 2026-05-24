from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.brain.llm_router import Brain, QuotaExhaustedError
from core.brain.session_state import ContextStore, SessionContext, UserPreferences
from core.event_bus import DexterEvents, EventBus
from core.pipeline import AsyncPipeline
from utils.config import ActivationConfig, DexterConfig, ProvidersConfig


@dataclass
class FallbackResponse:
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


class CountingContextStore(ContextStore):
    def __init__(self, db_path: str):
        super().__init__(db_path=db_path)
        self.save_calls = 0

    def save(self, session_context: SessionContext, user_scope: str | None = None) -> None:
        self.save_calls += 1
        super().save(session_context, user_scope=user_scope)


def _make_pipeline(
    *,
    db_path: str,
    transcript: str,
    event_bus: EventBus | None = None,
    context_store: ContextStore | None = None,
    session_context: SessionContext | None = None,
    brain: FakeBrain | None = None,
) -> AsyncPipeline:
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
    store = context_store or ContextStore(db_path=db_path)
    loaded_context = session_context or store.load(user_scope="pref_user")
    return AsyncPipeline(
        config=config,
        transcriber=FakeTranscriber([transcript]),
        vad_listener=FakeVAD(),
        tts_manager=FakeTTS(),
        memory_vault=FakeMemory(),
        brain=brain or FakeBrain(response="Here is the answer."),
        event_bus=bus,
        health_monitor=None,
        context_store=store,
        session_context=loaded_context,
    )


def _drain_events(queue: asyncio.Queue) -> list[dict]:
    events: list[dict] = []
    while True:
        try:
            events.append(queue.get_nowait())
        except asyncio.QueueEmpty:
            break
    return events


def _build_system_prompt(base_prompt: str, prefs: UserPreferences) -> str:
    # Adapter for current router API: the effective system prompt is produced by the `system_instruction` property.
    brain = Brain.__new__(Brain)
    brain.base_system_instruction = base_prompt
    brain.session_context = SessionContext(user_preferences=prefs)
    return brain.system_instruction


async def _complete_with_fallback(brain: Brain, prompt: str, session_context: SessionContext) -> FallbackResponse:
    # Adapter for current router API: `_process_text_fallback` is the provider fallback chain used by the router.
    brain.session_context = session_context
    text = await brain._process_text_fallback(prompt)
    provider = "fallback" if "can't reach" in (text or "").lower() or "unreachable" in (text or "").lower() else (brain.last_provider or "unknown")
    return FallbackResponse(text=text, provider=provider)


# SCENARIO 1 — Cold start and context persistence

def test_phase1_4b_s1_cold_start_and_context_persistence(context_store_with_tmp_db):
    store, db_path = context_store_with_tmp_db
    if os.path.exists(db_path):
        os.remove(db_path)

    context = store.load(user_scope="pref_user")

    assert context.project is None
    assert isinstance(context.user_preferences, UserPreferences)
    assert context.user_preferences.verbosity == "normal"
    assert context.user_preferences.tone == "neutral"

    context.user_preferences.verbosity = "brief"
    store.save(context, user_scope="pref_user")

    reloaded = ContextStore(db_path=db_path).load(user_scope="pref_user")

    assert isinstance(reloaded.user_preferences, UserPreferences)
    assert reloaded.user_preferences.verbosity == "brief"
    assert reloaded.user_preferences.tone == "neutral"


# SCENARIO 2 — Preference detection updates persistent context

@pytest.mark.asyncio
async def test_phase1_4b_s2_preference_detection_persists(context_store_with_tmp_db, caplog):
    base_store, db_path = context_store_with_tmp_db
    store = CountingContextStore(db_path=db_path)
    event_bus = EventBus(maxsize=0)
    event_queue = event_bus.subscribe(maxsize=0)

    pipeline = _make_pipeline(
        db_path=db_path,
        transcript="keep it short please",
        event_bus=event_bus,
        context_store=store,
        session_context=base_store.load(user_scope="pref_user"),
        brain=FakeBrain(response="Done."),
    )
    pipeline._loop = asyncio.get_running_loop()

    with caplog.at_level(logging.INFO):
        await pipeline._handle_once()

    assert pipeline.session_context.user_preferences.verbosity == "brief"
    assert store.save_calls == 1

    events = _drain_events(event_queue)
    pref_events = [e for e in events if e.get("type") == "preference_update"]
    assert pref_events, "Expected a preference_update event on event_bus"
    payload = pref_events[-1].get("payload", {})
    assert payload.get("source") == "explicit_command"

    restarted = ContextStore(db_path=db_path).load(user_scope="pref_user")
    assert restarted.user_preferences.verbosity == "brief"


# SCENARIO 3 — Dynamic prompt augmentation reaches the LLM

def test_phase1_4b_s3_prompt_augmentation_includes_preferences():
    base_prompt = "You are Dexter, a personal AI assistant."

    brief_prompt = _build_system_prompt(
        base_prompt,
        UserPreferences(verbosity="brief", tone="casual"),
    )

    assert "under 2 sentences" in brief_prompt.lower()
    assert "casual" in brief_prompt.lower()
    assert base_prompt in brief_prompt
    assert brief_prompt.index(base_prompt) < brief_prompt.index("Current user preferences")

    detailed_prompt = _build_system_prompt(
        base_prompt,
        UserPreferences(verbosity="detailed", tone="casual"),
    )

    assert "fuller explanations" in detailed_prompt.lower()
    assert "under 2 sentences" not in detailed_prompt.lower()


# SCENARIO 4 — Provider fallback chain with all providers mocked as failing

@pytest.mark.asyncio
async def test_phase1_4b_s4_provider_fallback_all_fail(caplog):
    bus = EventBus(maxsize=0)
    queue = bus.subscribe(maxsize=0)

    def custom_gemini_init(self):
        self.gemini_available = True

    def custom_groq_init(self):
        self.groq_available = True
        self.groq_tools = []

    def custom_ollama_init(self):
        self.ollama_available = True

    async def fail_quota(_prompt):
        raise QuotaExhaustedError("quota exhausted")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Brain, "_init_gemini", custom_gemini_init)
        mp.setattr(Brain, "_init_groq", custom_groq_init)
        mp.setattr(Brain, "_init_ollama", custom_ollama_init)
        brain = Brain(event_bus=bus, session_context=SessionContext())
        mp.setattr(brain, "_process_gemini", fail_quota)
        mp.setattr(brain, "_process_groq", fail_quota)
        mp.setattr(brain, "_process_ollama", fail_quota)

        with caplog.at_level(logging.ERROR):
            result = await _complete_with_fallback(brain, "hello", SessionContext())

    assert isinstance(result, FallbackResponse)
    assert isinstance(result.text, str)
    assert result.text.strip()

    events = _drain_events(queue)
    fallback_events = [e for e in events if e.get("type") == DexterEvents.PROVIDER_FALLBACK]
    providers = [e.get("payload", {}).get("provider") for e in fallback_events]
    assert "gemini" in providers
    assert "groq" in providers
    assert "ollama" in providers

    exhausted_events = [e for e in events if e.get("type") == "all_providers_exhausted"]
    assert len(exhausted_events) == 1


# SCENARIO 5 — Stage isolation: retrieval failure does not silence the turn

@pytest.mark.asyncio
async def test_phase1_4b_s5_retrieval_failure_still_speaks_error(context_store_with_tmp_db):
    _, db_path = context_store_with_tmp_db
    event_bus = EventBus(maxsize=0)
    queue = event_bus.subscribe(maxsize=0)

    pipeline = _make_pipeline(
        db_path=db_path,
        transcript="what is in my documents",
        event_bus=event_bus,
        brain=FakeBrain(response="Normal response."),
    )
    pipeline._loop = asyncio.get_running_loop()

    original_stage = pipeline.turn_controller._stage_retrieve_context

    async def broken_stage(ctx):
        raise RuntimeError("rag failed")

    pipeline.turn_controller._stage_retrieve_context = broken_stage
    await pipeline._handle_once()

    assert pipeline.tts.spoken
    assert all("runtimeerror" not in msg.lower() for msg in pipeline.tts.spoken)

    events = _drain_events(queue)
    stage_errors = [
        e for e in events
        if e.get("type") == "turn_stage_error"
        and (e.get("payload") or {}).get("stage") == "retrieve_context"
    ]
    assert stage_errors

    pipeline.turn_controller._stage_retrieve_context = original_stage
    pipeline.transcriber = FakeTranscriber(["what is in my documents"])
    previous_spoken_count = len(pipeline.tts.spoken)
    await pipeline._handle_once()

    assert len(pipeline.tts.spoken) > previous_spoken_count
    assert pipeline.state.name == "IDLE"


# SCENARIO 6 — Barge-in sets verbosity to brief and saves

@pytest.mark.asyncio
async def test_phase1_4b_s6_barge_in_sets_brief_and_saves(context_store_with_tmp_db):
    base_store, db_path = context_store_with_tmp_db
    store = CountingContextStore(db_path=db_path)
    loaded = base_store.load(user_scope="pref_user")
    loaded.user_preferences.verbosity = "normal"
    base_store.save(loaded, user_scope="pref_user")

    pipeline = _make_pipeline(
        db_path=db_path,
        transcript="dexter continue",
        context_store=store,
        session_context=loaded,
        brain=FakeBrain(response="Sure."),
    )
    pipeline._loop = asyncio.get_running_loop()

    # Simulate a barge-in flag captured before transcription is processed.
    pipeline._mark_response_interrupted()
    before_count = pipeline.session_context.user_preferences.preference_change_count
    await pipeline._handle_once()

    persisted = ContextStore(db_path=db_path).load(user_scope="pref_user")
    assert persisted.user_preferences.verbosity == "brief"
    assert store.save_calls >= 1
    assert persisted.user_preferences.preference_change_count == before_count + 1


# SCENARIO 7 — FeedbackStore write failure does not crash a turn

@pytest.mark.asyncio
async def test_phase1_4b_s7_feedback_write_failure_non_critical(context_store_with_tmp_db, caplog):
    _, db_path = context_store_with_tmp_db

    pipeline = _make_pipeline(
        db_path=db_path,
        transcript="what time is it",
        brain=FakeBrain(response="It is 2 PM."),
    )
    pipeline._loop = asyncio.get_running_loop()

    def failing_record(_feedback):
        raise IOError("Feedback write failed (non-critical): disk full")

    pipeline.feedback_store.record = failing_record
    pipeline._latest_retrieval_event = {
        "returned_path": "C:/docs/readme.md",
        "query": "readme",
    }

    original_stage_activate = pipeline.turn_controller._stage_activate

    async def stage_activate_with_feedback(ctx):
        pipeline._record_retrieval_feedback(ctx.cid, "wrong file")
        return await original_stage_activate(ctx)

    pipeline.turn_controller._stage_activate = stage_activate_with_feedback

    with caplog.at_level(logging.WARNING):
        await pipeline._handle_once()

    assert pipeline.tts.spoken
    assert any("it is 2 pm" in s.lower() for s in pipeline.tts.spoken)
    assert any("Feedback write failed" in rec.getMessage() or "retrieval_feedback_record_failed" in rec.getMessage() for rec in caplog.records)


# Conftest-style fixture requested in-file
@pytest.fixture
def context_store_with_tmp_db(tmp_path):
    db_path = str(Path(tmp_path) / "phase1_4b_session_context.sqlite3")
    if os.path.exists(db_path):
        os.remove(db_path)
    store = ContextStore(db_path=db_path)
    yield store, db_path
    if os.path.exists(db_path):
        os.remove(db_path)
