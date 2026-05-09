from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

from core.brain.llm_router import Brain, ProviderHealth
from core.pipeline import AsyncPipeline
from utils.config import DexterConfig, ActivationConfig, ProvidersConfig


def test_select_tools_for_provider_caps_groq_tools_at_configured_limit():
    brain = Brain.__new__(Brain)
    brain._cfg = SimpleNamespace(providers=SimpleNamespace(groq_max_tools=10))
    brain.groq_tools = [
        {"type": "function", "function": {"name": f"tool_{i}"}}
        for i in range(30)
    ] + [
        {"type": "function", "function": {"name": "get_current_datetime"}},
        {"type": "function", "function": {"name": "get_weather"}},
        {"type": "function", "function": {"name": "open_application"}},
    ]

    selected = Brain._select_tools_for_provider(
        brain,
        "what time is it and open calculator",
        "groq",
    )

    assert len(selected) <= 10


def test_provider_health_disable_gemini_skips_after_daily_quota_error():
    health = ProviderHealth()
    assert health.is_gemini_available() is True

    health.disable_gemini(3600.0)
    assert health.is_gemini_available() is False


def test_pipeline_turn_timeout_returns_to_idle_cleanly():
    config = DexterConfig(
        activation=ActivationConfig(
            mode="wake_word",
            wake_word="dexter",
            wake_words=["dexter"],
            min_command_words=1,
            start_active=True,
            fallback_to_always_on_after_failures=3,
        ),
        providers=ProvidersConfig(overall_turn_timeout_seconds=0.05),
    )

    class FakeTranscriber:
        def transcribe(self, _audio_file, on_partial=None):
            if on_partial:
                on_partial("dexter what time is it")
            return "dexter what time is it"

    class FakeVAD:
        def listen(self, output_file=None, on_speech_start=None, on_clap=None, clap_sensitivity=None):
            return "audio.wav"

    class FakeTTS:
        def __init__(self):
            self.spoken = []

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

    class HangingBrain:
        pending_action = None

        def __init__(self):
            self.intent_router = FakeIntentRouter()

        async def process_command_stream(self, command, long_term_memory="", indexed_context=""):
            await asyncio.sleep(1.0)
            yield "late"

    class FakeEventBus:
        def emit(self, *_a, **_kw):
            return None

    pipeline = AsyncPipeline(
        config=config,
        transcriber=FakeTranscriber(),
        vad_listener=FakeVAD(),
        tts_manager=FakeTTS(),
        memory_vault=FakeMemory(),
        brain=HangingBrain(),
        event_bus=FakeEventBus(),
        health_monitor=None,
    )
    pipeline._loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(pipeline._loop)
        pipeline.awake_until = time.time() + 10.0
        pipeline._loop.run_until_complete(pipeline._handle_once())
    finally:
        pipeline._loop.close()

    assert pipeline.state.name == "IDLE"
    assert any("didn't get a response in time" in msg.lower() for msg in pipeline.tts.spoken)
