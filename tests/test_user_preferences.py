from __future__ import annotations

from types import SimpleNamespace

from core.brain.llm_router import Brain
from core.brain.session_state import ContextStore, ProjectContext, SessionContext, UserPreferences
from core.pipeline import AsyncPipeline


def test_user_preferences_round_trip_through_context_store(tmp_path):
    store = ContextStore(db_path=str(tmp_path / "session_context.sqlite3"))
    context = SessionContext(
        project=ProjectContext(
            name="Demo",
            source_path="/tmp/demo",
            confidence=0.9,
            last_confirmed_ts=1.0,
            user_scope="pref_user",
        ),
        user_preferences=UserPreferences(
            verbosity="brief",
            tone="casual",
            correction_count=2,
            last_updated_ts=123.45,
        ),
    )

    store.save(context)
    loaded = store.load(user_scope="pref_user")

    assert isinstance(loaded.user_preferences, UserPreferences)
    assert loaded.user_preferences.verbosity == "brief"
    assert loaded.user_preferences.tone == "casual"
    assert loaded.user_preferences.correction_count == 2
    assert loaded.user_preferences.last_updated_ts == 123.45


def test_router_system_instruction_reflects_live_preferences():
    brain = Brain.__new__(Brain)
    brain.base_system_instruction = "Base prompt"
    brain.session_context = SessionContext(
        user_preferences=UserPreferences(verbosity="brief", tone="casual")
    )

    prompt = brain.system_instruction
    assert "Base prompt" in prompt
    assert "Keep all responses under 2 sentences unless asked to explain." in prompt
    assert "Use a casual, conversational tone." in prompt

    brain.session_context.user_preferences.verbosity = "detailed"
    brain.session_context.user_preferences.tone = "neutral"
    updated_prompt = brain.system_instruction
    assert "Provide fuller explanations when the user asks for more detail." in updated_prompt
    assert "Use a neutral, straightforward tone." in updated_prompt


def test_pipeline_updates_preferences_from_explicit_signal_and_saves(tmp_path):
    store = ContextStore(db_path=str(tmp_path / "session_context.sqlite3"))
    session_context = SessionContext(
        project=ProjectContext(
            name="Demo",
            source_path="/tmp/demo",
            confidence=0.9,
            last_confirmed_ts=1.0,
            user_scope="pref_user",
        )
    )

    pipeline = AsyncPipeline.__new__(AsyncPipeline)
    pipeline.context_store = store
    pipeline.session_context = session_context

    changed = AsyncPipeline._apply_preference_signals(
        pipeline,
        "keep it short and stop being so formal",
        source="explicit_command",
        reason="explicit preference request",
        turn_id="turn-1",
    )

    loaded = store.load(user_scope="pref_user")

    assert changed is True
    assert loaded.user_preferences.verbosity == "brief"
    assert loaded.user_preferences.tone == "casual"
    assert loaded.user_preferences.correction_count == 1
    assert loaded.user_preferences.last_updated_ts > 0.0
