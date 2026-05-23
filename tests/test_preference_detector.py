from __future__ import annotations

from core.brain.session_state import ContextStore, ProjectContext, SessionContext, UserPreferences
from core.pipeline import AsyncPipeline


def _make_pipeline(tmp_path):
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
    return pipeline, store


def test_brief_phrase_triggers_preference_change(tmp_path):
    pipeline, store = _make_pipeline(tmp_path)

    changed = AsyncPipeline._apply_preference_signals(
        pipeline,
        "keep it short",
        source="explicit_command",
        reason="explicit preference request",
        turn_id="turn-brief",
    )

    loaded = store.load(user_scope="pref_user")
    assert changed is True
    assert loaded.user_preferences.verbosity == "brief"
    assert loaded.user_preferences.preference_change_count == 1


def test_detailed_phrase_triggers_preference_change(tmp_path):
    pipeline, store = _make_pipeline(tmp_path)

    changed = AsyncPipeline._apply_preference_signals(
        pipeline,
        "can you elaborate",
        source="explicit_command",
        reason="explicit preference request",
        turn_id="turn-detailed",
    )

    loaded = store.load(user_scope="pref_user")
    assert changed is True
    assert loaded.user_preferences.verbosity == "detailed"
    assert loaded.user_preferences.preference_change_count == 1


def test_casual_phrase_triggers_preference_change(tmp_path):
    pipeline, store = _make_pipeline(tmp_path)

    changed = AsyncPipeline._apply_preference_signals(
        pipeline,
        "talk like a person",
        source="explicit_command",
        reason="explicit preference request",
        turn_id="turn-casual",
    )

    loaded = store.load(user_scope="pref_user")
    assert changed is True
    assert loaded.user_preferences.tone == "casual"
    assert loaded.user_preferences.preference_change_count == 1


def test_neutral_phrase_triggers_preference_change(tmp_path):
    pipeline, store = _make_pipeline(tmp_path)
    pipeline.session_context.user_preferences = UserPreferences(tone="casual")

    changed = AsyncPipeline._apply_preference_signals(
        pipeline,
        "formal please",
        source="explicit_command",
        reason="explicit preference request",
        turn_id="turn-neutral",
    )

    loaded = store.load(user_scope="pref_user")
    assert changed is True
    assert loaded.user_preferences.tone == "neutral"
    assert loaded.user_preferences.preference_change_count == 1


def test_contradictory_brevity_and_detail_request_is_ignored(tmp_path):
    pipeline, store = _make_pipeline(tmp_path)

    changed = AsyncPipeline._apply_preference_signals(
        pipeline,
        "keep it short but give me the details",
        source="explicit_command",
        reason="explicit preference request",
        turn_id="turn-negative-1",
    )

    loaded = store.load(user_scope="pref_user")
    assert changed is False
    assert loaded.user_preferences.verbosity == "normal"
    assert loaded.user_preferences.preference_change_count == 0


def test_topic_discussion_about_shortness_is_ignored(tmp_path):
    pipeline, store = _make_pipeline(tmp_path)

    changed = AsyncPipeline._apply_preference_signals(
        pipeline,
        "explain more about why to keep it short",
        source="explicit_command",
        reason="explicit preference request",
        turn_id="turn-negative-2",
    )

    loaded = store.load(user_scope="pref_user")
    assert changed is False
    assert loaded.user_preferences.verbosity == "normal"
    assert loaded.user_preferences.preference_change_count == 0


def test_shorter_word_inside_unrelated_question_is_ignored(tmp_path):
    pipeline, store = _make_pipeline(tmp_path)

    changed = AsyncPipeline._apply_preference_signals(
        pipeline,
        "Which route is shorter, the one through town or the bypass?",
        source="explicit_command",
        reason="explicit preference request",
        turn_id="turn-negative-3",
    )

    loaded = store.load(user_scope="pref_user")
    assert changed is False
    assert loaded.user_preferences.verbosity == "normal"
    assert loaded.user_preferences.preference_change_count == 0


def test_long_question_about_shorter_option_is_ignored(tmp_path):
    pipeline, store = _make_pipeline(tmp_path)

    changed = AsyncPipeline._apply_preference_signals(
        pipeline,
        "I am comparing these routes and want to know which one is shorter overall for the trip.",
        source="explicit_command",
        reason="explicit preference request",
        turn_id="turn-negative-4",
    )

    loaded = store.load(user_scope="pref_user")
    assert changed is False
    assert loaded.user_preferences.verbosity == "normal"
    assert loaded.user_preferences.preference_change_count == 0


def test_ambiguous_mixed_signal_is_ignored_case_one(tmp_path):
    pipeline, store = _make_pipeline(tmp_path)

    detection = pipeline._detect_preference_updates(
        "keep it short but also explain the background in detail please"
    )

    loaded = store.load(user_scope="pref_user")
    assert detection.ambiguous is True
    assert detection.confidence == 0.4
    assert detection.updates == {}
    assert loaded.user_preferences.preference_change_count == 0


def test_ambiguous_mixed_signal_is_ignored_case_two(tmp_path):
    pipeline, store = _make_pipeline(tmp_path)

    detection = pipeline._detect_preference_updates(
        "I know the short answer is fine, but can you explain more why it matters in the long run?"
    )

    loaded = store.load(user_scope="pref_user")
    assert detection.ambiguous is True
    assert detection.confidence == 0.4
    assert detection.updates == {}
    assert loaded.user_preferences.preference_change_count == 0


def test_preference_change_count_increments_for_multiple_changes(tmp_path):
    pipeline, store = _make_pipeline(tmp_path)

    first = AsyncPipeline._apply_preference_signals(
        pipeline,
        "keep it short",
        source="explicit_command",
        reason="explicit preference request",
        turn_id="turn-count-1",
    )
    second = AsyncPipeline._apply_preference_signals(
        pipeline,
        "talk like a person",
        source="explicit_command",
        reason="explicit preference request",
        turn_id="turn-count-2",
    )

    loaded = store.load(user_scope="pref_user")
    assert first is True
    assert second is True
    assert loaded.user_preferences.preference_change_count == 2


def test_correction_count_only_changes_on_explicit_corrections(tmp_path):
    pipeline, store = _make_pipeline(tmp_path)
    pipeline.session_context.user_preferences = UserPreferences(tone="casual")

    preference_changed = AsyncPipeline._apply_preference_signals(
        pipeline,
        "formal please",
        source="explicit_command",
        reason="explicit preference request",
        turn_id="turn-correction-1",
    )
    after_preference = store.load(user_scope="pref_user")
    correction_recorded = pipeline._record_explicit_correction("wrong answer", "turn-correction-2")
    after_correction = store.load(user_scope="pref_user")

    assert preference_changed is True
    assert after_preference.user_preferences.correction_count == 0
    assert correction_recorded is True
    assert after_correction.user_preferences.correction_count == 1
    assert after_correction.user_preferences.preference_change_count == 1
