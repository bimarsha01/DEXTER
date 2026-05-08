import pytest

from core.brain import session_state


def test_clear_if_stale_clears_after_expected_turns():
    user = "test_user_clear"
    session_state.set_current_project(
        name="MyProject",
        resolved_path="/tmp/MyProject",
        confidence=0.9,
        set_at_turn=3,
        user=user,
    )
    session_state.clear_if_stale(6, user=user)
    assert session_state.get_current_project(user=user) is None


def test_clear_if_stale_does_not_clear_immediately_after_set():
    user = "test_user_survive"
    session_state.set_current_project(
        name="MyProject",
        resolved_path="/tmp/MyProject",
        confidence=0.9,
        set_at_turn=10,
        user=user,
    )
    session_state.clear_if_stale(10, user=user)
    assert session_state.get_current_project(user=user) is not None


def test_clear_if_stale_survives_next_turns():
    user = "test_user_survive_window"
    session_state.set_current_project(
        name="MyProject",
        resolved_path="/tmp/MyProject",
        confidence=0.9,
        set_at_turn=3,
        user=user,
    )
    session_state.clear_if_stale(4, user=user)
    assert session_state.get_current_project(user=user) is not None
    session_state.clear_if_stale(5, user=user)
    assert session_state.get_current_project(user=user) is not None


def test_clear_if_stale_does_not_clear_none_sentinel():
    user = "test_user_sentinel_none"
    session_state.set_current_project(
        name="MyProject",
        resolved_path="/tmp/MyProject",
        confidence=0.9,
        set_at_turn=None,
        user=user,
    )
    session_state.clear_if_stale(999, user=user)
    assert session_state.get_current_project(user=user) is not None

