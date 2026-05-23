from __future__ import annotations

from pathlib import Path

from core.feedback import FeedbackStore, RetrievalFeedback
from core.pipeline import AsyncPipeline
from tools import document_tools


class _FakeBus:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def emit(self, event_type: str, payload: dict | None = None) -> None:
        self.events.append((event_type, dict(payload or {})))


class _NullBus:
    def emit(self, *_args, **_kwargs) -> None:
        return None


def test_feedback_store_penalizes_similar_negative_queries(tmp_path):
    store = FeedbackStore(db_path=str(tmp_path / "shared.sqlite3"))
    returned_path = str((tmp_path / "docs" / "wrong_file.md").resolve())

    store.record(
        RetrievalFeedback(
            turn_id="turn-1",
            query="wrong file",
            returned_path=returned_path,
            was_correct=False,
            user_note="not that one",
        )
    )
    store.record(
        RetrievalFeedback(
            turn_id="turn-2",
            query="wrong document",
            returned_path=returned_path,
            was_correct=False,
            user_note="use the other document",
        )
    )
    store.record(
        RetrievalFeedback(
            turn_id="turn-3",
            query="wrong file",
            returned_path=str((tmp_path / "docs" / "other.md").resolve()),
            was_correct=True,
        )
    )

    penalized = store.penalized_paths_for_query("that was the wrong file")
    assert returned_path in penalized
    assert penalized[returned_path] == 2


def test_document_tools_emits_retrieval_event_for_direct_file(tmp_path):
    fake_bus = _FakeBus()
    document_tools.set_event_bus(fake_bus)
    try:
        file_path = tmp_path / "readme.md"
        file_path.write_text("Dexter learns from retrieval feedback.", encoding="utf-8")

        result = document_tools.answer_document_question(str(file_path), "What does it say?")

        assert "Dexter" in result
        assert fake_bus.events
        event_type, payload = fake_bus.events[-1]
        assert event_type == "retrieval_event"
        assert Path(payload["returned_path"]).resolve() == file_path.resolve()
        assert payload["query"] == "What does it say?"
    finally:
        document_tools.set_event_bus(None)


def test_pipeline_records_retrieval_feedback_from_correction_phrase(tmp_path):
    store = FeedbackStore(db_path=str(tmp_path / "shared.sqlite3"))
    pipeline = AsyncPipeline.__new__(AsyncPipeline)
    pipeline.feedback_store = store
    pipeline._latest_retrieval_event = {
        "query": "wrong file",
        "returned_path": str((tmp_path / "docs" / "wrong.md").resolve()),
    }
    pipeline.event_bus = _NullBus()

    assert pipeline._looks_like_retrieval_correction("that's the wrong file") is True
    recorded = pipeline._record_retrieval_feedback("turn-9", "that's the wrong file")
    assert recorded is True

    entries = store.load()
    assert len(entries) == 1
    assert entries[0].returned_path.endswith("wrong.md")
    assert entries[0].was_correct is False
