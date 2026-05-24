from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
import getpass

import pytest

from core.brain.rag import PersonalRAGIndex, RagSearchHit
from core.feedback import FeedbackStore, RetrievalFeedback
from tools import document_tools


@pytest.fixture
def project_files(tmp_path):
    exact = tmp_path / "dexter-core.md"
    exact.write_text("Dexter core project documentation.", encoding="utf-8")

    partial_v2 = tmp_path / "dexter-v2.md"
    partial_v2.write_text("Dexter v2 project documentation.", encoding="utf-8")

    partial_core = tmp_path / "dexter-core-notes.md"
    partial_core.write_text("Dexter core notes.", encoding="utf-8")

    missing = tmp_path / "missing-project.md"

    return SimpleNamespace(
        exact=exact,
        partial_v2=partial_v2,
        partial_core=partial_core,
        missing=missing,
    )


@pytest.fixture
def fake_rag_index_factory():
    def _factory(results: list[dict[str, object]] | None = None, filenames: list[str] | None = None):
        class FakeIndex:
            def __init__(self) -> None:
                self._results = list(results or [])
                self._filenames = list(filenames or [])
                self.search_calls: list[tuple[str, int]] = []

            def search(self, query, limit=5):
                self.search_calls.append((query, limit))
                return list(self._results)

            def get_all_indexed_filenames(self):
                return list(self._filenames)

        return FakeIndex()

    return _factory


@pytest.fixture
def feedback_store(tmp_path):
    return FeedbackStore(db_path=str(tmp_path / "feedback.sqlite3"))


@pytest.fixture(autouse=True)
def _reset_document_tools_state(monkeypatch):
    monkeypatch.setattr(document_tools, "_RAG_MANAGER", None, raising=False)
    monkeypatch.setattr(document_tools, "_EVENT_BUS", None, raising=False)


def test_exact_project_match_returns_correct_source_path_with_high_confidence(
    monkeypatch,
    project_files,
    fake_rag_index_factory,
):
    fake_index = fake_rag_index_factory(
        results=[
            {
                "path": str(project_files.exact),
                "title": project_files.exact.name,
                "text": "Dexter core project documentation.",
                "score": 96.0,
            }
        ],
        filenames=[str(project_files.exact)],
    )
    monkeypatch.setattr(document_tools, "_get_rag_index", lambda: fake_index)
    monkeypatch.setattr(document_tools, "_read_file_as_text", lambda _path: "Dexter core project documentation.")

    result = document_tools.answer_document_question("dexter-core", "What is this project?")

    assert result.returned_path == str(project_files.exact)
    assert result.confidence >= 0.9
    assert Path(result.returned_path).exists()


def test_partial_name_match_prefers_highest_score_and_logs_ambiguity(
    monkeypatch,
    caplog,
    project_files,
    fake_rag_index_factory,
):
    fake_index = fake_rag_index_factory(
        results=[
            {
                "path": str(project_files.partial_v2),
                "title": project_files.partial_v2.name,
                "text": "Dexter v2 project documentation.",
                "score": 93.0,
            },
            {
                "path": str(project_files.partial_core),
                "title": project_files.partial_core.name,
                "text": "Dexter core notes.",
                "score": 92.0,
            },
        ],
        filenames=[str(project_files.partial_v2), str(project_files.partial_core)],
    )
    monkeypatch.setattr(document_tools, "_get_rag_index", lambda: fake_index)
    monkeypatch.setattr(document_tools, "_read_file_as_text", lambda _path: "Dexter project documentation.")

    with caplog.at_level("WARNING"):
        result = document_tools.answer_document_question("dexter", "What is the project about?")

    assert result.returned_path == str(project_files.partial_v2)
    assert result.confidence >= 0.9
    assert any("ambiguous" in record.message.lower() for record in caplog.records)


def test_wrong_project_correction_is_saved_as_negative_feedback(feedback_store, project_files):
    feedback_store.record(
        RetrievalFeedback(
            turn_id="turn-1",
            query="dexter",
            returned_path=str(project_files.partial_core),
            was_correct=False,
            user_note="that's the wrong project",
        )
    )

    entries = feedback_store.load()
    assert len(entries) == 1
    assert entries[0].was_correct is False
    assert entries[0].user_note == "that's the wrong project"
    assert entries[0].returned_path == str(project_files.partial_core.resolve())


def test_repeated_wrong_project_corrections_lower_the_next_retrieval_score(
    feedback_store,
    project_files,
):
    query = "dexter"
    penalized_path = str(project_files.partial_core.resolve())
    competitor_path = str(project_files.partial_v2.resolve())

    feedback_store.record(
        RetrievalFeedback(
            turn_id="turn-1",
            query=query,
            returned_path=penalized_path,
            was_correct=False,
            user_note="wrong project",
        )
    )
    feedback_store.record(
        RetrievalFeedback(
            turn_id="turn-2",
            query=query,
            returned_path=penalized_path,
            was_correct=False,
            user_note="still wrong project",
        )
    )

    hits = [
        RagSearchHit(path=penalized_path, content="Dexter core project documentation.", score=85.0),
        RagSearchHit(path=competitor_path, content="Dexter v2 project documentation.", score=85.0),
    ]
    fake_index = SimpleNamespace(
        user_id=getpass.getuser().lower(),
        _feedback_store=feedback_store,
        _feedback_penalty=0.15,
    )

    penalized = PersonalRAGIndex._apply_feedback_penalties(fake_index, hits, query)

    penalized_hit = next(hit for hit in penalized if hit.path == penalized_path)
    competitor_hit = next(hit for hit in penalized if hit.path == competitor_path)

    assert penalized_hit.score < competitor_hit.score
    assert competitor_hit.score == pytest.approx(85.0)


def test_empty_rag_index_returns_zero_confidence_document_result(
    monkeypatch,
    fake_rag_index_factory,
):
    fake_index = fake_rag_index_factory(results=[], filenames=[])
    monkeypatch.setattr(document_tools, "_get_rag_index", lambda: fake_index)

    result = document_tools.answer_document_question("empty-project", "What is it?")

    assert isinstance(result, document_tools.DocumentResult)
    assert result.confidence == 0.0
    assert result.returned_path == ""


def test_source_path_exists_or_result_is_marked_stale(
    monkeypatch,
    project_files,
    fake_rag_index_factory,
):
    existing_index = fake_rag_index_factory(
        results=[
            {
                "path": str(project_files.exact),
                "title": project_files.exact.name,
                "text": "Dexter core project documentation.",
                "score": 95.0,
            }
        ],
        filenames=[str(project_files.exact)],
    )
    monkeypatch.setattr(document_tools, "_get_rag_index", lambda: existing_index)
    monkeypatch.setattr(document_tools, "_read_file_as_text", lambda _path: "Dexter core project documentation.")

    existing_result = document_tools.answer_document_question("dexter-core", "What is this?")
    assert Path(existing_result.returned_path).exists()

    missing_index = fake_rag_index_factory(
        results=[
            {
                "path": str(project_files.missing),
                "title": project_files.missing.name,
                "text": "Missing project documentation.",
                "score": 95.0,
            }
        ],
        filenames=[str(project_files.missing)],
    )
    monkeypatch.setattr(document_tools, "_get_rag_index", lambda: missing_index)
    monkeypatch.setattr(
        document_tools,
        "_read_file_as_text",
        lambda _path: pytest.fail("stale path should be detected before reading"),
    )

    stale_result = document_tools.answer_document_question("missing-project", "What is this?")
    assert stale_result.metadata.get("stale") is True or not stale_result.returned_path
