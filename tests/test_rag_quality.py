"""Tests for RAG quality penalties and reranking hooks."""
from core.brain.rag import PersonalRAGIndex, RagSearchHit


def test_deprioritized_test_files_penalized():
    hits = [
        RagSearchHit(path="/proj/UserAuth.java", content="real auth", score=90.0),
        RagSearchHit(
            path="/proj/tests/test_rag_document_resolution.py",
            content="mock UserAuth fixture",
            score=88.0,
        ),
    ]
    index = PersonalRAGIndex.__new__(PersonalRAGIndex)
    penalized = index._apply_source_quality_penalties(hits)
    assert penalized[0].path.endswith("UserAuth.java")
    assert penalized[0].score > penalized[1].score


def test_format_context_header_code_mode():
    results = [{"path": "/x/service.py", "content": "def foo(): pass"}]
    header = PersonalRAGIndex.format_context_header(results)
    assert "CODE FILES" in header
