from __future__ import annotations

from tools import document_tools

import tempfile
from pathlib import Path
import uuid

import utils.config as dexter_config
from core.brain.rag import PersonalRAGIndex

_TMP_BASE = Path(__file__).resolve().parent / ".tmp"
_TMP_BASE.mkdir(parents=True, exist_ok=True)


def test_answer_document_question_resolves_partial_project_name(monkeypatch):
    project_file = _TMP_BASE / f"UserAuth-{uuid.uuid4().hex}.md"
    try:
        project_file.write_text(
            "UserAuth handles authentication, sessions, and user roles.\nIt uses JWT tokens.",
            encoding="utf-8",
        )

        class FakeIndex:
            def search(self, query, limit=5):
                assert query == "UserAuth"
                return [
                    {
                        "path": str(project_file),
                        "title": "UserAuth.md",
                        "text": project_file.read_text(encoding="utf-8"),
                        "score": 91.0,
                    }
                ]

        monkeypatch.setattr(document_tools, "_get_rag_index", lambda: FakeIndex())

        result = document_tools.answer_document_question("UserAuth", "What does it do?")

        assert "Summary:" in result or "Relevant excerpts:" in result
        assert "authentication" in result.lower()
        assert "JWT" in result
    finally:
        try:
            project_file.unlink(missing_ok=True)
        except Exception:
            pass


def test_answer_document_question_blends_multiple_candidates_for_summary(monkeypatch):
    first_file = _TMP_BASE / f"UserAuth-Overview-{uuid.uuid4().hex}.md"
    second_file = _TMP_BASE / f"UserAuth-Sessions-{uuid.uuid4().hex}.md"
    try:
        first_file.write_text("UserAuth overview: authentication, roles, and access control.", encoding="utf-8")
        second_file.write_text("Sessions are managed with JWT tokens and refresh logic.", encoding="utf-8")

        class FakeIndex:
            def search(self, query, limit=5):
                assert query == "UserAuth"
                return [
                    {
                        "path": str(first_file),
                        "title": first_file.name,
                        "text": first_file.read_text(encoding="utf-8"),
                        "score": 93.0,
                    },
                    {
                        "path": str(second_file),
                        "title": second_file.name,
                        "text": second_file.read_text(encoding="utf-8"),
                        "score": 91.0,
                    },
                ]

        monkeypatch.setattr(document_tools, "_get_rag_index", lambda: FakeIndex())

        result = document_tools.answer_document_question("UserAuth", "Give me a summary of the project")

        assert "Summary from the most relevant files:" in result
        assert first_file.name in result
        assert second_file.name in result
        assert "authentication" in result.lower()
        assert "JWT" in result
    finally:
        for file_path in (first_file, second_file):
            try:
                file_path.unlink(missing_ok=True)
            except Exception:
                pass


def test_truncate_rag_for_provider_groq_keeps_single_best_result():
    from core.brain.llm_router import Brain

    rag_context = (
        "RELEVANT CONTEXT FROM YOUR INDEXED FILES:\n"
        "(Use this to answer naturally and directly.)\n\n"
        "Source: First.md\n"
        "Path: C:/tmp/First.md\n"
        "Content: First content line.\n\n"
        "Source: Second.md\n"
        "Path: C:/tmp/Second.md\n"
        "Content: Second content line.\n\n"
        "Source: Third.md\n"
        "Path: C:/tmp/Third.md\n"
        "Content: Third content line.\n"
    )

    result = Brain._truncate_rag_for_provider(rag_context, "groq")
    assert result.count("Source: ") == 1
    assert "Second.md" not in result


def test_truncate_rag_for_provider_gemini_keeps_context():
    from core.brain.llm_router import Brain

    rag_context = (
        "RELEVANT CONTEXT FROM YOUR INDEXED FILES:\n"
        "(Use this to answer naturally and directly.)\n\n"
        "Source: First.md\n"
        "Path: C:/tmp/First.md\n"
        "Content: First content line.\n\n"
        "Source: Second.md\n"
        "Path: C:/tmp/Second.md\n"
        "Content: Second content line.\n"
    )

    result = Brain._truncate_rag_for_provider(rag_context, "gemini")
    assert result.count("Source: ") == 2


def test_build_context_respects_excerpt_max_chars_and_numbering(monkeypatch):
    # Ensure get_config() returns an excerpt_max_chars of 100.
    monkeypatch.setattr(
        dexter_config,
        "_CONFIG",
        dexter_config.DexterConfig(rag=dexter_config.RagConfig(excerpt_max_chars=100)),
    )

    class FakeIndex:
        def search(self, query, limit=4):
            return [
                {
                    "path": "/tmp/doc1.md",
                    "title": "doc1.md",
                    "parent_folder": "tmp",
                    "text": "A" * 250,
                    "score": 90.0,
                },
                {
                    "path": "/tmp/doc2.md",
                    "title": "doc2.md",
                    "parent_folder": "tmp",
                    "text": "B" * 250,
                    "score": 89.0,
                },
            ]

    fake = FakeIndex()
    # Provide helper used inside build_context.
    fake._is_import_only = PersonalRAGIndex._is_import_only

    ctx = PersonalRAGIndex.build_context(fake, "some query", limit=2)
    assert "[1]" in ctx
    assert "[2]" in ctx

    lines = ctx.splitlines()
    source_lines = [i for i, l in enumerate(lines) if l.startswith("[")]
    # The excerpt is expected to be the next non-empty line after each numbered source line.
    for i in source_lines:
        excerpt_line = lines[i + 1]
        assert len(excerpt_line) <= 100


def test_build_context_skips_import_only_chunks(monkeypatch):
    # Keep excerpt_max_chars large enough so truncation won't hide the test content.
    monkeypatch.setattr(
        dexter_config,
        "_CONFIG",
        dexter_config.DexterConfig(rag=dexter_config.RagConfig(excerpt_max_chars=500)),
    )

    class FakeIndex:
        def search(self, query, limit=4):
            return [
                {
                    "path": "/tmp/java_imports.java",
                    "title": "java_imports.java",
                    "parent_folder": "tmp",
                    "text": "import java.util.*;\nimport java.io.*;",
                    "score": 90.0,
                },
                {
                    "path": "/tmp/java_mixed.java",
                    "title": "java_mixed.java",
                    "parent_folder": "tmp",
                    "text": "import java.util.*;\npublic class Demo { int x; }",
                    "score": 89.0,
                },
            ]

    fake = FakeIndex()
    fake._is_import_only = PersonalRAGIndex._is_import_only

    ctx = PersonalRAGIndex.build_context(fake, "question", limit=2)
    assert "java_imports.java" not in ctx  # import-only snippet must be skipped entirely
    assert "public class Demo" in ctx  # mixed snippet must retain meaningful code


def test_extract_relevant_section_java_brace_languages_selects_top_level_class():
    java = """
public class Alpha {
    void foo() { System.out.println("foo"); }
}

public class Beta {
    void bar() { System.out.println("bar"); }
}
""".strip()

    excerpt = document_tools._extract_relevant_section(java, "What does Beta do with bar?", ".java")
    assert "class Beta" in excerpt
    assert "class Alpha" not in excerpt


def test_extract_relevant_section_python_keyword_languages_selects_top_level_def():
    py = """
def foo():
    return 1

def bar():
    return 2
""".strip()

    excerpt = document_tools._extract_relevant_section(py, "What does bar return?", ".py")
    assert "def bar" in excerpt
    assert "def foo" not in excerpt


def test_boost_filename_abbreviation_handles_camel_case(monkeypatch):
    monkeypatch.setattr(
        dexter_config,
        "_CONFIG",
        dexter_config.DexterConfig(rag=dexter_config.RagConfig(boost_cap=50.0)),
    )

    results = [
        {
            "path": "/tmp/misc/Main.java",
            "title": "OfficeReportingSystem.java",
            "parent_folder": "misc",
            "score": 10.0,
        }
    ]

    dummy = object()
    boosted = PersonalRAGIndex._boost_filename_matches(dummy, results, "ORS")
    assert boosted[0]["score"] >= 22.0  # base 10 + abbreviation bonus (+12) at minimum


def test_boost_filename_camel_case_tokens_match_spoken_forms(monkeypatch):
    monkeypatch.setattr(
        dexter_config,
        "_CONFIG",
        dexter_config.DexterConfig(rag=dexter_config.RagConfig(boost_cap=50.0)),
    )

    results = [
        {
            "path": "/tmp/misc/Main.java",
            "title": "OfficeReportingSystem.java",
            "parent_folder": "misc",
            "score": 10.0,
        }
    ]

    dummy = object()
    boosted = PersonalRAGIndex._boost_filename_matches(dummy, results, "office reporting system")
    assert boosted[0]["score"] >= 20.0  # base 10 + camelCase-token bonus (+10) at minimum


def test_boost_filename_directory_boost_checks_ancestor_path_components(monkeypatch):
    monkeypatch.setattr(
        dexter_config,
        "_CONFIG",
        dexter_config.DexterConfig(rag=dexter_config.RagConfig(boost_cap=50.0)),
    )

    results = [
        {
            "path": "/Projects/OfficeReportingSystem/src/Main.java",
            "title": "Main.java",
            "parent_folder": "misc",
            "score": 10.0,
        }
    ]

    dummy = object()
    boosted = PersonalRAGIndex._boost_filename_matches(dummy, results, "src")
    assert boosted[0]["score"] >= 25.0  # base 10 + (ancestor fuzz + keyword heuristic) at minimum


def test_read_file_as_text_truncates_large_java_and_extract_still_returns_nonempty():
    big_java = (
        "public class Alpha { "
            + ("A" * 8200)
        + " }\n\n"
        + "public class Beta { void bar() { System.out.println(\"bar\"); } }\n"
    )
    assert len(big_java) > 8000

    p = _TMP_BASE / f"Big-{uuid.uuid4().hex}.java"
    try:
        p.write_text(big_java, encoding="utf-8")

        text = document_tools._read_file_as_text(str(p))
        assert text[:8000] == big_java[:8000]
        assert "[truncated" in text

        excerpt = document_tools._extract_relevant_section(text, "What does Beta.bar do?", ".java")
        assert excerpt.strip()
    finally:
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass


def test_resolve_best_document_confidence_lt_05_returns_clarification_with_top3(monkeypatch):
    candidate1 = _TMP_BASE / f"ProjectAlpha-{uuid.uuid4().hex}.md"
    candidate2 = _TMP_BASE / f"ProjectBeta-{uuid.uuid4().hex}.md"
    candidate3 = _TMP_BASE / f"ProjectGamma-{uuid.uuid4().hex}.md"
    try:
        candidate1.write_text("Alpha project handles JWT and sessions.", encoding="utf-8")
        candidate2.write_text("Beta project handles OAuth and tokens.", encoding="utf-8")
        candidate3.write_text("Gamma project is unrelated.", encoding="utf-8")

        class FakeIndex:
            def search(self, query, limit=5):
                assert query == "ProjectAlpha-ish"
                return [
                    {"path": str(candidate1), "title": candidate1.name, "text": candidate1.read_text(encoding="utf-8"), "score": 45.0},
                    {"path": str(candidate2), "title": candidate2.name, "text": candidate2.read_text(encoding="utf-8"), "score": 44.0},
                    {"path": str(candidate3), "title": candidate3.name, "text": candidate3.read_text(encoding="utf-8"), "score": 43.0},
                ]

        monkeypatch.setattr(document_tools, "_get_rag_index", lambda: FakeIndex())

        result = document_tools.answer_document_question("ProjectAlpha-ish", "What does the project do?")
        assert "I'm not confident which file you mean." in result
        assert candidate1.name in result
        assert candidate2.name in result
        assert candidate3.name in result
        assert "Top candidates:" in result
    finally:
        for p in (candidate1, candidate2, candidate3):
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass


def test_rag_abbreviation_boost_ors_hits_office_reporting_system(monkeypatch):
    monkeypatch.setattr(
        dexter_config,
        "_CONFIG",
        dexter_config.DexterConfig(rag=dexter_config.RagConfig(boost_cap=50.0)),
    )

    results = [
        {
            "path": "/tmp/office-reporting-system.md",
            "title": "office-reporting-system.md",
            "parent_folder": "misc",
            "score": 10.0,
        }
    ]
    boosted = PersonalRAGIndex._boost_filename_matches(object(), results, "ORS")
    assert boosted[0]["score"] >= 22.0
