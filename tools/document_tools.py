from __future__ import annotations

import csv
import getpass
import json
import os
import re
import threading
from collections import Counter
from pathlib import Path

from core.brain.rag import MultiUserRAGManager
from core.brain import session_state
from utils.config import get_config
from utils.logger import get_logger

logger = get_logger("document_tools")


SUPPORTED_TEXT_EXTENSIONS = {".txt", ".md", ".py", ".json", ".yaml", ".yml", ".csv", ".log", ".ini", ".cfg", ".toml"}

_RAG_MANAGER: MultiUserRAGManager | None = None
_RAG_MANAGER_LOCK = threading.Lock()


def _get_rag_index():
    global _RAG_MANAGER
    if _RAG_MANAGER is None:
        with _RAG_MANAGER_LOCK:
            if _RAG_MANAGER is None:
                cfg = get_config()
                _RAG_MANAGER = MultiUserRAGManager(
                    persist_directory=cfg.rag.persist_directory,
                    default_roots=cfg.rag.personal_roots,
                    cfg={
                        "chunk_size": cfg.rag.chunk_size,
                        "chunk_overlap": cfg.rag.chunk_overlap,
                        "refresh_seconds": cfg.rag.refresh_seconds,
                        "exclude_patterns": cfg.rag.exclude_patterns,
                        "roots": cfg.rag.personal_roots,
                        "embedding_model": cfg.rag.embedding_model,
                        "index_schema_version": cfg.rag.index_schema_version,
                        "max_context_chars": cfg.rag.max_context_chars,
                        "batch_size": cfg.rag.batch_size,
                        "max_embedding_threads": cfg.rag.max_embedding_threads,
                    },
                )

    user_id = (getpass.getuser() or "default").lower()
    return _RAG_MANAGER.get_index_for_user(user_id)


def _read_file_as_text(path: str) -> str:
    """Read any file as text with sensible fallbacks; returns up to 8000 chars."""
    file_path = Path(path).expanduser()
    if not file_path.exists():
        return f"File not found: {path}"

    ext = file_path.suffix.lower()
    max_chars = 8000

    try:
        if ext == ".docx":
            try:
                from docx import Document  # type: ignore

                document = Document(str(file_path))
                text = "\n".join(paragraph.text for paragraph in document.paragraphs if paragraph.text.strip())
            except Exception as e:
                logger.warning("docx_read_unavailable", error=str(e))
                return "python-docx is not available or the document could not be read."

        elif ext == ".pdf":
            try:
                from pypdf import PdfReader  # type: ignore

                reader = PdfReader(str(file_path))
                pages = [page.extract_text() or "" for page in reader.pages]
                text = "\n".join(p for p in pages if p.strip())
            except Exception as e:
                logger.warning("pdf_read_unavailable", error=str(e))
                return "pypdf is not available or the PDF could not be read."

        else:
            # Plain text fallback for any other extension (always allowed)
            if ext not in SUPPORTED_TEXT_EXTENSIONS:
                logger.warning("reading_as_plain_text", ext=ext, path=str(file_path))
            with open(file_path, "r", encoding="utf-8", errors="replace") as handle:
                text = handle.read()

        if not text:
            return f"No readable text found in {file_path.name}."

        if len(text) > max_chars:
            omitted = len(text) - max_chars
            return text[:max_chars] + f"\n[truncated — {omitted} chars omitted]"
        return text

    except Exception as e:
        logger.error("file_read_failed", path=str(file_path), error=str(e), exc_info=True)
        return f"I could not read {file_path.name}."


def _extract_relevant_section(content: str, question: str, ext: str) -> str:
    """Extract a concise, relevant section from content for the given question.

    For code files this returns the top 3 scoring top-level blocks by
    `rapidfuzz.partial_ratio`. For non-code, returns the first 2000 chars.
    """
    from rapidfuzz import fuzz

    if not content:
        return ""

    ext_lower = (ext or "").lower()
    cleaned_lines = [
        l
        for l in content.splitlines()
        if not re.match(r"^(import\s+|package\s+|using\s+|#include\s+|require\s+)", l.strip(), re.IGNORECASE)
    ]
    cleaned = "\n".join(cleaned_lines).strip()
    if not cleaned:
        cleaned = content.strip()

    # Brace languages: use top-level brace tracking.
    brace_langs = {".java", ".cs", ".go", ".ts", ".js", ".cpp", ".kt", ".rs"}
    keyword_langs = {".py", ".rb"}

    def _blank_line_blocks(text: str) -> list[str]:
        return [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]

    def _take_top_blocks(blocks: list[str], cap_chars: int = 2000) -> str:
        q = question or ""
        scored = [(fuzz.partial_ratio(q, b), b) for b in blocks] if q else [(0, b) for b in blocks]
        scored.sort(key=lambda x: x[0], reverse=True)
        # Keep only blocks that are close enough to the best match.
        # This prevents returning multiple "top-level" structures when the query
        # clearly targets a single class/function.
        best_score = scored[0][0] if scored else 0.0
        score_threshold = max(0.0, best_score - 5.0)
        chosen = []
        total = 0
        filtered = [item for item in scored[:3] if item[0] >= score_threshold]
        if not filtered and scored:
            filtered = [scored[0]]

        for _score, blk in filtered:
            blk = blk.strip()
            if not blk:
                continue
            if total + len(blk) > cap_chars:
                remaining = cap_chars - total
                if remaining > 0:
                    chosen.append(blk[:remaining].rstrip())
                break
            chosen.append(blk)
            total += len(blk)
        return "\n\n".join(chosen).strip()

    if ext_lower in brace_langs:
        blocks: list[str] = []
        depth = 0
        start = None
        for i, ch in enumerate(cleaned):
            if ch == "{":
                if depth == 0:
                    start = cleaned.rfind("\n", 0, i) + 1
                depth += 1
            elif ch == "}" and depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    candidate = cleaned[start : i + 1].strip()
                    if candidate:
                        blocks.append(candidate)
                    start = None

        # If parsing found no top-level blocks, fall back to blank-line splitting.
        if not blocks:
            blocks = _blank_line_blocks(cleaned)

        out = _take_top_blocks(blocks)
        return out if out else cleaned[:2000]

    if ext_lower in keyword_langs:
        # Split on `def` / `class` at column 0.
        lines = cleaned.splitlines()
        starts: list[int] = []
        for idx, line in enumerate(lines):
            if line.startswith("def ") or line.startswith("class "):
                starts.append(idx)

        if not starts:
            blocks = _blank_line_blocks(cleaned)
            out = _take_top_blocks(blocks)
            return out if out else cleaned[:2000]

        blocks: list[str] = []
        for si, start_idx in enumerate(starts):
            end_idx = starts[si + 1] if si + 1 < len(starts) else len(lines)
            blk = "\n".join(lines[start_idx:end_idx]).strip()
            if blk:
                blocks.append(blk)

        out = _take_top_blocks(blocks)
        return out if out else cleaned[:2000]

    # Fallback (non-code / unsupported extension): return first 2000 chars.
    return cleaned[:2000] if cleaned else content[:2000]


def _resolve_best_document(query: str) -> tuple[str | None, float]:
    """Resolve `query` to a file path and confidence score using multiple strategies.

    Returns (path, confidence).
    """
    # 1) If exact path on disk
    candidate = Path(query).expanduser()
    if candidate.exists():
        return (str(candidate), 1.0)

    # 2) Boosted RAG search
    try:
        rag = _get_rag_index()
        if rag is not None:
            cfg = get_config()
            limit = int(getattr(cfg.rag, 'max_results', 5))
            results = rag.search(query, limit=limit)
            if results:
                top = results[0]
                score = float(top.get('score', 0.0))
                path = top.get('path') or None
                if path:
                    # Confidence is normalized to 0–1; callers may decide to
                    # ask for clarification if confidence is low.
                    return (path, score / 100.0)
    except Exception as e:
        logger.warning('rag_resolution_failed', error=str(e), exc_info=True)

    # 3) Fuzzy filename match across indexed results
    try:
        rag = _get_rag_index()
        if rag is not None:
            from rapidfuzz import fuzz

            filenames: list[str] = []
            try:
                filenames = rag.get_all_indexed_filenames()  # type: ignore[attr-defined]
            except Exception:
                filenames = []

            if not filenames:
                return (None, 0.0)

            best_path = None
            best_r = 0
            q = (query or "").lower()
            for path in filenames:
                base = os.path.splitext(os.path.basename(path))[0]
                r = fuzz.token_set_ratio(q, (base or "").lower())
                if r > best_r:
                    best_r = r
                    best_path = path

            if best_path and best_r >= 75:
                return (best_path, best_r / 100.0)
    except Exception:
        pass

    return (None, 0.0)


def read_document(path: str, max_chars: int = 12000) -> str:
    """Read a document using the universal file reader and return up to `max_chars`."""
    if not path:
        return "You must provide a file path."
    text = _read_file_as_text(path)
    if text.startswith("File not found") or text.startswith("I could not") or text.startswith("No readable"):
        return text
    if len(text) > max_chars:
        omitted = len(text) - max_chars
        return text[:max_chars] + f"\n...[truncated — {omitted} chars omitted]"
    return text


def summarize_document(path: str, max_bullets: int = 8) -> str:
    """Provide a lightweight extractive summary for common document formats."""
    text = read_document(path, max_chars=40000)
    if not text or text.startswith("I could not") or text.startswith("Unsupported"):
        return text

    sentences = _split_sentences(text)
    if not sentences:
        return text[:1000]

    scores = _score_sentences(sentences)
    ranked = sorted(range(len(sentences)), key=lambda idx: scores[idx], reverse=True)[: max(1, max_bullets)]
    ranked.sort()
    summary = [f"- {sentences[idx].strip()}" for idx in ranked if sentences[idx].strip()]
    if not summary:
        return text[:1000]
    return "Summary:\n" + "\n".join(summary)


def answer_document_question(path: str, question: str) -> str:
    """Answer a question about a file or project.

    Accepts a full path or a project name / partial description. If no exact path
    is given, the tool finds the most relevant file using semantic search.
    """
    # Resolve the best document path and a confidence score
    resolved_path, confidence = _resolve_best_document(path)
    if not resolved_path:
        # Try to offer top candidates from the RAG index
        try:
            rag = _get_rag_index()
            candidates = rag.search(path, limit=3) if rag is not None else []
            if candidates:
                names = [os.path.basename(c.get('path') or c.get('title') or '') for c in candidates]
                return f"I wasn't confident which file you meant. Top matches: {', '.join(names)}. Which one should I read?"
        except Exception:
            pass
        return f"I could not find a file matching {path!r}."

    # Read the file as text
    text = _read_file_as_text(resolved_path)
    if not text or text.startswith("I could not") or text.startswith("No readable"):
        return text

    ext = Path(resolved_path).suffix.lower()
    section = _extract_relevant_section(text, question, ext)

    # Confidence-based behavior
    if confidence < 0.5:
        # Clarify: list top 3 candidates for user to choose
        try:
            rag = _get_rag_index()
            candidates = rag.search(path, limit=3) if rag is not None else []
            names = [os.path.basename(c.get('path') or c.get('title') or '') for c in candidates]
            return f"I'm not confident which file you mean. Top candidates: {', '.join(names)}. Please confirm which file to read."
        except Exception:
            return "I'm not confident which file you mean. Could you provide the file path or more details?"

    prefix = ""
    if 0.5 <= confidence < 0.7:
        prefix = f"[Reading from {os.path.basename(resolved_path)} — confirm if that's not right]\n\n"

    # If confident enough, set session-scoped current_project (>= 0.65)
    try:
        if confidence >= 0.65:
            session_state.set_current_project(
                name=os.path.splitext(os.path.basename(resolved_path))[0],
                resolved_path=resolved_path,
                confidence=confidence,
                # Sentinel: "just set" so pipeline can record a real turn number later.
                set_at_turn=None,
            )
    except Exception:
        logger.debug('set_current_project_failed', path=resolved_path, confidence=confidence)

    # Produce an extractive answer; prefer concise summarization
    if not section:
        return prefix + summarize_document(resolved_path)

    # For code, present the extracted blocks as relevant excerpts
    code_exts = {".java", ".py", ".js", ".ts", ".cs", ".go", ".cpp", ".kt", ".rb", ".rs"}
    if ext in code_exts:
        excerpt = section.strip()
        # Short spoken preview first, then offer to read more
        preview = "\n".join(excerpt.splitlines()[:10])
        return prefix + "Relevant code excerpt:\n" + preview

    # Non-code: summarize the extracted section
    # Reuse sentence scoring heuristic
    sentences = _split_sentences(section)
    if not sentences:
        return prefix + section[:1000]
    scores = _score_sentences(sentences)
    ranked = sorted(range(len(sentences)), key=lambda idx: scores[idx], reverse=True)[:3]
    ranked.sort()
    summary = " ".join(sentences[idx].strip() for idx in ranked if sentences[idx].strip())
    return prefix + "Summary: " + summary


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
    return [part.strip() for part in parts if part.strip()]


def _score_sentences(sentences: list[str]) -> list[float]:
    words = Counter()
    for sentence in sentences:
        for token in re.findall(r"[A-Za-z0-9']+", sentence.lower()):
            if len(token) > 2:
                words[token] += 1
    scores = []
    for sentence in sentences:
        tokens = [token.lower() for token in re.findall(r"[A-Za-z0-9']+", sentence) if len(token) > 2]
        score = sum(words[token] for token in tokens)
        scores.append(float(score))
    return scores
