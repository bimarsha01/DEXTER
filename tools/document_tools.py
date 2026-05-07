from __future__ import annotations

import csv
import json
import os
import re
from collections import Counter
from pathlib import Path

from utils.logger import get_logger

logger = get_logger("document_tools")


SUPPORTED_TEXT_EXTENSIONS = {".txt", ".md", ".py", ".json", ".yaml", ".yml", ".csv", ".log", ".ini", ".cfg", ".toml"}


def read_document(path: str, max_chars: int = 12000) -> str:
    """Read text from common document formats with graceful optional dependency fallbacks."""
    if not path:
        return "You must provide a file path."

    file_path = Path(path).expanduser()
    if not file_path.exists():
        return f"File not found: {path}"

    try:
        ext = file_path.suffix.lower()
        if ext in SUPPORTED_TEXT_EXTENSIONS:
            if ext == ".csv":
                with open(file_path, "r", encoding="utf-8", errors="replace", newline="") as handle:
                    rows = list(csv.reader(handle))
                text = "\n".join([", ".join(row) for row in rows])
            else:
                with open(file_path, "r", encoding="utf-8", errors="replace") as handle:
                    text = handle.read()
        elif ext == ".docx":
            try:
                from docx import Document  # type: ignore

                document = Document(str(file_path))
                text = "\n".join(paragraph.text for paragraph in document.paragraphs if paragraph.text.strip())
            except Exception as e:
                logger.warning("docx_read_unavailable", error=str(e))
                return "python-docx is not available or the document could not be read."
        elif ext == ".xlsx":
            try:
                from openpyxl import load_workbook  # type: ignore

                workbook = load_workbook(str(file_path), data_only=True)
                lines = []
                for sheet in workbook.worksheets:
                    lines.append(f"Sheet: {sheet.title}")
                    for row in sheet.iter_rows(values_only=True):
                        values = [str(cell) for cell in row if cell is not None]
                        if values:
                            lines.append(" | ".join(values))
                text = "\n".join(lines)
            except Exception as e:
                logger.warning("xlsx_read_unavailable", error=str(e))
                return "openpyxl is not available or the spreadsheet could not be read."
        elif ext == ".pdf":
            try:
                from pypdf import PdfReader  # type: ignore

                reader = PdfReader(str(file_path))
                pages = []
                for page in reader.pages:
                    pages.append(page.extract_text() or "")
                text = "\n".join(page for page in pages if page.strip())
            except Exception as e:
                logger.warning("pdf_read_unavailable", error=str(e))
                return "pypdf is not available or the PDF could not be read."
        else:
            return f"Unsupported document format: {ext or 'unknown'}"

        if not text:
            return f"No readable text found in {file_path.name}."
        if len(text) > max_chars:
            return text[:max_chars] + "\n...[truncated]"
        return text
    except Exception as e:
        logger.error("document_read_failed", path=str(file_path), error=str(e), exc_info=True)
        return f"I could not read {file_path.name}."


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
    """Return the most relevant excerpts from a document for a question."""
    text = read_document(path, max_chars=50000)
    if not text or text.startswith("I could not") or text.startswith("Unsupported"):
        return text

    question_terms = {token.lower() for token in re.findall(r"[A-Za-z0-9']+", question) if len(token) > 2}
    if not question_terms:
        return summarize_document(path)

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    scored = []
    for line in lines:
        line_terms = {token.lower() for token in re.findall(r"[A-Za-z0-9']+", line) if len(token) > 2}
        overlap = len(question_terms & line_terms)
        if overlap:
            scored.append((overlap, line))

    if not scored:
        return summarize_document(path)

    scored.sort(key=lambda item: item[0], reverse=True)
    excerpts = [line for _, line in scored[:5]]
    return "Relevant excerpts:\n" + "\n".join(f"- {line}" for line in excerpts)


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
