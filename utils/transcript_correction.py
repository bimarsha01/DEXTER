from __future__ import annotations

import re
from dataclasses import dataclass

from rapidfuzz import fuzz, process

from utils.logger import get_logger
from utils.open_targets import get_open_target_index
from tools.pc_controls import APP_MAP

logger = get_logger("transcript_correction")


WAKE_WORD_ALIASES = {
    "next up": "dexter",
    "next star": "dexter",
    "nectar": "dexter",
    "decker": "dexter",
    "next, ": "dexter, ",
    "next up,": "dexter,",
    "next star,": "dexter,",
}


def apply_wake_word_aliases(text: str) -> str:
    """Normalize common wake-word ASR aliases before activation checks."""
    if not text:
        return text

    corrected = text
    for alias, canonical in WAKE_WORD_ALIASES.items():
        pattern = re.compile(rf"\b{re.escape(alias)}\b", re.IGNORECASE)
        corrected = pattern.sub(canonical, corrected)
    return corrected


@dataclass
class CorrectionResult:
    original: str
    corrected: str
    matched_name: str | None = None
    score: float | None = None


class TranscriptCorrector:
    def __init__(self, score_threshold: float = 86.0, max_span_tokens: int = 4) -> None:
        self._index = get_open_target_index()
        self._app_names = sorted({name.lower() for name in APP_MAP.keys()})
        self._score_threshold = float(score_threshold)
        self._max_span_tokens = max(1, int(max_span_tokens))
        self._trigger_verbs = {
            "open",
            "launch",
            "start",
            "close",
            "run",
            "show",
            "play",
            "find",
            "search",
            "open",
        }
        self._stopwords = {
            "the",
            "a",
            "an",
            "my",
            "your",
            "please",
            "me",
            "to",
            "for",
            "of",
            "on",
            "in",
            "at",
            "and",
            "with",
        }
        self._mixed_command_markers = {
            "in",
            "on",
            "from",
            "to",
            "with",
            "via",
            "into",
        }
        self._protected_terms = {
            "project",
            "folder",
            "system",
            "application",
            "directory",
            "file",
            "report",
            "module",
            "service",
            "database",
            "auth",
            "reporting",
            "manager",
            "controller",
        }

    def correct(self, text: str) -> CorrectionResult:
        if not text or not text.strip():
            return CorrectionResult(original=text, corrected=text)

        matches = list(re.finditer(r"[A-Za-z0-9']+", text))
        if not matches:
            return CorrectionResult(original=text, corrected=text)

        tokens = [m.group(0) for m in matches]
        lowered = [t.lower() for t in tokens]

        for idx, token in enumerate(lowered):
            if token not in self._trigger_verbs:
                continue
            start = idx + 1
            if start >= len(tokens):
                continue

            for end in range(min(len(tokens), start + self._max_span_tokens), start, -1):
                phrase_tokens = lowered[start:end]
                if not phrase_tokens or all(t in self._stopwords for t in phrase_tokens):
                    continue
                if any(token in self._mixed_command_markers for token in phrase_tokens[:-1]):
                    continue
                if len(phrase_tokens) == 1 and phrase_tokens[0] != tokens[start].lower():
                    continue
                phrase = " ".join(phrase_tokens)

                match = self._index.match_one(phrase, score_cutoff=self._score_threshold)
                if not match:
                    continue
                if match.candidate.source == "process":
                    candidate_name = match.candidate.name.lower().strip()
                    if candidate_name not in self._app_names and " " not in candidate_name:
                        continue

                if not self._should_apply_replacement(phrase, match.candidate.name):
                    continue

                corrected = self._replace_span(text, matches, start, end, match.candidate.name)
                if corrected != text:
                    logger.info("transcript_corrected", original=text, corrected=corrected, score=match.score)
                    return CorrectionResult(
                        original=text,
                        corrected=corrected,
                        matched_name=match.candidate.name,
                        score=match.score,
                    )

        app_corrected = self._correct_app_name_command(text, matches, lowered)
        if app_corrected and app_corrected.corrected != text:
            return app_corrected

        return CorrectionResult(original=text, corrected=text)

    def _correct_app_name_command(self, text: str, matches: list[re.Match], lowered: list[str]) -> CorrectionResult | None:
        if not lowered:
            return None

        if lowered[0] not in self._trigger_verbs:
            return None

        # Do not rewrite richer commands like "open YouTube in Google Chrome".
        # These should be handled by the intent router, not app-name correction.
        if any(token in {"in", "on", "from", "via"} for token in lowered[2:]):
            return None

        tail = " ".join(lowered[1:]).strip()
        if not tail:
            return None

        match = process.extractOne(tail, self._app_names, scorer=fuzz.WRatio)
        if not match:
            return None

        app_name, score, _ = match
        if score < self._score_threshold:
            return None

        if not self._should_apply_replacement(tail, app_name):
            return None

        corrected = self._replace_span(text, matches, 1, len(matches), app_name)
        if corrected == text:
            return None

        logger.info("transcript_corrected", original=text, corrected=corrected, score=score)
        return CorrectionResult(original=text, corrected=corrected, matched_name=app_name, score=score)

    def _should_apply_replacement(self, original_phrase: str, replacement: str) -> bool:
        original = (original_phrase or "").strip()
        candidate = (replacement or "").strip()
        if not original or not candidate:
            return False

        original_words = [token for token in re.findall(r"[A-Za-z0-9']+", original.lower()) if token]
        if any(token in self._protected_terms for token in original_words):
            logger.debug(
                "transcript_correction_rejected",
                reason="protected_term_present",
                original=original_phrase,
                candidate=replacement,
            )
            return False

        ratio = len(candidate) / max(1, len(original))
        if ratio < 0.6:
            logger.debug(
                "transcript_correction_rejected",
                reason="length_ratio_too_low",
                original=original_phrase,
                candidate=replacement,
            )
            return False

        replacement_words = re.findall(r"[A-Za-z0-9']+", candidate)
        if len(original_words) >= 3 and len(replacement_words) == 1:
            logger.debug(
                "transcript_correction_rejected",
                reason="multiword_to_single_word",
                original=original_phrase,
                candidate=replacement,
            )
            return False

        return True

    def _replace_span(
        self,
        text: str,
        matches: list[re.Match],
        start_token: int,
        end_token: int,
        replacement: str,
    ) -> str:
        start_idx = matches[start_token].start()
        end_idx = matches[end_token - 1].end()
        return (text[:start_idx] + replacement + text[end_idx:]).strip()
