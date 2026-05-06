from __future__ import annotations

import re
from dataclasses import dataclass

from utils.logger import get_logger
from utils.open_targets import get_open_target_index

logger = get_logger("transcript_correction")


@dataclass
class CorrectionResult:
    original: str
    corrected: str
    matched_name: str | None = None
    score: float | None = None


class TranscriptCorrector:
    def __init__(self, score_threshold: float = 86.0, max_span_tokens: int = 4) -> None:
        self._index = get_open_target_index()
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

                corrected = self._replace_span(text, matches, start, end, match.candidate.name)
                if corrected != text:
                    logger.info(
                        "transcript_corrected",
                        original=text,
                        corrected=corrected,
                        matched=match.candidate.name,
                        score=match.score,
                    )
                    return CorrectionResult(
                        original=text,
                        corrected=corrected,
                        matched_name=match.candidate.name,
                        score=match.score,
                    )

        return CorrectionResult(original=text, corrected=text)

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
