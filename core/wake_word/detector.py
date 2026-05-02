import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from utils.logger import get_logger

logger = get_logger("wake_word")


@dataclass
class WakeDetection:
    triggered: bool
    confidence: float
    phrase: str | None
    cleaned_text: str


class WakeWordDetector:
    def __init__(
        self,
        wake_phrases: list[str],
        match_mode: str = "prefix",
        min_confidence: float = 0.86,
        max_prefix_tokens: int = 4,
    ) -> None:
        self.wake_phrases = [self._normalize(p) for p in wake_phrases if p.strip()]
        self.match_mode = match_mode if match_mode in {"prefix", "any"} else "prefix"
        self.min_confidence = float(min_confidence)
        self.max_prefix_tokens = max(1, int(max_prefix_tokens))
        logger.info(
            "wake_word_detector_initialized",
            phrase_count=len(self.wake_phrases),
            match_mode=self.match_mode,
        )

    def detect(self, text: str) -> WakeDetection:
        if not text or not self.wake_phrases:
            return WakeDetection(False, 0.0, None, text or "")

        token_matches = list(re.finditer(r"[A-Za-z0-9']+", text))
        if not token_matches:
            return WakeDetection(False, 0.0, None, text)

        tokens = [m.group(0).lower() for m in token_matches]
        limit = len(tokens)
        start_positions = range(limit)
        if self.match_mode == "prefix":
            start_positions = range(min(limit, self.max_prefix_tokens))

        for phrase in self.wake_phrases:
            phrase_tokens = phrase.split()
            if not phrase_tokens:
                continue

            target = " ".join(phrase_tokens)
            for start in start_positions:
                end = start + len(phrase_tokens)
                if end > len(tokens):
                    continue

                candidate_tokens = tokens[start:end]
                candidate = " ".join(candidate_tokens)
                if candidate == target:
                    cleaned = self._remove_token_span(text, token_matches, start, len(phrase_tokens))
                    return WakeDetection(True, 1.0, phrase, cleaned)

                confidence = SequenceMatcher(None, target, candidate).ratio()
                if confidence >= self.min_confidence:
                    cleaned = self._remove_token_span(text, token_matches, start, len(phrase_tokens))
                    return WakeDetection(True, confidence, phrase, cleaned)

        return WakeDetection(False, 0.0, None, text)

    def _normalize(self, text: str) -> str:
        lowered = text.lower().strip()
        lowered = re.sub(r"[^a-z0-9'\s]+", " ", lowered)
        lowered = re.sub(r"\s+", " ", lowered).strip()
        return lowered

    def _remove_token_span(self, text: str, matches: list[re.Match], start: int, count: int) -> str:
        if start >= len(matches) or count <= 0:
            return text
        end = min(len(matches), start + count) - 1
        start_idx = matches[start].start()
        end_idx = matches[end].end()
        cleaned = (text[:start_idx] + text[end_idx:]).strip()
        cleaned = re.sub(r"^[\s,.\-!?]+", "", cleaned).strip()
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        return cleaned
