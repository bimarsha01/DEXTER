"""
Dynamic ASR correction engine.
Corrections are learned at runtime and stored in data/asr_corrections.json.
No corrections are hardcoded in source code.
"""

import json
import os
import time
from pathlib import Path

from rapidfuzz import fuzz, process

from utils.logger import get_logger

logger = get_logger("asr_corrections")

_DATA_DIR = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) / "data"
CORRECTIONS_FILE = _DATA_DIR / "asr_corrections.json"
VOCABULARY_FILE = _DATA_DIR / "known_vocabulary.json"


class ASRCorrectionEngine:
    """
    Learns and applies ASR corrections dynamically.

    Three sources of corrections (applied in priority order):
    1. User-confirmed corrections (highest priority, manually approved)
    2. Auto-learned corrections (from context inference)
    3. Fuzzy vocabulary matching (lowest priority, always on)
    """

    def __init__(self) -> None:
        self._confirmed: dict[str, str] = {}
        self._auto_learned: dict[str, str] = {}
        self._vocabulary: set[str] = set()
        self._correction_counts: dict[str, int] = {}
        self._load()

    # ── Persistence ────────────────────────────────────────────────

    def _load(self) -> None:
        """Load saved corrections and vocabulary from disk."""
        if CORRECTIONS_FILE.exists():
            try:
                data = json.loads(CORRECTIONS_FILE.read_text(encoding="utf-8"))
                self._confirmed = data.get("confirmed", {})
                self._auto_learned = data.get("auto_learned", {})
                self._correction_counts = data.get("counts", {})
                logger.info(
                    "asr_corrections_loaded",
                    confirmed=len(self._confirmed),
                    auto_learned=len(self._auto_learned),
                )
            except Exception as e:
                logger.warning("asr_corrections_load_failed", error=str(e))

        if VOCABULARY_FILE.exists():
            try:
                data = json.loads(VOCABULARY_FILE.read_text(encoding="utf-8"))
                self._vocabulary = set(data.get("vocabulary", []))
                logger.info("asr_vocabulary_loaded", terms=len(self._vocabulary))
            except Exception as e:
                logger.warning("asr_vocabulary_load_failed", error=str(e))

    def _save(self) -> None:
        """Persist corrections to disk."""
        CORRECTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
        CORRECTIONS_FILE.write_text(
            json.dumps(
                {
                    "confirmed": self._confirmed,
                    "auto_learned": self._auto_learned,
                    "counts": self._correction_counts,
                    "last_updated": time.time(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    # ── Vocabulary ─────────────────────────────────────────────────

    def add_vocabulary(self, terms: list[str]) -> None:
        """
        Add known terms to the vocabulary pool.
        Call this when indexing documents, scanning installed apps,
        reading folder names, etc.
        """
        before = len(self._vocabulary)
        self._vocabulary.update(t.lower() for t in terms if t and len(t) >= 3)
        added = len(self._vocabulary) - before
        if added > 0:
            VOCABULARY_FILE.parent.mkdir(parents=True, exist_ok=True)
            VOCABULARY_FILE.write_text(
                json.dumps(
                    {"vocabulary": sorted(self._vocabulary), "count": len(self._vocabulary)},
                    indent=2,
                ),
                encoding="utf-8",
            )
            logger.info("asr_vocabulary_updated", added=added, total=len(self._vocabulary))

    # ── Learning ───────────────────────────────────────────────────

    def confirm_correction(self, wrong: str, right: str) -> None:
        """
        User explicitly confirmed that 'wrong' should be 'right'.
        Called when user says "I meant X" or "no, X not Y".
        This correction has highest priority and never expires.
        """
        key = wrong.lower().strip()
        if not key or not right:
            return
        self._confirmed[key] = right
        self._correction_counts[key] = self._correction_counts.get(key, 0) + 1
        self._save()
        logger.info(
            "asr_correction_confirmed",
            wrong=wrong,
            right=right,
            total_confirmations=self._correction_counts[key],
        )

    def learn_correction(self, wrong: str, right: str, confidence: float) -> None:
        """
        System inferred a correction from context.
        Only stored if confidence > 0.85. Requires 3 occurrences before reliable.
        """
        if confidence < 0.85:
            return
        key = wrong.lower().strip()
        if not key:
            return
        self._correction_counts[key] = self._correction_counts.get(key, 0) + 1
        if self._correction_counts[key] >= 3:
            self._auto_learned[key] = right
            self._save()
            logger.info(
                "asr_correction_auto_learned",
                wrong=wrong,
                right=right,
                occurrences=self._correction_counts[key],
            )

    # ── Correction ─────────────────────────────────────────────────

    def correct(self, text: str) -> tuple[str, bool]:
        """
        Apply corrections to transcript text.
        Returns (corrected_text, was_corrected).

        Priority:
        1. Exact match in confirmed corrections
        2. Exact match in auto-learned corrections
        3. Fuzzy match against vocabulary (only for likely mishearings)
        """
        if not text or not text.strip():
            return text, False

        text_lower = text.lower()

        # Priority 1: confirmed corrections (exact substring match)
        for wrong, right in self._confirmed.items():
            if wrong in text_lower:
                corrected = text_lower.replace(wrong, right.lower())
                result = self._restore_case(text, corrected)
                logger.info("asr_correction_applied", source="confirmed", wrong=wrong, right=right)
                return result, True

        # Priority 2: auto-learned corrections
        for wrong, right in self._auto_learned.items():
            if wrong in text_lower:
                corrected = text_lower.replace(wrong, right.lower())
                result = self._restore_case(text, corrected)
                logger.info("asr_correction_applied", source="auto_learned", wrong=wrong, right=right)
                return result, True

        # Priority 3: fuzzy vocabulary matching
        if not self._vocabulary:
            return text, False

        words = text.split()
        corrected_words = []
        changed = False

        for word in words:
            clean = word.lower().strip(".,!?")
            if len(clean) < 4:
                corrected_words.append(word)
                continue

            if self._is_protected_word(clean):
                corrected_words.append(word)
                continue

            match = process.extractOne(
                clean,
                self._vocabulary,
                scorer=fuzz.ratio,
                score_cutoff=88,  # high threshold for unsupervised fuzzy
            )

            if match and match[0] != clean:
                corrected_words.append(match[0])
                changed = True
                logger.debug("asr_fuzzy_corrected", original=clean, corrected=match[0], score=match[1])
            else:
                corrected_words.append(word)

        result = " ".join(corrected_words)
        return result, changed

    # ── Helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _is_protected_word(word: str) -> bool:
        """Words that should never be fuzzy-corrected."""
        _PROTECTED = {
            "project", "folder", "system", "file", "report",
            "module", "service", "auth", "manager", "controller",
            "open", "close", "play", "stop", "the", "and", "for",
            "dexter", "weather", "time", "volume", "search",
        }
        return word in _PROTECTED

    @staticmethod
    def _restore_case(original: str, corrected: str) -> str:
        """Attempt to match the casing style of the original."""
        if not corrected:
            return corrected
        if original and original[0].isupper():
            return corrected[0].upper() + corrected[1:]
        return corrected

    def get_stats(self) -> dict:
        """Return current engine statistics."""
        return {
            "confirmed_corrections": len(self._confirmed),
            "auto_learned_corrections": len(self._auto_learned),
            "vocabulary_size": len(self._vocabulary),
        }
