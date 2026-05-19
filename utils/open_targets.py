from __future__ import annotations

import csv
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Iterable

from rapidfuzz import fuzz, process
from utils.logger import get_logger

logger = get_logger("open_targets")

_START_MENU_DIRS = [
    os.path.join(os.environ.get("ProgramData", "C:\\ProgramData"), "Microsoft", "Windows", "Start Menu", "Programs"),
    os.path.join(os.environ.get("APPDATA", ""), "Microsoft", "Windows", "Start Menu", "Programs"),
]

_DESKTOP_DIR = os.path.join(os.path.expanduser("~"), "Desktop")
_DOCUMENTS_DIR = os.path.join(os.path.expanduser("~"), "Documents")
_WORKSPACE_SKIP_NAMES = {".git", ".venv", "__pycache__", "node_modules"}

_SUPPORTED_WORKSPACE_FILE_EXTS = {
    ".md",
    ".txt",
    ".pdf",
    ".docx",
    ".json",
    ".yaml",
    ".yml",
}

_SUPPORTED_START_MENU_EXTS = {".lnk", ".appref-ms", ".url", ".exe"}


@dataclass
class OpenCandidate:
    name: str
    source: str
    path: str | None = None
    kind: str = "path"


@dataclass
class MatchResult:
    candidate: OpenCandidate
    score: float


class OpenTargetIndex:
    def __init__(self, refresh_seconds: int = 60) -> None:
        self._refresh_seconds = refresh_seconds
        self._last_refresh = 0.0
        self._candidates: list[OpenCandidate] = []
        self._name_map: dict[str, OpenCandidate] = {}

    def match_one(self, query: str, score_cutoff: float) -> MatchResult | None:
        matches = self.match(query, limit=1, score_cutoff=score_cutoff)
        return matches[0] if matches else None

    def match(self, query: str, limit: int = 3, score_cutoff: float = 0.0) -> list[MatchResult]:
        candidates = self._get_candidates()
        if not candidates:
            return []

        choices = list(self._name_map.keys())
        if not choices:
            return []

        query_text = query.lower().strip()
        if not query_text:
            return []

        results = process.extract(
            query_text,
            choices,
            scorer=fuzz.WRatio,
            limit=limit,
            score_cutoff=score_cutoff,
        )
        matches: list[MatchResult] = []
        for name, score, _ in results:
            candidate = self._name_map.get(name)
            if candidate:
                matches.append(MatchResult(candidate, float(score)))
        return matches

    def list_sources(self) -> dict[str, list[str]]:
        self._get_candidates()
        sources: dict[str, list[str]] = {}
        for candidate in self._candidates:
            sources.setdefault(candidate.source, []).append(candidate.name)
        return sources

    def _get_candidates(self) -> list[OpenCandidate]:
        if time.time() - self._last_refresh > self._refresh_seconds:
            self._refresh()
        return self._candidates

    def _refresh(self) -> None:
        self._candidates = []
        self._name_map = {}

        self._add_candidates(self._collect_start_menu_entries())
        self._add_candidates(self._collect_desktop_entries())
        self._add_candidates(self._collect_workspace_entries())
        self._add_candidates(self._collect_documents_entries())
        self._add_candidates(self._collect_process_entries())

        self._last_refresh = time.time()
        logger.debug("open_targets_refreshed", count=len(self._candidates))

    def _add_candidates(self, entries: Iterable[OpenCandidate]) -> None:
        for candidate in entries:
            key = candidate.name.lower().strip()
            if not key or key in self._name_map:
                continue
            self._name_map[key] = candidate
            self._candidates.append(candidate)

    def _collect_start_menu_entries(self) -> list[OpenCandidate]:
        entries: list[OpenCandidate] = []
        for root in _START_MENU_DIRS:
            if not root or not os.path.isdir(root):
                continue
            for dirpath, _, filenames in os.walk(root):
                for filename in filenames:
                    ext = os.path.splitext(filename)[1].lower()
                    if ext not in _SUPPORTED_START_MENU_EXTS:
                        continue
                    name = os.path.splitext(filename)[0]
                    path = os.path.join(dirpath, filename)
                    entries.append(OpenCandidate(name=name, source="start_menu", path=path, kind="path"))
        return entries

    def _collect_desktop_entries(self) -> list[OpenCandidate]:
        entries: list[OpenCandidate] = []
        if not os.path.isdir(_DESKTOP_DIR):
            return entries
        for name in os.listdir(_DESKTOP_DIR):
            path = os.path.join(_DESKTOP_DIR, name)
            if not os.path.exists(path):
                continue
            display = os.path.splitext(name)[0] if os.path.isfile(path) else name
            entries.append(OpenCandidate(name=display, source="desktop", path=path, kind="path"))
        return entries

    def _collect_documents_entries(self) -> list[OpenCandidate]:
        entries: list[OpenCandidate] = []
        if not os.path.isdir(_DOCUMENTS_DIR):
            return entries
        for name in os.listdir(_DOCUMENTS_DIR):
            path = os.path.join(_DOCUMENTS_DIR, name)
            if os.path.isdir(path):
                entries.append(OpenCandidate(name=name, source="documents", path=path, kind="path"))
        return entries

    def _collect_workspace_entries(self) -> list[OpenCandidate]:
        entries: list[OpenCandidate] = []
        try:
            from utils.config import get_workspace_root

            workspace_root = get_workspace_root()
        except Exception:
            workspace_root = ""

        if not workspace_root or not os.path.isdir(workspace_root):
            return entries

        for name in os.listdir(workspace_root):
            if name in _WORKSPACE_SKIP_NAMES:
                continue
            path = os.path.join(workspace_root, name)
            if os.path.isdir(path):
                entries.append(OpenCandidate(name=name, source="documents", path=path, kind="path"))
                continue
            ext = os.path.splitext(name)[1].lower()
            if ext in _SUPPORTED_WORKSPACE_FILE_EXTS:
                display = os.path.splitext(name)[0]
                entries.append(OpenCandidate(name=display, source="documents", path=path, kind="path"))
        return entries

    def _collect_process_entries(self) -> list[OpenCandidate]:
        entries: list[OpenCandidate] = []
        try:
            result = subprocess.run(
                ["tasklist", "/FO", "CSV", "/NH"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                return entries
            for row in csv.reader(result.stdout.splitlines()):
                if not row:
                    continue
                image = row[0]
                name = os.path.splitext(image)[0]
                if name:
                    entries.append(OpenCandidate(name=name, source="process", path=None, kind="process"))
        except Exception as e:
            logger.debug("open_targets_process_list_failed", error=str(e))
        return entries


_INDEX = OpenTargetIndex()


def get_open_target_index() -> OpenTargetIndex:
    return _INDEX


def get_start_menu_dirs() -> list[str]:
    return [d for d in _START_MENU_DIRS if d]
