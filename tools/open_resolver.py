import os
import time
import uuid
from typing import Any

from rapidfuzz import fuzz, process
from utils.config import get_config
from utils.logger import get_logger
from utils.open_targets import get_open_target_index, get_start_menu_dirs

logger = get_logger("open_resolver")

_STRONG_MATCH = 85.0
_REASONABLE_MATCH = 70.0
_OPTION_TTL_SECONDS = 45

_PENDING_MATCHES: dict[str, dict[str, Any]] = {}


def resolve_open_target(query: str | None = None, match_id: str | None = None, selection: str | None = None) -> dict:
    """Resolve and open an app/folder/file based on fuzzy matches across the system."""
    if match_id and selection:
        return _resolve_selection(match_id, selection)

    if not query or not query.strip():
        return {
            "status": "error",
            "message": "Please tell me what to open.",
        }

    index = get_open_target_index()
    matches = index.match(query, limit=3, score_cutoff=_REASONABLE_MATCH)
    if not matches:
        return {
            "status": "not_found",
            "message": "I could not find anything matching that name. Could you spell it out or describe it differently?",
        }

    best = matches[0]
    second_score = matches[1].score if len(matches) > 1 else 0.0
    if best.score >= _STRONG_MATCH and (len(matches) == 1 or (best.score - second_score) >= 8.0):
        opened = _open_candidate(best.candidate)
        return {
            "status": "open",
            "message": opened,
            "selected": _candidate_payload(best.candidate),
        }

    match_id = uuid.uuid4().hex
    _PENDING_MATCHES[match_id] = {
        "options": matches,
        "expires_at": time.time() + _OPTION_TTL_SECONDS,
    }
    prompt = _format_options(matches)
    return {
        "status": "ask",
        "message": prompt,
        "match_id": match_id,
        "options": [_candidate_payload(m.candidate, m.score) for m in matches],
    }


def _resolve_selection(match_id: str, selection: str) -> dict:
    payload = _PENDING_MATCHES.get(match_id)
    if not payload:
        return {
            "status": "not_found",
            "message": "That request expired. Please try again.",
        }

    if time.time() > payload.get("expires_at", 0):
        _PENDING_MATCHES.pop(match_id, None)
        return {
            "status": "not_found",
            "message": "That request expired. Please try again.",
        }

    options = payload.get("options", [])
    if not options:
        return {
            "status": "not_found",
            "message": "I could not find any options for that request.",
        }

    selection_text = selection.strip().lower()
    index = _parse_option_index(selection_text)
    if index is not None and 0 <= index < len(options):
        chosen = options[index]
        _PENDING_MATCHES.pop(match_id, None)
        message = _open_candidate(chosen.candidate)
        return {
            "status": "open",
            "message": message,
            "selected": _candidate_payload(chosen.candidate, chosen.score),
        }

    names = [m.candidate.name for m in options]
    match = process.extractOne(selection_text, names, scorer=fuzz.WRatio)
    if match:
        name, score, idx = match
        if score >= _REASONABLE_MATCH:
            chosen = options[idx]
            _PENDING_MATCHES.pop(match_id, None)
            message = _open_candidate(chosen.candidate)
            return {
                "status": "open",
                "message": message,
                "selected": _candidate_payload(chosen.candidate, chosen.score),
            }

    return {
        "status": "ask",
        "message": "Please say the option number or name.",
        "match_id": match_id,
    }


def _parse_option_index(text: str) -> int | None:
    mapping = {
        "one": 0,
        "1": 0,
        "option 1": 0,
        "two": 1,
        "2": 1,
        "option 2": 1,
        "three": 2,
        "3": 2,
        "option 3": 2,
    }
    return mapping.get(text)


def _format_options(matches) -> str:
    lines = ["I found a few matches. Did you mean:"]
    for idx, match in enumerate(matches, start=1):
        candidate = match.candidate
        label = _describe_candidate(candidate)
        lines.append(f"Option {idx}, {label}.")
    lines.append("Please say the number or the name.")
    return " ".join(lines)


def _describe_candidate(candidate) -> str:
    if candidate.source == "desktop":
        return f"{candidate.name} on Desktop"
    if candidate.source == "documents":
        return f"{candidate.name} in Documents"
    if candidate.source == "start_menu":
        return f"{candidate.name} in Programs"
    if candidate.source == "process":
        return f"{candidate.name} (already running)"
    return candidate.name


def _open_candidate(candidate) -> str:
    if candidate.source == "process":
        return f"{candidate.name} is already running."

    if candidate.path:
        if not _is_path_allowed(candidate.path):
            return "That location is not allowed by the current security policy."
        try:
            os.startfile(candidate.path)
            return f"Opened {candidate.name}."
        except Exception as e:
            logger.error("open_target_failed", name=candidate.name, error=str(e), exc_info=True)
            return f"I could not open {candidate.name}."

    return f"I could not open {candidate.name}."


def _is_path_allowed(path: str) -> bool:
    config = get_config()
    allowed_roots = config.security.allowed_file_roots or []
    expanded = []
    for root in allowed_roots:
        expanded.append(os.path.abspath(os.path.expandvars(os.path.expanduser(str(root)))))

    expanded.extend(get_start_menu_dirs())
    expanded.append(os.path.join(os.path.expanduser("~"), "Desktop"))
    expanded.append(os.path.join(os.path.expanduser("~"), "Documents"))

    path = os.path.abspath(path)
    for root in expanded:
        if not root:
            continue
        try:
            if os.path.commonpath([path, os.path.abspath(root)]) == os.path.abspath(root):
                return True
        except ValueError:
            continue
    return False


def _candidate_payload(candidate, score: float | None = None) -> dict:
    payload = {
        "name": candidate.name,
        "source": candidate.source,
        "path": candidate.path,
    }
    if score is not None:
        payload["score"] = score
    return payload
