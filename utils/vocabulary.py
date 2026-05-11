import os
from pathlib import Path
from typing import Set


def build_whisper_vocabulary(config) -> str:
    """
    Build a dynamic initial_prompt for Whisper by scanning the user's
    configured personal_roots (Desktop/Documents/Projects) and extracting
    top-level folder names and one-level subfolder names. Returns a comma
    separated prompt string limited to ~80 terms for efficiency.
    """
    vocabulary_terms: Set[str] = set()

    base_terms = [
        "Dexter", "Chrome", "Spotify", "Firefox", "YouTube", "Netflix",
        "GitHub", "IntelliJ", "VS Code", "Windows", "Desktop", "Documents",
        "screenshot", "volume", "weather", "summarize", "describe", "open",
        "close", "Spring Boot", "Java", "Python", "JavaScript",
    ]
    for t in base_terms:
        vocabulary_terms.add(t)

    try:
        roots = list(getattr(config, "rag", {}).personal_roots) if config else []
    except Exception:
        try:
            roots = getattr(config, "rag").personal_roots if config and hasattr(config, "rag") else []
        except Exception:
            roots = []

    scan_roots = []
    for root_str in (roots or []):
        try:
            expanded = str(Path(os.path.expandvars(root_str.replace('%USERPROFILE%', str(Path.home())))).expanduser())
            p = Path(expanded)
            if p.exists() and p.is_dir():
                scan_roots.append(p)
        except Exception:
            continue

    for root in scan_roots:
        try:
            for item in root.iterdir():
                if item.is_dir():
                    name = item.name.replace('-', ' ').replace('_', ' ')
                    vocabulary_terms.add(name)
                    try:
                        for sub in item.iterdir():
                            if sub.is_dir():
                                subname = sub.name.replace('-', ' ').replace('_', ' ')
                                vocabulary_terms.add(subname)
                    except PermissionError:
                        continue
        except PermissionError:
            continue
        except Exception:
            continue

    # Sort and trim to reasonable size (approx 80 terms)
    terms_list = sorted(vocabulary_terms)
    prompt = ", ".join(terms_list[:80])
    return prompt
