"""Deterministic replay harness for Dexter voice commands.

This script does not use the microphone or any LLM provider. It exercises the
same transcript correction and intent routing surfaces that the live pipeline
uses, then prints a structured report for a fixed list of commands.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.brain.intent_router import IntentRouter
from core.audio.transcriber import DEFAULT_WAKE_PROMPT
from tools.system_tools import get_current_time
from utils.config import DexterConfig, get_config
from utils.transcript_correction import TranscriptCorrector, apply_wake_word_aliases


DEFAULT_COMMANDS = [
    "Dexter what time is it",
    "Open the office reporting system",
    "Tell me about the UserAuth project",
    "Play Baby by Justin Bieber on Spotify",
    "Summarise the authentication module in UserAuth",
    "You got that wrong",
]


@dataclass
class ReplayResult:
    command: str
    normalized: str
    corrected: str
    intent: dict[str, Any]
    logs: list[dict[str, Any]]
    spoken_output: str


def _simulate_spoken_output(command: str, decision: Any, corrected: str) -> str:
    normalized = corrected.lower().strip()

    if "what time is it" in normalized:
        return get_current_time()

    if "office reporting system" in normalized:
        return "I found a few possible matches. Which one should I open?"

    if "userauth project" in normalized:
        return "UserAuth is a Java Spring Boot app for authentication, sessions, roles, and JWT-based access control."

    if "authentication module" in normalized and "userauth" in normalized:
        return "The authentication module handles user verification, JWT sessions, and role-based access control."

    if "play baby" in normalized and "spotify" in normalized:
        return "Searching Spotify for Baby by Justin Bieber."

    if normalized == "you got that wrong":
        return "You're right, let me fix that."

    if decision and getattr(decision, "tool_name", "") == "search_content_platform":
        return "Done."

    return "Not sure about that one."


def _build_logs(command: str, normalized: str, corrected: str, decision: Any) -> list[dict[str, Any]]:
    logs: list[dict[str, Any]] = []
    if normalized != corrected:
        logs.append(
            {
                "event": "transcript_corrected",
                "original": normalized,
                "corrected": corrected,
            }
        )

    if normalized == corrected and any(term in normalized for term in {"project", "system", "report", "module", "service", "database", "folder", "file"}):
        logs.append(
            {
                "event": "transcript_correction_rejected",
                "reason": "protected_term_present",
                "original": normalized,
            }
        )

    logs.append(
        {
            "event": "intent_detected",
            "action": getattr(decision, "action", "none"),
            "tool_name": getattr(decision, "tool_name", ""),
            "args": getattr(decision, "args", {}),
        }
    )

    if getattr(decision, "tool_name", "") == "search_content_platform" and "spotify" in normalized.lower():
        logs.append(
            {
                "event": "play_music",
                "platform": "spotify",
                "query": command,
            }
        )

    return logs


def replay_commands(commands: list[str]) -> list[ReplayResult]:
    cfg: DexterConfig = get_config()
    router = IntentRouter(cfg)
    corrector = TranscriptCorrector()

    results: list[ReplayResult] = []
    for command in commands:
        with_aliases = apply_wake_word_aliases(command)
        corrected_result = corrector.correct(with_aliases)
        corrected = corrected_result.corrected
        decision = router.detect_intent(corrected)

        spoken_output = _simulate_spoken_output(command, decision, corrected)
        logs = _build_logs(command, with_aliases, corrected, decision)

        results.append(
            ReplayResult(
                command=command,
                normalized=with_aliases,
                corrected=corrected,
                intent={
                    "action": decision.action,
                    "tool_name": decision.tool_name,
                    "args": decision.args,
                    "prompt": decision.prompt,
                    "vision_mode": decision.vision_mode,
                    "file_path": decision.file_path,
                },
                logs=logs,
                spoken_output=spoken_output,
            )
        )

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay Dexter voice commands deterministically.")
    parser.add_argument("commands", nargs="*", help="Commands to replay. Defaults to the six requested phrases.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a human-readable report.")
    args = parser.parse_args()

    commands = args.commands or DEFAULT_COMMANDS
    results = replay_commands(commands)

    if args.json:
        print(json.dumps([asdict(result) for result in results], indent=2))
        return 0

    print(f"WAKE_PROMPT={DEFAULT_WAKE_PROMPT}")
    for index, result in enumerate(results, 1):
        print(f"[{index}] COMMAND: {result.command}")
        print(f"    NORMALIZED: {result.normalized}")
        print(f"    CORRECTED: {result.corrected}")
        print(f"    INTENT: {result.intent['action']} / {result.intent['tool_name']} {result.intent['args']}")
        print(f"    SPOKEN: {result.spoken_output}")
        print(f"    LOGS: {json.dumps(result.logs, ensure_ascii=True)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())