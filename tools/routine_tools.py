from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

from utils.logger import get_logger

logger = get_logger("routine_tools")

ROUTINES_DIR = Path(os.path.expanduser("~")) / "Documents" / "DexterRoutines"


def save_automation_routine(name: str, steps: list[dict], description: str = "") -> str:
    """Persist a sequence of tool calls for later replay."""
    if not name or not name.strip():
        return "Please provide a routine name."
    if not isinstance(steps, list) or not steps:
        return "Please provide at least one step."

    ROUTINES_DIR.mkdir(parents=True, exist_ok=True)
    routine_path = ROUTINES_DIR / f"{_safe_name(name)}.json"
    payload = {
        "name": name.strip(),
        "description": description.strip(),
        "created_at": time.time(),
        "steps": steps,
    }
    try:
        with open(routine_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        logger.info("routine_saved", path=str(routine_path), steps=len(steps))
        return f"Saved routine '{name}'."
    except Exception as e:
        logger.error("routine_save_failed", error=str(e), exc_info=True)
        return f"I could not save the routine: {str(e)}"


def list_automation_routines() -> str:
    """List saved routines."""
    if not ROUTINES_DIR.exists():
        return "You have no saved routines yet."
    names = sorted(path.stem for path in ROUTINES_DIR.glob("*.json"))
    return "Saved routines: " + ", ".join(names) if names else "You have no saved routines yet."


def run_automation_routine(name: str) -> str:
    """Replay a saved routine by executing each stored step in order."""
    routine_path = ROUTINES_DIR / f"{_safe_name(name)}.json"
    if not routine_path.exists():
        return f"I could not find a routine named {name}."

    try:
        with open(routine_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        steps = payload.get("steps") or []
        if not steps:
            return f"Routine {name} has no steps."

        async def _execute_steps() -> list[str]:
            from tools.registry import EXECUTOR, load_tools

            load_tools()
            results = []
            for index, step in enumerate(steps, start=1):
                tool_name = step.get("tool")
                args = step.get("args") or {}
                if not tool_name:
                    results.append(f"Step {index}: missing tool name")
                    continue
                result = await EXECUTOR.execute(tool_name, args)
                if result.success:
                    results.append(f"Step {index}: {tool_name} -> ok")
                else:
                    results.append(f"Step {index}: {tool_name} -> {result.error or 'failed'}")
                    break
            return results

        results = asyncio.run(_execute_steps())
        logger.info("routine_run_completed", name=name, steps=len(steps))
        return "\n".join(results)
    except Exception as e:
        logger.error("routine_run_failed", error=str(e), exc_info=True)
        return f"I could not run the routine: {str(e)}"


def delete_automation_routine(name: str) -> str:
    path = ROUTINES_DIR / f"{_safe_name(name)}.json"
    if not path.exists():
        return f"I could not find a routine named {name}."
    try:
        path.unlink()
        return f"Deleted routine '{name}'."
    except Exception as e:
        logger.error("routine_delete_failed", error=str(e), exc_info=True)
        return f"I could not delete the routine: {str(e)}"


def _safe_name(name: str) -> str:
    return "".join(ch for ch in name.strip().lower().replace(" ", "_") if ch.isalnum() or ch in {"_", "-"})
