from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path

from core.health import get_global_health_monitor
from utils.logger import get_logger

logger = get_logger("proactive")


@dataclass
class Reminder:
    id: str
    message: str
    due_at: float
    repeat_seconds: float = 0.0
    completed: bool = False


class ProactiveAssistant:
    def __init__(self, event_bus, reminders_path: str | None = None, check_interval_seconds: int = 60, system_status_interval_seconds: int = 900) -> None:
        self.event_bus = event_bus
        self.check_interval_seconds = max(10, int(check_interval_seconds))
        self.system_status_interval_seconds = max(60, int(system_status_interval_seconds))
        self.reminders_path = Path(reminders_path or (Path(os.path.expanduser("~")) / "Documents" / "DexterReminders" / "reminders.json"))
        self.reminders_path.parent.mkdir(parents=True, exist_ok=True)
        self._last_status_emit = 0.0
        self._reminders: list[Reminder] = self._load_reminders()

    def add_reminder(self, message: str, due_in_seconds: int, repeat_seconds: int = 0) -> Reminder:
        reminder = Reminder(
            id=f"rem_{int(time.time() * 1000)}",
            message=message.strip(),
            due_at=time.time() + max(1, int(due_in_seconds)),
            repeat_seconds=max(0, int(repeat_seconds)),
        )
        self._reminders.append(reminder)
        self._save_reminders()
        return reminder

    async def run(self) -> None:
        logger.info("proactive_assistant_started")
        while True:
            try:
                self._emit_due_reminders()
                self._emit_periodic_health()
            except Exception as e:
                logger.warning("proactive_loop_failed", error=str(e), exc_info=True)
            await asyncio.sleep(self.check_interval_seconds)

    def _emit_due_reminders(self) -> None:
        now = time.time()
        changed = False
        for reminder in self._reminders:
            if reminder.completed or reminder.due_at > now:
                continue
            logger.info("proactive_reminder_due", reminder_id=reminder.id, message=reminder.message)
            self.event_bus.emit("proactive_reminder", {"id": reminder.id, "message": reminder.message})
            changed = True
            if reminder.repeat_seconds > 0:
                reminder.due_at = now + reminder.repeat_seconds
            else:
                reminder.completed = True
        if changed:
            self._save_reminders()

    def _emit_periodic_health(self) -> None:
        now = time.time()
        if now - self._last_status_emit < self.system_status_interval_seconds:
            return
        self._last_status_emit = now
        monitor = get_global_health_monitor()
        if monitor is not None:
            self.event_bus.emit("proactive_status", {"health": monitor.snapshot()})

    def _load_reminders(self) -> list[Reminder]:
        if not self.reminders_path.exists():
            return []
        try:
            with open(self.reminders_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            reminders = []
            for item in data:
                reminders.append(Reminder(**item))
            return reminders
        except Exception as e:
            logger.warning("reminders_load_failed", error=str(e), exc_info=True)
            return []

    def _save_reminders(self) -> None:
        try:
            with open(self.reminders_path, "w", encoding="utf-8") as handle:
                json.dump([asdict(reminder) for reminder in self._reminders], handle, indent=2)
        except Exception as e:
            logger.warning("reminders_save_failed", error=str(e), exc_info=True)
