import asyncio
import time
from typing import Any, Dict

from utils.logger import get_logger

logger = get_logger("event_bus")


class EventBus:
    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue] = set()
        logger.info("event_bus_initialized")

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._subscribers.discard(queue)

    def emit(self, event_type: str, payload: Dict[str, Any] | None = None) -> None:
        event = {
            "type": event_type,
            "payload": payload or {},
            "timestamp": time.time(),
        }
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning("event_bus_subscriber_queue_full", event_type=event_type)
