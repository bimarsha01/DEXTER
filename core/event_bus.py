import asyncio
import time
import uuid
from typing import Any, Dict

from utils.logger import get_logger, get_correlation_id

logger = get_logger("event_bus")


class EventBus:
    def __init__(self, maxsize: int = 200) -> None:
        self._subscribers: set[asyncio.Queue] = set()
        self._default_maxsize = maxsize
        logger.info("event_bus_initialized", maxsize=maxsize)

    def subscribe(self, maxsize: int | None = None) -> asyncio.Queue:
        size = self._default_maxsize if maxsize is None else maxsize
        if size is not None and size > 0:
            queue = asyncio.Queue(maxsize=size)
        else:
            queue = asyncio.Queue()
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._subscribers.discard(queue)

    def emit(self, event_type: str, payload: Dict[str, Any] | None = None) -> None:
        safe_payload = dict(payload or {})
        if "correlation_id" not in safe_payload:
            safe_payload["correlation_id"] = get_correlation_id()
        if "event_id" not in safe_payload:
            safe_payload["event_id"] = uuid.uuid4().hex
        event = {
            "type": event_type,
            "payload": safe_payload,
            "timestamp": time.time(),
        }
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning("event_bus_subscriber_queue_full", event_type=event_type)
