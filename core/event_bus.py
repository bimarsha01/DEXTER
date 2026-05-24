import asyncio
import time
import uuid
from typing import Any, Dict

from utils.logger import get_logger, get_correlation_id

logger = get_logger("event_bus")


class DexterEvents:
    """Event names for GUI/MCP subscribers."""

    STATE_CHANGED = "state_changed"
    TURN_STAGE = "turn_stage"
    RETRIEVAL_EVENT = "retrieval_event"
    TRANSCRIPT_READY = "transcript_ready"
    RAG_CONTEXT_USED = "rag_context_used"
    RAG_CONTEXT_EMPTY = "rag_context_empty"
    PROVIDER_USED = "provider_used"
    TOOL_CALLED = "tool_called"
    TOOL_RESULT = "tool_result"
    RESPONSE_CHUNK = "response_chunk"
    RESPONSE_COMPLETE = "response_complete"
    RESPONSE_COMPLETED = "response_completed"
    ACTIVATION_MODE_CHANGED = "activation_mode_changed"
    WAKE_WORD_DETECTED = "wake_word_detected"
    COMMAND_DROPPED = "command_dropped"
    PROVIDER_FALLBACK = "provider_fallback"
    RAG_SEARCH_FAILED = "rag_search_failed"


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
        try:
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
                except Exception as e:
                    logger.warning("event_bus_emit_subscriber_failed", event_type=event_type, error=str(e))
        except Exception as e:
            logger.warning("Event bus emit failed (non-critical)", event_type=event_type, error=str(e))
