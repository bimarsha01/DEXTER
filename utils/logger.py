import io
import json
import logging
from logging.handlers import TimedRotatingFileHandler
import os
import sys
import uuid
from contextvars import ContextVar

from pathlib import Path

try:
    import structlog
    from structlog.contextvars import bind_contextvars, clear_contextvars, merge_contextvars
except Exception:
    structlog = None
    bind_contextvars = None
    clear_contextvars = None
    merge_contextvars = None


CORRELATION_ID: ContextVar[str] = ContextVar("correlation_id", default="-")


def set_correlation_id(correlation_id: str | None = None) -> str:
    value = correlation_id or uuid.uuid4().hex
    CORRELATION_ID.set(value)
    if bind_contextvars is not None:
        bind_contextvars(correlation_id=value)
    return value


def clear_correlation_id() -> None:
    CORRELATION_ID.set("-")
    if clear_contextvars is not None:
        clear_contextvars()


def get_correlation_id() -> str:
    return CORRELATION_ID.get()

class CorrelationIdFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, "correlation_id"):
            record.correlation_id = get_correlation_id()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "timestamp": self.formatTime(record, self.datefmt),
            "name": record.name,
            "level": record.levelname,
            "message": record.getMessage(),
            "correlation_id": getattr(record, "correlation_id", get_correlation_id()),
            "module": record.module,
            "line": record.lineno,
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logger():
    # Force UTF-8 on Windows console to support Unicode characters (✓, →, ═, etc.)
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
            )
            sys.stderr = io.TextIOWrapper(
                sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True
            )
        except Exception:
            pass

    logger = logging.getLogger("Dexter")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    log_dir = Path(os.path.join(os.path.dirname(__file__), "..", "logs")).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.addFilter(CorrelationIdFilter())

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - [cid=%(correlation_id)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    ch.setFormatter(formatter)

    fh = TimedRotatingFileHandler(
        log_dir / "dexter.log",
        when="midnight",
        backupCount=14,
        encoding="utf-8",
        utc=False,
    )
    fh.setLevel(logging.DEBUG)
    fh.addFilter(CorrelationIdFilter())
    fh.setFormatter(JsonFormatter(datefmt="%Y-%m-%dT%H:%M:%S"))

    if not logger.handlers:
        logger.addHandler(ch)
        logger.addHandler(fh)

    if structlog is not None:
        structlog.configure(
            wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
            processors=[
                merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.JSONRenderer(),
            ],
        )

    return logger

logger = setup_logger()
