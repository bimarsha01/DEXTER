"""
Structured logging with structlog (preferred) or stdlib fallback.
Correlation IDs via contextvars; JSON rotation to logs/dexter.log.
"""
from __future__ import annotations

import io
import json
import logging
import logging.handlers
import sys
import uuid
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

CORRELATION_ID: ContextVar[str] = ContextVar("correlation_id", default="-")


def set_correlation_id(correlation_id: str | None = None) -> str:
    value = correlation_id or uuid.uuid4().hex
    CORRELATION_ID.set(value)
    try:
        import structlog

        structlog.contextvars.bind_contextvars(correlation_id=value)
    except ImportError:
        pass
    return value


def clear_correlation_id() -> None:
    CORRELATION_ID.set("-")
    try:
        import structlog

        structlog.contextvars.clear_contextvars()
    except ImportError:
        pass


def get_correlation_id() -> str:
    return CORRELATION_ID.get()


bind_correlation_id = set_correlation_id


@runtime_checkable
class ComponentLogger(Protocol):
    def debug(self, event: str, *args: Any, **kwargs: Any) -> None: ...
    def info(self, event: str, *args: Any, **kwargs: Any) -> None: ...
    def warning(self, event: str, *args: Any, **kwargs: Any) -> None: ...
    def error(self, event: str, *args: Any, **kwargs: Any) -> None: ...


class _StdlibComponentLogger:
    """Fallback when structlog is not installed: event + keyword fields in message."""

    __slots__ = ("_log",)

    def __init__(self, component: str) -> None:
        self._log = logging.getLogger(f"Dexter.{component}")

    def _emit(self, level: int, event: str, **kwargs: Any) -> None:
        exc_info = bool(kwargs.pop("exc_info", False))
        extra = {"correlation_id": get_correlation_id(), "component": self._log.name}
        if kwargs:
            self._log.log(
                level,
                "%s | %s",
                event,
                json.dumps(kwargs, default=str),
                extra=extra,
                exc_info=exc_info,
            )
        else:
            self._log.log(level, "%s", event, extra=extra, exc_info=exc_info)

    def debug(self, event: str, **kwargs: Any) -> None:
        self._emit(logging.DEBUG, event, **kwargs)

    def info(self, event: str, **kwargs: Any) -> None:
        self._emit(logging.INFO, event, **kwargs)

    def warning(self, event: str, **kwargs: Any) -> None:
        self._emit(logging.WARNING, event, **kwargs)

    def error(self, event: str, **kwargs: Any) -> None:
        exc_info = bool(kwargs.pop("exc_info", False))
        extra = {"correlation_id": get_correlation_id(), "component": self._log.name}
        if kwargs:
            self._log.error(
                "%s | %s",
                event,
                json.dumps(kwargs, default=str),
                extra=extra,
                exc_info=exc_info,
            )
        else:
            self._log.error("%s", event, extra=extra, exc_info=exc_info)


class CorrelationIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "correlation_id"):
            record.correlation_id = get_correlation_id()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
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


def _setup_fallback_logging() -> None:
    log_dir = Path(__file__).resolve().parent.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.addFilter(CorrelationIdFilter())
    ch.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - [cid=%(correlation_id)s] %(message)s",
            datefmt="%H:%M:%S",
        )
    )

    fh = logging.handlers.TimedRotatingFileHandler(
        log_dir / "dexter.log",
        when="midnight",
        backupCount=14,
        encoding="utf-8",
        utc=False,
    )
    fh.setLevel(logging.DEBUG)
    fh.addFilter(CorrelationIdFilter())
    fh.setFormatter(JsonFormatter(datefmt="%Y-%m-%dT%H:%M:%S"))

    root.addHandler(ch)
    root.addHandler(fh)


def _setup_structlog_logging() -> None:
    import structlog
    from structlog.stdlib import BoundLogger, LoggerFactory, ProcessorFormatter

    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer,
                encoding="utf-8",
                errors="replace",
                line_buffering=True,
            )
            sys.stderr = io.TextIOWrapper(
                sys.stderr.buffer,
                encoding="utf-8",
                errors="replace",
                line_buffering=True,
            )
        except OSError:
            pass

    log_dir = Path(__file__).resolve().parent.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    def add_correlation_id(_logger: Any, _method_name: str, event_dict: dict) -> dict:
        event_dict["correlation_id"] = CORRELATION_ID.get()
        return event_dict

    timestamper = structlog.processors.TimeStamper(fmt="iso")

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        add_correlation_id,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        timestamper,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    structlog.configure(
        processors=shared_processors + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=LoggerFactory(),
        wrapper_class=BoundLogger,
        cache_logger_on_first_use=True,
    )

    console_formatter = ProcessorFormatter(
        processor=structlog.dev.ConsoleRenderer(colors=False),
        foreign_pre_chain=shared_processors,
    )

    json_formatter = ProcessorFormatter(
        processor=structlog.processors.JSONRenderer(),
        foreign_pre_chain=shared_processors,
    )

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(console_formatter)

    fh = logging.handlers.TimedRotatingFileHandler(
        log_dir / "dexter.log",
        when="midnight",
        backupCount=14,
        encoding="utf-8",
        utc=False,
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(json_formatter)

    root.addHandler(ch)
    root.addHandler(fh)

    logging.getLogger("httpx").setLevel(logging.WARNING)


_USING_STRUCTLOG = False


def setup_logging() -> None:
    global _USING_STRUCTLOG
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer,
                encoding="utf-8",
                errors="replace",
                line_buffering=True,
            )
            sys.stderr = io.TextIOWrapper(
                sys.stderr.buffer,
                encoding="utf-8",
                errors="replace",
                line_buffering=True,
            )
        except OSError:
            pass

    try:
        import structlog  # noqa: F401

        _setup_structlog_logging()
        _USING_STRUCTLOG = True
    except ImportError:
        _setup_fallback_logging()
        _USING_STRUCTLOG = False


def get_logger(component: str) -> ComponentLogger:
    if _USING_STRUCTLOG:
        import structlog

        return structlog.get_logger(component)
    return _StdlibComponentLogger(component)


setup_logging()
