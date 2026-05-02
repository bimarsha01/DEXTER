import asyncio
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from jsonschema import ValidationError, validate

from utils.logger import logger
from utils.config import DexterConfig, get_workspace_root, get_config
from utils.metrics import metrics
from tools.schema_registry import get_tool_schema

UNSAFE_PATTERN = re.compile(r"(;|&&|\|\||`|\$\()")
PATH_ARG_NAMES = {"path", "file_path", "filepath", "root", "directory", "folder", "output_path"}
RELATIVE_PATH_NAMES = {"relative_path"}


@dataclass
class ToolResult:
    success: bool
    data: Any
    error: Optional[str]
    tool_name: str
    duration_ms: float
    timestamp: datetime


class ToolExecutor:
    def __init__(self, tools: List[Callable]):
        self._tools = {tool.__name__: tool for tool in tools}
        self._schemas: Dict[str, dict] = {}

    def _get_schema(self, tool_name: str, func: Callable) -> dict:
        if tool_name not in self._schemas:
            schema = get_tool_schema(tool_name)
            if not schema:
                logger.warning(f"No explicit schema for tool '{tool_name}'.")
                schema = {"type": "object", "properties": {}, "additionalProperties": True}
            self._schemas[tool_name] = schema
        return self._schemas[tool_name]

    def _sanitize_args(self, args: dict, schema: dict) -> dict:
        args = args or {}
        properties = schema.get("properties", {})
        if not properties and args:
            raise ValueError("This tool does not accept arguments.")

        clean = {}
        for key, value in args.items():
            if properties and key not in properties:
                continue
            if isinstance(value, str) and UNSAFE_PATTERN.search(value):
                raise ValueError(f"Unsafe characters detected in argument '{key}'.")
            clean[key] = value
        return clean

    def _get_allowed_roots(self, config: dict) -> List[str]:
        roots = config.get("security", {}).get("allowed_file_roots") or []
        workspace_root = get_workspace_root()
        if not roots:
            home = os.path.expanduser("~")
            roots = [
                workspace_root,
                os.path.join(home, "Documents"),
                os.path.join(home, "Desktop"),
            ]
        normalized = []
        for root in roots:
            expanded = os.path.expandvars(os.path.expanduser(str(root)))
            if not os.path.isabs(expanded):
                expanded = os.path.abspath(os.path.join(workspace_root, expanded))
            normalized.append(os.path.abspath(expanded))
        return normalized

    def _is_within_root(self, path: str, root: str) -> bool:
        try:
            return os.path.commonpath([path, root]) == root
        except ValueError:
            return False

    def _validate_paths(self, args: dict, config: DexterConfig) -> None:
        workspace_root = get_workspace_root()
        allowed_roots = self._get_allowed_roots(config)

        for key, value in args.items():
            if not isinstance(value, str):
                continue

            if key in RELATIVE_PATH_NAMES:
                abs_path = os.path.abspath(os.path.join(workspace_root, value))
                if not self._is_within_root(abs_path, workspace_root):
                    raise ValueError("Relative path is outside the workspace root.")
                continue

            if key in PATH_ARG_NAMES:
                abs_path = os.path.abspath(os.path.expanduser(os.path.expandvars(value)))
                if not any(self._is_within_root(abs_path, root) for root in allowed_roots):
                    raise ValueError("Path is outside allowed roots.")

    def _validate_args(self, args: dict, schema: dict) -> None:
        properties = schema.get("properties")
        if not properties and args:
            raise ValueError("This tool does not accept arguments.")
        if properties:
            validate(instance=args, schema=schema)

    async def execute(self, tool_name: str, args: dict) -> ToolResult:
        func = self._tools.get(tool_name)
        if not func:
            return ToolResult(
                success=False,
                data=None,
                error=f"Tool '{tool_name}' is not available.",
                tool_name=tool_name,
                duration_ms=0.0,
                timestamp=datetime.utcnow(),
            )

        config = get_config()
        timeout_sec = float(config.security.tool_timeout_sec)

        start = time.perf_counter()
        try:
            schema = self._get_schema(tool_name, func)
            clean_args = self._sanitize_args(args or {}, schema)
            self._validate_args(clean_args, schema)
            self._validate_paths(clean_args, config)

            result = await asyncio.wait_for(
                asyncio.to_thread(func, **clean_args),
                timeout=timeout_sec,
            )
            duration_ms = (time.perf_counter() - start) * 1000
            metrics.record_latency("tool_ms", duration_ms)
            return ToolResult(
                success=True,
                data=result,
                error=None,
                tool_name=tool_name,
                duration_ms=duration_ms,
                timestamp=datetime.utcnow(),
            )
        except ValidationError as e:
            duration_ms = (time.perf_counter() - start) * 1000
            metrics.record_latency("tool_ms", duration_ms)
            return ToolResult(
                success=False,
                data=None,
                error=f"Invalid arguments: {e.message}",
                tool_name=tool_name,
                duration_ms=duration_ms,
                timestamp=datetime.utcnow(),
            )
        except ValueError as e:
            duration_ms = (time.perf_counter() - start) * 1000
            metrics.record_latency("tool_ms", duration_ms)
            return ToolResult(
                success=False,
                data=None,
                error=str(e),
                tool_name=tool_name,
                duration_ms=duration_ms,
                timestamp=datetime.utcnow(),
            )
        except asyncio.TimeoutError:
            duration_ms = (time.perf_counter() - start) * 1000
            metrics.record_latency("tool_ms", duration_ms)
            return ToolResult(
                success=False,
                data=None,
                error=f"Tool '{tool_name}' timed out after {timeout_sec:.0f}s.",
                tool_name=tool_name,
                duration_ms=duration_ms,
                timestamp=datetime.utcnow(),
            )
        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000
            metrics.record_latency("tool_ms", duration_ms)
            logger.error(f"Tool execution failed: {tool_name}: {e}")
            return ToolResult(
                success=False,
                data=None,
                error=f"Execution of {tool_name} failed: {str(e)}",
                tool_name=tool_name,
                duration_ms=duration_ms,
                timestamp=datetime.utcnow(),
            )
