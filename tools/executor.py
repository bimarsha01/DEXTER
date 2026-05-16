import asyncio
import os
import re
import time
from enum import StrEnum
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from jsonschema import ValidationError, validate

from utils.logger import get_logger

logger = get_logger("tool_executor")
from utils.config import DexterConfig, get_workspace_root, get_config
from utils.metrics import metrics
from tools.schema_registry import get_tool_schema

UNSAFE_PATTERN = re.compile(r"(;|\||\|\||&&|`|\$\(|\n)")
PATH_ARG_NAMES = {"path", "file_path", "filepath", "root", "directory", "folder", "output_path"}
RELATIVE_PATH_NAMES = {"relative_path"}


class RiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class ToolResult:
    success: bool
    data: Any
    error: Optional[str]
    tool_name: str
    duration_ms: float
    timestamp: datetime
    risk_level: str = RiskLevel.LOW.value
    confirmation_required: bool = False
    policy_decision: str = "allowed"


class ToolExecutor:
    def __init__(self, tools: List[Callable], event_bus=None):
        self._tools = {tool.__name__: tool for tool in tools}
        self._schemas: Dict[str, dict] = {}
        self.event_bus = event_bus

    def get_tool_manifest(self) -> List[dict]:
        """
        Returns the JSON schema definitions for all registered tools.
        Used to export tool capabilities to MCP servers or external frontends.
        """
        manifest = []
        for name, func in self._tools.items():
            schema = self._get_schema(name, func)
            if schema:
                if "function" in schema:
                    manifest.append(schema)
                else:
                    manifest.append({"type": "function", "function": {"name": name, "parameters": schema}})
        return manifest

    def get_mcp_manifest(self) -> dict:
        """MCP-compatible tool registration payload."""
        tools = []
        for name in self._tools:
            schema = self._schemas.get(name) or self._get_schema(name, self._tools[name])
            description = ""
            func = self._tools.get(name)
            if func and func.__doc__:
                description = func.__doc__.strip().split("\n")[0]
            parameters = schema if isinstance(schema, dict) else {}
            if "properties" not in parameters and parameters.get("type") != "object":
                parameters = {"type": "object", "properties": parameters, "additionalProperties": False}
            tools.append({
                "name": name,
                "description": description,
                "inputSchema": parameters,
            })
        return {"tools": tools}

    def _get_schema(self, tool_name: str, func: Callable) -> dict:
        if tool_name not in self._schemas:
            schema = get_tool_schema(tool_name)
            if not schema:
                logger.warning("tool_schema_implicit_fallback", tool_name=tool_name)
                schema = {"type": "object", "properties": {}, "additionalProperties": True}
            self._schemas[tool_name] = schema
        return self._schemas[tool_name]

    def _sanitize_args(self, tool_name: str, args: dict, schema: dict) -> dict:
        args = args or {}
        properties = schema.get("properties", {})
        if not properties and args:
            raise ValueError("This tool does not accept arguments.")

        search_tools = {
            "search_google",
            "search_youtube",
            "search_content_platform",
            "play_youtube",
        }
        search_keys = {"query", "search_term", "platform", "content_type"}

        def _clean_search_text(value: str) -> str:
            cleaned = value.replace("\r", " ").replace("\n", " ")
            cleaned = re.sub(r"(\$\(|&&|\|\|)", " ", cleaned)
            cleaned = cleaned.replace("|", " ").replace(";", " ").replace("`", " ")
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            return cleaned

        clean = {}
        for key, value in args.items():
            if properties and key not in properties:
                continue
            if isinstance(value, str) and tool_name in search_tools and key in search_keys:
                value = _clean_search_text(value)
            if isinstance(value, str) and UNSAFE_PATTERN.search(value):
                raise ValueError(f"Unsafe characters detected in argument '{key}'.")
            clean[key] = value
        return clean

    def _get_allowed_roots(self, config: DexterConfig) -> List[str]:
        roots = config.security.allowed_file_roots or []
        if any(str(r).strip() in {".", "./"} for r in roots):
            logger.warning("unsafe_allowed_root_dot_present", allowed_roots=roots)
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

    def _assess_risk(self, tool_name: str, args: dict) -> tuple[str, bool, str]:
        high_risk_tools = {"shutdown_pc", "restart_pc", "sleep_pc"}
        medium_risk_tools = {
            "open_application",
            "open_url",
            "open_url_in_browser",
            "type_text",
            "press_shortcut",
            "copy_to_clipboard",
        }
        if tool_name in high_risk_tools:
            return RiskLevel.HIGH.value, True, "high_risk_power_action"
        if tool_name in medium_risk_tools:
            return RiskLevel.MEDIUM.value, False, "medium_risk_automation"
        if any(key in args for key in ("path", "file_path", "filepath", "root", "directory", "folder")):
            return RiskLevel.MEDIUM.value, False, "filesystem_access"
        return RiskLevel.LOW.value, False, "allowed"

    async def execute(self, tool_name: str, args: dict, event_bus: Any = None) -> ToolResult:
        def _emit(event_type: str, **fields: Any) -> None:
            if event_bus is not None:
                event_bus.emit(event_type, fields)

        func = self._tools.get(tool_name)
        if not func:
            logger.info(
                "tool_executed",
                tool_name=tool_name,
                success=False,
                duration_ms=0.0,
                error="tool_not_available",
            )
            _emit(
                "tool_execution_completed",
                tool_name=tool_name,
                success=False,
                duration_ms=0.0,
                error="tool_not_available",
                risk_level=RiskLevel.LOW.value,
                policy_decision="tool_not_available",
            )
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
        logger.info("tool_execution_started", tool_name=tool_name)
        args_keys = sorted((args or {}).keys())
        _emit("tool_execution_started", tool_name=tool_name, args_keys=args_keys)
        try:
            schema = self._get_schema(tool_name, func)
            clean_args = self._sanitize_args(tool_name, args or {}, schema)
            self._validate_args(clean_args, schema)
            self._validate_paths(clean_args, config)

            risk_level, confirmation_required, policy_decision = self._assess_risk(tool_name, clean_args)
            if confirmation_required and not clean_args.get("confirm", False):
                duration_ms = (time.perf_counter() - start) * 1000
                metrics.record_latency("tool_ms", duration_ms)
                err_msg = f"Tool '{tool_name}' requires explicit confirmation."
                logger.info(
                    "tool_executed",
                    tool_name=tool_name,
                    success=False,
                    duration_ms=duration_ms,
                    error=err_msg,
                    risk_level=risk_level,
                    policy_decision=policy_decision,
                )
                _emit(
                    "tool_execution_completed",
                    tool_name=tool_name,
                    success=False,
                    duration_ms=duration_ms,
                    error=err_msg,
                    risk_level=risk_level,
                    confirmation_required=True,
                    policy_decision=policy_decision,
                )
                return ToolResult(
                    success=False,
                    data=None,
                    error=err_msg,
                    tool_name=tool_name,
                    duration_ms=duration_ms,
                    timestamp=datetime.utcnow(),
                    risk_level=risk_level,
                    confirmation_required=True,
                    policy_decision=policy_decision,
                )

            if asyncio.iscoroutinefunction(func):
                result = await asyncio.wait_for(
                    func(**clean_args),
                    timeout=timeout_sec,
                )
            else:
                result = await asyncio.wait_for(
                    asyncio.to_thread(func, **clean_args),
                    timeout=timeout_sec,
                )
            duration_ms = (time.perf_counter() - start) * 1000
            metrics.record_latency("tool_ms", duration_ms)
            logger.info(
                "tool_executed",
                tool_name=tool_name,
                success=True,
                duration_ms=duration_ms,
                error=None,
            )
            _emit(
                "tool_execution_completed",
                tool_name=tool_name,
                success=True,
                duration_ms=duration_ms,
                error=None,
                risk_level=risk_level,
                confirmation_required=confirmation_required,
                policy_decision=policy_decision,
            )
            if event_bus is not None:
                event_bus.emit("tool_called", {"tool_name": tool_name, "args": clean_args})
            return ToolResult(
                success=True,
                data=result,
                error=None,
                tool_name=tool_name,
                duration_ms=duration_ms,
                timestamp=datetime.utcnow(),
                risk_level=risk_level,
                confirmation_required=confirmation_required,
                policy_decision=policy_decision,
            )
        except ValidationError as e:
            duration_ms = (time.perf_counter() - start) * 1000
            metrics.record_latency("tool_ms", duration_ms)
            err_msg = f"Invalid arguments: {e.message}"
            logger.info(
                "tool_executed",
                tool_name=tool_name,
                success=False,
                duration_ms=duration_ms,
                error=err_msg,
            )
            _emit(
                "tool_execution_completed",
                tool_name=tool_name,
                success=False,
                duration_ms=duration_ms,
                error=err_msg,
                risk_level=RiskLevel.LOW.value,
                policy_decision="invalid_args",
            )
            return ToolResult(
                success=False,
                data=None,
                error=err_msg,
                tool_name=tool_name,
                duration_ms=duration_ms,
                timestamp=datetime.utcnow(),
                risk_level=RiskLevel.LOW.value,
            )
        except ValueError as e:
            duration_ms = (time.perf_counter() - start) * 1000
            metrics.record_latency("tool_ms", duration_ms)
            err_msg = str(e)
            logger.info(
                "tool_executed",
                tool_name=tool_name,
                success=False,
                duration_ms=duration_ms,
                error=err_msg,
            )
            _emit(
                "tool_execution_completed",
                tool_name=tool_name,
                success=False,
                duration_ms=duration_ms,
                error=err_msg,
                risk_level=RiskLevel.LOW.value,
                policy_decision="invalid_args",
            )
            return ToolResult(
                success=False,
                data=None,
                error=err_msg,
                tool_name=tool_name,
                duration_ms=duration_ms,
                timestamp=datetime.utcnow(),
                risk_level=RiskLevel.LOW.value,
            )
        except asyncio.TimeoutError:
            duration_ms = (time.perf_counter() - start) * 1000
            metrics.record_latency("tool_ms", duration_ms)
            err_msg = f"Tool '{tool_name}' timed out after {timeout_sec:.0f}s."
            logger.info(
                "tool_executed",
                tool_name=tool_name,
                success=False,
                duration_ms=duration_ms,
                error=err_msg,
            )
            _emit(
                "tool_execution_completed",
                tool_name=tool_name,
                success=False,
                duration_ms=duration_ms,
                error=err_msg,
                risk_level=RiskLevel.MEDIUM.value,
                policy_decision="timeout",
            )
            return ToolResult(
                success=False,
                data=None,
                error=err_msg,
                tool_name=tool_name,
                duration_ms=duration_ms,
                timestamp=datetime.utcnow(),
                risk_level=RiskLevel.MEDIUM.value,
            )
        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000
            metrics.record_latency("tool_ms", duration_ms)
            logger.error(
                "tool_execution_exception",
                tool_name=tool_name,
                error=str(e),
                exc_info=True,
            )
            err_msg = f"Execution of {tool_name} failed: {str(e)}"
            logger.info(
                "tool_executed",
                tool_name=tool_name,
                success=False,
                duration_ms=duration_ms,
                error=err_msg,
            )
            _emit(
                "tool_execution_completed",
                tool_name=tool_name,
                success=False,
                duration_ms=duration_ms,
                error=err_msg,
                risk_level=RiskLevel.MEDIUM.value,
                policy_decision="execution_failed",
            )
            return ToolResult(
                success=False,
                data=None,
                error=err_msg,
                tool_name=tool_name,
                duration_ms=duration_ms,
                timestamp=datetime.utcnow(),
                risk_level=RiskLevel.MEDIUM.value,
            )
