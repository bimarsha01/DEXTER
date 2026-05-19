"""
Dexter Tool Registry — Central hub for all available tools.
Handles tool loading and dynamic execution for any LLM backend.
"""
import importlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from tools.executor import ToolExecutor, ToolResult
from utils.config import get_config, get_workspace_root
from utils.logger import get_logger

logger = get_logger("tool_registry")


# ─── Master list of all callable tools ───────────────────────────────────────
# Each entry maps to a real Python function with type hints + docstrings.
_TOOL_MODULES: dict[str, list[str]] = {
    "tools.pc_controls": [
        "open_application",
        "close_application",
        "lock_workstation",
        "set_system_volume",
    ],
    "tools.web_browser": [
        "search_google",
        "open_url",
        "open_url_in_browser",
        "search_youtube",
        "search_content_platform",
    ],
    "tools.media_tool": [
        "play_media",
        "play_music",
        "pause_music",
        "next_track",
    ],
    "tools.file_tools": [
        "create_note",
        "read_note",
        "list_notes",
    ],
    "tools.open_resolver": [
        "resolve_open_target",
    ],
    "tools.input_tools": [
        "type_text",
        "press_shortcut",
        "enter_key",
        "minimize_all_windows",
    ],
    "tools.system_tools": [
        "get_current_time",
        "get_current_datetime",
        "get_weather",
        "get_system_status",
        "read_clipboard",
        "copy_to_clipboard",
        "take_screenshot",
        "shutdown_pc",
        "restart_pc",
        "cancel_shutdown",
        "sleep_pc",
        "get_health_report",
    ],
    "tools.document_tools": [
        "read_document",
        "summarize_document",
        "answer_document_question",
    ],
    "tools.routine_tools": [
        "save_automation_routine",
        "list_automation_routines",
        "run_automation_routine",
        "delete_automation_routine",
    ],
    "tools.briefing": [
        "get_morning_briefing",
    ],
    "tools.vision_tools": [
        "read_workspace_file",
        "capture_screen",
    ],
    "tools.youtube_tool": [
        "play_youtube",
    ],
}

from mcp_server.client import MCP_TOOL_NAMES  # noqa: E402

RAW_TOOLS: list[Callable] = []
AVAILABLE_TOOLS: list[Callable] = []
EXECUTOR: ToolExecutor = ToolExecutor([])
_TOOLS_LOADED = False

_mcp_client: Optional["MCPClient"] = None
_mcp_ready = False


def _make_mcp_stub(name: str) -> Callable:
    """Placeholder so MCP tools appear in schema audit and LLM tool lists."""

    async def _mcp_stub(**_kwargs):
        return {
            "success": False,
            "error": "MCP tool must be invoked via execute_tool routing.",
        }

    _mcp_stub.__name__ = name
    _mcp_stub.__doc__ = f"MCP-backed tool: {name}"
    return _mcp_stub


def _load_module_tools(module_path: str, tool_names: list[str]) -> list[Callable]:
    try:
        module = importlib.import_module(module_path)
    except Exception as exc:
        logger.warning("tool_module_import_failed", module=module_path, error=str(exc))
        return []

    tools: list[Callable] = []
    for name in tool_names:
        func = getattr(module, name, None)
        if callable(func):
            tools.append(func)
        else:
            logger.warning("tool_missing", module=module_path, tool=name)
    return tools


def _build_tools() -> list[Callable]:
    tools: list[Callable] = []
    for module_path, tool_names in _TOOL_MODULES.items():
        tools.extend(_load_module_tools(module_path, tool_names))
    for mcp_name in MCP_TOOL_NAMES:
        tools.append(_make_mcp_stub(mcp_name))
    return tools


def load_tools():
    """
    Returns the list of Python functions that the LLM can call as tools.
    - Gemini SDK reads type hints + docstrings natively from these functions.
    - Groq/OpenAI schemas are auto-generated via inspect in the Brain class.
    """
    global RAW_TOOLS, AVAILABLE_TOOLS, EXECUTOR, _TOOLS_LOADED
    if _TOOLS_LOADED:
        return AVAILABLE_TOOLS

    RAW_TOOLS = _build_tools()
    AVAILABLE_TOOLS = list(RAW_TOOLS)
    EXECUTOR._tools = {tool.__name__: tool for tool in AVAILABLE_TOOLS}
    EXECUTOR._schemas = {}
    _TOOLS_LOADED = True
    logger.info("tools_loaded", count=len(AVAILABLE_TOOLS))
    return AVAILABLE_TOOLS


def _format_tool_result(result: ToolResult):
    if result.success:
        data = result.data
        if isinstance(data, (dict, list)):
            return json.dumps(data)
        return data
    return result.error or f"Execution of {result.tool_name} failed."


async def initialize_mcp(config) -> bool:
    """
    Start the MCP server and register its tools.
    Called after Dexter startup is complete.
    Returns True if MCP started successfully.
    """
    global _mcp_client, _mcp_ready

    if not getattr(config.mcp, "enabled", False):
        logger.info("mcp_disabled_in_config")
        return False

    try:
        from mcp_server.client import MCPClient

        allowed_roots = getattr(config.security, "allowed_file_roots", []) or []
        if not allowed_roots:
            home = Path.home()
            allowed_roots = [
                home / "Documents",
                home / "Desktop",
                Path(get_workspace_root()),
            ]
        expanded_roots = []
        for root in allowed_roots:
            expanded = Path(str(root).replace("%USERPROFILE%", str(Path.home())))
            expanded = Path(os.path.expandvars(os.path.expanduser(str(expanded))))
            if not expanded.is_absolute():
                expanded = Path(get_workspace_root()) / expanded
            expanded_roots.append(str(expanded.resolve()))

        _mcp_client = MCPClient(
            server_script=getattr(
                config.mcp,
                "server_script",
                "mcp_server/dexter_mcp_server.py",
            ),
            allowed_roots=expanded_roots,
            timeout=float(getattr(config.mcp, "timeout_seconds", 15.0)),
        )

        success = await _mcp_client.start()

        if success:
            _mcp_ready = True
            mcp_tools = _mcp_client.get_available_tools()
            logger.info(
                "mcp_tools_registered",
                count=len(mcp_tools),
                tools=mcp_tools,
            )

        return success

    except Exception as e:
        logger.error(
            "mcp_initialization_failed",
            error=str(e),
            exc_info=True,
        )
        _mcp_ready = False
        return False


async def shutdown_mcp() -> None:
    global _mcp_client, _mcp_ready
    if _mcp_client is not None:
        await _mcp_client.stop()
    _mcp_client = None
    _mcp_ready = False


async def execute_tool(func_name: str, arguments: dict, event_bus=None):
    """
    Dynamically finds and executes a tool by name with the given arguments.
    Called by the LLM router when the AI decides to use a tool.
    """
    if not _TOOLS_LOADED:
        load_tools()

    if func_name.startswith("mcp_"):
        if _mcp_ready and _mcp_client is not None:
            try:
                EXECUTOR._validate_paths(arguments or {}, get_config())
            except ValueError as exc:
                logger.warning(
                    "mcp_path_validation_failed",
                    tool_name=func_name,
                    error=str(exc),
                )
                return str(exc)

            mcp_tool_name = func_name[4:]
            mcp_result = await _mcp_client.call_tool(mcp_tool_name, arguments or {})
            tool_result = ToolResult(
                success=mcp_result.success,
                data=mcp_result.data,
                error=mcp_result.error,
                tool_name=func_name,
                duration_ms=mcp_result.duration_ms,
                timestamp=datetime.now(timezone.utc),
            )
            if event_bus is not None and tool_result.success:
                event_bus.emit("tool_called", {"tool_name": func_name, "args": arguments})
            if not tool_result.success:
                logger.warning(
                    "tool_dispatch_failed",
                    tool_name=func_name,
                    error=tool_result.error or "",
                )
            return _format_tool_result(tool_result)

        logger.warning("mcp_tool_unavailable", tool_name=func_name)
        return (
            "MCP server is not available. File and document tools are "
            "temporarily offline. Try again in a moment."
        )

    result = await EXECUTOR.execute(func_name, arguments, event_bus=event_bus)
    if result.success:
        return _format_tool_result(result)

    logger.warning(
        "tool_dispatch_failed",
        tool_name=func_name,
        error=result.error or "",
    )
    return result.error or f"Execution of {func_name} failed."
