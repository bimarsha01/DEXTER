"""
Dexter Tool Registry — Central hub for all available tools.
Handles tool loading and dynamic execution for any LLM backend.
"""
import importlib
import json
from typing import Callable

from tools.executor import ToolExecutor
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
    "tools.vision_tools": [
        "read_workspace_file",
        "capture_screen",
    ],
    "tools.youtube_tool": [
        "play_youtube",
    ],
}

RAW_TOOLS: list[Callable] = []
AVAILABLE_TOOLS: list[Callable] = []
EXECUTOR: ToolExecutor = ToolExecutor([])
_TOOLS_LOADED = False


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


async def execute_tool(func_name: str, arguments: dict):
    """
    Dynamically finds and executes a tool by name with the given arguments.
    Called by the LLM router when the AI decides to use a tool.
    """
    if not _TOOLS_LOADED:
        load_tools()
    result = await EXECUTOR.execute(func_name, arguments)
    if result.success:
        data = result.data
        if isinstance(data, (dict, list)):
            return json.dumps(data)
        return data

    logger.warning(
        "tool_dispatch_failed",
        tool_name=func_name,
        error=result.error or "",
    )
    return result.error or f"Execution of {func_name} failed."
