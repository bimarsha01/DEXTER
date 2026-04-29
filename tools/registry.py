"""
Dexter Tool Registry — Central hub for all available tools.
Handles tool loading and dynamic execution for any LLM backend.
"""
import time
from functools import wraps
from tools import pc_controls
from tools import web_browser
from tools import file_tools
from tools import input_tools
from tools import system_tools
from tools import vision_tools
from utils.logger import logger
from utils.metrics import metrics

def _wrap_tool(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            metrics.record_latency("tool_ms", (time.perf_counter() - start) * 1000)

    return wrapper


# ─── Master list of all callable tools ───────────────────────────────────────
# Each entry is a real Python function with proper type hints + docstrings.
# The LLM router reads these to auto-generate tool schemas for each backend.
RAW_TOOLS = [
    # PC Controls
    pc_controls.open_application,
    pc_controls.close_application,
    pc_controls.lock_workstation,
    pc_controls.set_system_volume,

    # Web Browser
    web_browser.search_google,
    web_browser.open_url,
    web_browser.search_youtube,

    # File / Notes
    file_tools.create_note,
    file_tools.read_note,
    file_tools.list_notes,

    # Keyboard / Input
    input_tools.type_text,
    input_tools.press_shortcut,
    input_tools.enter_key,
    input_tools.minimize_all_windows,

    # System Tools (Jarvis-style)
    system_tools.get_current_datetime,
    system_tools.get_weather,
    system_tools.get_system_status,
    system_tools.read_clipboard,
    system_tools.copy_to_clipboard,
    system_tools.take_screenshot,
    system_tools.shutdown_pc,
    system_tools.restart_pc,
    system_tools.cancel_shutdown,
    system_tools.sleep_pc,
    system_tools.get_health_report,

    # Vision / IDE
    vision_tools.read_workspace_file,
    vision_tools.capture_screen,
]

AVAILABLE_TOOLS = [_wrap_tool(tool) for tool in RAW_TOOLS]


def load_tools():
    """
    Returns the list of Python functions that the LLM can call as tools.
    - Gemini SDK reads type hints + docstrings natively from these functions.
    - Groq/OpenAI schemas are auto-generated via inspect in the Brain class.
    """
    logger.info(f"Loaded {len(AVAILABLE_TOOLS)} tools into Dexter's arsenal.")
    return AVAILABLE_TOOLS


async def execute_tool(func_name: str, arguments: dict):
    """
    Dynamically finds and executes a tool by name with the given arguments.
    Called by the LLM router when the AI decides to use a tool.
    """
    func = None
    for item in AVAILABLE_TOOLS:
        if item.__name__ == func_name:
            func = item
            break

    if not func:
        logger.error(f"LLM requested unknown tool: {func_name}")
        return f"Tool '{func_name}' is not available."

    try:
        logger.info(f"Executing tool: {func_name}({arguments})")
        result = func(**arguments)
        logger.debug(f"Tool result: {str(result)[:100]}")
        return result

    except TypeError as te:
        logger.error(f"Argument mismatch for {func_name}: {te}")
        return f"Tool {func_name} received incorrect arguments: {str(te)}"
    except Exception as e:
        logger.error(f"Tool execution failed — {func_name}: {e}")
        return f"Execution of {func_name} failed: {str(e)}"
