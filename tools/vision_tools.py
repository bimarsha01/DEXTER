import base64
import io
import os
from utils.logger import get_logger

logger = get_logger("vision_tools")
from utils.config import get_workspace_root


def read_workspace_file(relative_path: str, max_chars: int = 4000) -> str:
    """
    Reads a text file within the workspace. Rejects absolute paths or parent traversal.
    Returns a truncated string if the file is large.
    """
    if not relative_path or not relative_path.strip():
        return "You must provide a workspace-relative file path."

    clean_path = relative_path.strip().replace("\\", "/")
    if os.path.isabs(clean_path) or ":" in clean_path:
        return "Absolute paths are not allowed. Provide a relative path from the workspace root."
    if ".." in clean_path.split("/"):
        return "Parent path traversal is not allowed. Provide a safe relative path."

    root = get_workspace_root()
    full_path = os.path.abspath(os.path.join(root, clean_path))
    if not full_path.startswith(root):
        return "Path is outside the workspace."

    if not os.path.exists(full_path):
        return f"File not found: {clean_path}"

    try:
        with open(full_path, "r", encoding="utf-8", errors="replace") as file:
            data = file.read(max_chars + 1)
        if "\x00" in data:
            return "This file appears to be binary and cannot be read as text."
        if len(data) > max_chars:
            return data[:max_chars] + "\n...[truncated]"
        return data
    except Exception as e:
        logger.error("workspace_file_read_failed", path=clean_path, error=str(e), exc_info=True)
        return f"I could not read {clean_path}."


def capture_screen(max_dimension: int = 1280) -> str:
    """
    Captures the full screen and returns a base64-encoded PNG string.
    """
    try:
        import pyautogui
        image = pyautogui.screenshot()
        width, height = image.size
        if max(width, height) > max_dimension:
            scale = max_dimension / float(max(width, height))
            new_size = (int(width * scale), int(height * scale))
            image = image.resize(new_size)

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"image/png;base64,{encoded}"
    except Exception as e:
        logger.error("screen_capture_failed", error=str(e), exc_info=True)
        return "I was unable to capture the screen."
