import os
from utils.logger import get_logger

logger = get_logger("file_tools")

# Set a default directory for notes so Dexter doesn't clutter the main active directory
NOTES_DIR = os.path.join(os.path.expanduser("~"), "Documents", "DexterNotes")
if not os.path.exists(NOTES_DIR):
    os.makedirs(NOTES_DIR)


def _safe_note_filename(filename: str) -> str | None:
    raw = (filename or "").strip()
    if not raw:
        return None
    drive, _ = os.path.splitdrive(raw)
    name = os.path.basename(raw)
    if drive or name != raw or name in {".", ".."}:
        return None
    if not name.endswith(".txt"):
        name += ".txt"
    return name

def create_note(filename: str, content: str) -> str:
    """
    Creates a new text file or overwrites an existing one with the given content.
    """
    safe_name = _safe_note_filename(filename)
    if not safe_name:
        return "Invalid note name. Use a simple filename without folders."

    filepath = os.path.join(NOTES_DIR, safe_name)
    logger.info("note_create_started", path=filepath)
    
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully created note: {safe_name}."
    except Exception as e:
        logger.error("note_create_failed", filename=filename, error=str(e), exc_info=True)
        return f"Error creating note: {str(e)}"

def read_note(filename: str) -> str:
    """
    Reads the content of a previously created text note.
    """
    safe_name = _safe_note_filename(filename)
    if not safe_name:
        return "Invalid note name. Use a simple filename without folders."

    filepath = os.path.join(NOTES_DIR, safe_name)
    logger.info("note_read_started", path=filepath)
    
    if not os.path.exists(filepath):
        return f"I could not find a note named {safe_name}."
        
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        return f"Content of {safe_name}:\n{content}"
    except Exception as e:
        logger.error("note_read_failed", filename=filename, error=str(e), exc_info=True)
        return f"Error reading note: {str(e)}"

def list_notes() -> str:
    """
    Lists all saved notes in Dexter's notebook directory.
    """
    logger.info("note_list_requested")
    try:
        files = os.listdir(NOTES_DIR)
        txt_files = [f for f in files if f.endswith(".txt")]
        if not txt_files:
            return "You have no saved notes at the moment."
        return f"Here are your saved notes: {', '.join(txt_files)}"
    except Exception as e:
        logger.error("note_list_failed", error=str(e), exc_info=True)
        return f"Error listing notes: {str(e)}"
