import os
from utils.logger import get_logger

logger = get_logger("file_tools")

# Set a default directory for notes so Dexter doesn't clutter the main active directory
NOTES_DIR = os.path.join(os.path.expanduser("~"), "Documents", "DexterNotes")
if not os.path.exists(NOTES_DIR):
    os.makedirs(NOTES_DIR)

def create_note(filename: str, content: str) -> str:
    """
    Creates a new text file or overwrites an existing one with the given content.
    """
    if not filename.endswith(".txt"):
        filename += ".txt"
        
    filepath = os.path.join(NOTES_DIR, filename)
    logger.info("note_create_started", path=filepath)
    
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully created note: {filename}."
    except Exception as e:
        logger.error("note_create_failed", filename=filename, error=str(e), exc_info=True)
        return f"Error creating note: {str(e)}"

def read_note(filename: str) -> str:
    """
    Reads the content of a previously created text note.
    """
    if not filename.endswith(".txt"):
        filename += ".txt"
        
    filepath = os.path.join(NOTES_DIR, filename)
    logger.info("note_read_started", path=filepath)
    
    if not os.path.exists(filepath):
        return f"Sir, I could not find a note named {filename}."
        
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        return f"Content of {filename}:\n{content}"
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
            return "You have no saved notes at the moment, sir."
        return f"Here are your saved notes: {', '.join(txt_files)}"
    except Exception as e:
        logger.error("note_list_failed", error=str(e), exc_info=True)
        return f"Error listing notes: {str(e)}"
