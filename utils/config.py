import os
import yaml
from functools import lru_cache
from dotenv import load_dotenv

# 1. Get the root directory once
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

@lru_cache(maxsize=1)
def load_config() -> dict:
    """Load config.yaml, inject .env keys, and cache the result."""
    # Load the .env file from the root directory
    load_dotenv(os.path.join(ROOT_DIR, ".env"))
    
    config_path = os.path.join(ROOT_DIR, "config.yaml")
    
    try:
        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}
            
        # ─── THE FIX ───
        # We fetch by the VARIABLE NAME defined in your .env file
        config['api_keys']['gemini'] = os.getenv("GEMINI_API_KEY")
        config['api_keys']['groq'] = os.getenv("GROQ_API_KEY")
        
        return config
    except FileNotFoundError:
        print(f"Error: Could not find config.yaml at {config_path}")
        return {}

def get_workspace_root() -> str:
    """Return the workspace root directory (DEXTER)."""
    return ROOT_DIR