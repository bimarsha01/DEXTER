import json
import os
from functools import lru_cache
from utils.logger import get_logger

logger = get_logger("schema_registry")


@lru_cache(maxsize=1)
def load_tool_schemas() -> dict:
    base_dir = os.path.dirname(__file__)
    schema_path = os.path.join(base_dir, "schemas", "tool_schemas.json")
    try:
        with open(schema_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        logger.warning("tool_schema_file_not_found", path=schema_path)
        return {}
    except json.JSONDecodeError as e:
        logger.warning("tool_schema_json_invalid", error=str(e))
        return {}


def get_tool_schema(tool_name: str) -> dict:
    schemas = load_tool_schemas()
    return schemas.get(tool_name, {})
