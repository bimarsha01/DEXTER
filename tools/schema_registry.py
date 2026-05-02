import json
import os
from functools import lru_cache
from utils.logger import logger


@lru_cache(maxsize=1)
def load_tool_schemas() -> dict:
    base_dir = os.path.dirname(__file__)
    schema_path = os.path.join(base_dir, "schemas", "tool_schemas.json")
    try:
        with open(schema_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        logger.warning(f"Tool schema file not found: {schema_path}")
        return {}
    except json.JSONDecodeError as e:
        logger.warning(f"Tool schema JSON is invalid: {e}")
        return {}


def get_tool_schema(tool_name: str) -> dict:
    schemas = load_tool_schemas()
    return schemas.get(tool_name, {})
