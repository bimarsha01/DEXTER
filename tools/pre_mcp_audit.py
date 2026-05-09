"""Pre-MCP stability audit for Dexter.

Checks the core startup surfaces before moving on to MCP/FastMCP:
- typed config validation
- schema coverage
- importability of core modules
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
from dataclasses import asdict, dataclass

from tools.audit_tool_schemas import build_report
from utils.config import get_config
from utils.logger import get_logger

logger = get_logger("pre_mcp_audit")


CORE_IMPORTS = [ 
    "core.audio.vad",
    "core.audio.transcriber",
    "core.audio.speaker",
    "core.brain.llm_router",
    "core.brain.memory",
    "core.pipeline",
    "tools.registry",
    "tools.executor",
    "core.wake_word.detector",
]


@dataclass
class AuditResult:
    ok: bool
    config_ok: bool
    schemas_ok: bool
    imports_ok: bool
    import_failures: list[str]
    schema_report: dict


def run_audit() -> AuditResult:
    import_failures: list[str] = []
    config_ok = True
    schemas_ok = False
    imports_ok = True

    try:
        get_config()
    except Exception as e:
        config_ok = False
        logger.error("audit_config_load_failed", error=str(e), exc_info=True)
        import_failures.append(f"config: {e}")

    schema_report = build_report()
    schemas_ok = bool(schema_report.get("ok", False))
    if not schemas_ok:
        logger.warning(
            "audit_schema_coverage_incomplete",
            missing=schema_report.get("missing", []),
        )
        import_failures.append(
            f"schema coverage missing: {', '.join(schema_report.get('missing', [])) or 'unknown'}"
        )

    for module_name in CORE_IMPORTS:
        try:
            importlib.import_module(module_name)
        except Exception as e:
            imports_ok = False
            logger.error("audit_module_import_failed", module=module_name, error=str(e), exc_info=True)
            import_failures.append(f"{module_name}: {e}")

    ok = config_ok and schemas_ok and imports_ok
    return AuditResult(
        ok=ok,
        config_ok=config_ok,
        schemas_ok=schemas_ok,
        imports_ok=imports_ok,
        import_failures=import_failures,
        schema_report=schema_report,
    )


def main() -> int:
    result = run_audit()
    payload = asdict(result)
    logger.info("pre_mcp_audit_completed", **payload)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
