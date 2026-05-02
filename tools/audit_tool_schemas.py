"""Audit Dexter tool schema coverage.

This script verifies that every registered tool has an explicit JSON schema
entry and reports extra schema entries that do not map to a registered tool.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tools.registry import RAW_TOOLS
from tools.schema_registry import load_tool_schemas


def build_report() -> dict:
    tools = {tool.__name__ for tool in RAW_TOOLS}
    schemas = load_tool_schemas()
    schema_names = set(schemas.keys())

    missing = sorted(tools - schema_names)
    extra = sorted(schema_names - tools)
    covered = sorted(tools & schema_names)

    return {
        "tool_count": len(tools),
        "schema_count": len(schema_names),
        "covered_count": len(covered),
        "missing": missing,
        "extra": extra,
        "covered": covered,
        "ok": not missing,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Dexter tool schema coverage.")
    parser.add_argument("--json", action="store_true", help="Print the report as JSON.")
    args = parser.parse_args()

    report = build_report()

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Tools registered: {report['tool_count']}")
        print(f"Schemas defined:   {report['schema_count']}")
        print(f"Covered:           {report['covered_count']}")
        print(f"Missing:           {', '.join(report['missing']) if report['missing'] else 'none'}")
        print(f"Extra:             {', '.join(report['extra']) if report['extra'] else 'none'}")

    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
