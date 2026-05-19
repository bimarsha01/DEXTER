# Dexter MCP server package — subprocess tools for filesystem, documents, and Outlook.

# Pre-implementation architecture answers (from codebase read):
# 1. Tool registry routing: load_tools() imports native callables from _TOOL_MODULES into
#    EXECUTOR; execute_tool(func_name, arguments) delegates to EXECUTOR.execute().
# 2. executor.py: loads JSON schema per tool, sanitizes args, jsonschema-validates, checks
#    paths against allowed_file_roots, assesses risk, then runs sync/async callables with timeout.
# 3. config.yaml already has mcp: enabled: false (minimal); expanded in this change.
# 4. Tool count: 44 native tools in _TOOL_MODULES; 44 entries in tool_schemas.json.
# 5. fastmcp was not in requirements.txt before this change (added with pywin32).
