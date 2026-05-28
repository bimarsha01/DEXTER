"""Async MCP client — subprocess stdio transport for Dexter tool registry."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from utils.config import get_workspace_root
from utils.logger import get_logger

logger = get_logger("mcp_client")

MCP_TOOL_NAMES = [
    "mcp_read_file",
    "mcp_write_file",
    "mcp_list_directory",
    "mcp_search_files",
    "mcp_create_directory",
    "mcp_read_word_doc",
    "mcp_read_excel",
    "mcp_read_pdf",
    "mcp_list_emails",
    "mcp_create_email_draft",
    "mcp_list_calendar_events",
]


@dataclass
class MCPToolResult:
    success: bool
    data: Any
    error: Optional[str]
    tool_name: str
    duration_ms: float


class MCPClient:
    """
    Manages the lifecycle of the MCP server subprocess and provides async tool calling.
    The server is a separate Python process communicating via JSON over stdio.
    Auto-restarts if the server crashes.
    """

    def __init__(
        self,
        server_script: str,
        allowed_roots: list,
        timeout: float = 15.0,
    ):
        script_path = Path(server_script)
        if not script_path.is_absolute():
            script_path = Path(get_workspace_root()) / script_path
        self._server_script = script_path
        self._allowed_roots = allowed_roots
        self._timeout = timeout
        self._process: Optional[asyncio.subprocess.Process] = None
        self._ready = False
        self._lock = asyncio.Lock()
        self._restart_count = 0
        self._max_restarts = 3
        self._request_id = 0

    async def start(self) -> bool:
        """Launch the MCP server subprocess. Returns True if started successfully."""
        try:
            env = {
                "DEXTER_ALLOWED_ROOTS": json.dumps([str(r) for r in self._allowed_roots]),
                "PYTHONPATH": str(self._server_script.parent.parent),
            }
            full_env = {**os.environ, **env}

            self._process = await asyncio.create_subprocess_exec(
                sys.executable,
                str(self._server_script),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=full_env,
            )

            await asyncio.sleep(1.0)

            stderr_lines: list[str] = []
            if self._process.stderr:
                for _ in range(5):
                    try:
                        line = await asyncio.wait_for(self._process.stderr.readline(), timeout=0.05)
                    except asyncio.TimeoutError:
                        break
                    if not line:
                        break
                    stderr_lines.append(line.decode("utf-8", errors="ignore").strip())
            if stderr_lines:
                logger.warning("mcp_server_startup_stderr", lines=stderr_lines)

            if self._process.returncode is not None:
                stderr = b""
                if self._process.stderr:
                    try:
                        stderr = await asyncio.wait_for(
                            self._process.stderr.read(500),
                            timeout=2.0,
                        )
                    except asyncio.TimeoutError:
                        pass
                logger.error(
                    "mcp_server_failed_to_start",
                    stderr=stderr.decode("utf-8", errors="ignore"),
                )
                return False

            initialized = await self._initialize_session()
            if not initialized:
                return False

            self._ready = True
            logger.info(
                "mcp_server_started",
                pid=self._process.pid,
                script=str(self._server_script),
            )
            return True

        except Exception as e:
            logger.error("mcp_server_start_error", error=str(e), exc_info=True)
            return False

    async def _initialize_session(self) -> bool:
        """MCP handshake before tool calls."""
        try:
            response = await self._send_request(
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "dexter", "version": "1.0"},
                },
                timeout=10.0,
            )
            if response is None or "error" in response:
                return False
            await self._send_notification("notifications/initialized", {})
            return True
        except Exception as e:
            logger.error("mcp_initialize_failed", error=str(e), exc_info=True)
            return False

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    async def _send_notification(self, method: str, params: dict) -> None:
        if not self._process or not self._process.stdin:
            return
        payload = json.dumps(
            {"jsonrpc": "2.0", "method": method, "params": params}
        ) + "\n"
        self._process.stdin.write(payload.encode("utf-8"))
        await self._process.stdin.drain()

    async def _send_request(
        self,
        method: str,
        params: dict,
        timeout: Optional[float] = None,
    ) -> Optional[dict]:
        if not self._process or not self._process.stdin or not self._process.stdout:
            return None

        req_id = self._next_id()
        payload = json.dumps(
            {
                "jsonrpc": "2.0",
                "method": method,
                "params": params,
                "id": req_id,
            }
        ) + "\n"

        self._process.stdin.write(payload.encode("utf-8"))
        await self._process.stdin.drain()

        wait_timeout = timeout if timeout is not None else self._timeout
        while True:
            response_line = await asyncio.wait_for(
                self._process.stdout.readline(),
                timeout=wait_timeout,
            )
            if not response_line:
                return None
            try:
                response = json.loads(response_line.decode("utf-8"))
            except json.JSONDecodeError:
                continue
            if response.get("id") == req_id:
                return response
            # Ignore notifications / other responses

    async def call_tool(self, name: str, args: dict) -> MCPToolResult:
        """Call an MCP tool and return the result. Auto-restarts server on crash."""
        start_time = time.perf_counter()

        if not self._ready:
            return MCPToolResult(
                success=False,
                data=None,
                error="MCP server not ready. Try again in a moment.",
                tool_name=name,
                duration_ms=0,
            )

        async with self._lock:
            try:
                response = await self._send_request(
                    "tools/call",
                    {"name": name, "arguments": args or {}},
                )

                duration = (time.perf_counter() - start_time) * 1000

                if response is None:
                    await self._restart()
                    return MCPToolResult(
                        success=False,
                        data=None,
                        error="No response from MCP server.",
                        tool_name=name,
                        duration_ms=duration,
                    )

                if "error" in response:
                    err = response["error"]
                    message = err.get("message", "Unknown error") if isinstance(err, dict) else str(err)
                    return MCPToolResult(
                        success=False,
                        data=None,
                        error=message,
                        tool_name=name,
                        duration_ms=duration,
                    )

                result_payload = response.get("result", {})
                structured = result_payload
                if isinstance(result_payload, dict) and "structuredContent" in result_payload:
                    structured = result_payload["structuredContent"]
                elif isinstance(result_payload, dict) and "content" in result_payload:
                    blocks = result_payload.get("content") or []
                    if blocks and isinstance(blocks[0], dict) and "text" in blocks[0]:
                        try:
                            structured = json.loads(blocks[0]["text"])
                        except json.JSONDecodeError:
                            structured = {"success": True, "content": blocks[0]["text"]}

                success = True
                error = None
                if isinstance(structured, dict):
                    success = structured.get("success", True)
                    error = structured.get("error")

                logger.info(
                    "mcp_tool_called",
                    tool=name,
                    success=success,
                    duration_ms=round(duration, 1),
                )

                return MCPToolResult(
                    success=success,
                    data=structured,
                    error=error,
                    tool_name=name,
                    duration_ms=duration,
                )

            except asyncio.TimeoutError:
                logger.error("mcp_tool_timeout", tool=name, timeout=self._timeout)
                await self._restart()
                duration = (time.perf_counter() - start_time) * 1000
                return MCPToolResult(
                    success=False,
                    data=None,
                    error=f"Tool timed out after {self._timeout}s.",
                    tool_name=name,
                    duration_ms=duration,
                )

            except Exception as e:
                logger.error("mcp_tool_error", tool=name, error=str(e), exc_info=True)
                if self._process and self._process.returncode is not None:
                    await self._restart()
                duration = (time.perf_counter() - start_time) * 1000
                return MCPToolResult(
                    success=False,
                    data=None,
                    error=str(e),
                    tool_name=name,
                    duration_ms=duration,
                )

    async def _restart(self) -> None:
        if self._restart_count >= self._max_restarts:
            logger.error(
                "mcp_server_max_restarts_reached",
                count=self._restart_count,
            )
            self._ready = False
            return

        self._restart_count += 1
        self._ready = False

        logger.warning("mcp_server_restarting", attempt=self._restart_count)

        try:
            if self._process and self._process.returncode is None:
                self._process.terminate()
                await asyncio.sleep(1.0)
        except Exception:
            pass

        success = await self.start()
        if success:
            self._restart_count = 0

    async def stop(self) -> None:
        self._ready = False
        if self._process and self._process.returncode is None:
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=3.0)
            except asyncio.TimeoutError:
                self._process.kill()
        logger.info("mcp_server_stopped")

    def get_available_tools(self) -> list:
        """Return MCP tool names registered in Dexter (mcp_ prefix)."""
        return list(MCP_TOOL_NAMES)
