"""Tests for Dexter MCP filesystem tools and registry routing."""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import sys
from pathlib import Path
from unittest import mock

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _load_mcp_server(allowed_roots: list[str]):
    os.environ["DEXTER_ALLOWED_ROOTS"] = json.dumps(allowed_roots)
    import mcp_server.dexter_mcp_server as srv

    importlib.reload(srv)
    return srv


@pytest.fixture
def mcp_srv(tmp_path):
    return _load_mcp_server([str(tmp_path)])


class TestMCPFilesystemTools:
    @pytest.mark.asyncio
    async def test_read_file_reads_temp_file(self, mcp_srv, tmp_path):
        target = tmp_path / "hello.txt"
        target.write_text("line one\nline two", encoding="utf-8")
        result = await mcp_srv.read_file(str(target))
        assert result["success"] is True
        assert "line one" in result["content"]
        assert result["lines"] == 2

    @pytest.mark.asyncio
    async def test_read_file_rejects_outside_roots(self, mcp_srv, tmp_path):
        outside = tmp_path.parent / "outside_mcp.txt"
        outside.write_text("secret", encoding="utf-8")
        result = await mcp_srv.read_file(str(outside))
        assert result["success"] is False
        assert "outside" in result["error"].lower() or "allowed" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_read_file_rejects_over_10mb(self, mcp_srv, tmp_path):
        big = tmp_path / "big.bin"
        big.write_bytes(b"x" * (10 * 1024 * 1024 + 1))
        result = await mcp_srv.read_file(str(big))
        assert result["success"] is False
        assert "10MB" in result["error"] or "large" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_write_file_creates_content(self, mcp_srv, tmp_path):
        target = tmp_path / "out.txt"
        result = await mcp_srv.write_file(str(target), "saved text", "overwrite")
        assert result["success"] is True
        assert target.read_text(encoding="utf-8") == "saved text"

    @pytest.mark.asyncio
    async def test_write_file_rejects_invalid_mode(self, mcp_srv, tmp_path):
        target = tmp_path / "bad.txt"
        result = await mcp_srv.write_file(str(target), "x", "replace")
        assert result["success"] is False
        assert "mode" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_list_directory_metadata(self, mcp_srv, tmp_path):
        (tmp_path / "a.py").write_text("print(1)", encoding="utf-8")
        (tmp_path / "subdir").mkdir()
        result = await mcp_srv.list_directory(str(tmp_path), "*.py")
        assert result["success"] is True
        names = {item["name"] for item in result["items"]}
        assert "a.py" in names
        file_item = next(i for i in result["items"] if i["name"] == "a.py")
        assert file_item["type"] == "file"
        assert file_item["extension"] == ".py"

    @pytest.mark.asyncio
    async def test_search_files_filename_match(self, mcp_srv, tmp_path):
        nested = tmp_path / "proj"
        nested.mkdir()
        (nested / "findme.py").write_text("pass", encoding="utf-8")
        result = await mcp_srv.search_files(str(tmp_path), "findme", file_types=["py"])
        assert result["success"] is True
        assert any(m["name"] == "findme.py" for m in result["matches"])

    @pytest.mark.asyncio
    async def test_search_files_content_match(self, mcp_srv, tmp_path):
        doc = tmp_path / "notes.txt"
        doc.write_text("alpha UNIQUE_TOKEN beta", encoding="utf-8")
        result = await mcp_srv.search_files(
            str(tmp_path), "UNIQUE_TOKEN", search_content=True
        )
        assert result["success"] is True
        assert any(m["match_type"] == "content" for m in result["matches"])

    @pytest.mark.asyncio
    async def test_create_directory_idempotent(self, mcp_srv, tmp_path):
        target = tmp_path / "new" / "nested"
        first = await mcp_srv.create_directory(str(target))
        second = await mcp_srv.create_directory(str(target))
        assert first["success"] is True
        assert first["already_existed"] is False
        assert second["success"] is True
        assert second["already_existed"] is True

    @pytest.mark.asyncio
    async def test_path_traversal_rejected(self, mcp_srv, tmp_path):
        evil = tmp_path / ".." / ".." / "etc" / "passwd"
        result = await mcp_srv.read_file(str(evil))
        assert result["success"] is False


class TestMCPClientAndRegistry:
    @pytest.mark.asyncio
    async def test_client_not_ready_returns_error(self):
        from mcp_server.client import MCPClient

        client = MCPClient(
            server_script="mcp_server/dexter_mcp_server.py",
            allowed_roots=["/tmp"],
        )
        result = await client.call_tool("read_file", {"path": "x"})
        assert result.success is False
        assert "not ready" in (result.error or "").lower()

    @pytest.mark.asyncio
    async def test_registry_routes_mcp_tools(self):
        import tools.registry as registry

        registry._TOOLS_LOADED = False
        registry._mcp_ready = True
        registry._mcp_client = mock.AsyncMock()
        registry._mcp_client.call_tool.return_value = mock.Mock(
            success=True,
            data={"success": True, "content": "ok"},
            error=None,
            duration_ms=1.0,
        )

        with mock.patch.object(
            registry.EXECUTOR, "execute", new_callable=mock.AsyncMock
        ) as native_execute:
            with mock.patch.object(registry.EXECUTOR, "_validate_paths"):
                out = await registry.execute_tool(
                    "mcp_read_file", {"path": "C:/allowed/file.txt"}
                )

        registry._mcp_client.call_tool.assert_awaited_once_with(
            "read_file", {"path": "C:/allowed/file.txt"}
        )
        native_execute.assert_not_awaited()
        assert "ok" in str(out)

        registry._mcp_ready = False
        registry._mcp_client = None
        registry._TOOLS_LOADED = False
