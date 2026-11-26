"""Async and sync stdio MCP client bridge to the FastMCP tool server."""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPToolTimeoutError(TimeoutError):
    """Raised when an MCP stdio tool call exceeds tool_timeout_seconds."""


class MCPToolClient:
    """Invokes FastMCP tools over stdio subprocess with timeout, retry, and task-safe lifecycle."""

    def __init__(
        self,
        *,
        db_path: str | Path,
        rules_path: str | Path = "config/business_rules.yaml",
        timeout_seconds: float = 10.0,
        max_retries: int = 2,
    ) -> None:
        self.db_path = str(db_path)
        self.rules_path = str(rules_path)
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self._project_root = str(Path(__file__).resolve().parent.parent.parent)
        self._queue: asyncio.Queue[tuple[str, dict[str, Any], asyncio.Future[Any]] | None] | None = None
        self._worker_task: asyncio.Task[None] | None = None

    def _server_params(self) -> StdioServerParameters:
        env = dict(os.environ)
        env["APP_DB_PATH"] = self.db_path
        env["BUSINESS_RULES_PATH"] = self.rules_path
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{self._project_root}{os.pathsep}{existing_pp}" if existing_pp else self._project_root
        )
        return StdioServerParameters(
            command=sys.executable,
            args=["-m", "src.mcp_server.server"],
            env=env,
            cwd=self._project_root,
        )

    async def _session_worker(self, ready: asyncio.Future[None]) -> None:
        """Own stdio_client and ClientSession inside a single task for anyio cancel-scope safety."""
        assert self._queue is not None
        try:
            async with (
                stdio_client(self._server_params()) as (read_stream, write_stream),
                ClientSession(read_stream, write_stream) as session,
            ):
                async with asyncio.timeout(self.timeout_seconds):
                    await session.initialize()
                if not ready.done():
                    ready.set_result(None)

                while True:
                    item = await self._queue.get()
                    if item is None:
                        break
                    op, payload, fut = item
                    try:
                        if op == "__list_tools__":
                            async with asyncio.timeout(self.timeout_seconds):
                                resp = await session.list_tools()
                            if not fut.done():
                                fut.set_result([t.name for t in resp.tools])
                        else:
                            async with asyncio.timeout(self.timeout_seconds):
                                raw_res = await session.call_tool(op, payload)
                            if not fut.done():
                                fut.set_result(self._parse_tool_result(raw_res))
                    except TimeoutError as exc:
                        err = MCPToolTimeoutError(
                            f"MCP tool '{op}' timed out after {self.timeout_seconds}s"
                        )
                        err.__cause__ = exc
                        if not fut.done():
                            fut.set_exception(err)
                    except Exception as exc:  # noqa: BLE001
                        if not fut.done():
                            fut.set_exception(exc)
        except Exception as exc:  # noqa: BLE001
            if not ready.done():
                ready.set_exception(exc)

    async def __aenter__(self) -> MCPToolClient:
        loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue()
        ready: asyncio.Future[None] = loop.create_future()
        self._worker_task = loop.create_task(self._session_worker(ready))
        await ready
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._queue is not None:
            await self._queue.put(None)
        if self._worker_task is not None:
            with contextlib.suppress(Exception):
                await self._worker_task
        self._queue = None
        self._worker_task = None

    @staticmethod
    def _parse_tool_result(result: Any) -> dict[str, Any]:
        if getattr(result, "structuredContent", None) and isinstance(
            result.structuredContent, dict
        ):
            payload = result.structuredContent
            if "result" in payload and isinstance(payload["result"], dict):
                return payload["result"]
            return payload

        for block in getattr(result, "content", []) or []:
            text = getattr(block, "text", None)
            if text:
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    return {"ok": not bool(getattr(result, "isError", False)), "raw_text": text}
        return {"ok": not bool(getattr(result, "isError", False))}

    async def _send_to_worker(self, op: str, arguments: dict[str, Any]) -> Any:
        if self._queue is None or self._worker_task is None or self._worker_task.done():
            async with self:
                return await self._send_to_worker(op, arguments)
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[Any] = loop.create_future()
        await self._queue.put((op, arguments, fut))
        return await fut

    async def list_tools(self) -> list[str]:
        """Return tool names exposed by the FastMCP server."""
        return await self._send_to_worker("__list_tools__", {})

    async def alist_tools(self) -> list[str]:
        """Alias for list_tools."""
        return await self.list_tools()

    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> tuple[dict[str, Any], int]:
        """Execute an MCP tool over stdio with retries, exponential backoff, and timeout."""
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                payload = await self._send_to_worker(tool_name, arguments)
                return payload, attempt
            except TimeoutError as exc:
                last_error = exc
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                await self.__aexit__(None, None, None)
            if attempt < self.max_retries:
                await asyncio.sleep(0.05 * (2**attempt))
        raise last_error if last_error is not None else RuntimeError("MCP call failed")

    def call_tool_in_process(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Direct deterministic in-process tool dispatch helper."""
        from src.config import load_app_config
        from src.database.repositories import SupportRepository
        from src.mcp_server.order_tools import handle_get_order, handle_get_order_status
        from src.mcp_server.refund_tools import (
            handle_check_refund_eligibility,
            handle_create_refund,
            handle_get_refund_status,
        )
        from src.mcp_server.support_tools import (
            handle_create_support_ticket,
            handle_get_support_ticket,
        )

        repo = SupportRepository(self.db_path)
        rules = load_app_config(self.rules_path)
        dispatch = {
            "get_order": lambda a: handle_get_order(repo, rules, **a),
            "get_order_status": lambda a: handle_get_order_status(repo, rules, **a),
            "check_refund_eligibility": lambda a: handle_check_refund_eligibility(repo, rules, **a),
            "create_refund": lambda a: handle_create_refund(repo, rules, **a),
            "get_refund_status": lambda a: handle_get_refund_status(repo, rules, **a),
            "create_support_ticket": lambda a: handle_create_support_ticket(repo, rules, **a),
            "get_support_ticket": lambda a: handle_get_support_ticket(repo, rules, **a),
        }
        if tool_name not in dispatch:
            return {"ok": False, "reason": f"Unknown tool: {tool_name}"}
        return dispatch[tool_name](arguments)

    async def acall_tool(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> tuple[dict[str, Any], int]:
        """Alias for call_tool."""
        return await self.call_tool(tool_name, arguments)

    def call_tool_sync(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> tuple[dict[str, Any], int]:
        """Synchronous convenience wrapper for non-async callers."""
        try:
            asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, self.call_tool(tool_name, arguments)).result()
        except RuntimeError:
            return asyncio.run(self.call_tool(tool_name, arguments))
