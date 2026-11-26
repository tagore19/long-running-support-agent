"""LangGraph StateGraph compilation with SqliteSaver / AsyncSqliteSaver and MCP stdio."""

from __future__ import annotations

import asyncio
import contextlib
import sqlite3
from pathlib import Path
from typing import Any

import aiosqlite
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph

from src.agent.nodes import SupportGraphNodes
from src.agent.routing import route_after_agent, route_after_record_event
from src.agent.state import AgentState
from src.config import AppConfig, Settings, get_settings, load_app_config
from src.database.repositories import SupportRepository
from src.llm import LLMClient
from src.mcp_server.client import MCPToolClient
from src.observability.tracing import StructuredTracer


def build_support_graph(nodes: SupportGraphNodes, checkpointer: Any = None):
    """Compile the deterministic LangGraph customer support workflow."""
    graph = StateGraph(AgentState)
    graph.add_node("validate_input", nodes.validate_input)
    graph.add_node("load_state", nodes.load_state)
    graph.add_node("retrieve_memory", nodes.retrieve_memory)
    graph.add_node("build_context", nodes.build_context)
    graph.add_node("agent", nodes.agent)
    graph.add_node("validate_tool_call", nodes.validate_tool_call)
    graph.add_node("execute_tool", nodes.execute_tool)
    graph.add_node("record_event", nodes.record_event)
    graph.add_node("final_response", nodes.final_response)
    graph.add_node("compact_memory", nodes.compact_memory)

    graph.add_edge(START, "validate_input")
    graph.add_edge("validate_input", "load_state")
    graph.add_edge("load_state", "retrieve_memory")
    graph.add_edge("retrieve_memory", "build_context")
    graph.add_edge("build_context", "agent")
    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "validate_tool_call": "validate_tool_call",
            "final_response": "final_response",
        },
    )
    graph.add_edge("validate_tool_call", "execute_tool")
    graph.add_edge("execute_tool", "record_event")
    graph.add_conditional_edges(
        "record_event",
        route_after_record_event,
        {
            "agent": "agent",
            "final_response": "final_response",
        },
    )
    graph.add_edge("final_response", "compact_memory")
    graph.add_edge("compact_memory", END)

    return graph.compile(checkpointer=checkpointer)


class SupportAgentRunner:
    """High-level runner managing SQLite checkpointer and persistent MCP stdio client."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        rules: AppConfig | None = None,
        app_db_path: str | Path | None = None,
        checkpoint_db_path: str | Path | None = None,
        trace_log_path: str | Path | None = None,
        mode: str | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.rules = rules or load_app_config(self.settings.business_rules_path)
        self.mode = mode or self.settings.agent_mode

        self.app_db_path = (
            Path(app_db_path)
            if app_db_path
            else self.settings.resolve_path(self.settings.effective_db_path)
        )
        self.checkpoint_db_path = (
            Path(checkpoint_db_path)
            if checkpoint_db_path
            else self.settings.resolve_path(self.settings.checkpoint_db_path)
        )
        self.trace_log_path = (
            Path(trace_log_path)
            if trace_log_path
            else self.settings.resolve_path(self.settings.trace_log_path)
        )

        self.checkpoint_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.repo = SupportRepository(self.app_db_path)
        self.tracer = StructuredTracer(self.trace_log_path)
        self.llm = LLMClient(self.settings, self.rules)
        self.mcp_client = MCPToolClient(
            db_path=self.app_db_path,
            rules_path=self.settings.resolve_path(self.settings.business_rules_path),
            timeout_seconds=self.rules.guardrails.tool_timeout_seconds,
            max_retries=self.rules.guardrails.max_retries,
        )
        self.nodes = SupportGraphNodes(
            settings=self.settings,
            rules=self.rules,
            repo=self.repo,
            llm=self.llm,
            mcp_client=self.mcp_client,
            tracer=self.tracer,
        )

        self._ckpt_conn = sqlite3.connect(str(self.checkpoint_db_path), check_same_thread=False)
        self.checkpointer = SqliteSaver(self._ckpt_conn)
        self.graph = build_support_graph(self.nodes, checkpointer=self.checkpointer)

        self._async_ckpt_conn: aiosqlite.Connection | None = None
        self.async_checkpointer: AsyncSqliteSaver | None = None
        self.async_graph: Any = None

    async def __aenter__(self) -> SupportAgentRunner:
        self._async_ckpt_conn = await aiosqlite.connect(str(self.checkpoint_db_path))
        self.async_checkpointer = AsyncSqliteSaver(self._async_ckpt_conn)
        await self.async_checkpointer.setup()
        self.async_graph = build_support_graph(self.nodes, checkpointer=self.async_checkpointer)
        await self.mcp_client.__aenter__()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.mcp_client.__aexit__(exc_type, exc, tb)
        if self._async_ckpt_conn is not None:
            await self._async_ckpt_conn.close()
            self._async_ckpt_conn = None
        self.async_graph = None
        self.close()

    def close(self) -> None:
        """Close synchronous SQLite checkpointer connection."""
        with contextlib.suppress(Exception):
            self._ckpt_conn.close()

    async def arun_turn(
        self,
        *,
        thread_id: str,
        customer_id: str,
        user_message: str,
        mode: str | None = None,
        document_text: str | None = None,
        image_text: str | None = None,
    ) -> dict[str, Any]:
        """Execute a single conversational turn through LangGraph with checkpointing."""
        config = {"configurable": {"thread_id": thread_id}}
        initial_input: AgentState = {
            "thread_id": thread_id,
            "customer_id": customer_id,
            "user_message": user_message,
            "mode": mode or self.mode,
            "document_text": document_text,
            "image_text": image_text,
        }
        if self.async_graph is not None:
            return await self.async_graph.ainvoke(initial_input, config=config)
        async with aiosqlite.connect(str(self.checkpoint_db_path)) as conn:
            ckpt = AsyncSqliteSaver(conn)
            await ckpt.setup()
            compiled = build_support_graph(self.nodes, checkpointer=ckpt)
            return await compiled.ainvoke(initial_input, config=config)

    def run_turn(
        self,
        *,
        thread_id: str,
        customer_id: str,
        user_message: str,
        mode: str | None = None,
    ) -> dict[str, Any]:
        """Synchronous convenience wrapper around arun_turn."""
        return asyncio.run(
            self.arun_turn(
                thread_id=thread_id,
                customer_id=customer_id,
                user_message=user_message,
                mode=mode,
            )
        )
