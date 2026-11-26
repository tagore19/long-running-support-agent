"""Conditional routing functions for the LangGraph support workflow."""

from __future__ import annotations

from src.agent.state import AgentState


def route_after_agent(state: AgentState) -> str:
    """Route to tool validation if a tool call is pending, otherwise finalize."""
    if state.get("pending_tool_call") is not None:
        return "validate_tool_call"
    return "final_response"


def route_after_record_event(state: AgentState) -> str:
    """Loop back to agent if within step/tool guardrails, else finalize."""
    if state.get("final_response") is not None:
        return "final_response"
    return "agent"
