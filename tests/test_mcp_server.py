"""Tests for the MCP server: tool registration and one real round-trip call.

The round-trip test actually runs gpt2 through the `trace` tool the way an MCP client
would call it, rather than calling the underlying `run_trace` function directly, so a
regression in the FastMCP tool wiring itself would be caught here.
"""

import asyncio

import pytest

from neuronscope.mcp_server import mcp

EXPECTED_TOOLS = {"trace", "activations", "patch", "circuit"}


def test_all_four_tools_are_registered():
    tools = asyncio.run(mcp.list_tools())
    tool_names = {tool.name for tool in tools}
    assert EXPECTED_TOOLS.issubset(tool_names)


def test_tool_schemas_document_required_parameters():
    tools = {tool.name: tool for tool in asyncio.run(mcp.list_tools())}

    trace_schema = tools["trace"].inputSchema
    assert "model" in trace_schema["properties"]
    assert "prompt" in trace_schema["properties"]

    patch_schema = tools["patch"].inputSchema
    assert "layer" in patch_schema["properties"]
    assert "component" in patch_schema["properties"]


def test_trace_tool_real_round_trip():
    result = asyncio.run(
        mcp.call_tool("trace", {"model": "gpt2", "prompt": "The capital of France is"})
    )

    # FastMCP tools returning a dict get wrapped as structured content; unwrap either shape.
    payload = result[1] if isinstance(result, tuple) else result
    if isinstance(payload, dict) and "result" in payload:
        payload = payload["result"]

    assert payload["schema_version"] == 1
    assert payload["operation"] == "trace"
    assert payload["model"]["backend"] == "transformer_lens"
    assert len(payload["top_heads"]) > 0
    assert len(payload["top_neurons"]) > 0


def test_unsupported_model_returns_structured_error_not_an_exception():
    result = asyncio.run(
        mcp.call_tool("trace", {"model": "not-a-real-model-xyz", "prompt": "hello"})
    )
    payload = result[1] if isinstance(result, tuple) else result
    if isinstance(payload, dict) and "result" in payload:
        payload = payload["result"]

    assert payload["error_type"] == "UnsupportedModelError"
    assert "transformer_lens" in payload["message"]
