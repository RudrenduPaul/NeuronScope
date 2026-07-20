"""NeuronScope's MCP server: exposes trace, activations, patch, and circuit as MCP tools.

Every tool returns exactly the same pydantic-model-shaped JSON the CLI's --json flag
prints, via `.model_dump()`, so an agent calling this server and a script calling the CLI
get identical documents for identical inputs. Errors are returned as a structured
ErrorResponse dict rather than raised, so a calling agent gets a parseable result instead
of a bare tool-call failure.
"""

from __future__ import annotations

from typing import Annotated, Any

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from neuronscope.backends.transformer_lens import COMPONENT_HOOK_TEMPLATES
from neuronscope.core.registry import UnsupportedModelError
from neuronscope.core.trace import (
    DEFAULT_TOP_K,
    LayerOutOfRangeError,
    PromptTooLongError,
    run_activations,
    run_circuit,
    run_patch,
    run_trace,
)
from neuronscope.schema import ErrorResponse

mcp = FastMCP(
    name="neuronscope",
    instructions=(
        "Trace which neurons and attention heads in an open-weight HuggingFace language "
        "model were responsible for a given output, via TransformerLens. Use `trace` to "
        "find the top contributing heads/neurons for a prompt, `activations` to inspect "
        "raw activation summary stats, `patch` to zero-ablate one component and see the "
        "output delta, and `circuit` for a best-effort automated circuit sketch."
    ),
)


def _error_dict(operation: str, exc: Exception) -> dict[str, Any]:
    if isinstance(exc, UnsupportedModelError):
        error_type = "UnsupportedModelError"
    elif isinstance(exc, PromptTooLongError):
        error_type = "PromptTooLongError"
    elif isinstance(exc, LayerOutOfRangeError):
        error_type = "LayerOutOfRangeError"
    else:
        error_type = type(exc).__name__
    return ErrorResponse(operation=operation, error_type=error_type, message=str(exc)).model_dump()


@mcp.tool(description="Run a forward pass and report the layers/heads/neurons most responsible for the model's output.")
def trace(model: str, prompt: str, top_k: int = DEFAULT_TOP_K) -> dict[str, Any]:
    """Trace which attention heads and MLP neurons drove a model's next-token prediction.

    Args:
        model: Model name TransformerLens's HookedTransformer.from_pretrained supports
            (for example "gpt2", "EleutherAI/pythia-70m").
        prompt: Input text to run through the model.
        top_k: How many top heads and neurons to report.
    """
    try:
        return run_trace(model, prompt, top_k=top_k).model_dump()
    except Exception as exc:  # noqa: BLE001 - tool boundary, return structured error
        return _error_dict("trace", exc)


@mcp.tool(description="Dump summary statistics for raw activation tensors (shape, mean, max-activating position) for inspection.")
def activations(model: str, prompt: str) -> dict[str, Any]:
    """Inspect raw activation tensor summary stats for a prompt.

    Args:
        model: Model name TransformerLens's HookedTransformer.from_pretrained supports.
        prompt: Input text to run through the model.
    """
    try:
        return run_activations(model, prompt).model_dump()
    except Exception as exc:  # noqa: BLE001
        return _error_dict("activations", exc)


@mcp.tool(description="Zero-ablate one component (layer + component name) and report how the output logits changed.")
def patch(
    model: str,
    prompt: str,
    layer: int,
    # Sourced from COMPONENT_HOOK_TEMPLATES, the same dict the CLI's `--component`
    # click.Choice reads its options from, so this tool schema's enum can't drift from
    # what the CLI actually accepts.
    component: Annotated[str, Field(json_schema_extra={"enum": list(COMPONENT_HOOK_TEMPLATES)})],
) -> dict[str, Any]:
    """Ablate one activation and report the output delta.

    Args:
        model: Model name TransformerLens's HookedTransformer.from_pretrained supports.
        prompt: Input text to run through the model.
        layer: Zero-indexed transformer block to patch.
        component: One of resid_pre, resid_mid, resid_post, attn_out, mlp_out, mlp_post.
    """
    try:
        return run_patch(model, prompt, layer, component).model_dump()
    except Exception as exc:  # noqa: BLE001
        return _error_dict("patch", exc)


@mcp.tool(description="Best-effort automated circuit discovery for a single prompt: which heads/neurons chain together to produce the output.")
def circuit(model: str, prompt: str, top_k: int = DEFAULT_TOP_K) -> dict[str, Any]:
    """Sketch an approximate circuit for a prompt via ranked ablation.

    Args:
        model: Model name TransformerLens's HookedTransformer.from_pretrained supports.
        prompt: Input text to run through the model.
        top_k: How many circuit components to report.
    """
    try:
        return run_circuit(model, prompt, top_k=top_k).model_dump()
    except Exception as exc:  # noqa: BLE001
        return _error_dict("circuit", exc)


def run_server() -> None:
    """Start the MCP server on stdio transport. Blocks until the client disconnects."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    run_server()
