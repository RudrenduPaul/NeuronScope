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
from neuronscope.core.limits import ModelTooLargeError, TooManyConcurrentModelLoadsError
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
    elif isinstance(exc, ModelTooLargeError):
        error_type = "ModelTooLargeError"
    elif isinstance(exc, TooManyConcurrentModelLoadsError):
        error_type = "TooManyConcurrentModelLoadsError"
    else:
        error_type = type(exc).__name__
    return ErrorResponse(operation=operation, error_type=error_type, message=str(exc)).model_dump()


@mcp.tool(
    description=(
        "Run a forward pass of a small-to-medium open-weight language model (via "
        "TransformerLens) on one prompt, and report which attention heads and MLP neurons "
        "were most responsible for its predicted next token: heads ranked by direct logit "
        "attribution, neurons ranked by activation magnitude at the final prompt position. "
        "Call this to answer 'why did the model predict X' for a specific prompt. It only "
        "works on models TransformerLens's HookedTransformer.from_pretrained supports "
        "(GPT-2, Pythia, Llama, Gemma, Qwen, and similar open-weight checkpoints), not "
        "closed-source APIs like OpenAI or Anthropic models. Read-only and deterministic "
        "for a given model, prompt, and top_k: it writes nothing except the model's own "
        "weights, which HuggingFace Hub downloads to a local cache (~/.cache/huggingface) "
        "the first time a given model name is requested (needs network access that one "
        "time; later calls for the same model run offline from cache). Runs on CPU by "
        "default and can be slow for large models. On failure (an unsupported model name, "
        "or a prompt longer than the model's context window) it returns a structured error "
        "object instead of raising, so the tool call itself never fails silently. "
        "Parameters: model (str) is any name HookedTransformer.from_pretrained accepts, "
        "e.g. 'gpt2' or 'EleutherAI/pythia-70m'; prompt (str) is the input text; top_k "
        "(int, default 10) caps how many top heads and neurons are returned. Example call: "
        "model='gpt2', prompt='The capital of France is Paris. The capital of Japan is', "
        "top_k=5. Returns JSON with schema_version, operation, model (resolved name, "
        "backend, device, and layer/head/dimension counts), prompt, predicted_token, "
        "predicted_token_id, top_neurons (list of {layer, neuron_index, activation}), and "
        "top_heads (list of {layer, head_index, logit_attribution})."
    )
)
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


@mcp.tool(
    description=(
        "Dump raw per-layer activation summary statistics (shape, mean, std, min/max, and "
        "the max-activating sequence position) for one prompt run through an open-weight "
        "TransformerLens-supported model, covering every layer's residual stream, MLP "
        "neuron activations, and attention pattern. Call this when trace's top-k ranking "
        "isn't enough detail and you need the raw scale/shape of a specific hook point "
        "before deciding what to inspect further or patch with the patch tool. Same model "
        "constraint as trace: only models HookedTransformer.from_pretrained supports. "
        "Read-only and deterministic for a given model and prompt; the only side effect is "
        "HuggingFace Hub caching the model weights locally on first use of a given model "
        "name, which needs network access that one time. Runs on CPU by default. Output "
        "size scales with model depth since it returns stats for every layer, not a "
        "top-k subset, so it can be verbose for large models. On failure (unsupported "
        "model name, prompt too long for the context window) it returns a structured error "
        "object rather than raising. Parameters: model (str), any name "
        "HookedTransformer.from_pretrained accepts, e.g. 'gpt2'; prompt (str), the input "
        "text. Example call: model='gpt2', prompt='The capital of France is Paris. The "
        "capital of Japan is'. Returns JSON with schema_version, operation, model, prompt, "
        "n_tokens, and activations (list of {hook_name, layer, shape, mean, std, "
        "max_value, max_position, min_value}, one entry per hook point)."
    )
)
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


@mcp.tool(
    description=(
        "Zero-ablate one component (a single transformer block's layer plus a component "
        "type such as an attention output or MLP output) in an open-weight TransformerLens "
        "model's forward pass, and report how the predicted token and its logit changed "
        "relative to the unablated baseline. This is a minimal causal intervention: use it "
        "to test whether a component trace or circuit flagged as correlated with a "
        "prediction is actually causally responsible for it. Call it after trace or "
        "circuit has surfaced a candidate layer/component; it does not search for "
        "candidates itself. Read-only in the sense that it writes no files and the "
        "ablation only affects that single in-memory forward pass, nothing persists across "
        "calls; the same HuggingFace model-weight caching and CPU-by-default notes as "
        "trace apply. Deterministic for a given model, prompt, layer, and component. On "
        "failure it returns a structured error object instead of raising: an out-of-range "
        "layer raises LayerOutOfRangeError, an unsupported model name raises "
        "UnsupportedModelError, and a prompt exceeding the context window raises "
        "PromptTooLongError, all surfaced the same way. Parameters: model (str); prompt "
        "(str); layer (int), the zero-indexed transformer block to patch; component (str), "
        "one of resid_pre, resid_mid, resid_post, attn_out, mlp_out, mlp_post. Example "
        "call: model='gpt2', prompt='The capital of France is Paris. The capital of Japan "
        "is', layer=9, component='attn_out'. Returns JSON with schema_version, operation, "
        "model, prompt, layer, component, ablation_type ('zero'), "
        "baseline_predicted_token, baseline_predicted_token_id, baseline_top_logit, "
        "patched_predicted_token, patched_predicted_token_id, patched_top_logit, "
        "logit_delta, and prediction_changed (bool)."
    )
)
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


@mcp.tool(
    description=(
        "Sketch a best-effort automated circuit for one prompt on an open-weight "
        "TransformerLens model: ranks candidate attention heads and MLP neurons by direct "
        "logit attribution, then measures each candidate's individual causal effect via "
        "single-component zero-ablation, so the result reflects components that actually "
        "move the prediction, not just ones correlated with it. Call this when trace's "
        "correlational ranking isn't enough and you want a causal pass across multiple "
        "candidates without manually calling patch on each one. This is NOT full "
        "path-patching with clean/corrupted prompt pairs and does not capture interaction "
        "effects between components; the response's own method field restates this caveat "
        "so a caller doesn't have to trust prose alone. For rigorous transcoder-based "
        "circuit discovery on a fixed set of supported models, use a dedicated tool such "
        "as Anthropic's circuit-tracer instead. Read-only, with the same model-weight "
        "caching, network-on-first-use, and CPU-by-default behavior as trace; more "
        "expensive than trace since it runs one extra forward pass per candidate "
        "component being ablated. Deterministic for a given model, prompt, and top_k. On "
        "failure it returns a structured error object rather than raising. Parameters: "
        "model (str); prompt (str); top_k (int, default 10), how many top-attributed "
        "components to test via ablation. Example call: model='gpt2', prompt='The capital "
        "of France is Paris. The capital of Japan is', top_k=5. Returns JSON with "
        "schema_version, operation, model, prompt, predicted_token, predicted_token_id, "
        "components (list of {layer, component_type: 'head' or 'neuron', index, "
        "logit_drop_on_ablation}), and method (a string explaining the approximation)."
    )
)
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


# Alias for the `neuronscope-mcp` console-script entry point (pyproject.toml
# [project.scripts]), so this module works both as `neuronscope mcp-server`
# (the CLI subcommand) and as its own standalone binary.
main = run_server


if __name__ == "__main__":
    main()
