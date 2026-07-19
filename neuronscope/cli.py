"""NeuronScope's command-line interface.

Every command shares the same three exit codes: 0 on success, 1 on a general error
(including a prompt that is too long for the model's context window), 2 when the
requested model is not supported by any registered backend.
"""

from __future__ import annotations

import sys

import click
from rich.console import Console
from rich.table import Table

from neuronscope import __version__
from neuronscope.core.registry import UnsupportedModelError
from neuronscope.core.trace import (
    PromptTooLongError,
    run_activations,
    run_circuit,
    run_patch,
    run_trace,
)
from neuronscope.schema import (
    ActivationsResponse,
    CircuitResponse,
    ErrorResponse,
    PatchResponse,
    TraceResponse,
)

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_UNSUPPORTED_MODEL = 2

console = Console()
error_console = Console(stderr=True)


def _handle_error(operation: str, exc: Exception, as_json: bool) -> int:
    if isinstance(exc, UnsupportedModelError):
        error_type = "UnsupportedModelError"
        exit_code = EXIT_UNSUPPORTED_MODEL
    elif isinstance(exc, PromptTooLongError):
        error_type = "PromptTooLongError"
        exit_code = EXIT_ERROR
    else:
        error_type = type(exc).__name__
        exit_code = EXIT_ERROR

    if as_json:
        response = ErrorResponse(operation=operation, error_type=error_type, message=str(exc))
        click.echo(response.model_dump_json(indent=2))
    else:
        error_console.print(f"[bold red]Error ({error_type}):[/bold red] {exc}")
    return exit_code


@click.group()
@click.version_option(version=__version__, prog_name="neuronscope")
def main() -> None:
    """NeuronScope: trace which neurons and attention heads drove a language model's output.

    Wraps TransformerLens with a CLI and an MCP server, for open-weight HuggingFace models
    that TransformerLens's HookedTransformer supports (run `neuronscope trace --help` to
    see how unsupported models are reported).
    """


@main.command()
@click.argument("model")
@click.argument("prompt")
@click.option("--top-k", default=10, show_default=True, help="How many top neurons/heads to report.")
@click.option("--json", "as_json", is_flag=True, help="Output a schema-versioned JSON document instead of a table.")
def trace(model: str, prompt: str, top_k: int, as_json: bool) -> None:
    """Run a forward pass and report the layers/heads/neurons most responsible for the output.

    MODEL is any model name TransformerLens's HookedTransformer.from_pretrained supports
    (for example gpt2, EleutherAI/pythia-70m). PROMPT is the input text to run through the
    model. Heads are ranked by direct logit attribution to the predicted next token; neurons
    are ranked by activation magnitude at the final prompt position.
    """
    try:
        result: TraceResponse = run_trace(model, prompt, top_k=top_k)
    except Exception as exc:  # noqa: BLE001 - CLI boundary, must not leak raw tracebacks
        sys.exit(_handle_error("trace", exc, as_json))

    if as_json:
        click.echo(result.model_dump_json(indent=2))
        return

    console.print(f"[bold]Prompt:[/bold] {result.prompt}")
    console.print(f"[bold]Predicted next token:[/bold] {result.predicted_token!r}")

    heads_table = Table(title="Top attention heads (by direct logit attribution)")
    heads_table.add_column("Layer", justify="right")
    heads_table.add_column("Head", justify="right")
    heads_table.add_column("Logit attribution", justify="right")
    for h in result.top_heads:
        heads_table.add_row(str(h.layer), str(h.head_index), f"{h.logit_attribution:.4f}")
    console.print(heads_table)

    neurons_table = Table(title="Top MLP neurons (by activation magnitude)")
    neurons_table.add_column("Layer", justify="right")
    neurons_table.add_column("Neuron", justify="right")
    neurons_table.add_column("Activation", justify="right")
    for n in result.top_neurons:
        neurons_table.add_row(str(n.layer), str(n.neuron_index), f"{n.activation:.4f}")
    console.print(neurons_table)


@main.command()
@click.argument("model")
@click.argument("prompt")
@click.option("--json", "as_json", is_flag=True, help="Output a schema-versioned JSON document instead of a table.")
def activations(model: str, prompt: str, as_json: bool) -> None:
    """Dump summary statistics for raw activation tensors, for inspection.

    Reports shape, mean, std, min/max, and the sequence position of the max-magnitude
    activation for each layer's residual stream, MLP neuron activations, and attention
    pattern.
    """
    try:
        result: ActivationsResponse = run_activations(model, prompt)
    except Exception as exc:  # noqa: BLE001
        sys.exit(_handle_error("activations", exc, as_json))

    if as_json:
        click.echo(result.model_dump_json(indent=2))
        return

    console.print(f"[bold]Prompt:[/bold] {result.prompt} ({result.n_tokens} tokens)")
    table = Table(title="Activation summary")
    table.add_column("Hook")
    table.add_column("Shape")
    table.add_column("Mean", justify="right")
    table.add_column("Std", justify="right")
    table.add_column("Max", justify="right")
    table.add_column("Max position", justify="right")
    for a in result.activations:
        table.add_row(
            a.hook_name,
            "x".join(str(d) for d in a.shape),
            f"{a.mean:.4f}",
            f"{a.std:.4f}",
            f"{a.max_value:.4f}",
            str(a.max_position),
        )
    console.print(table)


@main.command()
@click.argument("model")
@click.argument("prompt")
@click.option("--layer", required=True, type=int, help="Zero-indexed transformer block to patch.")
@click.option(
    "--component",
    required=True,
    type=click.Choice(
        ["resid_pre", "resid_mid", "resid_post", "attn_out", "mlp_out", "mlp_post"]
    ),
    help="Which component's activation to zero-ablate at the given layer.",
)
@click.option("--json", "as_json", is_flag=True, help="Output a schema-versioned JSON document instead of a table.")
def patch(model: str, prompt: str, layer: int, component: str, as_json: bool) -> None:
    """Zero-ablate one component and report how the output logits changed.

    Runs PROMPT through MODEL once cleanly and once with the activation at --layer /
    --component zeroed out, then reports the predicted token and top logit before and
    after, so you can see how much that component mattered for this specific output.
    """
    try:
        result: PatchResponse = run_patch(model, prompt, layer, component)
    except Exception as exc:  # noqa: BLE001
        sys.exit(_handle_error("patch", exc, as_json))

    if as_json:
        click.echo(result.model_dump_json(indent=2))
        return

    console.print(f"[bold]Prompt:[/bold] {result.prompt}")
    console.print(f"[bold]Ablated:[/bold] layer {result.layer}, component {result.component}")
    console.print(
        f"Baseline prediction: {result.baseline_predicted_token!r} "
        f"(logit {result.baseline_top_logit:.4f})"
    )
    console.print(
        f"Patched prediction:  {result.patched_predicted_token!r} "
        f"(logit {result.patched_top_logit:.4f})"
    )
    console.print(f"Logit delta on baseline token: {result.logit_delta:+.4f}")
    console.print(f"Prediction changed: {result.prediction_changed}")


@main.command()
@click.argument("model")
@click.argument("prompt")
@click.option("--top-k", default=10, show_default=True, help="How many circuit components to report.")
@click.option("--json", "as_json", is_flag=True, help="Output a schema-versioned JSON document instead of a table.")
def circuit(model: str, prompt: str, top_k: int, as_json: bool) -> None:
    """Best-effort automated circuit discovery for a single prompt.

    Ranks candidate heads and neurons by direct logit attribution, then measures each
    candidate's individual causal effect via single-component zero-ablation. This is an
    approximate sketch of a circuit, not a full path-patching analysis with clean/corrupted
    prompt pairs, and it will not capture interaction effects between components -- see the
    "method" field in --json output for the exact caveat.
    """
    try:
        result: CircuitResponse = run_circuit(model, prompt, top_k=top_k)
    except Exception as exc:  # noqa: BLE001
        sys.exit(_handle_error("circuit", exc, as_json))

    if as_json:
        click.echo(result.model_dump_json(indent=2))
        return

    console.print(f"[bold]Prompt:[/bold] {result.prompt}")
    console.print(f"[bold]Predicted next token:[/bold] {result.predicted_token!r}")
    console.print(f"[dim]{result.method}[/dim]")
    table = Table(title="Approximate circuit components (by measured causal effect)")
    table.add_column("Layer", justify="right")
    table.add_column("Type")
    table.add_column("Index", justify="right")
    table.add_column("Logit drop on ablation", justify="right")
    for c in result.components:
        table.add_row(str(c.layer), c.component_type, str(c.index), f"{c.logit_drop_on_ablation:+.4f}")
    console.print(table)


@main.command("mcp-server")
def mcp_server_cmd() -> None:
    """Start the NeuronScope MCP server over stdio.

    Exposes trace, activations, patch, and circuit as MCP tools with the same parameters
    and the same JSON schema as this CLI's --json output, so any MCP-capable agent can
    call them directly.
    """
    from neuronscope.mcp_server import run_server

    run_server()


if __name__ == "__main__":
    main()
