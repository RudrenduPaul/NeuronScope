"""End-to-end smoke tests that actually invoke the CLI against a real small model.

Uses gpt2 (124M params) so these run on CPU without a GPU or a large download, matching
what CI can afford. These tests exercise the real TransformerLens forward pass, not a
mock, so a broken integration between the CLI, core/trace.py, and the backend will show
up here even if the unit tests all pass.
"""

import json

from click.testing import CliRunner

from neuronscope.cli import main

PROMPT = "The capital of France is"


def test_version_matches_installed_package():
    from importlib.metadata import version

    runner = CliRunner()
    result = runner.invoke(main, ["--version"])

    assert result.exit_code == 0, result.output
    assert version("neuronscope-cli") in result.output


def test_trace_json_smoke():
    runner = CliRunner()
    result = runner.invoke(main, ["trace", "gpt2", PROMPT, "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["schema_version"] == 1
    assert payload["operation"] == "trace"
    assert payload["prompt"] == PROMPT
    assert payload["model"]["backend"] == "transformer_lens"
    assert payload["model"]["resolved_name"] == "gpt2"
    assert len(payload["top_heads"]) > 0
    assert len(payload["top_neurons"]) > 0
    assert isinstance(payload["predicted_token_id"], int)


def test_trace_human_readable_smoke():
    runner = CliRunner()
    result = runner.invoke(main, ["trace", "gpt2", PROMPT])

    assert result.exit_code == 0, result.output
    assert "Predicted next token" in result.output
    assert "Top attention heads" in result.output
    assert "Top MLP neurons" in result.output


def test_activations_json_smoke():
    runner = CliRunner()
    result = runner.invoke(main, ["activations", "gpt2", PROMPT, "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["operation"] == "activations"
    assert payload["n_tokens"] > 0
    assert len(payload["activations"]) > 0
    assert all("hook_name" in a for a in payload["activations"])


def test_patch_json_smoke():
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["patch", "gpt2", PROMPT, "--layer", "5", "--component", "mlp_out", "--json"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["operation"] == "patch"
    assert payload["layer"] == 5
    assert payload["component"] == "mlp_out"
    assert payload["ablation_type"] == "zero"
    assert isinstance(payload["prediction_changed"], bool)


def test_circuit_json_smoke():
    runner = CliRunner()
    result = runner.invoke(main, ["circuit", "gpt2", PROMPT, "--top-k", "3", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["operation"] == "circuit"
    assert len(payload["components"]) <= 3
    assert "approximate" in payload["method"].lower()


def test_unsupported_model_exits_3():
    runner = CliRunner()
    result = runner.invoke(main, ["trace", "not-a-real-model-xyz", PROMPT, "--json"])

    assert result.exit_code == 3, result.output
    payload = json.loads(result.output)
    assert payload["error_type"] == "UnsupportedModelError"
    assert "transformer_lens" in payload["message"]


def test_bad_component_choice_exits_2_click_usage_error():
    # A bad --component value never reaches NeuronScope's own error handling -- Click
    # rejects it during argument parsing and exits 2 itself. This must stay distinct from
    # exit code 3 (unsupported model), which is a NeuronScope domain error, not a usage
    # mistake.
    runner = CliRunner()
    result = runner.invoke(
        main, ["patch", "gpt2", PROMPT, "--layer", "0", "--component", "not_a_component"]
    )

    assert result.exit_code == 2, result.output
    assert "not_a_component" in result.output
    assert "Traceback" not in result.output


def test_missing_required_argument_exits_2_click_usage_error():
    # Same distinction from the other direction: a missing PROMPT argument is also a
    # Click usage error (exit 2), not an "unsupported model" domain error (exit 3).
    runner = CliRunner()
    result = runner.invoke(main, ["trace", "gpt2"])

    assert result.exit_code == 2, result.output
    assert "Traceback" not in result.output


def test_layer_out_of_range_exits_1_plain():
    runner = CliRunner()
    result = runner.invoke(
        main, ["patch", "gpt2", PROMPT, "--layer", "99", "--component", "mlp_out"]
    )

    assert result.exit_code == 1, result.output
    assert "99" in result.output
    assert "12" in result.output
    assert "Traceback" not in result.output
    assert "KeyError" not in result.output


def test_layer_out_of_range_exits_1_json():
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["patch", "gpt2", PROMPT, "--layer", "99", "--component", "mlp_out", "--json"],
    )

    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert payload["error_type"] == "LayerOutOfRangeError"
    assert payload["operation"] == "patch"
    assert "99" in payload["message"]
    assert "12" in payload["message"]
    assert "gpt2" in payload["message"]


def test_help_text_is_present_for_every_command():
    runner = CliRunner()
    for command in ["trace", "activations", "patch", "circuit", "mcp-server"]:
        result = runner.invoke(main, [command, "--help"])
        assert result.exit_code == 0
        assert len(result.output.strip()) > 0


def test_prompt_too_long_exits_1_not_a_raw_traceback():
    # gpt2's context window is 1024 tokens; this repeats well past that. model.to_tokens()
    # silently truncates by default, so this also guards against that truncation hiding
    # the over-length prompt from the check in core/trace.py.
    runner = CliRunner()
    long_prompt = "the quick brown fox jumps over the lazy dog " * 400
    result = runner.invoke(main, ["trace", "gpt2", long_prompt, "--json"])

    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert payload["error_type"] == "PromptTooLongError"
    assert "gpt2" in payload["message"]
    assert "Traceback" not in result.output
