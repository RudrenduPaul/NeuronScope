# Contributing to NeuronScope

Issues and pull requests are welcome. This document covers the parts a first-time
contributor actually needs: how to get a working dev setup, how the code is organized,
and what a PR needs before it can merge.

## Setup

```bash
git clone https://github.com/RudrenduPaul/NeuronScope
cd NeuronScope
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Running the test suite

```bash
pytest -v
```

`tests/test_cli_smoke.py` and `tests/test_mcp_server.py` run a real forward pass against
`gpt2` (no mocking), so the first run downloads the model from HuggingFace Hub (about
500MB) and takes longer than subsequent runs.

To check coverage:

```bash
pytest --cov=neuronscope --cov-report=term-missing
```

## Where things live

- `neuronscope/cli.py`: the Click CLI, one function per subcommand (`trace`,
  `activations`, `patch`, `circuit`, `mcp-server`).
- `neuronscope/mcp_server.py`: the MCP server, exposing the same four operations as MCP
  tools. If you change a CLI subcommand's behavior, check whether the matching MCP tool
  needs the same change; they should never drift apart.
- `neuronscope/core/trace.py`: the actual orchestration logic (resolve model, pick a
  backend, run the operation, serialize the result). Both the CLI and the MCP server call
  into this, so most real logic changes belong here, not in `cli.py` or `mcp_server.py`
  directly.
- `neuronscope/backends/`: the `Backend` interface and its one v1 implementation,
  `TransformerLensBackend`. A new backend (nnsight, SAELens) implements `backends/base.py`'s
  interface and gets registered in `core/registry.py`.
- `neuronscope/schema.py`: every response type, each carrying a `schema_version` field.
  Bump the version if you change a response shape in a way that would break an existing
  consumer.

## Adding a backend

The `Backend` interface (`neuronscope/backends/base.py`) is deliberately small:
`load_model`, `get_activations`, `patch_activations`, `list_supported_architectures`. A new
backend implements these four methods and registers itself in
`neuronscope/core/registry.py`'s model-to-backend resolution. It should not require
changes to `cli.py`, `mcp_server.py`, or `schema.py`.

## Before opening a PR

- Run the full test suite and make sure it passes.
- Add a test for any new behavior, especially a new edge case (unsupported model, bad
  input, resource limit).
- If you added or changed a CLI flag or MCP tool parameter, update the README's CLI
  reference / MCP section to match.
- Keep the scope honest: NeuronScope orchestrates TransformerLens (and, later, other
  backends). It does not aim to reimplement what those libraries already do well.

## Reporting a bug

Open a GitHub issue with the exact command you ran, the model name, and the full output
(including any traceback). If it's reproducible with `gpt2`, that's the easiest case for
someone else to debug quickly.
