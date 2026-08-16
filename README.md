# NeuronScope

<!-- mcp-name: io.github.RudrenduPaul/neuronscope -->
<!-- Ownership-proof string for registry.modelcontextprotocol.io publishing. Do not remove. -->

[![CI](https://github.com/RudrenduPaul/NeuronScope/actions/workflows/ci.yml/badge.svg)](https://github.com/RudrenduPaul/NeuronScope/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/neuronscope-cli.svg)](https://pypi.org/project/neuronscope-cli/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/RudrenduPaul/NeuronScope/blob/main/LICENSE)

<a href="https://www.producthunt.com/products/neuronscope?embed=true&utm_source=badge-featured&utm_medium=badge&utm_campaign=badge-neuronscope" target="_blank" rel="noopener noreferrer"><img alt="NeuronScope - Traces LLM outputs to the neurons and heads that caused them | Product Hunt" width="250" height="54" src="https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=1222875&theme=light&t=1786882007537"></a>

Ask a language model "why did you say that" and get back the actual attention heads and
neurons responsible, as JSON, from the command line or from an agent over MCP.

![NeuronScope tracing a real gpt2 prediction from the command line, showing the top attention heads and MLP neurons responsible for the output](https://raw.githubusercontent.com/RudrenduPaul/NeuronScope/main/docs/demo.gif)

## Install

```bash
pip install neuronscope-cli
```

That gets you the `neuronscope` command. To install from source instead (for development or to
track `main`):

```bash
git clone https://github.com/RudrenduPaul/NeuronScope
cd NeuronScope
pip install -e .
```

> [!NOTE]
> The first run of any command downloads the requested model from the HuggingFace Hub
> (`gpt2` is about 500MB) and prints two lines to stderr that are expected, not errors: a
> CPU-fallback notice if you don't have a CUDA GPU, and an unauthenticated-HF-Hub
> rate-limit notice. Neither one means anything broke.

## Quickstart

```bash
neuronscope trace gpt2 "The capital of France is Paris. The capital of Japan is" --top-k 5
```

Real output from this exact command (stderr trimmed to the two expected warnings mentioned
above):

```
Prompt: The capital of France is Paris. The capital of Japan is
Predicted next token: ' Tokyo'
Top attention heads (by direct logit
            attribution)
┏━━━━━━━┳━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃ Layer ┃ Head ┃ Logit attribution ┃
┡━━━━━━━╇━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
│     9 │    8 │            4.0679 │
│     8 │   11 │            2.9028 │
│    10 │    7 │           -1.4782 │
│     8 │   10 │           -1.3999 │
│    10 │    0 │            1.1424 │
└───────┴──────┴───────────────────┘
Top MLP neurons (by activation
          magnitude)
┏━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━┓
┃ Layer ┃ Neuron ┃ Activation ┃
┡━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━┩
│    10 │     97 │     7.8394 │
│    11 │    611 │     4.6954 │
│    11 │   2997 │     4.6468 │
│    10 │   1793 │     4.5443 │
│     9 │   1460 │     4.4196 │
└───────┴────────┴────────────┘
```

gpt2 predicts `Tokyo` correctly, and head `L9H8` is the single biggest contributor to that
prediction. Add `--json` to get the machine-readable version of the same result:

```bash
neuronscope trace gpt2 "The capital of France is Paris. The capital of Japan is" --top-k 3 --json
```

```json
{
  "schema_version": 1,
  "operation": "trace",
  "model": {
    "requested_name": "gpt2",
    "resolved_name": "gpt2",
    "backend": "transformer_lens",
    "device": "cpu",
    "n_layers": 12,
    "n_heads": 12,
    "d_model": 768,
    "d_mlp": 3072
  },
  "prompt": "The capital of France is Paris. The capital of Japan is",
  "predicted_token": " Tokyo",
  "predicted_token_id": 11790,
  "top_neurons": [
    { "layer": 10, "neuron_index": 97, "activation": 7.839381217956543 },
    { "layer": 11, "neuron_index": 611, "activation": 4.695372581481934 },
    { "layer": 11, "neuron_index": 2997, "activation": 4.646785736083984 }
  ],
  "top_heads": [
    { "layer": 9, "head_index": 8, "logit_attribution": 4.067923545837402 },
    { "layer": 8, "head_index": 11, "logit_attribution": 2.9028172492980957 },
    { "layer": 10, "head_index": 7, "logit_attribution": -1.4781968593597412 }
  ]
}
```

## What it does

NeuronScope is a CLI and [MCP](https://modelcontextprotocol.io) server built on top of
[TransformerLens](https://github.com/TransformerLensOrg/TransformerLens). TransformerLens
does the actual model loading, hooking, and activation math; NeuronScope adds a stable CLI,
a versioned JSON schema, and an MCP server around it, so a script or an agent can ask "which
components drove this output" without writing TransformerLens code directly.

- **`trace`**: runs a prompt through the model and ranks attention heads by direct logit
  attribution to the predicted token, and MLP neurons by activation magnitude at the final
  prompt position.
- **`activations`**: dumps shape, mean, std, min/max, and the max-activating sequence
  position for every layer's residual stream, MLP neuron activations, and attention pattern.
- **`patch`**: zero-ablates one component (`resid_pre`, `resid_mid`, `resid_post`,
  `attn_out`, `mlp_out`, or `mlp_post`) at a given layer and reports how the predicted token
  and its logit changed.
- **`circuit`**: a best-effort automated circuit sketch. Ranks candidate heads/neurons by
  logit attribution, then measures each one's individual causal effect via single-component
  ablation. This is not full path-patching with clean/corrupted prompt pairs and does not
  capture interaction effects between components. The `--json` output says so explicitly in
  its `method` field.
- Every command supports `--json` for a `schema_version`-stamped document instead of a
  table, and the same four operations are exposed as MCP tools returning the identical
  shape via `.model_dump()`, so a CLI call and an MCP tool call produce the same document
  for the same input.
- Model support is whatever `transformer_lens.HookedTransformer.from_pretrained` supports.
  Installing `neuronscope-cli` today pulls TransformerLens 3.6.0, which supports 249
  pretrained checkpoints and aliases (`OFFICIAL_MODEL_NAMES`), covering GPT-2, Pythia,
  Llama, Gemma, Qwen, and more. Small models like `gpt2` run comfortably on CPU.

NeuronScope does not replace TransformerLens, [nnsight](https://nnsight.net/),
[SAELens](https://github.com/jbloomAus/SAELens), Anthropic's
[circuit-tracer](https://github.com/decoderesearch/circuit-tracer), or
[Neuronpedia](https://www.neuronpedia.org/). It wraps TransformerLens for one narrower job:
fast, scriptable, agent-callable component tracing on a single prompt. It leaves deeper
mechanistic work (SAE training, transcoder-based circuit graphs, hosted feature browsing) to
those tools.

## CLI reference

Every command takes `MODEL` (any name `HookedTransformer.from_pretrained` accepts, for
example `gpt2` or `EleutherAI/pythia-70m`) and `PROMPT` as positional arguments.

| Command | Extra flags | What it does |
|---|---|---|
| `neuronscope trace MODEL PROMPT` | `--top-k INTEGER` (default 10), `--json` | Ranks top attention heads (logit attribution) and MLP neurons (activation magnitude) for the predicted next token |
| `neuronscope activations MODEL PROMPT` | `--json` | Dumps per-layer activation summary stats (residual stream, MLP, attention pattern) |
| `neuronscope patch MODEL PROMPT` | `--layer INTEGER` (required), `--component [resid_pre\|resid_mid\|resid_post\|attn_out\|mlp_out\|mlp_post]` (required), `--json` | Zero-ablates one component and reports the logit/prediction delta |
| `neuronscope circuit MODEL PROMPT` | `--top-k INTEGER` (default 10), `--json` | Best-effort circuit sketch via ranked single-component ablation |
| `neuronscope mcp-server` | none | Starts the MCP server over stdio |

Global: `neuronscope --version`, `neuronscope <command> --help`. Exit codes: `0` success,
`1` a runtime error (prompt too long for the model's context window, `--layer` out of range,
etc.), `2` a Click usage error (bad flags), `3` an unsupported model name.

![neuronscope circuit ranking candidate heads/neurons by logit attribution and measuring each one's causal effect via single-component ablation](https://raw.githubusercontent.com/RudrenduPaul/NeuronScope/main/docs/demo-circuit.gif)

![neuronscope patch zero-ablating one component at a given layer and reporting how the predicted token and its logit changed](https://raw.githubusercontent.com/RudrenduPaul/NeuronScope/main/docs/demo-patch.gif)

## MCP Server

NeuronScope ships a [Model Context Protocol](https://modelcontextprotocol.io) server so an AI
agent (Claude, Cursor, or any MCP-compatible client) can trace, inspect, ablate, and sketch
circuits directly, without a human invoking the CLI by hand.

Install the extra:

```bash
pip install "neuronscope-cli[mcp]"
```

Add it to your MCP client's config (for Claude Desktop, `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "neuronscope": {
      "command": "uvx",
      "args": ["--from", "neuronscope-cli", "neuronscope-mcp"]
    }
  }
}
```

The server exposes four tools, `trace`, `activations`, `patch`, and `circuit`, each returning
the identical pydantic-model-shaped JSON the CLI's `--json` flag prints, via `.model_dump()`, so
an agent calling this server and a script calling the CLI get the same document for the same
input. A real `trace` call and its response:

```
trace(model="gpt2", prompt="The capital of France is Paris. The capital of Japan is", top_k=3)

{
  "schema_version": 1,
  "operation": "trace",
  "predicted_token": " Tokyo",
  "predicted_token_id": 11790,
  "top_neurons": [
    { "layer": 10, "neuron_index": 97, "activation": 7.839381217956543 }
  ],
  "top_heads": [
    { "layer": 9, "head_index": 8, "logit_attribution": 4.067923545837402 }
  ]
}
```

Errors never raise across the tool boundary: every handler catches its exceptions and returns a
structured `ErrorResponse` dict instead, so a calling agent always gets a parseable result.

> [!WARNING]
> NeuronScope puts no size cap or timeout on model loading or forward passes. If you expose
> this MCP server somewhere an untrusted agent can call it, put a resource limit around the
> process (a cgroup, `ulimit`, or a container memory/CPU cap) rather than relying on
> NeuronScope to refuse an oversized request on its own.

Transport is stdio, so there is nothing to host: the MCP client spawns the server as a local
subprocess. Source: [`neuronscope/mcp_server.py`](neuronscope/mcp_server.py).

- **Claude Code** reads this from a project-level `.mcp.json` in your repo root, or you can
  add it with `claude mcp add neuronscope -- neuronscope-mcp`.
- **Claude Desktop** reads this from its `claude_desktop_config.json`
  (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS,
  `%APPDATA%\Claude\claude_desktop_config.json` on Windows), under the same
  `"mcpServers"` key.
- The `neuronscope mcp-server` CLI subcommand still works as a local, non-`uvx` alternative
  that runs the same server over stdio from an existing install.

## How it compares

All five of these are real, actively maintained projects doing different jobs. This table
compares CLI/JSON-agent-output surface and model coverage, not depth of interpretability
research, where TransformerLens, nnsight, SAELens, circuit-tracer, and Neuronpedia are all
more mature than NeuronScope. Star counts, release info, and last-push dates below were
pulled from each project's GitHub API on 2026-08-03 and will drift over time; check the repos
directly for current numbers.

| Project | Stars | Last activity | CLI | Agent-callable structured output | Model coverage |
|---|---|---|---|---|---|
| [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) | 3,750 | v3.6.0 released 2026-07-28, pushed 2026-08-03 | No (Python library) | No | 249 pretrained checkpoints/aliases (its own official list) |
| [nnsight](https://github.com/ndif-team/nnsight) | 1,014 | v0.7.0 released 2026-05-05, pushed 2026-07-30 | No (Python library) | No (returns tensors/Python objects) | Any HuggingFace or PyTorch model generically, no fixed list |
| [circuit-tracer](https://github.com/decoderesearch/circuit-tracer) (Anthropic-authored, moved from `safety-research/circuit-tracer`) | 2,882 | v0.5.2 released 2026-07-18, pushed 2026-07-18 | Yes | JSON attribution-graph export; no MCP server | Fixed transcoder allowlist: Gemma-2 (2B), Gemma-3 (270M-27B), Llama-3.2 (1B), Llama-3.1 (8B Instruct), Qwen-3 (0.6B-14B), GPT-OSS (20B) |
| [SAELens](https://github.com/jbloomAus/SAELens) | 1,492 | v6.47.0 released 2026-07-28, pushed 2026-07-28 | No (Python library) | No | Any PyTorch model generically; deepest integration is with TransformerLens |
| [Neuronpedia](https://github.com/hijohnnylin/neuronpedia) | 1,093 | continuously deployed, tag v1.0.795 | No (hosted web app + REST API) | REST API returns JSON; MCP access exists only via an unofficial third-party wrapper, not the official repo | Models loadable through TransformerLens's model table (GPT-2, Gemma-2, Llama, DeepSeek, etc.) |
| **NeuronScope** (this project) | 1 | this commit | Yes | Yes: `--json` on every command, plus a native MCP server returning the same schema | Whatever TransformerLens's `HookedTransformer.from_pretrained` supports: 249 checkpoints/aliases |

The honest differentiation is narrow: NeuronScope is the only one of these with a CLI, a
native MCP server, and a versioned JSON schema together in one package, and it's
model-agnostic across whatever TransformerLens supports rather than pinned to a fixed
transcoder allowlist like circuit-tracer. It is not more capable, more mature, or more
widely used than any of these projects.

## What is NeuronScope and why does it exist

TransformerLens gives you a Python API for loading a model and running hooked forward
passes. That's the right interface for a research notebook. It's the wrong interface for a
script that needs a subprocess call and a JSON document back, or for an agent that needs a
tool it can call over MCP. NeuronScope exists to be that second interface: the same
underlying computation, wrapped so a CLI invocation or an MCP tool call gets back a
schema-versioned document instead of a Python object graph.

## FAQ

**Is this a replacement for TransformerLens, nnsight, SAELens, circuit-tracer, or
Neuronpedia?**
No. NeuronScope is built directly on TransformerLens and does not do anything TransformerLens
itself can't already do at a lower level. It doesn't train SAEs (SAELens), do full
path-patching circuit discovery with transcoders (circuit-tracer), give you a Python-native
tracing context manager for arbitrary PyTorch models (nnsight), or host a browsable feature
database (Neuronpedia). It's a CLI and MCP wrapper around one slice of TransformerLens's
functionality.

**What models are supported?**
Anything `transformer_lens.HookedTransformer.from_pretrained` supports, which today is 249
checkpoints and aliases spanning GPT-2, Pythia, Llama, Gemma, Qwen, and others. Run
`python -c "from transformer_lens.loading_from_pretrained import OFFICIAL_MODEL_NAMES; print(len(OFFICIAL_MODEL_NAMES))"`
in your own environment to get the exact count for your installed version, since
TransformerLens adds models over time.

**Does it need a GPU?**
No. Small models like `gpt2` run fine on CPU; that's what the test suite and the quickstart
above run on. Larger models will be slow on CPU. NeuronScope does not auto-select Apple
Silicon's MPS backend even when available, because PyTorch's MPS backend can silently
produce incorrect values for some ops that this project's activation-patching math depends
on being exact. Pass `device="mps"` explicitly in your own code if you want it anyway.

**Is it safe to expose the MCP server to an untrusted agent?**
Only with resource limits in place. See Known limitations below.

**How is NeuronScope different from circuit-tracer, the other CLI tool in this list?**
circuit-tracer does deeper circuit analysis (full attribution graphs from trained
transcoders) but only for a fixed allowlist of models: Gemma-2, Gemma-3, Llama-3.1/3.2,
Qwen-3, and GPT-OSS. NeuronScope trades that depth for breadth: it works with any of
TransformerLens's 249 supported checkpoints with no transcoder training step, and ships an
MCP server so an agent can call it directly. The cost is that NeuronScope does
single-component logit attribution and zero-ablation, not transcoder-based path patching.

**Does the installed version always match what's on PyPI?**
Run `neuronscope --version` after installing to check. `pip install neuronscope-cli` pulls
whatever release PyPI has published most recently; the code on this repo's `main` branch can
be ahead of that between releases. Installing from source (`pip install -e .`) always tracks
`main` exactly, including whatever hasn't been released yet.

**What license is NeuronScope under, and can I use it commercially?**
MIT. You can use, modify, and redistribute it in commercial and closed-source projects,
with attribution and the license notice kept intact. The dependencies it pulls in
(TransformerLens, PyTorch, the `mcp` package) carry their own licenses; check those
separately if you're redistributing a bundled product rather than just calling
`neuronscope-cli` as a dependency.

## Known limitations

- **`circuit` is an approximation.** It ranks components by logit attribution and measures
  each one's individual causal effect via single-component zero-ablation on one prompt. It
  does not do full path-patching with clean/corrupted prompt pairs, and it will not catch
  interaction effects between components. The `--json` output states this in its `method`
  field so a caller doesn't have to trust prose to know the caveat.
- **No size cap or timeout on model loading or forward passes.** NeuronScope loads whatever
  model weights the caller asks for and runs the forward pass to completion, with no built-in
  limit on model size or wall-clock time. If you run the MCP server somewhere an untrusted
  agent can call it, put a resource limit around the process (a cgroup, `ulimit`, or a
  container memory/CPU cap) rather than relying on NeuronScope to refuse an oversized
  request on its own.
- **`HookedTransformer.from_pretrained` is deprecated upstream.** TransformerLens 3.6.0
  emits a `DeprecationWarning` pointing at `TransformerBridge.boot_transformers` as the
  replacement. It still works today, and every command shown in this README ran on it, but
  NeuronScope's backend hasn't migrated yet. Tracked as an open item; migrating would be a
  change inside `neuronscope/backends/transformer_lens.py`, not a change to any CLI command
  or MCP tool signature.

## Contributing

Issues and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for dev setup,
where the code lives, and what a PR needs before it merges. Quick version:

```bash
pip install -e ".[dev,mcp]"
pytest -v
```

CI runs the same suite on Python 3.10, 3.11, and 3.12 on every push and pull request against
`main`. The suite covers 87% of `neuronscope/` (`pytest --cov=neuronscope`), with the MCP
server's less-exercised paths (specific error branches) the main gap.

## License

MIT. See [LICENSE](LICENSE).
