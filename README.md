# NeuronScope

Ask a language model "why did you say that" and get back the actual attention heads and
neurons responsible, as JSON, from the command line or from an agent over MCP.

[![CI](https://github.com/RudrenduPaul/NeuronScope/actions/workflows/ci.yml/badge.svg)](https://github.com/RudrenduPaul/NeuronScope/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/RudrenduPaul/NeuronScope/blob/main/LICENSE)

## Install

Not on PyPI yet, so install from source:

```bash
git clone https://github.com/RudrenduPaul/NeuronScope
cd NeuronScope
pip install -e .
```

That gets you the `neuronscope` command. A published package (`pip install neuronscope-cli`)
is planned but not live yet; if you try it today you'll get a 404 from PyPI.

**First run of any command** downloads the requested model from the HuggingFace Hub (`gpt2`
is about 500MB) and prints two lines to stderr that are expected, not errors: a CPU-fallback
notice if you don't have a CUDA GPU, and an unauthenticated-HF-Hub rate-limit notice. Neither
one means anything broke.

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
  As of the `transformer-lens` version this repo currently depends on, that's 247 pretrained
  checkpoints and aliases (`OFFICIAL_MODEL_NAMES`), covering GPT-2, Pythia, Llama, Gemma,
  Qwen, and more. Small models like `gpt2` run comfortably on CPU.

NeuronScope does not replace TransformerLens, [nnsight](https://nnsight.net/),
[SAELens](https://github.com/jbloomAus/SAELens), Anthropic's
[circuit-tracer](https://github.com/decoderesearch/circuit-tracer), or
[Neuronpedia](https://www.neuronpedia.org/). It wraps TransformerLens for one narrower job:
fast, scriptable, agent-callable component tracing on a single prompt. It leaves deeper
mechanistic work (SAE training, transcoder-based circuit graphs, hosted feature browsing) to
those tools.

## Quickstart

```bash
neuronscope trace gpt2 "The capital of France is Paris. The capital of Japan is"
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
    { "layer": 10, "neuron_index": 97, "activation": 7.839381217956543 }
  ],
  "top_heads": [
    { "layer": 9, "head_index": 8, "logit_attribution": 4.067923545837402 }
  ]
}
```

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

## MCP server

```bash
neuronscope mcp-server
```

Starts an MCP server over stdio that exposes `trace`, `activations`, `patch`, and `circuit`
as MCP tools, with the same arguments and the same JSON schema as the CLI's `--json` output.

To register it with an MCP host, add:

```json
{
  "mcpServers": {
    "neuronscope": {
      "command": "neuronscope",
      "args": ["mcp-server"]
    }
  }
}
```

- **Claude Code** reads this from a project-level `.mcp.json` in your repo root, or you can
  add it with `claude mcp add neuronscope -- neuronscope mcp-server`.
- **Claude Desktop** reads this from its `claude_desktop_config.json`
  (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS,
  `%APPDATA%\Claude\claude_desktop_config.json` on Windows), under the same
  `"mcpServers"` key.

## How it compares

All five of these are real, actively maintained projects doing different jobs. This table
compares CLI/JSON-agent-output surface and model coverage, not depth of interpretability
research, where TransformerLens, nnsight, SAELens, circuit-tracer, and Neuronpedia are all
more mature than NeuronScope. Star counts, release info, and last-push dates below were
pulled from each project's GitHub API on 2026-07-19 and will drift over time; check the repos
directly for current numbers.

| Project | Stars | Last activity | CLI | Agent-callable structured output | Model coverage |
|---|---|---|---|---|---|
| [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) | 3,689 | v3.5.1 released 2026-07-01, pushed 2026-07-16 | No (Python library) | No | 247 pretrained checkpoints/aliases (its own official list) |
| [nnsight](https://github.com/ndif-team/nnsight) | 995 | v0.7.0 released 2026-05-05, pushed 2026-07-14 | No (Python library) | No (returns tensors/Python objects) | Any HuggingFace or PyTorch model generically, no fixed list |
| [circuit-tracer](https://github.com/decoderesearch/circuit-tracer) (Anthropic-authored, moved from `safety-research/circuit-tracer`) | 2,864 | v0.5.2 released 2026-07-18 | Yes | JSON attribution-graph export; no MCP server | Fixed transcoder allowlist: Gemma-2 (2B), Gemma-3 (270M-27B), Llama-3.2 (1B), Llama-3.1 (8B Instruct), Qwen-3 (0.6B-14B), GPT-OSS (20B) |
| [SAELens](https://github.com/jbloomAus/SAELens) | 1,476 | v6.46.0 released 2026-07-13, pushed 2026-07-13 | No (Python library) | No | Any PyTorch model generically; deepest integration is with TransformerLens |
| [Neuronpedia](https://github.com/hijohnnylin/neuronpedia) | 1,070 | continuously deployed, tag v1.0.795 on 2026-07-17 | No (hosted web app + REST API) | REST API returns JSON; MCP access exists only via an unofficial third-party wrapper, not the official repo | Models loadable through TransformerLens's model table (GPT-2, Gemma-2, Llama, DeepSeek, etc.) |
| **NeuronScope** (this project) | New, pre-release | this commit | Yes | Yes: `--json` on every command, plus a native MCP server returning the same schema | Whatever TransformerLens's `HookedTransformer.from_pretrained` supports: 247 checkpoints/aliases |

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
Anything `transformer_lens.HookedTransformer.from_pretrained` supports, which today is 247
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

## Contributing

Issues and pull requests are welcome. To run the test suite locally:

```bash
pip install -e ".[dev]"
pytest -v
```

CI runs the same suite on Python 3.10, 3.11, and 3.12 on every push and pull request against
`main`.

## License

MIT. See [LICENSE](LICENSE).
