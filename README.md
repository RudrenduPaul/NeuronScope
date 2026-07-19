# NeuronScope

NeuronScope is a CLI and [MCP](https://modelcontextprotocol.io) server for tracing which
neurons and attention heads inside an open-weight language model were most responsible for
a given output. It is built directly on top of
[TransformerLens](https://github.com/TransformerLensOrg/TransformerLens), which does the
actual model loading, hooking, and activation math — NeuronScope adds a CLI, a stable JSON
schema, and an MCP server so an agent or a script can ask "why did the model say that" without
writing TransformerLens code by hand.

NeuronScope does not replace TransformerLens, [nnsight](https://nnsight.net/),
[SAELens](https://github.com/jbloomAus/SAELens), Anthropic's
[circuit-tracer](https://github.com/safety-research/circuit-tracer), or
[Neuronpedia](https://www.neuronpedia.org/) — it wraps TransformerLens for a narrower job
(fast, scriptable, agent-callable component tracing) and leaves deeper mechanistic work to
those tools.

## Install

```bash
pip install neuronscope-cli
```

## CLI

```bash
neuronscope trace gpt2 "The capital of France is"
neuronscope activations gpt2 "The capital of France is"
neuronscope patch gpt2 "The capital of France is" --layer 5 --component mlp_out
neuronscope circuit gpt2 "The capital of France is"
```

Add `--json` to any command for a schema-versioned JSON document instead of the human-readable
table. Run `neuronscope <command> --help` for full flag documentation.

## MCP server

```bash
neuronscope mcp-server
```

Starts an MCP server over stdio exposing `trace`, `activations`, `patch`, and `circuit` as MCP
tools, so any MCP-capable agent can call them directly with the same JSON schema the CLI's
`--json` flag produces.

## Supported models

NeuronScope supports whatever `transformer_lens.HookedTransformer.from_pretrained` supports —
see `neuronscope.backends.transformer_lens.TransformerLensBackend.list_supported_architectures()`
for the exact list installed in your environment. Small models like `gpt2` run comfortably on
CPU.

## Status

Early and under active development. Interfaces (CLI flags, JSON schema fields) may still
change between minor versions until `1.0`.

## License

MIT
