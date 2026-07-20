# Changelog

All notable changes to NeuronScope are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.2] - 2026-07-20

### Fixed
- README's demo GIF now uses an absolute URL instead of a repo-relative path, so it
  renders on PyPI's project page too, not just on GitHub.

## [0.1.1] - 2026-07-20

### Added
- Second author credit (Sourav Nandy) on the PyPI package manifest, matching the actual
  authorship of the project.
- `CONTRIBUTING.md` with real dev setup, test, and coverage instructions.
- This changelog.

### Fixed
- Package metadata author field previously listed only one of the two real authors.

## [0.1.0] - 2026-07-19

Initial release.

### Added
- `neuronscope trace`: trace which attention heads and MLP neurons most influenced a
  model's next-token prediction for a given prompt.
- `neuronscope activations`: dump raw activation summary statistics for a prompt.
- `neuronscope patch`: zero-ablate a specific attention head or MLP neuron and report
  the resulting change in output.
- `neuronscope circuit`: best-effort automated circuit sketch combining top heads and
  neurons for a prompt (documented as approximate, not a full path-patching analysis).
- `neuronscope mcp-server`: an MCP server (stdio transport) exposing the same four
  operations as MCP tools, so an agent can call them directly.
- `--json` output on every data-returning command, with a `schema_version` field on every
  response.
- `TransformerLensBackend`, the first of a pluggable `Backend` interface, covering the
  247 model families TransformerLens's `HookedTransformer` natively supports.
- Real, distinct exit codes: `0` success, `1` general runtime error, `2` CLI usage error
  (Click's own convention), `3` unsupported model.
- Layer-bounds validation on `patch` (a `--layer` outside the model's real range fails
  with a clear message, not a raw `KeyError`).
- CPU fallback when no CUDA GPU is available, with a logged warning (Apple Silicon MPS is
  available via an explicit `device=` argument but not auto-selected, since it can
  silently produce incorrect activation values for some models).
