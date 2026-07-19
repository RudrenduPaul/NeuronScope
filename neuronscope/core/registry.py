"""Maps a requested model name to the backend that can serve it.

TransformerLens is the only backend in v1, but this indirection means adding a second
backend later is a matter of appending to ``BACKENDS`` and does not touch the CLI, the
MCP server, or ``core/trace.py``.
"""

from __future__ import annotations

from neuronscope.backends.base import Backend
from neuronscope.backends.transformer_lens import TransformerLensBackend

BACKENDS: list[Backend] = [TransformerLensBackend()]


class UnsupportedModelError(Exception):
    """Raised when no registered backend can load the requested model.

    Carries enough detail for the caller to print a message naming both what was tried
    and what is actually supported, instead of a bare "not found".
    """

    def __init__(self, model_name: str, backends_tried: list[str]):
        self.model_name = model_name
        self.backends_tried = backends_tried
        examples = ", ".join(sorted(TransformerLensBackend().list_supported_architectures())[:8])
        message = (
            f"Model '{model_name}' is not supported by any registered backend "
            f"(tried: {', '.join(backends_tried)}). "
            f"TransformerLens supports a fixed list of pretrained checkpoints and their "
            f"short aliases, for example: {examples}, ... "
            f"Run `TransformerLensBackend().list_supported_architectures()` for the full, "
            f"exact list for your installed transformer_lens version."
        )
        super().__init__(message)


def resolve_backend(model_name: str) -> Backend:
    """Return the backend that supports ``model_name``, or raise UnsupportedModelError."""
    backends_tried = []
    for backend in BACKENDS:
        backends_tried.append(backend.name)
        if backend.supports_model(model_name):
            return backend
    raise UnsupportedModelError(model_name, backends_tried)
