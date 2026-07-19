"""Abstract interface every NeuronScope backend must implement.

TransformerLens is the only concrete backend in v1 (``transformer_lens.py``), but keeping
this as an explicit interface means a future backend (for example one built on nnsight)
can be added without touching ``core/trace.py`` or the CLI.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Backend(ABC):
    """A backend loads a model and exposes activation inspection / patching primitives."""

    name: str

    @abstractmethod
    def supports_model(self, model_name: str) -> bool:
        """Return True if this backend can load the given model name."""
        raise NotImplementedError

    @abstractmethod
    def list_supported_architectures(self) -> list[str]:
        """Return the model names this backend can load, as reported by the underlying
        library. Must reflect the real capability of the installed library version, not
        an aspirational or hand-maintained list."""
        raise NotImplementedError

    @abstractmethod
    def load_model(self, model_name: str, device: str | None = None) -> Any:
        """Load and return a handle to the model, ready for activation inspection."""
        raise NotImplementedError

    @abstractmethod
    def get_activations(self, model: Any, prompt: str) -> Any:
        """Run a forward pass on ``prompt`` and return the model's full activation cache."""
        raise NotImplementedError

    @abstractmethod
    def patch_activations(
        self,
        model: Any,
        prompt: str,
        layer: int,
        component: str,
    ) -> Any:
        """Run ``prompt`` twice: once clean, once with the given layer/component
        zero-ablated. Return whatever the backend needs to report the output delta."""
        raise NotImplementedError
