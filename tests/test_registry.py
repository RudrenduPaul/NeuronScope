"""Tests for backend resolution and the UnsupportedModelError edge case."""

import pytest

from neuronscope.backends.transformer_lens import TransformerLensBackend
from neuronscope.core.registry import UnsupportedModelError, resolve_backend


def test_resolve_backend_for_known_model():
    backend = resolve_backend("gpt2")
    assert backend.name == "transformer_lens"


def test_resolve_backend_for_known_alias():
    # "gpt2-small" is an alias TransformerLens resolves to the official name "gpt2".
    backend = resolve_backend("gpt2-small")
    assert backend.name == "transformer_lens"


def test_resolve_backend_raises_for_unsupported_model():
    with pytest.raises(UnsupportedModelError) as exc_info:
        resolve_backend("definitely-not-a-real-model-xyz-123")

    error = exc_info.value
    assert error.model_name == "definitely-not-a-real-model-xyz-123"
    assert "transformer_lens" in error.backends_tried
    # The message must name what was tried and hint at what IS supported.
    message = str(error)
    assert "transformer_lens" in message
    assert "definitely-not-a-real-model-xyz-123" in message
    assert "gpt2" in message.lower() or "supported" in message.lower()


def test_list_supported_architectures_is_nonempty_and_sorted():
    backend = TransformerLensBackend()
    architectures = backend.list_supported_architectures()
    assert len(architectures) > 0
    assert architectures == sorted(architectures)
    assert "gpt2" in architectures


def test_supports_model_matches_list():
    backend = TransformerLensBackend()
    assert backend.supports_model("gpt2") is True
    assert backend.supports_model("definitely-not-a-real-model-xyz-123") is False
