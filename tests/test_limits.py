"""Tests for the resource caps on model loading: parameter-count ceiling and the
concurrency ceiling on in-flight loads (neuronscope/core/limits.py).

These caps close a self-disclosed gap: NeuronScope loads a fresh model on every call
with nothing caching a loaded model across calls, so an uncapped caller could either
name an arbitrarily large checkpoint or fire off several concurrent tool calls and
exhaust the host's memory either way. Both caps must fail fast with a clear error
rather than blocking or crashing -- that's what these tests assert.
"""

from __future__ import annotations

import threading

import pytest
import torch

from neuronscope.core.limits import (
    DEFAULT_MAX_CONCURRENT_LOADS,
    DEFAULT_MAX_MODEL_PARAMS,
    ModelTooLargeError,
    TooManyConcurrentModelLoadsError,
    check_model_size,
    max_concurrent_loads,
    max_model_params,
    model_load_slot,
)
from neuronscope.core.trace import _load_and_validate


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


def test_max_model_params_default(monkeypatch):
    monkeypatch.delenv("NEURONSCOPE_MAX_MODEL_PARAMS", raising=False)
    assert max_model_params() == DEFAULT_MAX_MODEL_PARAMS


def test_max_model_params_respects_env(monkeypatch):
    monkeypatch.setenv("NEURONSCOPE_MAX_MODEL_PARAMS", "123456")
    assert max_model_params() == 123456


def test_max_model_params_falls_back_on_invalid_env(monkeypatch):
    monkeypatch.setenv("NEURONSCOPE_MAX_MODEL_PARAMS", "not-a-number")
    assert max_model_params() == DEFAULT_MAX_MODEL_PARAMS


def test_max_model_params_falls_back_on_nonpositive_env(monkeypatch):
    monkeypatch.setenv("NEURONSCOPE_MAX_MODEL_PARAMS", "0")
    assert max_model_params() == DEFAULT_MAX_MODEL_PARAMS


def test_max_concurrent_loads_default(monkeypatch):
    monkeypatch.delenv("NEURONSCOPE_MAX_CONCURRENT_LOADS", raising=False)
    assert max_concurrent_loads() == DEFAULT_MAX_CONCURRENT_LOADS


def test_max_concurrent_loads_respects_env(monkeypatch):
    monkeypatch.setenv("NEURONSCOPE_MAX_CONCURRENT_LOADS", "4")
    assert max_concurrent_loads() == 4


# ---------------------------------------------------------------------------
# Parameter-count cap
# ---------------------------------------------------------------------------


def test_check_model_size_raises_when_over_cap(monkeypatch):
    import transformer_lens.loading_from_pretrained as tlp

    monkeypatch.setattr(tlp, "get_num_params_of_pretrained", lambda name: 5_000_000_000)
    monkeypatch.setenv("NEURONSCOPE_MAX_MODEL_PARAMS", "2000000000")

    with pytest.raises(ModelTooLargeError) as exc_info:
        check_model_size("huge-model")

    error = exc_info.value
    assert error.model_name == "huge-model"
    assert error.n_params == 5_000_000_000
    assert error.max_params == 2_000_000_000
    assert "huge-model" in str(error)
    assert "NEURONSCOPE_MAX_MODEL_PARAMS" in str(error)


def test_check_model_size_allows_when_under_cap(monkeypatch):
    import transformer_lens.loading_from_pretrained as tlp

    monkeypatch.setattr(tlp, "get_num_params_of_pretrained", lambda name: 124_000_000)
    monkeypatch.setenv("NEURONSCOPE_MAX_MODEL_PARAMS", "2000000000")

    check_model_size("small-model")  # must not raise


def test_check_model_size_fails_open_when_lookup_errors(monkeypatch):
    import transformer_lens.loading_from_pretrained as tlp

    def _raise(model_name: str) -> int:
        raise ValueError("unknown model, can't resolve a config")

    monkeypatch.setattr(tlp, "get_num_params_of_pretrained", _raise)

    # Can't determine the size (e.g. an unresolvable name) -> best-effort check fails
    # open rather than blocking the caller; UnsupportedModelError is what actually
    # rejects a bad model name, raised later in resolve_backend.
    check_model_size("not-a-real-model-xyz")


# ---------------------------------------------------------------------------
# Concurrency cap
# ---------------------------------------------------------------------------


def test_model_load_slot_allows_a_single_load(monkeypatch):
    monkeypatch.setenv("NEURONSCOPE_MAX_CONCURRENT_LOADS", "1")
    with model_load_slot():
        pass  # must not raise


def test_model_load_slot_rejects_when_cap_is_already_saturated(monkeypatch):
    monkeypatch.setenv("NEURONSCOPE_MAX_CONCURRENT_LOADS", "1")

    with model_load_slot():
        with pytest.raises(TooManyConcurrentModelLoadsError) as exc_info:
            with model_load_slot():
                pass  # never reached
        assert exc_info.value.max_concurrent == 1
        assert "NEURONSCOPE_MAX_CONCURRENT_LOADS" in str(exc_info.value)


def test_model_load_slot_frees_its_slot_on_exit(monkeypatch):
    monkeypatch.setenv("NEURONSCOPE_MAX_CONCURRENT_LOADS", "1")

    with model_load_slot():
        pass
    # The first slot was released on exit, so a second, sequential load must succeed.
    with model_load_slot():
        pass


def test_model_load_slot_respects_a_higher_configured_cap(monkeypatch):
    monkeypatch.setenv("NEURONSCOPE_MAX_CONCURRENT_LOADS", "2")

    with model_load_slot():
        with model_load_slot():
            pass  # two concurrent loads are fine under a cap of 2


def test_model_load_slot_rejects_true_concurrent_contention(monkeypatch):
    # Same as the nested-context test above, but across real threads, to prove the
    # limiter is actually thread-safe rather than just correct in a single-threaded
    # nested-call shape.
    monkeypatch.setenv("NEURONSCOPE_MAX_CONCURRENT_LOADS", "1")

    first_slot_held = threading.Event()
    release_first_slot = threading.Event()
    results: list[bool] = []

    def hold_first_slot():
        with model_load_slot():
            first_slot_held.set()
            release_first_slot.wait(timeout=5)

    thread = threading.Thread(target=hold_first_slot)
    thread.start()
    assert first_slot_held.wait(timeout=5)

    try:
        with pytest.raises(TooManyConcurrentModelLoadsError):
            with model_load_slot():
                results.append(True)  # never reached
    finally:
        release_first_slot.set()
        thread.join(timeout=5)

    assert results == []
    # The slot is free again once the first thread's context exits.
    with model_load_slot():
        pass


# ---------------------------------------------------------------------------
# Wired into the actual model-loading path (core/trace.py's _load_and_validate)
# ---------------------------------------------------------------------------


class _StubCfg:
    n_ctx = 1024
    model_name = "stub-model"


class _StubModel:
    cfg = _StubCfg()

    def to_tokens(self, prompt: str, truncate: bool = True) -> torch.Tensor:
        return torch.zeros((1, 3))


class _StubBackend:
    name = "stub"

    def load_model(self, model_name: str, device: str | None = None) -> _StubModel:
        return _StubModel()


def test_load_and_validate_enforces_the_concurrency_cap(monkeypatch):
    monkeypatch.setenv("NEURONSCOPE_MAX_CONCURRENT_LOADS", "1")
    backend = _StubBackend()

    with model_load_slot():
        with pytest.raises(TooManyConcurrentModelLoadsError):
            _load_and_validate(backend, "not-a-real-model-xyz", "hello")


def test_load_and_validate_enforces_the_size_cap(monkeypatch):
    import transformer_lens.loading_from_pretrained as tlp

    monkeypatch.setattr(tlp, "get_num_params_of_pretrained", lambda name: 5_000_000_000)
    monkeypatch.setenv("NEURONSCOPE_MAX_MODEL_PARAMS", "2000000000")
    backend = _StubBackend()

    with pytest.raises(ModelTooLargeError):
        _load_and_validate(backend, "huge-model", "hello")


def test_load_and_validate_succeeds_when_under_both_caps(monkeypatch):
    monkeypatch.setenv("NEURONSCOPE_MAX_CONCURRENT_LOADS", "1")
    monkeypatch.setenv("NEURONSCOPE_MAX_MODEL_PARAMS", "2000000000")
    import transformer_lens.loading_from_pretrained as tlp

    monkeypatch.setattr(tlp, "get_num_params_of_pretrained", lambda name: 124_000_000)
    backend = _StubBackend()

    model = _load_and_validate(backend, "stub-model", "hello")
    assert model.cfg.model_name == "stub-model"
