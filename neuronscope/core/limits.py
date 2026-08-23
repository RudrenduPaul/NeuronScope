"""Resource caps on MCP/CLI model loading.

NeuronScope loads a fresh copy of whatever model a caller names on every single
`trace`/`activations`/`patch`/`circuit` call -- nothing here caches a loaded model
across calls (see `core/trace.py`'s module docstring). That means two things are
otherwise completely unbounded:

- **Model size.** A caller can name an arbitrarily large checkpoint and NeuronScope
  will happily try to load the whole thing into memory.
- **Concurrency.** Nothing stops several tool calls in flight at once from each
  bringing a full model into memory at the same time, multiplying the memory hit.

Either one alone is enough for a malicious or merely careless MCP client to exhaust
the host's memory. This module adds two independent, configurable caps that close
both: a parameter-count ceiling checked *before* any weights are downloaded/loaded,
and a concurrency ceiling on in-flight loads. Both fail fast with a clear, structured
error (surfaced the same way as `UnsupportedModelError` etc.) instead of blocking or
crashing -- this is a resource *cap*, not a queue.

This is defense in depth, not a replacement for the process-level resource limit
(cgroup/ulimit/container memory cap) the README still recommends for untrusted
deployments -- it catches the common case cheaply, in-process, with no operator setup
required.
"""

from __future__ import annotations

import os
import threading

# ~2B parameters is comfortably loadable in float32 on a typical 16GB-RAM machine
# (roughly 8GB resident for the weights alone, leaving headroom for activations and
# the activation cache) and covers every model named as a first-class example in the
# README and tool descriptions (GPT-2, Pythia up to ~1.4B, small Llama/Gemma/Qwen
# checkpoints). Bigger checkpoints still work fine -- raise the cap via
# NEURONSCOPE_MAX_MODEL_PARAMS on hardware that can actually hold them.
DEFAULT_MAX_MODEL_PARAMS = 2_000_000_000

# Model loading -- and the forward pass that follows it -- is memory-hungry and not
# meaningfully parallelizable on the CPU-by-default, single-process deployment this
# ships as, so the default only allows one load in flight at a time. Raise
# NEURONSCOPE_MAX_CONCURRENT_LOADS on a machine with enough memory to genuinely hold
# several models at once.
DEFAULT_MAX_CONCURRENT_LOADS = 1

MAX_MODEL_PARAMS_ENV = "NEURONSCOPE_MAX_MODEL_PARAMS"
MAX_CONCURRENT_LOADS_ENV = "NEURONSCOPE_MAX_CONCURRENT_LOADS"


class ModelTooLargeError(Exception):
    """Raised when a requested model's parameter count exceeds the configured cap."""

    def __init__(self, model_name: str, n_params: int, max_params: int):
        self.model_name = model_name
        self.n_params = n_params
        self.max_params = max_params
        super().__init__(
            f"Model '{model_name}' has {n_params:,} parameters, which exceeds the "
            f"configured cap of {max_params:,} (set via {MAX_MODEL_PARAMS_ENV}). Raise "
            f"{MAX_MODEL_PARAMS_ENV} if your hardware can hold this model, or request a "
            f"smaller one."
        )


class TooManyConcurrentModelLoadsError(Exception):
    """Raised when a load request arrives while the concurrency cap is already saturated."""

    def __init__(self, max_concurrent: int):
        self.max_concurrent = max_concurrent
        super().__init__(
            f"Already {max_concurrent} model load(s) in flight, which is the configured "
            f"cap (set via {MAX_CONCURRENT_LOADS_ENV}). Retry once the in-flight request "
            f"completes, or raise {MAX_CONCURRENT_LOADS_ENV} if your hardware can hold "
            f"more models loaded at once."
        )


def _positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def max_model_params() -> int:
    """The configured parameter-count cap (NEURONSCOPE_MAX_MODEL_PARAMS, or the default)."""
    return _positive_int_env(MAX_MODEL_PARAMS_ENV, DEFAULT_MAX_MODEL_PARAMS)


def max_concurrent_loads() -> int:
    """The configured concurrency cap (NEURONSCOPE_MAX_CONCURRENT_LOADS, or the default)."""
    return _positive_int_env(MAX_CONCURRENT_LOADS_ENV, DEFAULT_MAX_CONCURRENT_LOADS)


def check_model_size(model_name: str) -> None:
    """Raise ModelTooLargeError if ``model_name``'s parameter count exceeds the cap.

    Uses TransformerLens's own config lookup (``get_num_params_of_pretrained``), which
    reads only the model's config, not its weights, so an oversized request is rejected
    before any multi-gigabyte download happens. If the parameter count can't be
    determined at all (offline with nothing cached yet, or a model type TransformerLens
    doesn't report ``n_params`` for), this fails open and lets the load proceed rather
    than blocking a legitimate request on a best-effort check -- the concurrency cap
    below still bounds how many such loads can be in flight at once.
    """
    from transformer_lens.loading_from_pretrained import get_num_params_of_pretrained

    try:
        n_params = get_num_params_of_pretrained(model_name)
    except Exception:
        return
    limit = max_model_params()
    if n_params > limit:
        raise ModelTooLargeError(model_name, n_params, limit)


class _LoadSlotLimiter:
    """Thread-safe counting limiter.

    ``try_acquire`` never blocks or queues: it returns False immediately once the limit
    is reached, so a caller over the cap gets an immediate, clear error instead of a
    hang -- callers of ``model_load_slot`` turn that into ``TooManyConcurrentModelLoadsError``.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active = 0

    def try_acquire(self, limit: int) -> bool:
        with self._lock:
            if self._active >= limit:
                return False
            self._active += 1
            return True

    def release(self) -> None:
        with self._lock:
            self._active = max(0, self._active - 1)


# Process-wide: the whole point is to bound concurrent loads across every call into this
# process, whichever thread (MCP tool dispatch, CLI invocation, or a test) makes it.
_limiter = _LoadSlotLimiter()


class model_load_slot:
    """Context manager guarding one model load against the concurrency cap.

    Raises ``TooManyConcurrentModelLoadsError`` on ``__enter__`` instead of blocking if
    ``max_concurrent_loads()`` in-flight loads are already active.
    """

    def __enter__(self) -> "model_load_slot":
        limit = max_concurrent_loads()
        if not _limiter.try_acquire(limit):
            raise TooManyConcurrentModelLoadsError(limit)
        return self

    def __exit__(self, *exc_info: object) -> None:
        _limiter.release()
