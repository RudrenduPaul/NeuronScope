"""The TransformerLens backend: the only concrete backend NeuronScope ships in v1.

All model loading, activation extraction, and activation patching goes through
``transformer_lens.HookedTransformer``. This module deliberately stays a thin wrapper —
the actual interpretability primitives (hooks, activation caching, logit attribution)
live in TransformerLens itself.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

# Keep Hugging Face Hub's weight-download progress bars off stdout. NeuronScope's --json
# CLI output and MCP tool results must be clean, parseable JSON with nothing else mixed
# into the stream a caller reads.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import torch
from transformer_lens import ActivationCache, HookedTransformer
from transformer_lens.loading_from_pretrained import OFFICIAL_MODEL_NAMES, get_official_model_name

from neuronscope.backends.base import Backend

logger = logging.getLogger("neuronscope.backends.transformer_lens")

# Component names NeuronScope accepts for `neuronscope patch`, mapped to the TransformerLens
# hook-name template each one addresses. These are real HookedTransformer hook points, not
# invented names — see HookedTransformer's hook_dict for the full set per model.
COMPONENT_HOOK_TEMPLATES: dict[str, str] = {
    "resid_pre": "blocks.{layer}.hook_resid_pre",
    "resid_mid": "blocks.{layer}.hook_resid_mid",
    "resid_post": "blocks.{layer}.hook_resid_post",
    "attn_out": "blocks.{layer}.hook_attn_out",
    "mlp_out": "blocks.{layer}.hook_mlp_out",
    "mlp_post": "blocks.{layer}.mlp.hook_post",
}


@dataclass
class ForwardResult:
    """Everything downstream code needs from a single forward pass."""

    tokens: torch.Tensor
    logits: torch.Tensor
    cache: ActivationCache
    predicted_token_id: int
    predicted_token_str: str


class TransformerLensBackend(Backend):
    name = "transformer_lens"

    def supports_model(self, model_name: str) -> bool:
        try:
            get_official_model_name(model_name)
            return True
        except ValueError:
            return False

    def list_supported_architectures(self) -> list[str]:
        # OFFICIAL_MODEL_NAMES is TransformerLens's own source of truth for which
        # pretrained checkpoints HookedTransformer.from_pretrained can load. Reported
        # directly from the installed package so this list can never drift from reality.
        return sorted(OFFICIAL_MODEL_NAMES)

    def _resolve_device(self, device: str | None) -> str:
        if device is not None:
            return device
        if torch.cuda.is_available():
            return "cuda"
        # Deliberately not auto-selecting MPS even when available: PyTorch's MPS backend
        # can silently produce incorrect results for some ops (TransformerLens itself
        # warns about this), and NeuronScope's activation-patching and logit-attribution
        # math depends on exact values, not just plausible-looking ones. CPU is slower
        # but correct. Pass device="mps" explicitly if you want it anyway.
        logger.warning(
            "No CUDA GPU detected — falling back to CPU. This works but will be slow for "
            "anything larger than a small model. (Apple Silicon MPS is available but not "
            "auto-selected: it can silently produce incorrect activation values; pass "
            "device='mps' explicitly if you want it.)"
        )
        return "cpu"

    def load_model(self, model_name: str, device: str | None = None) -> HookedTransformer:
        resolved_device = self._resolve_device(device)
        model = HookedTransformer.from_pretrained(model_name, device=resolved_device)
        model.eval()
        return model

    def get_activations(self, model: HookedTransformer, prompt: str) -> ForwardResult:
        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            logits, cache = model.run_with_cache(tokens)
        predicted_token_id = int(logits[0, -1].argmax().item())
        predicted_token_str = model.to_string(torch.tensor([predicted_token_id]))
        return ForwardResult(
            tokens=tokens,
            logits=logits,
            cache=cache,
            predicted_token_id=predicted_token_id,
            predicted_token_str=predicted_token_str,
        )

    def patch_activations(
        self,
        model: HookedTransformer,
        prompt: str,
        layer: int,
        component: str,
    ) -> tuple[ForwardResult, torch.Tensor]:
        """Run ``prompt`` clean, then again with the given layer/component zero-ablated.

        Returns the clean ForwardResult plus the patched run's final-position logits, so
        callers can compare both without re-running the clean pass.
        """
        if component not in COMPONENT_HOOK_TEMPLATES:
            raise ValueError(
                f"Unknown component '{component}'. Supported components: "
                f"{', '.join(sorted(COMPONENT_HOOK_TEMPLATES))}"
            )
        clean = self.get_activations(model, prompt)
        hook_name = COMPONENT_HOOK_TEMPLATES[component].format(layer=layer)

        def zero_ablate(activation: torch.Tensor, hook: Any) -> torch.Tensor:
            return torch.zeros_like(activation)

        with torch.no_grad():
            patched_logits = model.run_with_hooks(
                clean.tokens, fwd_hooks=[(hook_name, zero_ablate)]
            )
        return clean, patched_logits[0, -1]

    def ablate_single_component(
        self,
        model: HookedTransformer,
        tokens: torch.Tensor,
        layer: int,
        component_type: str,
        index: int,
    ) -> torch.Tensor:
        """Zero-ablate one attention head's output (component_type='head') or one MLP
        neuron (component_type='neuron') and return the patched final-position logits."""
        if component_type == "head":
            hook_name = f"blocks.{layer}.attn.hook_z"

            def hook_fn(activation: torch.Tensor, hook: Any) -> torch.Tensor:
                activation = activation.clone()
                activation[:, :, index, :] = 0.0
                return activation

        elif component_type == "neuron":
            hook_name = f"blocks.{layer}.mlp.hook_post"

            def hook_fn(activation: torch.Tensor, hook: Any) -> torch.Tensor:
                activation = activation.clone()
                activation[:, :, index] = 0.0
                return activation

        else:
            raise ValueError(f"Unknown component_type '{component_type}'")

        with torch.no_grad():
            patched_logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, hook_fn)])
        return patched_logits[0, -1]
