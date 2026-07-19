"""Orchestration layer: resolve model -> pick backend -> run operation -> serialize.

Every public function here (`run_trace`, `run_activations`, `run_patch`, `run_circuit`)
takes a model name and a prompt, does the same three steps (resolve backend, load model,
validate the prompt fits the model's context window), then delegates the actual tensor
work to the backend and packages the result into the matching pydantic model from
`neuronscope.schema`. This is the one place the CLI and the MCP server both call into, so
their behavior can never drift apart.
"""

from __future__ import annotations

import contextlib
import io

import torch

from neuronscope.backends.base import Backend
from neuronscope.backends.transformer_lens import TransformerLensBackend
from neuronscope.core.registry import resolve_backend
from neuronscope.schema import (
    ActivationsResponse,
    ActivationSummary,
    CircuitComponent,
    CircuitResponse,
    HeadScore,
    ModelInfo,
    NeuronScore,
    PatchResponse,
    TraceResponse,
)

DEFAULT_TOP_K = 10


class PromptTooLongError(Exception):
    """Raised when a prompt tokenizes to more positions than the model's context window."""

    def __init__(self, model_name: str, n_tokens: int, n_ctx: int):
        self.model_name = model_name
        self.n_tokens = n_tokens
        self.n_ctx = n_ctx
        super().__init__(
            f"Prompt tokenizes to {n_tokens} tokens, which exceeds {model_name}'s context "
            f"window of {n_ctx} tokens. Shorten the prompt and try again."
        )


def _load_and_validate(backend: Backend, model_name: str, prompt: str):
    """Load the model and make sure the prompt fits before any heavy backend work runs."""
    # TransformerLens prints a "Loaded pretrained model ..." line straight to stdout on
    # every load, with no way to disable it via from_pretrained's own arguments. Silence
    # it here so it can never end up mixed into --json output or an MCP tool result.
    with contextlib.redirect_stdout(io.StringIO()):
        model = backend.load_model(model_name)
    # model.to_tokens() silently truncates to n_ctx by default (truncate=True), which
    # would make this check never fire. Count untruncated tokens explicitly so an
    # over-length prompt is caught here instead of quietly analyzing a truncated prompt.
    untruncated_tokens = model.to_tokens(prompt, truncate=False)
    n_tokens = untruncated_tokens.shape[-1]
    n_ctx = model.cfg.n_ctx
    if n_tokens > n_ctx:
        raise PromptTooLongError(model_name, n_tokens, n_ctx)
    return model


def _model_info(backend: Backend, model_name: str, model) -> ModelInfo:
    return ModelInfo(
        requested_name=model_name,
        resolved_name=model.cfg.model_name,
        backend=backend.name,
        device=str(model.cfg.device),
        n_layers=model.cfg.n_layers,
        n_heads=model.cfg.n_heads,
        d_model=model.cfg.d_model,
        d_mlp=model.cfg.d_mlp,
    )


def run_trace(model_name: str, prompt: str, top_k: int = DEFAULT_TOP_K) -> TraceResponse:
    backend = resolve_backend(model_name)
    model = _load_and_validate(backend, model_name, prompt)
    result = backend.get_activations(model, prompt)
    cache = result.cache

    head_stack, head_labels = cache.stack_head_results(
        layer=-1, pos_slice=-1, return_labels=True
    )
    head_attrs = cache.logit_attrs(
        head_stack, tokens=torch.tensor([result.predicted_token_id]), pos_slice=-1
    ).squeeze(-1)

    head_scores: list[HeadScore] = []
    for label, attr in zip(head_labels, head_attrs.tolist()):
        # Labels look like "L{layer}H{head}".
        layer_str, head_str = label[1:].split("H")
        head_scores.append(
            HeadScore(layer=int(layer_str), head_index=int(head_str), logit_attribution=attr)
        )
    head_scores.sort(key=lambda h: abs(h.logit_attribution), reverse=True)

    neuron_scores: list[NeuronScore] = []
    for layer in range(model.cfg.n_layers):
        post = cache[f"blocks.{layer}.mlp.hook_post"][0, -1, :]
        top_vals, top_idx = post.abs().topk(min(top_k, post.shape[-1]))
        for idx, _ in zip(top_idx.tolist(), top_vals.tolist()):
            neuron_scores.append(
                NeuronScore(layer=layer, neuron_index=idx, activation=float(post[idx].item()))
            )
    neuron_scores.sort(key=lambda n: abs(n.activation), reverse=True)

    return TraceResponse(
        model=_model_info(backend, model_name, model),
        prompt=prompt,
        predicted_token=result.predicted_token_str,
        predicted_token_id=result.predicted_token_id,
        top_neurons=neuron_scores[:top_k],
        top_heads=head_scores[:top_k],
    )


def run_activations(model_name: str, prompt: str) -> ActivationsResponse:
    backend = resolve_backend(model_name)
    model = _load_and_validate(backend, model_name, prompt)
    result = backend.get_activations(model, prompt)
    cache = result.cache
    n_tokens = result.tokens.shape[-1]

    summaries: list[ActivationSummary] = []
    for layer in range(model.cfg.n_layers):
        for hook_suffix in ("hook_resid_post", "mlp.hook_post"):
            hook_name = f"blocks.{layer}.{hook_suffix}"
            tensor = cache[hook_name][0]  # drop batch dim -> [pos, ...]
            flat = tensor.reshape(tensor.shape[0], -1)
            per_pos_max = flat.abs().max(dim=-1).values
            max_position = int(per_pos_max.argmax().item())
            summaries.append(
                ActivationSummary(
                    hook_name=hook_name,
                    layer=layer,
                    shape=list(tensor.shape),
                    mean=float(tensor.mean().item()),
                    std=float(tensor.std().item()),
                    max_value=float(tensor.max().item()),
                    min_value=float(tensor.min().item()),
                    max_position=max_position,
                )
            )
        pattern = cache[f"blocks.{layer}.attn.hook_pattern"][0]  # [head, pos, pos]
        summaries.append(
            ActivationSummary(
                hook_name=f"blocks.{layer}.attn.hook_pattern",
                layer=layer,
                shape=list(pattern.shape),
                mean=float(pattern.mean().item()),
                std=float(pattern.std().item()),
                max_value=float(pattern.max().item()),
                min_value=float(pattern.min().item()),
                max_position=int(pattern.reshape(-1).argmax().item()),
            )
        )

    return ActivationsResponse(
        model=_model_info(backend, model_name, model),
        prompt=prompt,
        n_tokens=int(n_tokens),
        activations=summaries,
    )


def run_patch(model_name: str, prompt: str, layer: int, component: str) -> PatchResponse:
    backend = resolve_backend(model_name)
    model = _load_and_validate(backend, model_name, prompt)
    clean, patched_final_logits = backend.patch_activations(model, prompt, layer, component)

    baseline_logits = clean.logits[0, -1]
    baseline_id = int(baseline_logits.argmax().item())
    patched_id = int(patched_final_logits.argmax().item())

    return PatchResponse(
        model=_model_info(backend, model_name, model),
        prompt=prompt,
        layer=layer,
        component=component,
        baseline_predicted_token=model.to_string(torch.tensor([baseline_id])),
        baseline_predicted_token_id=baseline_id,
        baseline_top_logit=float(baseline_logits[baseline_id].item()),
        patched_predicted_token=model.to_string(torch.tensor([patched_id])),
        patched_predicted_token_id=patched_id,
        patched_top_logit=float(patched_final_logits[patched_id].item()),
        logit_delta=float(
            (patched_final_logits[baseline_id] - baseline_logits[baseline_id]).item()
        ),
        prediction_changed=baseline_id != patched_id,
    )


def run_circuit(
    model_name: str, prompt: str, top_k: int = DEFAULT_TOP_K
) -> CircuitResponse:
    backend = resolve_backend(model_name)
    if not isinstance(backend, TransformerLensBackend):
        # Only the TransformerLens backend implements single-component ablation today.
        raise NotImplementedError("circuit discovery requires the transformer_lens backend")

    model = _load_and_validate(backend, model_name, prompt)
    result = backend.get_activations(model, prompt)
    cache = result.cache
    tokens = result.tokens
    target_id = result.predicted_token_id
    baseline_logit = float(result.logits[0, -1, target_id].item())

    head_stack, head_labels = cache.stack_head_results(
        layer=-1, pos_slice=-1, return_labels=True
    )
    head_attrs = cache.logit_attrs(
        head_stack, tokens=torch.tensor([target_id]), pos_slice=-1
    ).squeeze(-1)
    head_candidates = sorted(
        zip(head_labels, head_attrs.tolist()), key=lambda p: abs(p[1]), reverse=True
    )[:top_k]

    neuron_candidates: list[tuple[int, int, float]] = []
    for layer in range(model.cfg.n_layers):
        post = cache[f"blocks.{layer}.mlp.hook_post"][0, -1, :]
        top_vals, top_idx = post.abs().topk(min(3, post.shape[-1]))
        for idx in top_idx.tolist():
            neuron_candidates.append((layer, idx, float(post[idx].item())))
    neuron_candidates.sort(key=lambda t: abs(t[2]), reverse=True)
    neuron_candidates = neuron_candidates[:top_k]

    components: list[CircuitComponent] = []
    for label, _attr in head_candidates:
        layer_str, head_str = label[1:].split("H")
        layer_i, head_i = int(layer_str), int(head_str)
        patched_logits = backend.ablate_single_component(model, tokens, layer_i, "head", head_i)
        drop = baseline_logit - float(patched_logits[target_id].item())
        components.append(
            CircuitComponent(
                layer=layer_i, component_type="head", index=head_i, logit_drop_on_ablation=drop
            )
        )

    for layer_i, neuron_i, _act in neuron_candidates:
        patched_logits = backend.ablate_single_component(
            model, tokens, layer_i, "neuron", neuron_i
        )
        drop = baseline_logit - float(patched_logits[target_id].item())
        components.append(
            CircuitComponent(
                layer=layer_i,
                component_type="neuron",
                index=neuron_i,
                logit_drop_on_ablation=drop,
            )
        )

    components.sort(key=lambda c: abs(c.logit_drop_on_ablation), reverse=True)

    return CircuitResponse(
        model=_model_info(backend, model_name, model),
        prompt=prompt,
        predicted_token=result.predicted_token_str,
        predicted_token_id=target_id,
        components=components[:top_k],
    )
