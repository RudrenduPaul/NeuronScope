"""Pydantic response models for every NeuronScope operation.

Every response carries ``schema_version`` so downstream consumers (the CLI's ``--json``
output and the MCP server's tool results share the same models) can detect breaking
changes without guessing at field presence.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = 1


class ModelInfo(BaseModel):
    """Identifies which model and device actually produced a result."""

    requested_name: str
    resolved_name: str
    backend: str
    device: str
    n_layers: int
    n_heads: int
    d_model: int
    d_mlp: int


class NeuronScore(BaseModel):
    """One MLP neuron's activation at the final prompt position."""

    layer: int
    neuron_index: int
    activation: float


class HeadScore(BaseModel):
    """One attention head's direct logit attribution to the predicted next token."""

    layer: int
    head_index: int
    logit_attribution: float


class TraceResponse(BaseModel):
    schema_version: int = Field(default=SCHEMA_VERSION)
    operation: Literal["trace"] = "trace"
    model: ModelInfo
    prompt: str
    predicted_token: str
    predicted_token_id: int
    top_neurons: list[NeuronScore]
    top_heads: list[HeadScore]


class ActivationSummary(BaseModel):
    """Summary statistics for one hook point's activation tensor."""

    hook_name: str
    layer: int | None
    shape: list[int]
    mean: float
    std: float
    max_value: float
    max_position: int
    min_value: float


class ActivationsResponse(BaseModel):
    schema_version: int = Field(default=SCHEMA_VERSION)
    operation: Literal["activations"] = "activations"
    model: ModelInfo
    prompt: str
    n_tokens: int
    activations: list[ActivationSummary]


class PatchResponse(BaseModel):
    schema_version: int = Field(default=SCHEMA_VERSION)
    operation: Literal["patch"] = "patch"
    model: ModelInfo
    prompt: str
    layer: int
    component: str
    ablation_type: Literal["zero"] = "zero"
    baseline_predicted_token: str
    baseline_predicted_token_id: int
    baseline_top_logit: float
    patched_predicted_token: str
    patched_predicted_token_id: int
    patched_top_logit: float
    logit_delta: float
    prediction_changed: bool


class CircuitComponent(BaseModel):
    """One component identified as part of the approximate circuit, with its measured
    causal effect from single-component ablation."""

    layer: int
    component_type: Literal["head", "neuron"]
    index: int
    logit_drop_on_ablation: float


class CircuitResponse(BaseModel):
    schema_version: int = Field(default=SCHEMA_VERSION)
    operation: Literal["circuit"] = "circuit"
    model: ModelInfo
    prompt: str
    predicted_token: str
    predicted_token_id: int
    components: list[CircuitComponent]
    method: str = (
        "Ranks components by direct logit attribution, then measures each candidate's "
        "individual causal effect via single-component zero-ablation on this one prompt. "
        "This is an approximate, best-effort circuit sketch, not a full path-patching "
        "analysis with clean/corrupted prompt pairs, and it does not capture interaction "
        "effects between components."
    )


class ErrorResponse(BaseModel):
    schema_version: int = Field(default=SCHEMA_VERSION)
    operation: str
    error_type: str
    message: str
