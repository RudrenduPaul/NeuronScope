"""Tests for the pydantic response schemas: versioning and (de)serialization round-trips."""

from neuronscope.schema import (
    ActivationsResponse,
    ActivationSummary,
    CircuitComponent,
    CircuitResponse,
    ErrorResponse,
    HeadScore,
    ModelInfo,
    NeuronScore,
    PatchResponse,
    SCHEMA_VERSION,
    TraceResponse,
)


def _model_info() -> ModelInfo:
    return ModelInfo(
        requested_name="gpt2",
        resolved_name="gpt2",
        backend="transformer_lens",
        device="cpu",
        n_layers=12,
        n_heads=12,
        d_model=768,
        d_mlp=3072,
    )


def test_schema_version_defaults_to_one():
    assert SCHEMA_VERSION == 1
    response = TraceResponse(
        model=_model_info(),
        prompt="hello",
        predicted_token=" world",
        predicted_token_id=995,
        top_neurons=[],
        top_heads=[],
    )
    assert response.schema_version == 1
    assert response.operation == "trace"


def test_trace_response_round_trip():
    response = TraceResponse(
        model=_model_info(),
        prompt="The capital of France is",
        predicted_token=" Paris",
        predicted_token_id=6342,
        top_neurons=[NeuronScore(layer=3, neuron_index=42, activation=1.23)],
        top_heads=[HeadScore(layer=5, head_index=1, logit_attribution=0.87)],
    )
    payload = response.model_dump_json()
    restored = TraceResponse.model_validate_json(payload)
    assert restored == response


def test_activations_response_round_trip():
    response = ActivationsResponse(
        model=_model_info(),
        prompt="hello world",
        n_tokens=3,
        activations=[
            ActivationSummary(
                hook_name="blocks.0.hook_resid_post",
                layer=0,
                shape=[3, 768],
                mean=0.01,
                std=0.5,
                max_value=2.3,
                min_value=-2.1,
                max_position=2,
            )
        ],
    )
    restored = ActivationsResponse.model_validate_json(response.model_dump_json())
    assert restored == response


def test_patch_response_prediction_changed_flag():
    response = PatchResponse(
        model=_model_info(),
        prompt="hello",
        layer=2,
        component="mlp_out",
        baseline_predicted_token=" world",
        baseline_predicted_token_id=1,
        baseline_top_logit=5.0,
        patched_predicted_token=" there",
        patched_predicted_token_id=2,
        patched_top_logit=4.0,
        logit_delta=-1.0,
        prediction_changed=True,
    )
    assert response.prediction_changed is True
    assert response.ablation_type == "zero"


def test_circuit_response_carries_method_caveat():
    response = CircuitResponse(
        model=_model_info(),
        prompt="hello",
        predicted_token=" world",
        predicted_token_id=1,
        components=[
            CircuitComponent(layer=0, component_type="head", index=3, logit_drop_on_ablation=0.4)
        ],
    )
    assert "approximate" in response.method.lower()
    assert response.components[0].component_type == "head"


def test_error_response_shape():
    response = ErrorResponse(
        operation="trace", error_type="UnsupportedModelError", message="not supported"
    )
    assert response.schema_version == 1
    data = response.model_dump()
    assert data["error_type"] == "UnsupportedModelError"
