import pytest
import torch
import torch.distributed.checkpoint as dcp

from prime_rl.utils import dcp_export
from prime_rl.utils.dcp_export import iter_cpu_hf_groups


@pytest.mark.parametrize(
    ("model_type", "config_fields", "sensitive"),
    [
        ("glm4_moe", {}, ["model.layers.0.mlp.gate.e_score_correction_bias"]),
        ("glm_moe_dsa", {}, ["model.layers.0.mlp.gate.e_score_correction_bias"]),
        (
            "nemotron_h",
            {"layer_types": ["linear_attention", "moe"], "num_hidden_layers": 2},
            [
                "backbone.layers.0.mixer.A_log",
                "backbone.layers.0.mixer.D",
                "backbone.layers.1.mixer.gate.e_score_correction_bias",
            ],
        ),
    ],
)
def test_hf_identity_uses_canonical_fp32_policy(tmp_path, monkeypatch, model_type, config_fields, sensitive):
    exact = torch.tensor([0.1234567, -0.7654321], dtype=torch.float32)
    state = {name: exact.clone() for name in sensitive}
    state["lm_head.weight"] = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    schema = {name: tuple(tensor.shape) for name, tensor in state.items()}
    monkeypatch.setattr(
        dcp_export,
        "hf_schema_details",
        lambda *args: (schema, {}, {name: tensor.dtype for name, tensor in state.items()}),
    )
    checkpoint = tmp_path / "checkpoint"
    dcp.save({"app": {"model": state}}, checkpoint_id=checkpoint, no_dist=True)
    config = {"model_type": model_type, "num_hidden_layers": 1, **config_fields}
    output = {name: tensor for group in iter_cpu_hf_groups(checkpoint, config) for name, tensor in group.items()}
    assert output.keys() == state.keys()
    for name in sensitive:
        assert output[name].dtype == torch.float32
        torch.testing.assert_close(output[name], exact, rtol=0, atol=0)
    assert output["lm_head.weight"].dtype == torch.bfloat16


@pytest.mark.parametrize("integer_dtype", [torch.int8, torch.uint8])
def test_hf_identity_rejects_integer_payload_for_floating_state(tmp_path, monkeypatch, integer_dtype):
    state = {"lm_head.weight": torch.arange(6, dtype=integer_dtype).reshape(2, 3)}
    schema = {"lm_head.weight": (2, 3)}
    monkeypatch.setattr(dcp_export, "hf_schema_details", lambda *args: (schema, {}, {"lm_head.weight": torch.float32}))
    checkpoint = tmp_path / "checkpoint"
    dcp.save({"app": {"model": state}}, checkpoint_id=checkpoint, no_dist=True)
    with pytest.raises(ValueError):
        list(iter_cpu_hf_groups(checkpoint, {"model_type": "gpt2"}))
