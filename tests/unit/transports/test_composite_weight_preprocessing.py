from types import SimpleNamespace

import torch
from torch import nn

from prime_rl.trainer.models.nemotron_h.configuration_nemotron_h import NemotronHConfig
from prime_rl.trainer.models.nemotron_h_omni import NemotronHOmniForCausalLM
from prime_rl.transports.weights.nccl import preprocess_layer_checkpoint
from prime_rl.transports.weights.nixl.graph import WeightLoadRecorder, make_hf_lazy_weights
from prime_rl.transports.weights.nixl.trainer_tensor_table import (
    TrainerGroup,
    TrainerTensor,
    TrainerTensorTable,
)


def _config():
    return SimpleNamespace(
        model_type="nemotron_h_omni",
        llm_config=NemotronHConfig(
            hidden_size=8,
            vocab_size=16,
            layers_block_type=["moe"],
            n_routed_experts=2,
            num_experts_per_tok=1,
            moe_intermediate_size=4,
            moe_shared_expert_intermediate_size=4,
            moe_latent_size=4,
        ),
        vision_config=SimpleNamespace(num_hidden_layers=0),
    )


def _model_without_weights():
    model = object.__new__(NemotronHOmniForCausalLM)
    nn.Module.__init__(model)
    model.config = _config()
    return model


def _table(*tensors: TrainerTensor) -> TrainerTensorTable:
    return TrainerTensorTable(
        agents=[],
        staging_buffer_count=1,
        groups=[TrainerGroup(name="model", tensors=list(tensors))],
    )


def _tensor(name: str, shape: tuple[int, ...]) -> TrainerTensor:
    return TrainerTensor(
        name=name,
        wire_dtype="bfloat16",
        shape=shape,
        shards=[],
    )


def test_nccl_preprocess_converts_composite_decoder_layer_names():
    layer = {
        "model.language_model.layers.0.mlp.router.gate.weight": torch.empty(2, 8),
        "model.language_model.layers.0.mlp.experts.up_proj": torch.empty(2, 4, 8),
    }

    converted = preprocess_layer_checkpoint(_model_without_weights(), layer, layer_idx=0)

    assert set(converted) == {
        "language_model.backbone.layers.0.mixer.gate.weight",
        "language_model.backbone.layers.0.mixer.experts.0.up_proj.weight",
        "language_model.backbone.layers.0.mixer.experts.1.up_proj.weight",
    }


def test_nixl_lazy_weights_use_composite_hf_names():
    table = _table(
        _tensor("model.language_model.embed_tokens.weight", (16, 8)),
        _tensor("model.language_model.layers.0.mlp.router.gate.weight", (2, 8)),
        _tensor("model.language_model.layers.0.mlp.experts.up_proj", (2, 4, 8)),
        _tensor("model.vision_model.embeddings.patch_projection.weight", (8, 12)),
        _tensor("model.vision_projector.mlp1.linear1.weight", (8, 8)),
    )

    weights = make_hf_lazy_weights(
        table,
        device=torch.device("cpu"),
        recorder=WeightLoadRecorder(),
        hf_config=_config(),
    )

    assert [name for name, _ in weights] == [
        "language_model.backbone.embeddings.weight",
        "language_model.backbone.layers.0.mixer.experts.0.up_proj.weight",
        "language_model.backbone.layers.0.mixer.experts.1.up_proj.weight",
        "language_model.backbone.layers.0.mixer.gate.weight",
        "mlp1.1.weight",
        "vision_model.radio_model.model.patch_generator.embedder.weight",
    ]


def test_composite_detection_accepts_single_decoder_layer_dictionary():
    assert NemotronHOmniForCausalLM.is_prime_state_dict(
        {"model.language_model.layers.3.mamba.in_proj.weight": torch.empty(8, 8)}
    )


def test_nixl_lazy_weights_preserve_base_nemotron_h_conversion():
    config = _config().llm_config
    weights = make_hf_lazy_weights(
        _table(
            _tensor("model.embed_tokens.weight", (16, 8)),
            _tensor("model.layers.0.mlp.router.gate.weight", (2, 8)),
        ),
        device=torch.device("cpu"),
        recorder=WeightLoadRecorder(),
        hf_config=config,
    )

    assert [name for name, _ in weights] == [
        "backbone.embeddings.weight",
        "backbone.layers.0.mixer.gate.weight",
    ]
