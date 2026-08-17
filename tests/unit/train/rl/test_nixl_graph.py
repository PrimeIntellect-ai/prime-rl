from types import SimpleNamespace

import torch

from prime_rl.trainer.rl.broadcast.nixl.graph import (
    Destination,
    LazyWeight,
    TensorOperation,
    WeightLoadRecorder,
    make_hf_lazy_weights,
)
from prime_rl.trainer.rl.broadcast.nixl.trainer_tensor_table import (
    TrainerGroup,
    TrainerTensor,
    TrainerTensorTable,
)


def test_lazy_weight_records_slice_assignment() -> None:
    destination = torch.zeros((4, 2), dtype=torch.bfloat16)
    module = SimpleNamespace(weight=destination)
    recorder = WeightLoadRecorder(active_destination=Destination(module, "weight", destination))
    source = LazyWeight(
        "model.language_model.layers.0.linear_attn.conv1d.weight",
        torch.Size((4, 2)),
        torch.bfloat16,
        torch.device("cpu"),
        recorder,
    )

    destination[:2] = source[:2]

    assert len(recorder.copies) == 1
    copy = recorder.copies[0]
    assert copy.source_name == "model.language_model.layers.0.linear_attn.conv1d.weight"
    assert copy.destination_module is module
    assert copy.destination_name == "weight"
    assert copy.destination_offset == 0
    assert copy.destination_shape == (2, 2)
    assert copy.ops == (TensorOperation(name="__getitem__", args=(slice(None, 2, None),)),)


def test_hf_training_weights_do_not_require_a_custom_conversion_chain() -> None:
    table = TrainerTensorTable(
        agents=[],
        staging_buffer_count=1,
        groups=[
            TrainerGroup(
                name="non_layer",
                tensors=[
                    TrainerTensor(
                        name="model.language_model.embed_tokens.weight",
                        wire_dtype="bfloat16",
                        shape=(2, 2),
                        shards=[],
                    )
                ],
            )
        ],
    )

    weights = make_hf_lazy_weights(
        table,
        device=torch.device("meta"),
        recorder=WeightLoadRecorder(),
        hf_config=SimpleNamespace(model_type="qwen3_vl_text"),
    )

    assert [name for name, _ in weights] == ["model.language_model.embed_tokens.weight"]
