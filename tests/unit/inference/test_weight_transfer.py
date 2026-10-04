import struct

import pytest
import torch
from safetensors.torch import save_file

from prime_rl.inference.vllm.worker.weight_transfer import load_sparse_delta_weights
from prime_rl.utils.delta import (
    DELTA_INDEX_SUFFIX,
    DELTA_METADATA_FORMAT_KEY,
    DELTA_METADATA_FORMAT_V1,
    DELTA_VALUE_SUFFIX,
    ModelDeltaManager,
)


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_load_sparse_delta_weights_applies_delta(tmp_path, streaming, dtype) -> None:
    module = torch.nn.Linear(2, 2, bias=True, dtype=dtype)
    with torch.no_grad():
        module.weight.fill_(1.0)
        module.bias.zero_()

    base = {name: tensor.clone() for name, tensor in module.state_dict().items()}
    target = {"weight": torch.tensor([[-0.001, 1.0], [1.0, 2.0]], dtype=dtype), "bias": torch.ones(2, dtype=dtype)}
    delta_path = tmp_path / "delta.safetensors"
    manager = ModelDeltaManager()
    extract = (
        manager.extract_sparse_delta_streaming_from_state_dicts
        if streaming
        else manager.extract_sparse_delta_from_state_dicts
    )
    extract(base, target, delta_path)

    load_sparse_delta_weights(module, delta_path.as_posix())

    assert torch.equal(module.weight, target["weight"])
    assert torch.equal(module.bias, target["bias"])


def test_load_sparse_delta_weights_applies_full_tensor_delta(tmp_path) -> None:
    module = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        module.weight.zero_()

    base = {"weight": module.weight.detach().clone()}
    target = {"weight": torch.ones_like(module.weight)}
    delta_path = tmp_path / "delta.safetensors"
    ModelDeltaManager().extract_sparse_delta_from_state_dicts(base, target, delta_path)

    load_sparse_delta_weights(module, delta_path.as_posix())

    assert torch.equal(module.weight, target["weight"])


def test_load_sparse_delta_weights_accepts_empty_delta(tmp_path) -> None:
    module = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        module.weight.fill_(1.0)

    base = {"weight": module.weight.detach().clone()}
    target = {"weight": module.weight.detach().clone()}
    delta_path = tmp_path / "delta.safetensors"
    ModelDeltaManager().extract_sparse_delta_from_state_dicts(base, target, delta_path)

    load_sparse_delta_weights(module, delta_path.as_posix())

    assert torch.equal(module.weight, target["weight"])


def test_load_sparse_delta_weights_applies_streaming_delta(tmp_path) -> None:
    module = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        module.weight.zero_()

    base = {"weight": module.weight.detach().clone()}
    target = {"weight": torch.tensor([[0.0, 1.0], [0.0, 2.0]])}
    delta_path = tmp_path / "delta.stream"
    ModelDeltaManager().extract_sparse_delta_streaming_from_state_dicts(base, target, delta_path)

    load_sparse_delta_weights(module, delta_path.as_posix())

    assert torch.equal(module.weight, target["weight"])


def test_load_sparse_delta_weights_accepts_legacy_v1_safetensors(tmp_path) -> None:
    module = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        module.weight.zero_()
    delta_path = tmp_path / "delta.safetensors"
    save_file(
        {
            f"weight{DELTA_INDEX_SUFFIX}": torch.tensor([1, 3], dtype=torch.int32),
            f"weight{DELTA_VALUE_SUFFIX}": torch.tensor([2.0, 4.0]),
        },
        delta_path,
        metadata={DELTA_METADATA_FORMAT_KEY: DELTA_METADATA_FORMAT_V1},
    )

    load_sparse_delta_weights(module, delta_path.as_posix())

    assert torch.equal(module.weight, torch.tensor([[0.0, 2.0], [0.0, 4.0]]))


def test_load_sparse_delta_weights_accepts_legacy_v1_stream(tmp_path) -> None:
    module = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        module.weight.zero_()
    delta_path = tmp_path / "delta.stream"
    name = b"weight"
    indices = torch.tensor([1, 3], dtype=torch.int32)
    values = torch.tensor([2.0, 4.0], dtype=torch.float32)
    payload = b"".join(
        (
            struct.pack("<8sI", b"PDELSTRM", 1),
            struct.pack("<I B B H Q Q", len(name), 1, 3, 0, indices.nbytes, values.numel()),
            name,
            indices.numpy().tobytes(),
            values.numpy().tobytes(),
        )
    )
    delta_path.write_bytes(payload)

    load_sparse_delta_weights(module, delta_path.as_posix())

    assert torch.equal(module.weight, torch.tensor([[0.0, 2.0], [0.0, 4.0]]))
