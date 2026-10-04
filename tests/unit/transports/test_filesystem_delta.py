from types import SimpleNamespace

import pytest
import torch

from prime_rl.transports.weights.filesystem import FileSystemWeightSender
from prime_rl.utils.delta import ModelDeltaManager, verify_sparse_delta_state_dicts


@pytest.mark.parametrize(
    ("streaming", "filename"),
    [(False, "delta.safetensors"), (True, "delta.stream")],
)
def test_filesystem_sender_writes_base_marker_then_exact_delta(
    tmp_path,
    monkeypatch,
    streaming: bool,
    filename: str,
) -> None:
    base = {"model.layers.0.weight": torch.zeros((2, 2), dtype=torch.bfloat16)}
    target = {"model.layers.0.weight": torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.bfloat16)}
    states = [base, target]

    sender = object.__new__(FileSystemWeightSender)
    sender.lora_config = None
    sender.mode = "delta"
    sender.delta_index_encoding = "optimized"
    sender.delta_streaming_enabled = streaming
    sender.delta_stream_group_size = 4
    sender.delta_manager = ModelDeltaManager()
    sender._previous_state = None
    sender.world = SimpleNamespace(is_master=True)
    sender.logger = SimpleNamespace(debug=lambda _message: None)
    sender._gather_delta_state = lambda _model: states.pop(0)
    monkeypatch.setattr("prime_rl.transports.weights.filesystem.dist.barrier", lambda: None)

    model = torch.nn.Linear(2, 2, bias=False)
    base_dir = tmp_path / "step_0"
    delta_dir = tmp_path / "step_1"
    base_dir.mkdir()
    delta_dir.mkdir()

    sender._broadcast(model, 0, base_dir)
    sender._broadcast(model, 1, delta_dir)

    assert verify_sparse_delta_state_dicts(base, base, base_dir / filename).ok
    assert verify_sparse_delta_state_dicts(base, target, delta_dir / filename).ok
