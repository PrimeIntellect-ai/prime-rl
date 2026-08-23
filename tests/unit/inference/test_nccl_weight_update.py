import pickle
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from prime_rl.inference.vllm.worker import nccl, weight_transfer


def test_receiver_preserves_state_dict_boundaries(monkeypatch):
    receiver = object.__new__(nccl.NCCLWeightBroadcastReceiver)
    receiver.communicator = object()
    streams = [iter([("model.embed_tokens.weight", 0)]), iter([("model.layers.0.input_layernorm.weight", 1)])]

    monkeypatch.setattr(nccl, "receive_integer", lambda _communicator: len(streams))
    monkeypatch.setattr(nccl, "receive_state_dict", lambda _communicator: streams.pop(0))

    assert [list(stream) for stream in receiver.receive_state_dicts()] == [
        [("model.embed_tokens.weight", 0)],
        [("model.layers.0.input_layernorm.weight", 1)],
    ]


def test_receive_state_dict_returns_views_of_receive_buffer():
    metadata = {
        torch.float32: [
            ("model.embed_tokens.weight", torch.Size([2]), 2),
            ("model.norm.weight", torch.Size([1]), 1),
        ]
    }
    serialized_metadata = pickle.dumps(metadata)
    payload = torch.tensor([1.0, 2.0, 3.0])

    class FakeCommunicator:
        device = torch.device("cpu")

        def __init__(self):
            self.call = 0
            self.receive_buffer = None

        def broadcast(self, tensor, src):
            assert src == 0
            if self.call == 0:
                tensor.fill_(len(serialized_metadata))
            elif self.call == 1:
                tensor.copy_(torch.tensor(list(serialized_metadata), dtype=torch.uint8))
            else:
                tensor.copy_(payload)
                self.receive_buffer = tensor
            self.call += 1

    communicator = FakeCommunicator()
    received = list(nccl.receive_state_dict(communicator))

    assert [key for key, _ in received] == ["model.embed_tokens.weight", "model.norm.weight"]
    assert torch.equal(received[0][1], torch.tensor([1.0, 2.0]))
    assert torch.equal(received[1][1], torch.tensor([3.0]))
    receive_storage = communicator.receive_buffer.untyped_storage().data_ptr()
    assert all(tensor.untyped_storage().data_ptr() == receive_storage for _, tensor in received)


def test_layerwise_reload_consumes_each_state_dict_separately(monkeypatch):
    events = []
    model = Mock()
    model.parameters.return_value = iter([SimpleNamespace(device="cpu")])
    model.load_weights.side_effect = lambda state_iter: events.append(("load", list(state_iter)))

    monkeypatch.setattr(weight_transfer, "set_current_vllm_config", lambda _config: nullcontext())
    monkeypatch.setattr(weight_transfer, "initialize_layerwise_reload", lambda _model: events.append("initialize"))
    monkeypatch.setattr(
        weight_transfer,
        "finalize_layerwise_reload",
        lambda _model, _model_config: events.append("finalize"),
    )

    weight_transfer.load_weight_groups_checkpoint_layerwise(
        model,
        [
            iter([("model.embed_tokens.weight", 0)]),
            iter([("model.layers.0.input_layernorm.weight", 1)]),
        ],
        model_config=object(),
        vllm_config=object(),
    )

    assert events == [
        "initialize",
        ("load", [("model.embed_tokens.weight", 0)]),
        ("load", [("model.layers.0.input_layernorm.weight", 1)]),
        "finalize",
    ]
