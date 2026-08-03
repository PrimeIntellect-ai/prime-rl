from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

from prime_rl.inference.vllm.worker import nccl, weight_transfer


def test_receiver_preserves_state_dict_boundaries(monkeypatch):
    receiver = object.__new__(nccl.NCCLWeightBroadcastReceiver)
    receiver.communicator = object()
    streams = [iter([("layer.0", 0)]), iter([("layer.1", 1)])]

    monkeypatch.setattr(nccl, "receive_integer", lambda _communicator: len(streams))
    monkeypatch.setattr(nccl, "receive_state_dict", lambda _communicator: streams.pop(0))

    assert [list(stream) for stream in receiver.receive_state_dicts()] == [
        [("layer.0", 0)],
        [("layer.1", 1)],
    ]


def test_layerwise_reload_brackets_all_state_dict_groups(monkeypatch):
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
        [iter([("layer.0", 0)]), iter([("layer.1", 1)])],
        model_config=object(),
        vllm_config=object(),
    )

    assert events == [
        "initialize",
        ("load", [("layer.0", 0)]),
        ("load", [("layer.1", 1)]),
        "finalize",
    ]
