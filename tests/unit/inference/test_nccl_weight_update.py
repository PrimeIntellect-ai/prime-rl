from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from prime_rl.inference.vllm.worker import nccl


def test_receive_state_dicts_preserves_broadcast_boundaries(monkeypatch):
    receiver = object.__new__(nccl.NCCLWeightBroadcastReceiver)
    receiver.communicator = object()
    first = [("model.layers.0.weight", torch.tensor([1.0]))]
    second = [("model.layers.1.weight", torch.tensor([2.0]))]
    chunks = iter((iter(first), iter(second)))

    monkeypatch.setattr(nccl, "receive_integer", lambda communicator: 2)
    monkeypatch.setattr(nccl, "receive_state_dict", lambda communicator: next(chunks))

    assert [list(chunk) for chunk in receiver.receive_state_dicts()] == [first, second]


def test_checkpoint_reload_finalizes_each_broadcast_state_dict():
    model = torch.nn.Linear(1, 1)
    first = iter([("model.layers.0.weight", torch.tensor([1.0]))])
    second = iter([("model.layers.1.weight", torch.tensor([2.0]))])
    receiver = SimpleNamespace(
        receive_state_dicts=MagicMock(return_value=iter((first, second))),
        receive_state_dict=MagicMock(side_effect=AssertionError("flat stream used")),
    )
    worker = object.__new__(nccl.NCCLWeightUpdateWorker)
    worker.model_runner = SimpleNamespace(model=model, model_config=object())
    worker.vllm_config = object()
    worker.nccl_broadcast_receiver = receiver
    worker.quantize_in_weight_transfer = False

    with patch.object(nccl, "load_weights_checkpoint_layerwise") as load:
        worker.update_weights_from_path("unused")

    assert load.call_count == 2
    assert load.call_args_list[0].args[1] is first
    assert load.call_args_list[1].args[1] is second
    receiver.receive_state_dict.assert_not_called()


def test_kernel_reload_keeps_flat_state_dict_stream():
    model = torch.nn.Linear(1, 1)
    state_iter = iter([("weight", torch.tensor([[1.0]]))])
    receiver = SimpleNamespace(
        receive_state_dict=MagicMock(return_value=state_iter),
        receive_state_dicts=MagicMock(side_effect=AssertionError("chunked stream used")),
    )
    worker = object.__new__(nccl.NCCLWeightUpdateWorker)
    worker.model_runner = SimpleNamespace(model=model, model_config=object())
    worker.vllm_config = object()
    worker.nccl_broadcast_receiver = receiver
    worker.quantize_in_weight_transfer = True

    with (
        patch.object(nccl, "load_weights_kernel") as load,
        patch.object(nccl, "update_mla_absorbed_weights") as update,
    ):
        worker.update_weights_from_path("unused")

    load.assert_called_once_with(model, state_iter)
    update.assert_called_once_with(model)
    receiver.receive_state_dicts.assert_not_called()
