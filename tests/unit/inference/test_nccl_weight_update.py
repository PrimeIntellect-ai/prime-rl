from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
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


def test_weight_update_rejects_missing_rank_metadata_before_receiving_weights():
    receiver = SimpleNamespace(
        receive_state_dict=MagicMock(),
        receive_state_dicts=MagicMock(),
    )
    worker = object.__new__(nccl.NCCLWeightUpdateWorker)
    worker.model_runner = SimpleNamespace(model=torch.nn.Linear(1, 1), model_config=object())
    worker.vllm_config = object()
    worker.nccl_broadcast_receiver = receiver
    worker.quantize_in_weight_transfer = False

    with pytest.raises(AttributeError, match="_prime_rl_global_inference_rank"):
        worker.update_weights_from_path("unused")

    receiver.receive_state_dict.assert_not_called()
    receiver.receive_state_dicts.assert_not_called()


def test_checkpoint_reload_rejects_empty_broadcast():
    receiver = SimpleNamespace(
        receive_state_dicts=MagicMock(return_value=iter(())),
        receive_state_dict=MagicMock(side_effect=AssertionError("flat stream used")),
    )
    worker = object.__new__(nccl.NCCLWeightUpdateWorker)
    worker.model_runner = SimpleNamespace(model=torch.nn.Linear(1, 1), model_config=object())
    worker.vllm_config = object()
    worker.nccl_broadcast_receiver = receiver
    worker.quantize_in_weight_transfer = False
    worker._prime_rl_global_inference_rank = 0
    worker._prime_rl_inference_world_size = 2

    with (
        patch.object(nccl, "load_weights_checkpoint_layerwise") as load,
        patch.object(nccl.logger, "info") as log_info,
        pytest.raises(RuntimeError, match="received no weight tensors"),
    ):
        worker.update_weights_from_path("unused")

    load.assert_not_called()
    log_info.assert_not_called()
    receiver.receive_state_dict.assert_not_called()


def test_checkpoint_reload_rejects_empty_state_dict():
    receiver = SimpleNamespace(
        receive_state_dicts=MagicMock(return_value=iter((iter(()),))),
        receive_state_dict=MagicMock(side_effect=AssertionError("flat stream used")),
    )
    worker = object.__new__(nccl.NCCLWeightUpdateWorker)
    worker.model_runner = SimpleNamespace(model=torch.nn.Linear(1, 1), model_config=object())
    worker.vllm_config = object()
    worker.nccl_broadcast_receiver = receiver
    worker.quantize_in_weight_transfer = False
    worker._prime_rl_global_inference_rank = 0
    worker._prime_rl_inference_world_size = 2

    with (
        patch.object(
            nccl,
            "load_weights_checkpoint_layerwise",
            side_effect=lambda model, state_iter, model_config, vllm_config: list(state_iter),
        ),
        patch.object(nccl.logger, "info") as log_info,
        pytest.raises(RuntimeError, match="received no weight tensors"),
    ):
        worker.update_weights_from_path("unused")

    log_info.assert_not_called()
    receiver.receive_state_dict.assert_not_called()


def test_kernel_reload_rejects_empty_state_dict():
    receiver = SimpleNamespace(
        receive_state_dict=MagicMock(return_value=iter(())),
        receive_state_dicts=MagicMock(side_effect=AssertionError("chunked stream used")),
    )
    worker = object.__new__(nccl.NCCLWeightUpdateWorker)
    worker.model_runner = SimpleNamespace(model=torch.nn.Linear(1, 1), model_config=object())
    worker.vllm_config = object()
    worker.nccl_broadcast_receiver = receiver
    worker.quantize_in_weight_transfer = True
    worker._prime_rl_global_inference_rank = 0
    worker._prime_rl_inference_world_size = 2

    with (
        patch.object(nccl, "load_weights_kernel", side_effect=lambda model, state_iter: list(state_iter)),
        patch.object(nccl, "update_mla_absorbed_weights") as update,
        patch.object(nccl.logger, "info") as log_info,
        pytest.raises(RuntimeError, match="received no weight tensors"),
    ):
        worker.update_weights_from_path("unused")

    update.assert_not_called()
    log_info.assert_not_called()
    receiver.receive_state_dicts.assert_not_called()


def test_checkpoint_reload_finalizes_each_broadcast_state_dict():
    model = torch.nn.Linear(1, 1)
    first_data = [("model.layers.0.weight", torch.tensor([1.0]))]
    second_data = [("model.layers.1.weight", torch.tensor([2.0]))]
    receiver = SimpleNamespace(
        receive_state_dicts=MagicMock(return_value=iter((iter(first_data), iter(second_data)))),
        receive_state_dict=MagicMock(side_effect=AssertionError("flat stream used")),
    )
    worker = object.__new__(nccl.NCCLWeightUpdateWorker)
    worker.model_runner = SimpleNamespace(model=model, model_config=object())
    worker.vllm_config = object()
    worker.nccl_broadcast_receiver = receiver
    worker.quantize_in_weight_transfer = False
    worker._prime_rl_global_inference_rank = 0
    worker._prime_rl_inference_world_size = 2

    loaded_state_dicts = []
    with patch.object(
        nccl,
        "load_weights_checkpoint_layerwise",
        side_effect=lambda model, state_iter, model_config, vllm_config: loaded_state_dicts.append(list(state_iter)),
    ) as load:
        worker.update_weights_from_path("unused")

    assert load.call_count == 2
    assert loaded_state_dicts == [first_data, second_data]
    receiver.receive_state_dict.assert_not_called()


def test_kernel_reload_keeps_flat_state_dict_stream():
    model = torch.nn.Linear(1, 1)
    state_data = [("weight", torch.tensor([[1.0]]))]
    receiver = SimpleNamespace(
        receive_state_dict=MagicMock(return_value=iter(state_data)),
        receive_state_dicts=MagicMock(side_effect=AssertionError("chunked stream used")),
    )
    worker = object.__new__(nccl.NCCLWeightUpdateWorker)
    worker.model_runner = SimpleNamespace(model=model, model_config=object())
    worker.vllm_config = object()
    worker.nccl_broadcast_receiver = receiver
    worker.quantize_in_weight_transfer = True
    worker._prime_rl_global_inference_rank = 1
    worker._prime_rl_inference_world_size = 2

    loaded_state_dicts = []
    with (
        patch.object(
            nccl,
            "load_weights_kernel",
            side_effect=lambda model, state_iter: loaded_state_dicts.append(list(state_iter)),
        ) as load,
        patch.object(nccl, "update_mla_absorbed_weights") as update,
    ):
        worker.update_weights_from_path("unused")

    load.assert_called_once()
    assert load.call_args.args[0] is model
    assert loaded_state_dicts == [state_data]
    update.assert_called_once_with(model)
    receiver.receive_state_dicts.assert_not_called()
