from collections import deque

import pytest
import torch
from torch.multiprocessing.reductions import StorageWeakRef
from vllm.model_executor.model_loader.reload import record_metadata_for_reloading
from vllm.model_executor.model_loader.reload.layerwise import LAYERWISE_INFO
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import AutoWeightsLoader

from prime_rl.inference.vllm.worker.nccl import receive_state_dict
from prime_rl.inference.vllm.worker.weight_transfer import load_weights_checkpoint_layerwise
from prime_rl.utils.nccl import encode_tensor_metadata, iter_tensor_buckets


class LoopbackCommunicator:
    """Fills each receive buffer with what the trainer broadcasts for ``state_dict``."""

    device = torch.device("cpu")

    def __init__(self, state_dict: dict[str, torch.Tensor]):
        metadata: dict[torch.dtype, list[tuple[str, torch.Size, int]]] = {}
        for key, value in state_dict.items():
            metadata.setdefault(value.dtype, []).append((key, value.shape, value.numel()))
        encoded = encode_tensor_metadata(metadata)
        self.sent = deque([torch.tensor([len(encoded)]), torch.frombuffer(bytearray(encoded), dtype=torch.uint8)])
        for dtype, tensor_info in metadata.items():
            for bucket in iter_tensor_buckets(tensor_info, dtype):
                self.sent.append(torch.cat([state_dict[key].flatten() for key, _, _ in bucket]))
        self.receive_buffers: list[StorageWeakRef] = []

    def broadcast(self, tensor: torch.Tensor, src: int) -> None:
        tensor.copy_(self.sent.popleft().view_as(tensor))
        self.receive_buffers.append(StorageWeakRef(tensor.untyped_storage()))


def test_received_tensors_do_not_keep_a_shared_bucket_alive():
    state_dict = {
        "attn.scale": torch.rand(4).to(torch.bfloat16),
        "attn.proj": torch.rand(16, 16).to(torch.bfloat16),
        "experts": torch.rand(2, 8, 8),
    }
    communicator = LoopbackCommunicator(state_dict)

    received = dict(receive_state_dict(communicator))

    # A bucket holds one dtype: the bfloat16 tensors share one, the float32 tensor is alone in the next.
    shared_bucket, lone_bucket = communicator.receive_buffers[-2:]
    assert shared_bucket.expired()
    assert not lone_bucket.expired()
    assert received.keys() == state_dict.keys()
    assert all(torch.equal(received[key], value) for key, value in state_dict.items())


class Pair(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.Parameter(torch.zeros(4, 4))
        self.second = torch.nn.Parameter(torch.zeros(4))


class TinyModel(torch.nn.Module):
    """Loads weights the way vLLM models do, through each parameter's weight loader."""

    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(8, 4)
        self.pair = Pair()

    def load_weights(self, weights):
        for name, weight in weights:
            parameter = self.get_parameter(name)
            getattr(parameter, "weight_loader", default_weight_loader)(parameter, weight)


def checkpoint(seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        name: torch.rand(parameter.shape, generator=generator) for name, parameter in TinyModel().named_parameters()
    }


def breaks_off_before(update: dict[str, torch.Tensor], key: str):
    """The stream of ``update``, which fails when it reaches ``key``."""
    for name, weight in update.items():
        if name == key:
            raise ConnectionError("the trainer went away")
        yield name, weight


def test_failed_layerwise_reload_leaves_no_open_layer():
    model = TinyModel()
    record_metadata_for_reloading(model)
    serving = checkpoint(0)
    model.load_weights(serving.items())
    serving_parameters = dict(model.named_parameters())

    update = checkpoint(1)

    # ``embed`` completes, ``pair`` has one of its two tensors buffered when the stream breaks off.
    with pytest.raises(ConnectionError):
        load_weights_checkpoint_layerwise(model, breaks_off_before(update, "pair.second"), None, None)

    assert all(parameter is serving_parameters[name] for name, parameter in model.named_parameters())
    assert not any(parameter.is_meta for parameter in model.parameters())
    assert not any(LAYERWISE_INFO[module].can_load() for module in model.modules())
    assert not any(LAYERWISE_INFO[module].loaded_weights for module in model.modules())
    assert torch.equal(model.embed.weight, update["embed.weight"])
    assert torch.equal(model.pair.first, serving["pair.first"])
    assert torch.equal(model.pair.second, serving["pair.second"])

    complete = checkpoint(2)
    load_weights_checkpoint_layerwise(model, complete.items(), None, None)

    assert all(parameter is serving_parameters[name] for name, parameter in model.named_parameters())
    assert all(torch.equal(parameter, complete[name]) for name, parameter in model.named_parameters())


class Triple(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.up = torch.nn.Parameter(torch.zeros(6, 4))
        self.down = torch.nn.Parameter(torch.zeros(4, 6))
        self.gate = torch.nn.Parameter(torch.zeros(6, 4))


class AutoLoadedModel(torch.nn.Module):
    """Resolves each module's parameters once per load, as the vLLM models that use ``AutoWeightsLoader`` do."""

    def __init__(self):
        super().__init__()
        self.triple = Triple()

    def load_weights(self, weights):
        return AutoWeightsLoader(self).load_weights(weights)


def test_update_sent_again_after_a_failed_reload_applies_every_tensor():
    model = AutoLoadedModel()
    record_metadata_for_reloading(model)
    generator = torch.Generator().manual_seed(0)
    update = {name: torch.rand(parameter.shape, generator=generator) for name, parameter in model.named_parameters()}

    # ``triple`` has one of its three tensors buffered when the stream breaks off.
    with pytest.raises(ConnectionError):
        load_weights_checkpoint_layerwise(model, breaks_off_before(update, "triple.down"), None, None)

    # A layer left open with that tensor would count it again: the layer would complete one tensor early,
    # and the tensor that arrives after it would be dropped and stay at zero, without an error.
    load_weights_checkpoint_layerwise(model, update.items(), None, None)

    assert all(torch.equal(parameter, update[name]) for name, parameter in model.named_parameters())
