"""CPU tests for the torch-only NCCL weight broadcast wire protocol.

A fake in-process communicator pairs the sender-side serialization with the
receiver-side deserialization, so the layered stream can be checked for exact
symmetry — the property whose loss wedges receiver workers in a blocking
NCCL read: the receiver must consume exactly the items and broadcasts the
sender emitted per layer, no more, no fewer.
"""

import torch

from prime_rl.trainer.models.conversion_ops import apply_prime_to_hf
from prime_rl.trainer.models.glm4_moe.converting_glm4_moe import glm_moe_layer_ops
from prime_rl.trainer.models.glm_moe_dsa.converting_glm_moe_dsa import quantize_tt_layer_to_vllm_fp8_checkpoint
from prime_rl.transports.wire import (
    broadcast_integer,
    broadcast_state_dict,
    receive_integer,
    receive_state_dict,
)


class FakeWire:
    """In-process NCCL communicator stand-in.

    ``sender`` and ``receiver`` expose the same ``broadcast(tensor, src)``
    interface as vLLM's PyNcclCommunicator: the sender endpoint enqueues every
    broadcast, the receiver endpoint pops from the queue. Any mismatch
    between the number of broadcasts on the two sides shows up as a leftover
    item or an immediate pop from an empty queue.
    """

    def __init__(self):
        self.device = torch.device("cpu")
        self.queue: list[torch.Tensor] = []
        self.excess_receives = 0
        self.sender = self.Sender(self)
        self.receiver = self.Receiver(self)

    class Sender:
        def __init__(self, wire: "FakeWire"):
            self.device = wire.device
            self._wire = wire

        def broadcast(self, tensor: torch.Tensor, src: int) -> None:
            self._wire.queue.append(tensor.clone())

    class Receiver:
        def __init__(self, wire: "FakeWire"):
            self.device = wire.device
            self._wire = wire

        def broadcast(self, tensor: torch.Tensor, src: int) -> None:
            if not self._wire.queue:
                self._wire.excess_receives += 1
                raise AssertionError("receiver consumed more broadcasts than the sender emitted")
            tensor.copy_(self._wire.queue.pop(0))

    def assert_drained(self) -> None:
        assert self.queue == [], "sender emitted broadcasts the receiver never consumed"
        assert self.excess_receives == 0, "receiver waited for broadcasts the sender never emitted"


def _bytes_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.dtype == b.dtype and a.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        return torch.equal(a, b)
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def _prime_moe_layer(layer_idx: int, num_experts: int = 3) -> dict[str, torch.Tensor]:
    p = f"model.layers.{layer_idx}"
    return {
        f"{p}.input_layernorm.weight": torch.randn(64),
        f"{p}.self_attn.q_a_proj.weight": torch.randn(32, 64),
        f"{p}.self_attn.kv_a_proj_with_mqa.weight": torch.randn(16, 64),
        f"{p}.mlp.router.gate.weight": torch.randn(num_experts, 64),
        f"{p}.mlp.router.selection_bias": torch.randn(num_experts),
        f"{p}.mlp.experts.gate_proj": torch.randn(num_experts, 32, 64),
        f"{p}.mlp.experts.up_proj": torch.randn(num_experts, 32, 64),
        f"{p}.mlp.experts.down_proj": torch.randn(num_experts, 64, 32),
    }


def _quantized_wire_layers(num_layers: int = 3) -> list[dict[str, torch.Tensor]]:
    """Per-layer wire state dicts of a quantized NCCL broadcast, mirroring the
    transports-layer composition: prime -> HF checkpoint naming, then fp8."""
    layers = []
    for layer_idx in range(num_layers):
        hf = apply_prime_to_hf(_prime_moe_layer(layer_idx), glm_moe_layer_ops(layer_idx))
        layers.append(quantize_tt_layer_to_vllm_fp8_checkpoint(hf, layer_idx))
    return layers


def _send_layer(wire: FakeWire, layer: dict[str, torch.Tensor]) -> None:
    broadcast_state_dict(layer, wire.sender)


def _receive_layer(wire: FakeWire) -> list[tuple[str, torch.Tensor]]:
    return [(name, tensor.clone()) for name, tensor in receive_state_dict(wire.receiver)]


def test_layered_stream_roundtrip_quantized_moe_layers():
    """Sender serialization and receiver deserialization agree exactly, layer by
    layer, for a quantized MoE wire stream (bf16/fp32 norms, fp8 weights, fp32
    scales — several dtype groups per layer)."""
    layers = _quantized_wire_layers(num_layers=3)
    wire = FakeWire()

    broadcast_integer(len(layers), wire.sender)
    assert receive_integer(wire.receiver) == len(layers)

    for layer in layers:
        _send_layer(wire, layer)
        received = _receive_layer(wire)
        # The wire groups items by dtype, so the receiver's item order differs
        # from the sender's dict order — but the item COUNT and every
        # (name, shape, dtype, value) must agree exactly.
        assert len(received) == len(layer)
        assert sorted(name for name, _ in received) == sorted(layer.keys())
        by_name = dict(received)
        for name, sent in layer.items():
            assert by_name[name].shape == sent.shape, name
            assert by_name[name].dtype == sent.dtype, name
            assert _bytes_equal(by_name[name], sent), name

    wire.assert_drained()


def test_receiver_drain_is_metadata_driven_for_mixed_dtypes():
    """The receiver reads only the broadcasts the sender's metadata announces —
    the item-count agreement that keeps a full-drain consumer from waiting on a
    NCCL read the trainer never performs."""
    state = {
        "norm.weight": torch.randn(64),  # fp32
        "proj.weight": torch.randn(32, 64).to(torch.bfloat16),  # bf16
        "proj.weight_scale_inv": torch.randn(1, 1),  # fp32
    }
    wire = FakeWire()
    _send_layer(wire, state)

    received = {name: tensor for name, tensor in receive_state_dict(wire.receiver)}
    assert set(received.keys()) == set(state.keys())
    for name, tensor in state.items():
        assert _bytes_equal(received[name], tensor), name
    wire.assert_drained()


def test_fp8_dtype_group_roundtrips_bit_exact():
    """A dtype group of fp8 e4m3 tensors concatenates, broadcasts, and splits
    back bit-exactly."""
    state = {
        "a.weight": torch.randn(16, 16).to(torch.float8_e4m3fn),
        "b.weight": torch.randn(8, 32).to(torch.float8_e4m3fn),
    }
    wire = FakeWire()
    _send_layer(wire, state)

    received = _receive_layer(wire)
    assert [(name, tensor.shape) for name, tensor in received] == [
        ("a.weight", (16, 16)),
        ("b.weight", (8, 32)),
    ]
    assert all(tensor.dtype == torch.float8_e4m3fn for _, tensor in received)
    assert _bytes_equal(received[0][1], state["a.weight"])
    assert _bytes_equal(received[1][1], state["b.weight"])
    wire.assert_drained()


def test_zero_element_tensors_roundtrip():
    state = {"empty.weight": torch.zeros(0, 8, dtype=torch.bfloat16), "norm.weight": torch.randn(8)}
    wire = FakeWire()
    _send_layer(wire, state)
    received = _receive_layer(wire)
    assert [(name, tuple(tensor.shape)) for name, tensor in received] == [
        ("empty.weight", (0, 8)),
        ("norm.weight", (8,)),
    ]
    wire.assert_drained()
