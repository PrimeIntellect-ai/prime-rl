"""Trainer-side per-rank publisher: allocate slots, register them with NIXL,
publish the agent metadata + tensor descriptors through Model Express.

Setup-time only — does not perform any transfer. After construction the
publisher exposes its slots, NIXL agent, and rendezvous handle so the
transport layer (next step) can build the write table and post RDMA
WRITEs against inference peer descriptors.
"""

from __future__ import annotations

from typing import Callable, Mapping

import torch
from modelexpress import p2p_pb2
from modelexpress.client import MxClient

from prime_rl.trainer.models.conversion_spec import ConversionSpec
from prime_rl.trainer.models.slots import Shape, Slot, allocate_slots
from prime_rl.transport.mx_rendezvous import MxRendezvous
from prime_rl.transport.nixl_agent import NixlAgentWrapper, make_agent_name


class TrainerPublisher:
    """One trainer rank's slot table + NIXL agent + MX rendezvous handle.

    Construct with the rank's view of the trainer state's *shapes* and the
    inference target's resolved conversion settings; the publisher allocates
    slot buffers, pins them with NIXL, and is ready to call :meth:`publish`.

    The publisher does not retain references to the trainer's live state
    dict — only shapes are needed up front. The actual source tensors are
    passed in later (transport step) when materializing slots before posting
    RDMA WRITEs.
    """

    def __init__(
        self,
        *,
        client: MxClient,
        rank: int,
        peer_world_size: int,
        inference_model_name: str,
        default_conversion: str,
        base_dtype: torch.dtype,
        layer_specs_fn: Callable[[int, bool], tuple[ConversionSpec, ...]],
        non_layer_specs: tuple[ConversionSpec, ...],
        is_dense_fn: Callable[[int], bool],
        num_layers: int,
        state_shapes: Mapping[str, Shape],
        expert_parallel_size: int = 0,
    ) -> None:
        self.slots: list[Slot] = allocate_slots(
            state_shapes,
            layer_specs_fn=layer_specs_fn,
            non_layer_specs=non_layer_specs,
            is_dense_fn=is_dense_fn,
            num_layers=num_layers,
            default_conversion=default_conversion,
            base_dtype=base_dtype,
        )

        self.agent = NixlAgentWrapper(name=make_agent_name("trainer", rank))
        for slot in self.slots:
            self.agent.register_tensor(slot.weight)
            if slot.scale is not None:
                self.agent.register_tensor(slot.scale)

        self.rendezvous = MxRendezvous(
            client=client,
            role="trainer",
            rank=rank,
            peer_world_size=peer_world_size,
            model_name=inference_model_name,
            expert_parallel_size=expert_parallel_size,
            quantization="fp8" if default_conversion == "fp8_128x128" else "",
        )

    def _tensor_descriptors(self) -> list[p2p_pb2.TensorDescriptor]:
        descs: list[p2p_pb2.TensorDescriptor] = []
        for slot in self.slots:
            descs.append(self.agent.make_tensor_descriptor(slot.full_name, slot.weight))
            if slot.scale is not None:
                assert slot.scale_name is not None  # paired with the scale buffer
                descs.append(self.agent.make_tensor_descriptor(slot.scale_name, slot.scale))
        return descs

    def publish(self) -> str:
        """Push NIXL agent metadata + slot tensor descriptors through MX. Returns the assigned ``mx_source_id``."""
        return self.rendezvous.publish(
            nixl_metadata=self.agent.get_metadata(),
            tensors=self._tensor_descriptors(),
        )
