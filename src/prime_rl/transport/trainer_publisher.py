"""Trainer-side per-rank publisher: build slots, register them with NIXL,
publish the agent metadata + tensor descriptors through Model Express.

Setup-time only — does not perform any transfer. After construction the
publisher exposes its slots, NIXL agent, and rendezvous handle so the
transport layer can build the write table and post RDMA WRITEs against
inference peer descriptors.
"""

from __future__ import annotations

from typing import Callable

import msgspec
import torch
from modelexpress import p2p_pb2
from modelexpress.client import MxClient
from torch import Tensor

from prime_rl.trainer.models.conversion_spec import ConversionSpec
from prime_rl.trainer.models.slots import Slot, build_slots
from prime_rl.trainer.parallel_dims import ParallelDims
from prime_rl.transport.classic_cuda_pool import classic_cuda_alloc
from prime_rl.transport.mx_rendezvous import MxRendezvous
from prime_rl.transport.nixl_agent import NixlAgentWrapper, make_agent_name
from prime_rl.transport.wire import RendezvousPayload


class TrainerPublisher:
    """One trainer rank's slots + NIXL agent + MX rendezvous handle.

    Construct with the rank's view of the trainer state dict and the
    inference target's resolved conversion settings; the publisher builds
    slot buffers via :func:`build_slots` (wrapped in the classic-cudaMalloc
    pool), pins their memory with NIXL, and is ready to call :meth:`publish`.
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
        state_dict: dict[str, Tensor],
        parallel_dims: ParallelDims,
    ) -> None:
        with classic_cuda_alloc():
            self.slots: list[Slot] = build_slots(
                state_dict,
                layer_specs_fn=layer_specs_fn,
                non_layer_specs=non_layer_specs,
                is_dense_fn=is_dense_fn,
                num_layers=num_layers,
                parallel_dims=parallel_dims,
                default_conversion=default_conversion,
                base_dtype=base_dtype,
            )

        self.agent = NixlAgentWrapper(name=make_agent_name("trainer", rank))
        for slot in self.slots:
            for _, tensor, _ in slot.buffers:
                self.agent.register_tensor(tensor)

        self.rendezvous = MxRendezvous(
            client=client,
            role="trainer",
            rank=rank,
            peer_world_size=peer_world_size,
            model_name=inference_model_name,
            expert_parallel_size=parallel_dims.ep,
            quantization="fp8" if default_conversion == "fp8_128x128" else "",
        )

    def publish(self) -> str:
        """Push NIXL agent metadata + slot tensor descriptors + the layout
        manifest through MX. The layout rides inside the
        :class:`RendezvousPayload` packed into ``WorkerMetadata.nixl_metadata``
        so inference can narrow + chunk its destinations after one
        ``GetMetadata`` call. Returns the assigned ``mx_source_id``.
        """
        descriptors: list[p2p_pb2.TensorDescriptor] = []
        for slot in self.slots:
            for buf_key, tensor, _ in slot.buffers:
                descriptors.append(self.agent.make_tensor_descriptor(buf_key, tensor))

        layout = []
        for slot in self.slots:
            layout.extend(slot.layout_payload())

        payload = RendezvousPayload(
            agent_metadata=self.agent.get_metadata(),
            agent_name=self.agent.name,
            layout=layout,
        )
        return self.rendezvous.publish(
            nixl_metadata=msgspec.msgpack.encode(payload),
            tensors=descriptors,
        )
