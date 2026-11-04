import asyncio
import os
import pickle
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Callable, Generator

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor
from torch.distributed import PrefixStore, TCPStore
from torch.distributed.tensor import DTensor, Replicate, Shard
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup

from prime_rl.configs.shared import NCCLWeightBroadcastConfig
from prime_rl.orchestrator.clients import Engine, EngineLost
from prime_rl.trainer.models import PreTrainedModelPrimeRL
from prime_rl.trainer.utils import get_world
from prime_rl.trainer.vlm import get_layer_prefix
from prime_rl.trainer.world import wait_for_master
from prime_rl.transports.weights.base import (
    ABORTED_MARKER,
    RECEIVED_MARKER,
    BroadcastAborted,
    WeightReceiver,
    WeightSender,
    write_marker,
)
from prime_rl.transports.weights.nccl_communicator import WeightBroadcastCommunicator
from prime_rl.utils.logger import get_logger
from prime_rl.utils.nccl import disable_nccl_p2p_if_unavailable, iter_tensor_buckets
from prime_rl.utils.weights import get_chunked_conversion_ops, resolve_wire_dtype


def _gather_row_chunk_to_master(value: Tensor, start: int, stop: int, dtype: torch.dtype) -> Tensor | None:
    """Assemble one fragment on trainer rank 0 by reducing over its replica's row owners.

    Each owner contributes its rows in an otherwise zero buffer, so the sum on rank 0 is the fragment.
    Requires trainer rank 0 at the origin of `value`'s device mesh, as laid out by `init_device_mesh`.
    """
    rank = dist.get_rank()
    if not isinstance(value, DTensor):
        return value[start:stop].to(dtype) if rank == 0 else None
    mesh = value.device_mesh
    active_dims = [
        i for i, placement in enumerate(value.placements) if mesh.size(i) > 1 and not isinstance(placement, Replicate)
    ]
    if len(active_dims) > 1 or (active_dims and value.placements[active_dims[0]] != Shard(0)):
        raise ValueError("Embedding transfer requires replicated weights or a single row-sharding dimension")
    if not active_dims:
        return value.to_local()[start:stop].to(dtype) if rank == 0 else None
    mesh_dim = active_dims[0]
    index = [0] * mesh.ndim
    index[mesh_dim] = slice(None)
    owners = mesh.mesh[tuple(index)].tolist()
    if rank not in owners:
        return None
    rows_per_rank = (value.shape[0] + len(owners) - 1) // len(owners)
    local = value.to_local()
    offset = min(rows_per_rank * owners.index(rank), value.shape[0])
    left, right = max(start, offset), min(stop, offset + local.shape[0])
    chunk = torch.zeros((stop - start, *value.shape[1:]), dtype=dtype, device=value.device)
    if left < right:
        chunk[left - start : right - start].copy_(local[left - offset : right - offset])
    dist.reduce(chunk, dst=0, group=mesh.get_group(mesh_dim))
    return chunk if rank == 0 else None


def get_max_layer_num(state_dict: dict[str, Tensor], layer_prefix: str = "model.layers.") -> int:
    """Get the maximum number of layers in the model."""
    max_num = -1
    for key in state_dict:
        if not key.startswith(layer_prefix):
            continue
        layer_num_str = key[len(layer_prefix) :].split(".")[0]
        if layer_num_str.isdigit():
            max_num = max(max_num, int(layer_num_str))
    return max_num + 1


def broadcast_integer(integer: int, communicator: PyNcclCommunicator | WeightBroadcastCommunicator) -> None:
    """Broadcast an integer to a process group using NCCL communicator."""
    integer_tensor = torch.tensor([integer], dtype=torch.long).cuda()
    communicator.broadcast(integer_tensor, src=0)


def broadcast_state_dict(
    state_dict: dict[str, Tensor], communicator: PyNcclCommunicator | WeightBroadcastCommunicator
) -> None:
    """Broadcast a state dict to the inference NCCL process group."""
    # Group tensors by dtype
    dtype_groups: dict[torch.dtype, list[tuple[str, Tensor]]] = {}
    for key, value in state_dict.items():
        assert not isinstance(value, DTensor), (
            "DTensor is not supported for broadcast, should have been converted to tensor already"
        )
        dtype = value.dtype
        if dtype not in dtype_groups:
            dtype_groups[dtype] = []
        dtype_groups[dtype].append((key, value))

    # Build metadata: for each dtype group, store keys and shapes
    metadata = {}
    for dtype, items in dtype_groups.items():
        metadata[dtype] = [(key, value.shape, value.numel()) for key, value in items]

    # Send metadata
    state = pickle.dumps(metadata)
    size_tensor = torch.tensor([len(state)], dtype=torch.long).cuda()
    communicator.broadcast(size_tensor, src=0)
    state_tensor = torch.ByteTensor(list(state)).cuda()
    communicator.broadcast(state_tensor, src=0)

    # Concatenate and broadcast tensors grouped by dtype
    for dtype, tensor_info in metadata.items():
        for bucket in iter_tensor_buckets(tensor_info, dtype):
            flat_tensors = [state_dict[key].flatten() for key, _, _ in bucket]
            concatenated = flat_tensors[0].contiguous() if len(flat_tensors) == 1 else torch.cat(flat_tensors)
            if concatenated.numel():
                communicator.broadcast(concatenated, src=0)
            del concatenated, flat_tensors


def filter_state_dict_by_layers(
    state_dict: dict[str, torch.Tensor], num_layers: int, layer_prefix: str
) -> Generator[tuple[int, dict[str, torch.Tensor]], None, None]:
    """Yield non-layer weights first, then each layer's weights.

    Yields (layer_idx, layer_state_dict) where layer_idx is -1 for the non-layer
    dict and the actual layer index (0, 1, ...) for layer dicts.
    """
    yield -1, {key: value for key, value in state_dict.items() if not key.startswith(layer_prefix)}

    for i in range(num_layers):
        yield (
            i,
            {key: value for key, value in state_dict.items() if key.startswith(f"{layer_prefix}{i}.")},
        )


def resolve_dtensors(
    state_dict: dict[str, Tensor],
    keep_in_fp32: Callable[[str], bool] | None,
    default_dtype: torch.dtype,
) -> dict[str, Tensor]:
    """Replace every sharded tensor with its full tensor, at the dtype it goes on the wire in.

    Only DTensors are touched, since only they need gathering. A buffer is never sharded, so it
    goes on the wire in whatever dtype it already holds and its fp32 declaration, if it has one,
    is never consulted. TODO: NIXL transport does not use this function; unify logic.
    """
    for key, value in list(state_dict.items()):
        if isinstance(value, DTensor):
            # only gather after the downcast as it will be faster
            target_dtype = resolve_wire_dtype(keep_in_fp32, key, default_dtype)
            state_dict[key] = value.to(target_dtype).full_tensor()
    return state_dict


def preprocess_layer_checkpoint(
    model: nn.Module,
    layer_state_dict: dict[str, Tensor],
    layer_idx: int,
) -> dict[str, Tensor]:
    if not layer_state_dict:
        return layer_state_dict
    if isinstance(model, PreTrainedModelPrimeRL) and model.is_prime_state_dict(layer_state_dict):
        model.convert_layer_to_hf(layer_state_dict, layer_idx)
        return layer_state_dict

    from transformers.core_model_loading import revert_weight_conversion

    return revert_weight_conversion(model, layer_state_dict)


class NCCLBroadcaster:
    def __init__(
        self,
        host: str,
        port: int,
        rank: int,
        world_size: int,
        device: int | str | torch.device,
        timeout: int,
        recoverable: bool = False,
        elastic: bool = False,
    ):
        self.logger = get_logger()
        self.world = get_world()
        self.dtype = torch.bfloat16
        self.device = device
        self.timeout = timeout
        self.elastic = elastic
        self.session: str | None = None
        """The elastic transfer group the communicator belongs to."""
        self.communicator: PyNcclCommunicator | WeightBroadcastCommunicator | None = None

        if not self.world.is_master:
            self.logger.debug("Initialized NCCL broadcast on non-master rank (no communicator)")
            return
        disable_nccl_p2p_if_unavailable()
        if elastic:
            # Inference workers dial this store for every group the consumer forms; each group's keys
            # live under its session, so a worker never reads a previous group's NCCL ID.
            self.store = TCPStore(
                host_name=host,
                port=port,
                is_master=True,
                timeout=timedelta(seconds=timeout),
                wait_for_workers=False,
                use_libuv=False,
            )
            self.logger.debug("Initialized the elastic NCCL broadcast store on master rank")
            return
        # Trainer is on rank 0 in process group with all inference GPUs
        pg = StatelessProcessGroup.create(host=host, port=port, rank=rank, world_size=world_size, store_timeout=timeout)
        self.communicator = (
            WeightBroadcastCommunicator(pg, device=device, timeout=timeout)
            if recoverable
            else PyNcclCommunicator(pg, device=device)
        )
        self.logger.debug("Initialized NCCL broadcast on master rank")

    def join_group(self, session: str, world_size: int, cancelled: Callable[[], bool]) -> None:
        """Form the elastic group ``session`` of the trainer and ``world_size - 1`` inference workers,
        unless the communicator already belongs to it."""
        if self.session == session and self.communicator is not None and self.communicator.comm:
            return
        if self.communicator is not None:
            self.communicator.close()
        self.communicator = None
        self.session = None
        self.logger.info(f"Forming NCCL transfer group {session} with {world_size - 1} inference workers")
        pg = StatelessProcessGroup(rank=0, world_size=world_size, store=PrefixStore(f"{session}/", self.store))
        self.communicator = WeightBroadcastCommunicator(
            pg, device=self.device, timeout=self.timeout, cancelled=cancelled
        )
        self.session = session

    def join(self, ack: dict, cancelled: Callable[[], bool]) -> bool:
        """Form the elastic group ``ack`` names (master only), giving up once ``cancelled``. Returns
        whether the group formed."""
        try:
            self.join_group(ack["session"], ack["world_size"], cancelled)
        except Exception as error:
            self.logger.warning(f"Stopped sending weights to inference: {error!r}")
            return False
        self.communicator.cancelled = cancelled
        return True

    @torch.no_grad()
    def send(self, model: nn.Module, joined: bool = True) -> bool:
        """Broadcast the state dict of a model into the inference pool using NCCL. With an elastic
        group (formed by ``join`` beforehand, ``joined``), the master stops sending once a send fails
        (e.g. an inference server was lost); the other ranks' gathers go on, and ``send`` returns False."""
        state_dict = model.state_dict()
        layer_prefix = get_layer_prefix(model)
        num_layers = get_max_layer_num(state_dict, layer_prefix)
        chunked_ops = get_chunked_conversion_ops(model, state_dict)
        num_state_dict_to_send = num_layers + 1 + sum(op.shard_count for op in chunked_ops.values())
        stopped = not joined

        def master_sends(send: Callable[[], None]) -> None:
            nonlocal stopped
            if not self.elastic:
                send()
                return
            if stopped:
                return
            try:
                send()
            except Exception as error:
                stopped = True
                self.logger.warning(f"Stopped sending weights to inference: {error!r}")

        if self.world.is_master:
            master_sends(lambda: broadcast_integer(num_state_dict_to_send, self.communicator))

        self.logger.debug(f"Broadcasting {num_state_dict_to_send} layer state dicts")
        keep_in_fp32 = getattr(model, "keep_in_fp32_for_weight_transfer", None)
        for layer_id, layer_state_dict in filter_state_dict_by_layers(state_dict, num_layers, layer_prefix):
            chunked_weights = {key: layer_state_dict.pop(key) for key in chunked_ops if key in layer_state_dict}
            layer_state_dict = resolve_dtensors(layer_state_dict, keep_in_fp32, self.dtype)
            layer_state_dict = preprocess_layer_checkpoint(model, layer_state_dict, layer_id)
            if self.world.is_master:
                master_sends(partial(broadcast_state_dict, layer_state_dict, self.communicator))
            del layer_state_dict
            for key, value in chunked_weights.items():
                op = chunked_ops[key]
                dtype = resolve_wire_dtype(keep_in_fp32, key, self.dtype) if isinstance(value, DTensor) else value.dtype
                resolve_rows = partial(_gather_row_chunk_to_master, dtype=dtype)
                for packet in op.iter_prime_to_hf(value, resolve_rows):
                    if self.world.is_master:
                        assert packet is not None
                        master_sends(partial(broadcast_state_dict, packet, self.communicator))
                    del packet
        if self.elastic and self.communicator is not None:
            self.communicator.cancelled = None
        return not stopped


class NCCLWeightSender(WeightSender):
    """Broadcast weights into the inference engine using NCCL."""

    def __init__(
        self,
        output_dir: Path,
        config: NCCLWeightBroadcastConfig,
        device: int | str | torch.device,
    ):
        super().__init__(output_dir, config.timeout)
        self.nccl_broadcast_sender = NCCLBroadcaster(
            config.host,
            config.port,
            0,
            config.inference_world_size + 1,
            device,
            config.timeout,
            config.recoverable,
            config.elastic,
        )
        self.confirms_delivery = config.elastic
        self.joins = 0

    @torch.no_grad()
    def _broadcast(self, model: nn.Module, step: int, step_dir: Path) -> bool:
        if not self.confirms_delivery:
            return self.nccl_broadcast_sender.send(model)
        joined = True
        if self.world.is_master:
            joined = self.nccl_broadcast_sender.join(self.ack, partial(os.path.exists, step_dir / ABORTED_MARKER))
        # Forming the group can outlast the trainer's collective timeout, so the other ranks wait for
        # it on the store, not in the gathers of ``send``, and longer than the group's own deadline.
        self.joins += 1
        wait_for_master(f"weight_broadcast/join/{self.joins}", 2 * self.timeout)
        return self.nccl_broadcast_sender.send(model, joined)


class NoServerLeft(Exception):
    """Every server taking part in an update was lost before its transfer group formed."""


class NCCLWeightReceiver(WeightReceiver):
    """Joins the trainer's NCCL collective. The receiver pauses the engines,
    acknowledges, and sends them into the receive RPC — only then does the
    trainer enter the collective, so the handshake can never race a stale
    marker."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.group: list[tuple[Engine, int]] | None = None
        """The servers of the current elastic transfer group, in rank order, each with its loss count when
        the group formed; None until one forms."""
        self.generation = 0

    async def initialize(self) -> None:
        if self.config.elastic:
            # Elastic groups form per update, around the servers taking part.
            return
        await self.admin_plane.initialize_nccl(
            host=self.config.host,
            port=self.config.port,
            timeout=self.config.timeout,
            inference_world_size=self.config.inference_world_size,
            recoverable=self.config.recoverable,
        )

    # The trainer sends its group's NCCL ID once: joining again would abort the communicators of
    # workers that already joined.
    reconnect_retryable = False

    async def reconnect(self) -> None:
        if self.config.elastic:
            # The replacement trainer joins the group the next update forms.
            self.group = None
            return
        # Each worker aborts its communicator with the replaced trainer and joins a fresh one.
        await self.initialize()

    async def receive(self, step: int) -> None:
        if not self.config.elastic:
            await self.admin_plane.update_weights(
                self.step_dir(step),
                transport="nccl",
                step=step,
                on_paused=lambda: self._ack(step),
                abort_path=self.step_dir(step) / ABORTED_MARKER,
            )
            return
        while True:
            attempt = self.offered_attempt(step)
            try:
                await self._receive_elastic(step, attempt)
                return
            except EngineLost as error:
                # The other servers stopped with part of v{step}; the trainer offers it again.
                self.group = None
                self.logger.warning(f"{error}; receiving policy v{step} again without them")
                await self.wait_for_new_offer(step, attempt)

    async def _receive_elastic(self, step: int, attempt: str) -> None:
        """Pause the servers taking part, acknowledge with the group they form (re-formed if the
        servers changed), receive, and confirm delivery to the trainer."""
        step_dir = self.step_dir(step)
        abort_path = step_dir / ABORTED_MARKER
        gpus_per_server = self.config.inference_world_size // len(self.admin_plane.engines)

        async def form_group(engines: list[Engine]) -> None:
            if not engines:
                # Not acknowledged yet: a group of the trainer alone would confirm a version no
                # server holds.
                raise NoServerLeft
            # A server lost since the group formed runs no communicator of it, even once it is back.
            group = [(engine, engine.losses) for engine in engines]
            reform = group != self.group
            if reform:
                self.generation += 1
                self.group = None
            session = f"g{self.generation}"
            self._ack(step, attempt=attempt, session=session, world_size=1 + len(engines) * gpus_per_server)
            if reform:
                await self.admin_plane.form_nccl_group(
                    engines,
                    host=self.config.host,
                    port=self.config.port,
                    timeout=self.config.timeout,
                    gpus_per_server=gpus_per_server,
                    session_id=session,
                    abort_path=abort_path,
                )
                self.group = group

        while True:
            while not self.admin_plane.members():
                if self.aborted.is_set():
                    raise BroadcastAborted(f"Stopped waiting for inference servers to receive v{step}")
                await asyncio.sleep(1.0)
            try:
                await self.admin_plane.update_weights(
                    step_dir,
                    transport="nccl",
                    step=step,
                    abort_path=abort_path,
                    before_transfer=form_group,
                    all_or_nothing=True,
                )
            except NoServerLeft:
                # Every server was lost while pausing: receive the offer once one is back.
                continue
            break
        write_marker(step_dir / RECEIVED_MARKER, attempt)
