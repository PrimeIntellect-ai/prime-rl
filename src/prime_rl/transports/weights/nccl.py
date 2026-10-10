import asyncio
import json
import pickle
import threading
import time
from pathlib import Path
from typing import Callable, Generator

import torch
import torch.distributed as dist
import torch.nn as nn
from httpx import AsyncClient
from torch import Tensor
from torch.distributed.tensor import DTensor
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup

from prime_rl.configs.shared import NCCLWeightBroadcastConfig
from prime_rl.orchestrator.clients import ADMIN_TIMEOUT_S, _admin_post
from prime_rl.trainer.models import PreTrainedModelPrimeRL
from prime_rl.trainer.utils import get_world
from prime_rl.transports.weights.base import FINISHED_MARKER, WeightReceiver, WeightSender
from prime_rl.utils.logger import get_logger
from prime_rl.utils.nccl import disable_nccl_p2p_if_unavailable, iter_tensor_buckets
from prime_rl.utils.pathing import get_broadcast_dir
from prime_rl.utils.vlm import get_layer_prefix
from prime_rl.utils.weights import resolve_wire_dtype


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


def broadcast_integer(integer: int, communicator: PyNcclCommunicator) -> None:
    """Broadcast an integer to a process group using NCCL communicator."""
    integer_tensor = torch.tensor([integer], dtype=torch.long).cuda()
    communicator.broadcast(integer_tensor, src=0)


def broadcast_state_dict(state_dict: dict[str, Tensor], communicator: PyNcclCommunicator) -> None:
    """Broadcast a state dict to NCCL process group using the PyNcclCommunicator."""
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
    ):
        self.logger = get_logger()
        self.world = get_world()
        self.dtype = torch.bfloat16
        self.port = port

        if self.world.is_master:
            disable_nccl_p2p_if_unavailable()
            # Trainer is on rank 0 in process group with all inference GPUs
            pg = StatelessProcessGroup.create(
                host=host, port=port, rank=rank, world_size=world_size, store_timeout=timeout
            )
            self.communicator = PyNcclCommunicator(pg, device=device)
            self.logger.debug("Initialized NCCL broadcast on master rank")
        else:
            self.logger.debug("Initialized NCCL broadcast on non-master rank (no communicator)")

    @torch.no_grad()
    def send(self, model: nn.Module, abort_marker: Path | None = None) -> bool:
        """Broadcast the state dict of a model into the inference pool using NCCL. On the master,
        a thread aborts the communicator once ``abort_marker`` appears (an engine died during
        the transfer); later broadcasts are then no-ops, but every rank still walks all layers,
        since the DTensor gathers are trainer-wide collectives. Returns whether it aborted."""
        done = threading.Event()
        watchdog = None
        if self.world.is_master and abort_marker is not None:
            watchdog = threading.Thread(target=self._abort_on, args=(abort_marker, done), daemon=True)
            watchdog.start()
        try:
            self._send(model)
            # The broadcasts run asynchronously: wait for them while the watchdog can still abort
            torch.cuda.current_stream().synchronize()
        finally:
            done.set()
            if watchdog is not None:
                watchdog.join()
        return self.world.is_master and self.communicator.disabled

    def _abort_on(self, marker: Path, done: threading.Event) -> None:
        while not done.wait(0.5):
            if marker.exists():
                self.logger.warning("Aborting the NCCL weight broadcast: an inference engine died")
                self.communicator.destroy()
                return

    def _send(self, model: nn.Module) -> None:
        state_dict = model.state_dict()
        layer_prefix = get_layer_prefix(model.config)
        num_layers = get_max_layer_num(state_dict, layer_prefix)
        num_state_dict_to_send = num_layers + 1  # we send all layer plus the remaining weights

        if self.world.is_master:
            broadcast_integer(num_state_dict_to_send, self.communicator)

        self.logger.debug(f"Broadcasting {num_state_dict_to_send} layer state dicts")
        keep_in_fp32 = getattr(model, "keep_in_fp32_for_weight_transfer", None)
        for layer_id, layer_state_dict in filter_state_dict_by_layers(state_dict, num_layers, layer_prefix):
            layer_state_dict = resolve_dtensors(layer_state_dict, keep_in_fp32, self.dtype)
            layer_state_dict = preprocess_layer_checkpoint(model, layer_state_dict, layer_id)
            if self.world.is_master:
                broadcast_state_dict(layer_state_dict, self.communicator)


def nccl_group_file(broadcast_dir: Path) -> Path:
    """Where the orchestrator publishes a new NCCL group after evicting or re-adding an engine."""
    return broadcast_dir / "nccl_group.json"


class NCCLWeightSender(WeightSender):
    """Broadcast weights into the inference engine using NCCL."""

    def __init__(
        self,
        output_dir: Path,
        config: NCCLWeightBroadcastConfig,
        device: int | str | torch.device,
    ):
        super().__init__(output_dir, config.timeout)
        self.config = config
        self.device = device
        self.generation = 0
        self.nccl_broadcast_sender = NCCLBroadcaster(
            config.host,
            config.port,
            0,
            config.inference_world_size + 1,
            device,
            config.timeout,
        )

    @torch.no_grad()
    def _broadcast(self, model: nn.Module, step: int, step_dir: Path) -> None:
        # The master enters only after the receiver acknowledged the handshake,
        # but all ranks must be held back until then: the broadcast preparation
        # (DTensor resolution, checkpoint conversion) enqueues collectives on non-master
        # ranks, and if those start before the receiver has paused inference,
        # the collectives sit unmatched until NCCL's watchdog kills the process.
        while True:
            if self.world.world_size > 1:
                dist.barrier()
            abort_marker = step_dir / f".abort_{self.nccl_broadcast_sender.port}" if self.world.is_master else None
            aborted = torch.tensor([self.nccl_broadcast_sender.send(model, abort_marker)], device="cuda")
            if self.world.world_size > 1:
                dist.broadcast(aborted, src=0)
            if not aborted.item():
                return
            if self.world.is_master:
                # Tell the orchestrator, then join the group it rebuilds from the survivors
                abort_marker.with_name(f".aborted_{self.nccl_broadcast_sender.port}").touch()
                deadline = time.monotonic() + self.timeout
                while not self._maybe_rebuild():
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"No NCCL group to resend v{step} to within {self.timeout}s")
                    time.sleep(0.1)
            self.logger.info(f"Resending policy v{step} to the remaining inference engines")

    def _wait_hook(self) -> None:
        self._maybe_rebuild()

    def _maybe_rebuild(self) -> bool:
        """Join the group the orchestrator published after evicting or re-adding an engine. It
        writes the group before acknowledging the version and then initializes the engines, so
        this runs while waiting for the receiver. Returns whether it rebuilt."""
        group_file = nccl_group_file(get_broadcast_dir(self.output_dir))
        if not group_file.exists():
            return False
        group = json.loads(group_file.read_text())
        if group["generation"] <= self.generation:
            return False
        self.logger.info(
            f"Rebuilding the NCCL weight broadcast group (generation {group['generation']}, "
            f"{group['inference_world_size']} inference ranks)"
        )
        self.nccl_broadcast_sender.communicator.destroy()
        self.nccl_broadcast_sender = NCCLBroadcaster(
            self.config.host,
            group["port"],
            0,
            group["inference_world_size"] + 1,
            self.device,
            self.config.timeout,
        )
        self.generation = group["generation"]
        return True


class NCCLWeightReceiver(WeightReceiver):
    """Joins the trainer's NCCL collective. The receiver pauses the engines,
    acknowledges, and sends them into the receive RPC — only then does the
    trainer enter the collective, so the handshake can never race a stale
    marker."""

    async def initialize(self) -> None:
        # A group left by a previous run must not steer the new trainer off generation 0
        nccl_group_file(self.broadcast_dir).unlink(missing_ok=True)
        await self.admin_plane.initialize_nccl(
            host=self.config.host,
            port=self.config.port,
            timeout=self.config.timeout,
            inference_world_size=self.config.inference_world_size,
        )
        self.gpus_per_server = self.config.inference_world_size // len(self.admin_plane.clients)
        self.port = self.config.port

    async def receive(self, step: int) -> None:
        await self.admin_plane.update_weights(
            self.step_dir(step),
            transport="nccl",
            step=step,
            on_paused=lambda: self._ack(step),
            on_membership_change=self.on_membership_change,
            abort_marker=lambda: self.step_dir(step) / f".abort_{self.port}",
            resend=lambda: self._resend(step),
        )

    async def _resend(self, step: int) -> bool:
        """After an abort, wait for the trainer's verdict: it either aborted too and will resend
        once the group is rebuilt, or it had already finished the transfer."""
        aborted = self.step_dir(step) / f".aborted_{self.port}"
        finished = self.step_dir(step) / FINISHED_MARKER
        while not (aborted.exists() or finished.exists()):
            await asyncio.sleep(0.1)
        return aborted.exists()

    async def on_membership_change(self, members: list[AsyncClient], generation: int) -> None:
        """Rebuild the communicator over ``members`` on a fresh port: publish the group for the
        trainer, which joins it while it waits for this version's acknowledgement (or for a
        resend after an abort), and initialize the engines' side."""
        self.port = self.config.port + generation
        inference_world_size = self.gpus_per_server * len(members)
        self.logger.info(
            f"Rebuilding the NCCL weight broadcast group (generation {generation}): {len(members)} engines"
        )
        group_file = nccl_group_file(self.broadcast_dir)
        tmp = group_file.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"generation": generation, "port": self.port, "inference_world_size": inference_world_size})
        )
        tmp.replace(group_file)
        # Not evicting: an engine missing from the rendezvous would hang it, so a failure is fatal
        await asyncio.gather(
            *(
                _admin_post(
                    client,
                    "/init_broadcaster",
                    timeout_s=max(ADMIN_TIMEOUT_S, self.config.timeout),
                    json={
                        "host": self.config.host,
                        "port": self.port,
                        "rank_offset": index * self.gpus_per_server,
                        "inference_world_size": inference_world_size,
                        "timeout": self.config.timeout,
                    },
                )
                for index, client in enumerate(members)
            )
        )
