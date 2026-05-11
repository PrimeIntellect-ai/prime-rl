import ctypes
import pickle
from dataclasses import dataclass
from typing import Generator, cast

import torch
from sglang.srt.distributed.device_communicators.pynccl import PyNcclCommunicator
from sglang.srt.distributed.device_communicators.pynccl_wrapper import (
    NCCLLibrary,
    ncclComm_t,
    ncclUniqueId,
)
from sglang.srt.distributed.utils import StatelessProcessGroup
from sglang.srt.managers.io_struct import BaseReq, UpdateWeightFromDiskReqOutput
from torch import Tensor


@dataclass
class PrimeRLInitBroadcasterReqInput(BaseReq):
    host: str
    port: int
    rank_offset: int
    inference_world_size: int
    timeout: int


@dataclass
class PrimeRLUpdateWeightsReqInput(BaseReq):
    flush_cache: bool = True


def _unique_id_from_bytes(data: bytes) -> ncclUniqueId:
    if len(data) != ctypes.sizeof(ncclUniqueId):
        raise ValueError(f"Expected NCCL unique id with {ctypes.sizeof(ncclUniqueId)} bytes, got {len(data)}")

    unique_id = ncclUniqueId()
    ctypes.memmove(ctypes.addressof(unique_id), data, len(data))
    return unique_id


def _unique_id_to_bytes(unique_id: ncclUniqueId) -> bytes:
    return ctypes.string_at(ctypes.addressof(unique_id), ctypes.sizeof(unique_id))


class PrimeRLNeutralPyNcclCommunicator(PyNcclCommunicator):
    """SGLang PyNccl communicator that accepts a plain-byte NCCL unique id."""

    def __init__(
        self,
        group: StatelessProcessGroup,
        device: int | str | torch.device,
        library_path: str | None = None,
        use_current_stream: bool = False,
    ):
        self.rank = group.rank
        self.world_size = group.world_size
        self.group = group

        if self.world_size == 1:
            self.available = False
            self.disabled = True
            self.stream = None
            return

        self.nccl = NCCLLibrary(library_path)
        self.available = True
        self.disabled = False
        self.use_current_stream = use_current_stream
        self.nccl_version = self.nccl.ncclGetRawVersion()

        if self.rank == 0:
            unique_id = self.nccl.ncclGetUniqueId()
            self.unique_id = group.broadcast_obj(_unique_id_to_bytes(unique_id), src=0)
        else:
            self.unique_id = group.broadcast_obj(None, src=0)
        self.unique_id = _unique_id_from_bytes(self.unique_id)

        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device

        with torch.cuda.device(device):
            self.comm: ncclComm_t = self.nccl.ncclCommInitRank(self.world_size, self.unique_id, self.rank)
            self.stream = torch.cuda.Stream()

        self.disabled = True


def _receive_integer(communicator: PyNcclCommunicator) -> int:
    integer_tensor = torch.tensor([10], dtype=torch.long, device=communicator.device)
    communicator.broadcast(integer_tensor, src=0)
    communicator.stream.synchronize()
    return cast(int, integer_tensor.item())


def _receive_state_dict(communicator: PyNcclCommunicator) -> Generator[tuple[str, Tensor], None, None]:
    size_tensor = torch.tensor([10], dtype=torch.long, device=communicator.device)
    communicator.broadcast(size_tensor, src=0)
    communicator.stream.synchronize()

    state_tensor = torch.empty(cast(int, size_tensor.item()), dtype=torch.uint8, device=communicator.device)
    communicator.broadcast(state_tensor, src=0)
    communicator.stream.synchronize()

    metadata = pickle.loads(bytes(state_tensor.cpu().numpy()))
    for dtype, tensor_info_list in metadata.items():
        total_elements = sum(numel for _, _, numel in tensor_info_list)
        concatenated = torch.empty(total_elements, dtype=dtype, device=communicator.device)
        communicator.broadcast(concatenated, src=0)
        communicator.stream.synchronize()

        offset = 0
        for key, shape, numel in tensor_info_list:
            tensor = concatenated[offset : offset + numel].view(shape).clone()
            offset += numel
            try:
                yield key, tensor
            finally:
                del tensor

        del concatenated


class PrimeRLNCCLWeightBroadcastReceiver:
    def __init__(
        self,
        host: str,
        port: int,
        rank: int,
        world_size: int,
        device: int | str | torch.device,
    ):
        pg = StatelessProcessGroup.create(host=host, port=port, rank=rank, world_size=world_size)
        self.communicator = PrimeRLNeutralPyNcclCommunicator(pg, device=device)

    @torch.no_grad()
    def receive_state_dict(self) -> Generator[tuple[str, Tensor], None, None]:
        with self.communicator.change_state(enable=True):
            num_state_dicts = _receive_integer(self.communicator)
            for _ in range(num_state_dicts):
                yield from _receive_state_dict(self.communicator)


def init_broadcaster(scheduler, recv_req: PrimeRLInitBroadcasterReqInput):
    model_runner = scheduler.tp_worker.model_runner
    local_rank = scheduler.tp_worker.tp_rank
    global_rank = recv_req.rank_offset + local_rank + 1
    world_size = recv_req.inference_world_size + 1
    device = torch.device(f"cuda:{getattr(model_runner, 'gpu_id', 0)}")

    scheduler.prime_rl_nccl_receiver = PrimeRLNCCLWeightBroadcastReceiver(
        host=recv_req.host,
        port=recv_req.port,
        rank=global_rank,
        world_size=world_size,
        device=device,
    )
    return UpdateWeightFromDiskReqOutput(True, "Succeeded to initialize prime-rl NCCL receiver.", 0)


def update_weights(scheduler, recv_req: PrimeRLUpdateWeightsReqInput):
    receiver = getattr(scheduler, "prime_rl_nccl_receiver", None)
    if receiver is None:
        return UpdateWeightFromDiskReqOutput(False, "prime-rl NCCL receiver is not initialized.", 0)

    try:
        scheduler.tp_worker.model_runner.model.load_weights(receiver.receive_state_dict())
        if recv_req.flush_cache:
            flush_cache_success = scheduler.flush_cache()
            assert flush_cache_success, "Cache flush failed after updating weights"
        return UpdateWeightFromDiskReqOutput(True, "Succeeded to update weights from prime-rl NCCL broadcast.", 0)
    except Exception as e:
        return UpdateWeightFromDiskReqOutput(False, f"Failed to update weights from prime-rl NCCL broadcast: {e}", 0)
