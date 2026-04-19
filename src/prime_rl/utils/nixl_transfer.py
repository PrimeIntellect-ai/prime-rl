"""Shared helpers for NIXL-based weight transfer.

Used by both the trainer-side sender (``prime_rl.trainer.rl.broadcast.nixl``)
and the inference-side receiver (``prime_rl.inference.vllm.worker.nixl``).
Keeps NIXL-specific code here so neither side depends on the other.

``nixl_cu13`` is imported lazily inside :class:`NixlAgentWrapper` so that
modules that simply reference this file can still be imported on machines
without NIXL installed (CI, CPU-only developer laptops).
"""

from __future__ import annotations

import functools
import os
import socket
import subprocess
from dataclasses import dataclass
from typing import Any, Sequence

from torch import Tensor


@functools.lru_cache(maxsize=1)
def _nvidia_smi_topo() -> str:
    return subprocess.check_output(["nvidia-smi", "topo", "-m"]).decode("utf-8")


@functools.lru_cache(maxsize=8)
def map_gpu_to_nic(gpu: int) -> str:
    """Resolve the PIX-attached IB NIC for a GPU from ``nvidia-smi topo -m``.

    Returns a string suitable for ``UCX_NET_DEVICES``, e.g. ``mlx5_0:1``.
    """
    topo = _nvidia_smi_topo()

    legend = topo.split("NIC Legend:")[1].strip()
    legend_lines = [line.strip().split(":") for line in legend.split("\n") if line.strip()]
    legend_map = {kv[0].strip(): kv[1].strip() for kv in legend_lines}

    header = topo.split("\n")[0]
    header_nics = {i: x for i, x in enumerate(header.split("\t")) if x.startswith("NIC")}

    gpu_lines = [x for x in topo.split("\n") if x.startswith(f"GPU{gpu}")]
    assert len(gpu_lines) == 1, f"GPU{gpu} mapping not found"
    cols = gpu_lines[0].split("\t")
    pix_cols = [i for i, x in enumerate(cols) if x.strip() == "PIX"]
    if not pix_cols:
        raise RuntimeError(f"no PIX-attached NIC for GPU{gpu}")
    return f"{legend_map[header_nics[pix_cols[0]]]}:1"


def pin_ucx_rail(local_rank: int) -> None:
    """Set per-rank UCX env *before* the nixl agent is created.

    Uses ``setdefault`` so anything in ``.env`` takes precedence.
    """
    try:
        nic = map_gpu_to_nic(local_rank)
    except Exception:
        # Fallback: let UCX auto-discover if topology probing fails (e.g. in CI).
        nic = None
    if nic is not None:
        os.environ.setdefault("UCX_NET_DEVICES", nic)
    os.environ.setdefault("UCX_TLS", "rc_mlx5,ud,cuda_copy,cuda_ipc")
    os.environ.setdefault("UCX_IB_GPU_DIRECT_RDMA", "y")
    os.environ.setdefault("UCX_RNDV_SCHEME", "put_zcopy")
    os.environ.setdefault("UCX_RNDV_THRESH", "8192")
    os.environ.setdefault("UCX_MEMTYPE_CACHE", "y")
    os.environ.setdefault("UCX_WARN_UNUSED_ENV_VARS", "n")


class NixlAgentWrapper:
    """Thin wrapper around ``nixl_cu13._api.nixl_agent`` with the argument shapes
    prime-rl needs. Mirrors the pattern in the rdma-playground script
    ``nixl/put_bw_nxn_konig.py``.
    """

    def __init__(
        self,
        name: str,
        local_rank: int,
        backends: Sequence[str] = ("UCX",),
    ) -> None:
        from nixl_cu13._api import nixl_agent, nixl_agent_config  # type: ignore

        pin_ucx_rail(local_rank)
        self.name = name
        self.backends: list[str] = list(backends)
        self._agent = nixl_agent(name, nixl_agent_config(backends=self.backends))

    def register_tensor(self, tensor: Tensor):
        """Pin a tensor's device memory for RDMA and return its xfer descriptor."""
        self._agent.register_memory(tensor)
        return self._agent.get_xfer_descs(tensor)

    def chunked_descs(self, tensor: Tensor, num_chunks: int):
        """Split a registered tensor into ``num_chunks`` equal-size xfer descriptors."""
        nbytes = tensor.numel() * tensor.element_size()
        if nbytes % num_chunks != 0:
            raise ValueError(f"tensor nbytes {nbytes} not divisible by num_chunks {num_chunks}")
        chunk = nbytes // num_chunks
        base = tensor.data_ptr()
        dev = tensor.get_device()
        tuples = [(base + i * chunk, chunk, dev) for i in range(num_chunks)]
        return self._agent.get_xfer_descs(tuples, mem_type="cuda")

    def get_metadata(self) -> bytes:
        return self._agent.get_agent_metadata()

    def serialize_descs(self, descs) -> bytes:
        return self._agent.get_serialized_descs(descs)

    def deserialize_descs(self, serialized):
        return self._agent.deserialize_descs(serialized)

    def add_remote(self, peer_metadata: bytes) -> None:
        self._agent.add_remote_agent(peer_metadata)

    def prep_local(self, descs):
        return self._agent.prep_xfer_dlist("NIXL_INIT_AGENT", descs, backends=self.backends)

    def prep_remote(self, peer_name: str, descs):
        return self._agent.prep_xfer_dlist(peer_name, descs, backends=self.backends)

    def make_connection(self, peer_name: str) -> None:
        """Eagerly establish the UCX connection to a peer so the first transfer
        doesn't race connection setup (see playground commentary)."""
        self._agent.make_connection(peer_name, backends=self.backends)

    def post_write(self, local_prep, local_idx: int, remote_prep, remote_idx: int):
        handle = self._agent.make_prepped_xfer(
            "WRITE",
            local_prep,
            [local_idx],
            remote_prep,
            [remote_idx],
            backends=self.backends,
        )
        state = self._agent.transfer(handle)
        if state == "ERR":
            raise RuntimeError("nixl transfer post returned ERR")
        return handle

    def wait(self, handle) -> None:
        # Busy-wait mirrors the playground. NIXL does not currently expose a blocking
        # wait; the tight loop is cheap because UCX completions land quickly.
        while self._agent.check_xfer_state(handle) == "PROC":
            pass
        state = self._agent.check_xfer_state(handle)
        if state != "DONE":
            raise RuntimeError(f"nixl transfer ended with state {state}")
        handle.release()


def make_agent_name(role: str, global_rank: int) -> str:
    return f"{role}-{socket.gethostname()}-r{global_rank}"


@dataclass
class KonigAssignment:
    trainer_rank: int
    inference_rank: int
    chunk_idx: int


def konig_schedule(trainer_ws: int, inference_ws: int) -> list[list[KonigAssignment]]:
    """König-style bipartite schedule for non-expert FSDP-sharded tensors.

    Supports divisible R:I ratios only (v1). Three regimes:

    - ``R == I``: direct rotation (``R`` rounds, one trainer writes one chunk to one inference
      per round — the pattern used in ``put_bw_nxn_konig.py``).
    - ``R > I`` with ``R % I == 0``: group trainers into ``I`` buckets of ``R/I``; over ``I`` rounds
      each inference rank receives the appropriate trainer's chunk from every bucket.
    - ``I > R`` with ``I % R == 0``: each trainer writes to ``I/R`` inference ranks per round.

    Returns ``list[list[KonigAssignment]]`` — one list of assignments per round.
    """
    if trainer_ws == inference_ws:
        n = trainer_ws
        return [[KonigAssignment(t, (t + k) % n, chunk_idx=t) for t in range(n)] for k in range(n)]
    if trainer_ws > inference_ws and trainer_ws % inference_ws == 0:
        r, i = trainer_ws, inference_ws
        group = r // i
        return [
            [
                KonigAssignment(
                    trainer_rank=t,
                    inference_rank=((t // group) + k) % i,
                    chunk_idx=t,
                )
                for t in range(r)
            ]
            for k in range(i)
        ]
    if inference_ws > trainer_ws and inference_ws % trainer_ws == 0:
        r, i = trainer_ws, inference_ws
        fan = i // r
        rounds: list[list[KonigAssignment]] = []
        for k in range(r):
            round_list: list[KonigAssignment] = []
            for t in range(r):
                for j in range(fan):
                    round_list.append(
                        KonigAssignment(
                            trainer_rank=t,
                            inference_rank=(t * fan + j + k * fan) % i,
                            chunk_idx=t,
                        )
                    )
            rounds.append(round_list)
        return rounds
    raise ValueError(
        f"NIXL König schedule requires divisible trainer/inference world sizes "
        f"(R={trainer_ws}, I={inference_ws}). Non-divisible ratios are a future phase."
    )


def peer_table_from_gather(gathered: list[dict[str, Any]], trainer_ws: int) -> tuple[list[dict], list[dict]]:
    """Split an ``all_gather_obj`` result into (trainer_infos, inference_infos)."""
    return gathered[:trainer_ws], gathered[trainer_ws:]
