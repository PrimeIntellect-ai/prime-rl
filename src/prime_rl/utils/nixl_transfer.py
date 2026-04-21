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
from typing import Sequence

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

    ``UCX_NET_DEVICES`` is hard-overridden to the GPU's PIX-attached NIC so
    that inference decode workers — where vLLM pre-sets UCX_NET_DEVICES=mlx5_0:1
    for its PD KV connector — do not funnel every weight-transfer WRITE through
    a single NIC per node. This only affects UCX agents created AFTER this
    point in the process; the PD connector's UCP worker is already up with
    its own env snapshot and keeps using mlx5_0.
    """
    try:
        nic = map_gpu_to_nic(local_rank)
    except Exception:
        # Fallback: let UCX auto-discover if topology probing fails (e.g. in CI).
        nic = None
    if nic is not None:
        os.environ["UCX_NET_DEVICES"] = nic
    # cuda_ipc is disabled: inter-node transfers go over RDMA (rc_mlx5) via
    # GPUDirect, and cuda_ipc's memh packing trips an assertion when torch's
    # caching allocator hands out a tensor that spans across CUDA allocation
    # segments. We don't need intra-node IPC for the trainer↔inference path.
    os.environ.setdefault("UCX_TLS", "rc_mlx5,ud,cuda_copy")
    os.environ.setdefault("UCX_IB_GPU_DIRECT_RDMA", "y")
    os.environ.setdefault("UCX_RNDV_SCHEME", "put_zcopy")
    os.environ.setdefault("UCX_RNDV_THRESH", "8192")
    os.environ.setdefault("UCX_MEMTYPE_CACHE", "n")
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
        """Pin a tensor's device memory for RDMA and return its xfer descriptor.

        Explicit ``backends=self.backends`` mirrors how vLLM's NixlConnector
        registers KV caches — without it, the backend the rkey is bound to can
        mismatch the one used for the actual transfer and trigger mlx5 local
        protection errors on WRITE landing.
        """
        self._agent.register_memory(tensor, backends=self.backends)
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

    def descs_from_tuples(self, tuples: Sequence[tuple[int, int, int]]):
        """Build xfer descriptors from raw ``(ptr, size, device)`` tuples. Used to
        construct descriptors pointing into a remote agent's registered memory
        from per-tensor ``(ptr, size, dev)`` info published at rendezvous."""
        return self._agent.get_xfer_descs(list(tuples), mem_type="cuda")

    def get_metadata(self) -> bytes:
        return self._agent.get_agent_metadata()

    def serialize_descs(self, descs) -> bytes:
        return self._agent.get_serialized_descs(descs)

    def deserialize_descs(self, serialized):
        return self._agent.deserialize_descs(serialized)

    def add_remote(self, peer_metadata: bytes) -> None:
        self._agent.add_remote_agent(peer_metadata)

    def prep_local(self, descs):
        return self._agent.prep_xfer_dlist("NIXL_INIT_AGENT", descs)

    def prep_remote(self, peer_name: str, descs):
        return self._agent.prep_xfer_dlist(peer_name, descs)

    def make_connection(self, peer_name: str) -> None:
        """Eagerly establish the UCX connection to a peer so the first transfer
        doesn't race connection setup (see playground commentary)."""
        self._agent.make_connection(peer_name)

    def post_write(self, local_prep, local_idx: int, remote_prep, remote_idx: int):
        handle = self._agent.make_prepped_xfer(
            "WRITE",
            local_prep,
            [local_idx],
            remote_prep,
            [remote_idx],
        )
        state = self._agent.transfer(handle)
        if state == "ERR":
            raise RuntimeError("nixl transfer post returned ERR")
        return handle

    def post_write_dlist(self, local_dlist, remote_dlist, remote_agent_name: str):
        """Post a WRITE using raw xfer dlists (no prep step)."""
        handle = self._agent.initialize_xfer(
            "WRITE", local_dlist, remote_dlist, remote_agent_name
        )
        state = self._agent.transfer(handle)
        if state == "ERR":
            raise RuntimeError("nixl transfer post returned ERR")
        return handle

    def describe_prep(self, prep_handle) -> str:
        """Best-effort debug repr for a prepped dlist handle."""
        return repr(prep_handle)

    def wait(self, handle, context: str = "") -> None:
        # Busy-wait mirrors the playground. NIXL does not currently expose a blocking
        # wait; the tight loop is cheap because UCX completions land quickly.
        try:
            while self._agent.check_xfer_state(handle) == "PROC":
                pass
            state = self._agent.check_xfer_state(handle)
        except Exception as exc:
            raise RuntimeError(
                f"nixl check_xfer_state raised {type(exc).__name__} ({exc}) "
                f"from {self.name} context={context!r}"
            ) from exc
        if state != "DONE":
            raise RuntimeError(f"nixl transfer ended with state {state} from {self.name} context={context!r}")
        handle.release()


def make_agent_name(role: str, global_rank: int) -> str:
    return f"{role}-{socket.gethostname()}-r{global_rank}"
