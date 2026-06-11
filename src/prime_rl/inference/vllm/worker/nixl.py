"""vLLM worker extension that pulls sharded weight updates over NIXL.

The trainer is a passive, sharded weight store: it publishes one table to
Model Express describing, per state-dict tensor, which dim-0 row range lives
on which rank's NIXL buffer. This worker discovers — once — exactly which
bytes of which trainer tensor each of its parameter slices needs:

1. **Bake**: drive ``model.load_weights`` with zero-storage
   :class:`LazyWeight` placeholders built from the trainer's table (via the
   generic prime-naming adapter). vLLM's own loaders (fused QKV, merged
   gate/up, FusedMoE expert routing) slice the placeholders and ``copy_``
   them into views of live parameters; each ``copy_`` records
   ``(trainer tensor, op chain, destination view)``. No data moves.
2. **Route**: each op chain is resolved to a strided region of the *full
   logical* trainer tensor, decomposed into runs, and mapped onto the trainer
   shards that own those dim-0 rows — yielding, per source rank, the exact
   ``(src_addr -> dst_addr, nbytes)`` reads.
3. **Pull**: per sync, one batched NIXL READ per trainer rank moves the bytes
   straight into this worker's live parameter storage, then
   ``process_weights_after_loading`` runs so any kernel-format repacking
   happens in place — exactly as a normal vLLM weight reload would.

No gather, no conversion, and no per-consumer state on the trainer, so
inference workers can scale out, restart, or fail without trainer coordination.
"""

from __future__ import annotations

import bisect
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch.nn import Module
from vllm.logger import init_logger

from prime_rl.inference.vllm.worker.weight_transfer import update_mla_absorbed_weights
from prime_rl.weight_transfer.adapter import make_hf_named_lazy_weights
from prime_rl.weight_transfer.chains import region_elem_runs, resolve_chain_region, tensor_runs
from prime_rl.weight_transfer.lazy import BakeRecorder
from prime_rl.weight_transfer.mx import MX_MODEL_NAME, MxRendezvous
from prime_rl.weight_transfer.nixl import NixlAgent, make_agent_name, set_ucx_env_defaults
from prime_rl.weight_transfer.sharding import route_region, zip_src_dst
from prime_rl.weight_transfer.wire import NIXL_DONE_MARKER, NIXL_PULLED_MARKER, TrainerTable, decode_table

# This is to get type hints for the Worker class but not actually extend it at runtime as this is required by vLLM worker extension
if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

    Worker = Worker
else:
    Worker = object

logger = init_logger("vllm.inference.vllm.worker_nixl")


def _check_rdma_registrable_allocator() -> None:
    """Expandable-segments (VMM) allocations cannot be pinned by nvidia_peermem;
    RDMA into them fails at the HCA. The inference process must run with the
    classic caching allocator so model params are registrable."""
    conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if "expandable_segments:True" in conf.replace(" ", ""):
        raise RuntimeError(
            "NIXL weight broadcast requires classic cudaMalloc-backed model parameters, but this "
            f"process runs with PYTORCH_CUDA_ALLOC_CONF={conf!r}. Unset expandable_segments for inference."
        )


class NIXLWeightUpdateWorker(Worker):
    """vLLM worker extension that pulls sharded in-place weight updates over NIXL."""

    @property
    def raw_model(self) -> Module:
        model_runner = self.model_runner
        model = model_runner.model.runnable if hasattr(model_runner.model, "runnable") else model_runner.model
        assert isinstance(model, Module)
        return model

    def liveness_probe(self) -> None:
        """No-op RPC used by the API server liveness endpoint."""
        return None

    def init_broadcaster(
        self,
        host: str,
        port: int,
        rank_offset: int,
        inference_world_size: int,
        timeout: int,
        quantize_in_weight_transfer: bool = False,
    ) -> None:
        """Validate this worker can receive pulls and remember the MX endpoint.

        The pull plan is built lazily on the first weight update, once the
        trainer has published its table (which happens after this RPC runs).
        """
        if self.vllm_config.parallel_config.enable_eplb:
            # EPLB rearranges experts at runtime, invalidating the baked plan.
            raise NotImplementedError("NIXL weight broadcast does not support EPLB")
        _check_rdma_registrable_allocator()
        self._mx_url = f"{host}:{port}"
        self._global_rank = rank_offset + self.device.index
        self._timeout = timeout
        self._initialized = False
        logger.info(f"NIXL pull worker ready (rank={self._global_rank}, mx={self._mx_url})")

    # ------------------------------ plan ------------------------------ #

    @torch.no_grad()
    def _lazy_init(self) -> None:
        if self._initialized:
            return
        from modelexpress.client import MxClient

        start = time.perf_counter()
        set_ucx_env_defaults()
        self.nixl_agent = NixlAgent(name=make_agent_name("inference", self._global_rank))

        rendezvous = MxRendezvous(
            client=MxClient(server_url=self._mx_url),
            role="inference",
            rank=self._global_rank,
            peer_world_size=1,  # the trainer master publishes one combined table
            model_name=MX_MODEL_NAME,
        )
        refs = rendezvous.wait_for_peers(timeout=self._timeout)
        table = decode_table(rendezvous.fetch_peer(refs[0]).nixl_metadata)

        model = self.raw_model
        registered = self._register_params(model)
        copies = self._bake(model, table)
        self._build_pulls(table, copies, registered, model)
        self._initialized = True
        logger.info(
            f"NIXL pull plan ready in {time.perf_counter() - start:.2f}s: rank={self._global_rank} "
            f"copies={len(copies)} bytes={self._total_pull_bytes:,} agents={len(self._pull_specs)} "
            f"process_modules={len(self._process_modules)}"
        )

    def _register_params(self, model: Module) -> dict[str, torch.Tensor]:
        registered: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if not param.is_contiguous():
                raise RuntimeError(f"non-contiguous param {name} cannot be NIXL-registered")
            self.nixl_agent.register_tensor(param.data)
            registered[name] = param.data
        return registered

    def _bake(self, model: Module, table: TrainerTable) -> list:
        """One dry-run ``load_weights`` over the trainer's tensor table.

        Lazy placeholders carry the full logical shape/dtype; vLLM's loaders
        slice them into views of the live params and ``copy_`` (recorded, no
        data moved)."""
        recorder = BakeRecorder()
        metas = [(t.name, getattr(torch, t.dtype), tuple(t.shape)) for t in table.tensors]
        model.load_weights(make_hf_named_lazy_weights(metas, self.device, recorder))
        if not recorder.copies:
            raise RuntimeError("NIXL bake recorded no copies — load_weights consumed no lazy placeholders")
        return recorder.copies

    def _build_pulls(
        self, table: TrainerTable, copies: list, registered: dict[str, torch.Tensor], model: Module
    ) -> None:
        """Resolve op chains to trainer shards and prebuild per-agent READ descriptors."""
        tensors = {t.name: t for t in table.tensors}
        # Map a destination address to its owning leaf module so we can run
        # process_weights_after_loading on exactly the touched modules.
        param_intervals = sorted(
            (p.data_ptr(), p.data_ptr() + p.numel() * p.element_size(), name) for name, p in registered.items()
        )
        module_by_param = {
            name: module for module in model.modules() for name, _ in module.named_parameters(recurse=False)
        }

        # One agent == one trainer rank == one GPU, so its device id is constant.
        agent_device = {shard.agent: shard.device_id for t in table.tensors for shard in t.shards}

        local_descs: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
        remote_descs: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
        device_index = self.device.index
        self._total_pull_bytes = 0
        process_modules: dict[int, Module] = {}

        for copy in copies:
            src = tensors.get(copy.src_name)
            if src is None:
                raise RuntimeError(f"bake recorded a copy from {copy.src_name!r}, absent from the trainer table")
            src_dtype = getattr(torch, src.dtype)
            if src_dtype != copy.dst.dtype:
                raise RuntimeError(
                    f"dtype mismatch for {copy.src_name!r}: trainer serves {src_dtype}, dest param is "
                    f"{copy.dst.dtype} — raw RDMA cannot cast"
                )
            row_numel = 1
            for d in src.shape[1:]:
                row_numel *= d
            offset, shape, stride = resolve_chain_region(tuple(src.shape), src_dtype, copy.ops)
            src_runs = region_elem_runs(offset, shape, stride)
            src_pieces = route_region(src_runs, src.shards, row_numel, src_dtype.itemsize)
            dst_runs = tensor_runs(copy.dst)
            for addr, _ in dst_runs:
                if not _owning_param(param_intervals, addr):
                    raise RuntimeError(
                        f"bake for {copy.src_name!r} recorded a copy_ into memory outside any registered param"
                    )
            for agent, src_addr, dst_addr, nbytes in zip_src_dst(src_pieces, dst_runs):
                local_descs[agent].append((dst_addr, nbytes, device_index))
                remote_descs[agent].append((src_addr, nbytes, agent_device[agent]))
                self._total_pull_bytes += nbytes

            owner = _owning_param(param_intervals, copy.dst.data_ptr())
            if owner is not None and owner in module_by_param:
                module = module_by_param[owner]
                process_modules[id(module)] = module

        self._log_param_coverage(registered, copies, param_intervals)

        self._pull_specs: list[tuple[Any, Any, list[int]]] = []
        for agent_idx, descs in sorted(remote_descs.items()):
            agent = table.agents[agent_idx]
            peer_name = self.nixl_agent.add_remote_agent(agent.metadata)
            self.nixl_agent.make_connection(peer_name)
            local_prep = self.nixl_agent.prep_local(local_descs[agent_idx])
            remote_prep = self.nixl_agent.prep_remote(peer_name, descs)
            self._pull_specs.append((local_prep, remote_prep, list(range(len(descs)))))
        self._process_modules = list(process_modules.values())

    def _log_param_coverage(self, registered: dict[str, torch.Tensor], copies: list, param_intervals: list) -> None:
        """Warn about params the bake left (partially) unwritten — usually a
        name mismatch, surfaced loudly so it is never silently stale weights."""
        written: dict[str, int] = defaultdict(int)
        for copy in copies:
            owner = _owning_param(param_intervals, copy.dst.data_ptr())
            if owner is not None:
                written[owner] += sum(n for _, n in tensor_runs(copy.dst))
        uncovered = [n for n, t in registered.items() if written.get(n, 0) == 0]
        partial = [
            f"{n} ({written[n]}/{t.numel() * t.element_size()}B)"
            for n, t in registered.items()
            if 0 < written.get(n, 0) < t.numel() * t.element_size()
        ]
        if uncovered:
            logger.warning(f"NIXL bake covered no bytes of {len(uncovered)} params: {uncovered}")
        if partial:
            logger.warning(f"NIXL bake partially covered {len(partial)} params: {partial}")

    # ------------------------------ sync ------------------------------ #

    @torch.no_grad()
    def update_weights_from_path(self, weight_dir: str) -> None:
        """Wait for the trainer's shards to hold this step's weights, then pull."""
        self._lazy_init()

        done_marker = Path(weight_dir) / NIXL_DONE_MARKER
        deadline = time.monotonic() + self._timeout
        while not done_marker.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out after {self._timeout}s waiting for {done_marker}")
            time.sleep(0.1)

        start = time.perf_counter()
        handles = [
            self.nixl_agent.post_read(local_prep, idxs, remote_prep, idxs)
            for local_prep, remote_prep, idxs in self._pull_specs
        ]
        for handle in handles:
            self.nixl_agent.wait(handle, context="weight pull")
        torch.cuda.synchronize(self.device)
        self._process_after_load()
        update_mla_absorbed_weights(self.raw_model)
        (Path(weight_dir) / f"{NIXL_PULLED_MARKER}.{self._global_rank}").touch()
        logger.info(
            f"Weight update pulled over NIXL: {self._total_pull_bytes / 1e9:.2f} GB in {time.perf_counter() - start:.2f}s"
        )

    def _process_after_load(self) -> None:
        """Run each touched module's ``process_weights_after_loading`` so any
        kernel-format repacking happens in place — mirrors vLLM's reload path."""
        from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase

        for module in self._process_modules:
            quant_method = getattr(module, "quant_method", None)
            if not isinstance(quant_method, QuantizeMethodBase):
                continue
            if hasattr(module, "_already_called_process_weights_after_loading"):
                delattr(module, "_already_called_process_weights_after_loading")
            quant_method.process_weights_after_loading(module)


def _owning_param(intervals: list[tuple[int, int, str]], addr: int) -> str | None:
    """Name of the registered param whose storage contains ``addr``, or None."""
    i = bisect.bisect_right(intervals, (addr, float("inf"), "")) - 1
    if i >= 0 and intervals[i][0] <= addr < intervals[i][1]:
        return intervals[i][2]
    return None
