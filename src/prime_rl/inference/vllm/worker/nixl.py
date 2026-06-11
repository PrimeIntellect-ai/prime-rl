"""vLLM worker extension that pulls weight updates over NIXL.

The trainer is a passive weight-store server: it publishes one table to
Model Express describing every tensor of its *native* state dict (name,
shape, dtype, device address, owning NIXL agent) and refreshes the store
contents each sync. This worker discovers — once — exactly which bytes of
which trainer tensor it needs:

1. **Bake**: drive ``model.load_weights`` with zero-storage
   :class:`LazyWeight` placeholders built from the trainer's table (via the
   generic prime-naming adapter). vLLM's own loaders (fused QKV, merged
   gate/up, FusedMoE EP routing) slice the placeholders and ``copy_`` them
   into views of live parameters; each ``copy_`` records
   ``(trainer tensor, op chain, destination view)``.
2. **Resolve**: every chain is resolved to a strided region of the trainer
   tensor (meta simulation; pure view ops only), decomposed into contiguous
   runs, and matched run-by-run against the destination view's runs.
3. **Pull**: per sync, one batched NIXL READ per trainer agent moves the
   bytes straight from the trainer's store into this worker's parameter
   memory. No staging, no load_weights, no conversion anywhere.

Because the trainer knows nothing about its consumers, inference workers
can scale out, restart, or fail without any trainer-side coordination.
Direct in-place writes into live params are valid only when the kernel
weight format equals the loaded HF layout, which
:func:`_check_direct_write_safe` enforces at init.
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
from prime_rl.weight_transfer.chains import contiguous_runs, match_runs, resolve_chain_region, tensor_runs
from prime_rl.weight_transfer.lazy import BakeRecorder
from prime_rl.weight_transfer.mx import MX_MODEL_NAME, MxRendezvous
from prime_rl.weight_transfer.nixl import NixlAgent, make_agent_name, set_ucx_env_defaults
from prime_rl.weight_transfer.wire import NIXL_DONE_MARKER, TrainerTable, decode_table

# This is to get type hints for the Worker class but not actually extend it at runtime as this is required by vLLM worker extension
if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

    Worker = Worker
else:
    Worker = object

logger = init_logger("vllm.inference.vllm.worker_nixl")


def _check_direct_write_safe(model: Module, quantization: str | None) -> None:
    """Pulling straight into live params skips ``process_weights_after_loading``.

    That is only correct when post-loading processing is a no-op on the weight
    bytes: unquantized models whose MoE kernel format equals the loaded HF
    layout (``convert_to_unquantized_kernel_format`` is identity for the
    triton backends). Anything else must fail here, at init, not as silent
    weight corruption.
    """
    if quantization is not None:
        raise NotImplementedError(
            f"NIXL weight broadcast supports unquantized models only (got quantization={quantization!r})"
        )

    from vllm.model_executor.layers.fused_moe.layer import FusedMoE
    from vllm.model_executor.layers.fused_moe.oracle.unquantized import UnquantizedMoeBackend
    from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import UnquantizedFusedMoEMethod

    identity_backends = (UnquantizedMoeBackend.TRITON, UnquantizedMoeBackend.BATCHED_TRITON)
    for name, module in model.named_modules():
        if not isinstance(module, FusedMoE):
            continue
        quant_method = module.quant_method
        if not isinstance(quant_method, UnquantizedFusedMoEMethod):
            raise NotImplementedError(
                f"NIXL weight broadcast: FusedMoE {name!r} uses {type(quant_method).__name__}, "
                "which may repack weights after loading; only UnquantizedFusedMoEMethod is supported"
            )
        if quant_method.unquantized_backend not in identity_backends:
            raise NotImplementedError(
                f"NIXL weight broadcast: FusedMoE {name!r} uses backend "
                f"{quant_method.unquantized_backend}, which shuffles weights into a kernel format; "
                f"supported identity-format backends: {[b.value for b in identity_backends]}"
            )


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
    """vLLM worker extension that pulls in-place weight updates over NIXL."""

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

        The pull plan itself is built lazily on the first weight update: the
        trainer publishes its table on its first broadcast, which happens
        after this RPC runs.
        """
        if quantize_in_weight_transfer:
            raise NotImplementedError("NIXL weight broadcast does not support quantize_in_weight_transfer")
        if self.vllm_config.parallel_config.enable_eplb:
            # EPLB rearranges experts at runtime, which would silently
            # invalidate the baked expert-to-rank pull plan.
            raise NotImplementedError("NIXL weight broadcast does not support EPLB")
        _check_rdma_registrable_allocator()
        _check_direct_write_safe(self.raw_model, self.model_runner.model_config.quantization)

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
        self._build_pulls(table, copies, registered)
        self._initialized = True
        logger.info(
            f"NIXL pull plan ready in {time.perf_counter() - start:.2f}s: rank={self._global_rank} "
            f"copies={len(copies)} bytes={self._total_pull_bytes:,} agents={len(self._pull_handles_spec)}"
        )

    def _register_params(self, model: Module) -> dict[str, torch.Tensor]:
        """Pin every parameter's live storage for RDMA. Returns name -> tensor."""
        registered: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if not param.is_contiguous():
                raise RuntimeError(f"non-contiguous param {name} cannot be NIXL-registered")
            self.nixl_agent.register_tensor(param.data)
            registered[name] = param.data
        return registered

    def _bake(self, model: Module, table: TrainerTable) -> list:
        """One dry-run ``load_weights`` pass over the trainer's tensor table."""
        recorder = BakeRecorder()
        metas = [(t.name, getattr(torch, t.dtype), tuple(t.shape)) for t in table.tensors]
        model.load_weights(make_hf_named_lazy_weights(metas, self.device, recorder))
        if not recorder.copies:
            raise RuntimeError("NIXL bake recorded no copies — load_weights consumed no lazy placeholders")
        return recorder.copies

    def _build_pulls(self, table: TrainerTable, copies: list, registered: dict[str, torch.Tensor]) -> None:
        """Resolve op chains to trainer regions and prebuild READ descriptors."""
        tensors_by_name = {t.name: t for t in table.tensors}
        intervals = sorted((t.data_ptr(), t.data_ptr() + t.numel() * t.element_size()) for t in registered.values())

        # Per trainer agent: matched (local dst, remote src) descriptor lists.
        local_descs: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
        remote_descs: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
        device_index = self.device.index
        self._total_pull_bytes = 0

        for copy in copies:
            src = tensors_by_name.get(copy.src_name)
            if src is None:
                raise RuntimeError(f"bake recorded a copy from {copy.src_name!r}, which is not in the trainer table")
            src_dtype = getattr(torch, src.dtype)
            if src_dtype != copy.dst.dtype:
                raise RuntimeError(
                    f"dtype mismatch for {copy.src_name!r}: trainer serves {src_dtype}, "
                    f"destination param is {copy.dst.dtype} — raw RDMA cannot cast"
                )
            offset, shape, stride = resolve_chain_region(tuple(src.shape), src_dtype, copy.ops)
            src_runs = contiguous_runs(src.addr, src_dtype.itemsize, offset, shape, stride)
            dst_runs = tensor_runs(copy.dst)
            for addr, nbytes in dst_runs:
                if not _within_intervals(intervals, addr, nbytes):
                    raise RuntimeError(
                        f"bake for {copy.src_name!r} recorded a copy_ into memory outside any registered "
                        "parameter (loader copied into a temporary?)"
                    )
            for src_addr, dst_addr, nbytes in match_runs(src_runs, dst_runs):
                local_descs[src.agent].append((dst_addr, nbytes, device_index))
                remote_descs[src.agent].append((src_addr, nbytes, src.device_id))
                self._total_pull_bytes += nbytes

        self._log_param_coverage(registered, copies)

        # One prepared (local, remote) descriptor pair per trainer agent; every
        # sync posts one batched READ per agent over the full index range.
        self._pull_handles_spec: list[tuple[Any, Any, list[int]]] = []
        for agent_idx, descs in sorted(remote_descs.items()):
            agent = table.agents[agent_idx]
            peer_name = self.nixl_agent.add_remote_agent(agent.metadata)
            self.nixl_agent.make_connection(peer_name)
            local_prep = self.nixl_agent.prep_local(local_descs[agent_idx])
            remote_prep = self.nixl_agent.prep_remote(peer_name, descs)
            self._pull_handles_spec.append((local_prep, remote_prep, list(range(len(descs)))))

    def _log_param_coverage(self, registered: dict[str, torch.Tensor], copies: list) -> None:
        """Warn about params the bake left (partially) unwritten.

        Partial coverage can be legitimate (padded vocab rows); zero coverage
        usually means a name mismatch between trainer and vLLM — surfaced
        loudly so it is never silently stale weights.
        """
        written: dict[int, int] = defaultdict(int)
        for copy in copies:
            written[copy.dst.data_ptr() - copy.dst.storage_offset() * copy.dst.element_size()] += sum(
                n for _, n in tensor_runs(copy.dst)
            )
        uncovered = []
        partial = []
        for name, tensor in registered.items():
            got = written.get(tensor.data_ptr(), 0)
            want = tensor.numel() * tensor.element_size()
            if got == 0:
                uncovered.append(name)
            elif got < want:
                partial.append(f"{name} ({got}/{want}B)")
        if uncovered:
            logger.warning(f"NIXL bake covered no bytes of {len(uncovered)} params: {uncovered}")
        if partial:
            logger.warning(f"NIXL bake partially covered {len(partial)} params: {partial}")

    # ------------------------------ sync ------------------------------ #

    @torch.no_grad()
    def update_weights_from_path(self, weight_dir: str) -> None:
        """Wait for the trainer's store to hold this step's weights, then pull.

        The trainer master touches the step-scoped marker in the broadcast
        step directory (shared filesystem) once every rank's store is filled.
        """
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
            for local_prep, remote_prep, idxs in self._pull_handles_spec
        ]
        for handle in handles:
            self.nixl_agent.wait(handle, context="weight pull")
        torch.cuda.synchronize(self.device)
        update_mla_absorbed_weights(self.raw_model)
        logger.info(
            f"Weight update pulled over NIXL: {self._total_pull_bytes / 1e9:.2f} GB "
            f"in {time.perf_counter() - start:.2f}s"
        )


def _within_intervals(intervals: list[tuple[int, int]], addr: int, nbytes: int) -> bool:
    """True iff [addr, addr+nbytes) lies inside one of the sorted intervals."""
    i = bisect.bisect_right(intervals, (addr, float("inf"))) - 1
    return i >= 0 and intervals[i][0] <= addr and addr + nbytes <= intervals[i][1]
