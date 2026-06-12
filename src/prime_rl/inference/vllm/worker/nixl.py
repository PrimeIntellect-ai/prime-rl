"""vLLM worker extension that pulls sharded weight updates over NIXL and runs
vLLM's own weight processing — so kernel formats (e.g. online block-FP8) work.

Mirrors upstream vLLM #43375 (sharded RDT), with pure NIXL + Model Express
instead of Ray RDT:

1. **Bake** (once): drive ``model.load_weights`` through vLLM's layerwise
   reload with zero-storage :class:`LazyWeight` placeholders. The layer params
   are on meta in their *load-time* (pre-``process_weights_after_loading``)
   layout — bf16 for an online-fp8 model — and each loader's ``copy_`` records
   ``(trainer tensor, op chain, dest module/param, offset/shape/stride)``.
2. **Plan**: allocate one persistent bf16 staging buffer per destination param
   (the load-time layout), register it with NIXL, and resolve every op chain to
   a region of the full logical trainer tensor routed onto the trainer's dim-0
   shards. Destinations and sources are both fixed, so the per-rank READ plan
   is static.
3. **Per sync**: one batched NIXL READ per trainer rank fills the staging
   buffers with bf16 slices (sharded, no gather), then for each layer
   ``process_weights_after_loading`` runs — re-quantizing bf16->fp8-blockwise
   into the persistent kernel storage, exactly as a vLLM weight reload. For an
   unquantized model that step is a no-op and the staging buffer simply becomes
   the kernel weight.

No gather, no conversion, no per-consumer state on the trainer.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from math import prod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from vllm.config import set_current_vllm_config
from vllm.logger import init_logger

from prime_rl.inference.vllm.worker.weight_transfer import update_mla_absorbed_weights
from prime_rl.weight_transfer.adapter import make_hf_named_lazy_weights
from prime_rl.weight_transfer.chains import region_elem_runs, resolve_chain_region, tensor_runs
from prime_rl.weight_transfer.cuda_pool import classic_cuda_alloc
from prime_rl.weight_transfer.lazy import BakeRecorder, RecordedCopy
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
    RDMA into them fails at the HCA. Staging buffers come from the classic
    cudaMalloc pool, so this only needs the inference process not to force VMM
    globally in a way that would also affect that pool."""
    conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if "expandable_segments:True" in conf.replace(" ", ""):
        raise RuntimeError(
            "NIXL weight broadcast requires classic cudaMalloc-backed staging buffers, but this "
            f"process runs with PYTORCH_CUDA_ALLOC_CONF={conf!r}. Unset expandable_segments for inference."
        )


class _BakedGroup:
    """A leaf module fully covered by the bake and the copies that fill it."""

    def __init__(self, layer: nn.Module, copies: list[RecordedCopy]):
        self.layer = layer
        self.copies = copies
        self.param_names = sorted({c.param_name for c in copies})


class NIXLWeightUpdateWorker(Worker):
    """vLLM worker extension: sharded NIXL pull + vLLM weight processing."""

    @property
    def raw_model(self) -> nn.Module:
        model_runner = self.model_runner
        model = model_runner.model.runnable if hasattr(model_runner.model, "runnable") else model_runner.model
        assert isinstance(model, nn.Module)
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
        """Fetch the trainer table, bake the plan, allocate + register staging,
        and build the static per-rank READ plan. Runs once, on the first sync
        (the trainer publishes its table on its first broadcast)."""
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
        self._table = decode_table(rendezvous.fetch_peer(refs[0]).nixl_metadata)

        self._groups = self._bake()
        self._allocate_staging(self._groups)
        self._build_pulls(self._table, self._groups)
        self._initialized = True
        logger.info(
            f"NIXL pull plan ready in {time.perf_counter() - start:.2f}s: rank={self._global_rank} "
            f"groups={len(self._groups)} bytes={self._total_pull_bytes:,} agents={len(self._pull_specs)}"
        )

    def _bake(self) -> list[_BakedGroup]:
        """One meta dry-run ``load_weights`` through vLLM's layerwise reload.

        Records, per leaf module, the copies that fill it (source op chain +
        destination offset/shape/stride in the load-time layout). Only modules
        the bake fully covers are kept; a partially covered module is a loud
        error (it would otherwise keep stale weights)."""
        from vllm.model_executor.model_loader.reload.layerwise import (
            LAYERWISE_INFO,
            _get_original_loader,
            _place_kernel_tensors,
            initialize_layerwise_reload,
        )
        from vllm.model_executor.model_loader.reload.utils import get_layer_size, get_layer_tensors

        model = self.raw_model
        table = self._table
        recorder = BakeRecorder()
        metas = [(t.name, getattr(torch, t.dtype), tuple(t.shape)) for t in table.tensors]

        with torch.device(self.device), set_current_vllm_config(self.vllm_config):
            initialize_layerwise_reload(model)
            # Stamp the ORIGINAL loaders (bypassing online_process_loader) so the
            # single pass records via the lazy copy_ with no inline processing.
            for module in model.modules():
                for name, tensor in get_layer_tensors(module).items():
                    if getattr(tensor, "weight_loader", None) is None:
                        continue
                    tensor.weight_loader = self._stamp(recorder, module, name, _get_original_loader(tensor))
            model.load_weights(make_hf_named_lazy_weights(metas, self.device, recorder))

            by_layer: dict[int, list[RecordedCopy]] = defaultdict(list)
            layer_of: dict[int, nn.Module] = {}
            for c in recorder.copies:
                by_layer[id(c.layer)].append(c)
                layer_of[id(c.layer)] = c.layer
            groups: list[_BakedGroup] = []
            for lid, copies in by_layer.items():
                layer = layer_of[lid]
                covered = sum(prod(c.shape) for c in copies)
                if covered < get_layer_size(layer):
                    raise RuntimeError(
                        f"NIXL bake covered {covered}/{get_layer_size(layer)} of {type(layer).__name__} "
                        "— partial coverage would leave stale weights (unsupported loader?)"
                    )
                groups.append(_BakedGroup(layer, copies))

            # Capture the load-time (bf16) shape/dtype of each destination param
            # before restoring, so staging buffers match what materialize would make.
            self._param_layout: dict[tuple[int, str], tuple[torch.Size, torch.dtype]] = {}
            for g in groups:
                for pname in g.param_names:
                    p = getattr(g.layer, pname)
                    self._param_layout[(id(g.layer), pname)] = (p.shape, p.dtype)

            # Restore without materializing (mirror the engine's abort path).
            for layer in model.modules():
                info = LAYERWISE_INFO.get(layer)
                if info is not None and info.can_load():
                    if info.kernel_tensors is not None:
                        _place_kernel_tensors(layer, info)
                    info.reset()
            if hasattr(model, "_original_do_torchao_reload"):
                model._do_torchao_reload = model._original_do_torchao_reload

        if not groups:
            raise RuntimeError("NIXL bake recorded no copies — load_weights consumed no lazy placeholders")
        return groups

    @staticmethod
    def _stamp(recorder: BakeRecorder, layer: nn.Module, name: str, inner: Any):
        def stamp(*args, **kwargs):
            recorder.current = (layer, name)
            try:
                return inner(*args, **kwargs)
            finally:
                recorder.current = None

        return stamp

    def _allocate_staging(self, groups: list[_BakedGroup]) -> None:
        """One persistent, NIXL-registered bf16 buffer per destination param."""
        self._staging: dict[tuple[int, str], torch.Tensor] = {}
        with classic_cuda_alloc():
            for g in groups:
                for pname in g.param_names:
                    shape, dtype = self._param_layout[(id(g.layer), pname)]
                    buf = torch.empty(tuple(shape), dtype=dtype, device=self.device)
                    self._staging[(id(g.layer), pname)] = buf
                    self.nixl_agent.register_tensor(buf)

    def _build_pulls(self, table: TrainerTable, groups: list[_BakedGroup]) -> None:
        """Resolve op chains to trainer shards; prebuild the static per-agent READ plan."""
        tensors = {t.name: t for t in table.tensors}
        agent_device = {shard.agent: shard.device_id for t in table.tensors for shard in t.shards}
        local_descs: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
        remote_descs: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
        device_index = self.device.index
        self._total_pull_bytes = 0

        for g in groups:
            for c in g.copies:
                src = tensors.get(c.src_name)
                if src is None:
                    raise RuntimeError(f"bake recorded a copy from {c.src_name!r}, absent from the trainer table")
                src_dtype = getattr(torch, src.dtype)
                buf = self._staging[(id(g.layer), c.param_name)]
                if src_dtype != buf.dtype:
                    raise RuntimeError(
                        f"dtype mismatch for {c.src_name!r}: trainer serves {src_dtype}, load-time param is "
                        f"{buf.dtype} — raw RDMA cannot cast"
                    )
                row_numel = prod(src.shape[1:]) if len(src.shape) > 1 else 1
                offset, shape, stride = resolve_chain_region(tuple(src.shape), src_dtype, c.ops)
                src_pieces = route_region(
                    region_elem_runs(offset, shape, stride), src.shards, row_numel, src_dtype.itemsize
                )
                dst = buf.as_strided(c.shape, c.stride, c.offset)
                for agent, src_addr, dst_addr, nbytes in zip_src_dst(src_pieces, tensor_runs(dst)):
                    local_descs[agent].append((dst_addr, nbytes, device_index))
                    remote_descs[agent].append((src_addr, nbytes, agent_device[agent]))
                    self._total_pull_bytes += nbytes

        self._pull_specs: list[tuple[Any, Any, list[int]]] = []
        for agent_idx, descs in sorted(remote_descs.items()):
            peer_name = self.nixl_agent.add_remote_agent(table.agents[agent_idx].metadata)
            self.nixl_agent.make_connection(peer_name)
            local_prep = self.nixl_agent.prep_local(local_descs[agent_idx])
            remote_prep = self.nixl_agent.prep_remote(peer_name, descs)
            self._pull_specs.append((local_prep, remote_prep, list(range(len(descs)))))

    # ------------------------------ sync ------------------------------ #

    @torch.no_grad()
    def update_weights_from_path(self, weight_dir: str) -> None:
        """Wait for the trainer's shards, pull into staging, then process."""
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
        self._process_and_commit()
        update_mla_absorbed_weights(self.raw_model)
        (Path(weight_dir) / f"{NIXL_PULLED_MARKER}.{self._global_rank}").touch()
        logger.info(
            f"Weight update over NIXL: {self._total_pull_bytes / 1e9:.2f} GB pulled + processed "
            f"in {time.perf_counter() - start:.2f}s"
        )

    def _process_and_commit(self) -> None:
        """Drive each baked layer through vLLM's process+kernel-copy with the
        staging buffers as its load-time params — re-quantizing as needed."""
        from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
        from vllm.model_executor.model_loader.reload.layerwise import (
            LAYERWISE_INFO,
            _copy_and_restore_kernel_tensors,
            initialize_layerwise_reload,
        )

        model = self.raw_model
        with torch.device(self.device), set_current_vllm_config(self.vllm_config):
            initialize_layerwise_reload(model)  # save current kernel params, params -> meta
            for g in self._groups:
                info = LAYERWISE_INFO.get(g.layer)
                # Install the freshly pulled staging buffers as the layer's
                # load-time params (instead of materialize_layer's empty ones).
                for pname in g.param_names:
                    buf = self._staging[(id(g.layer), pname)]
                    setattr(g.layer, pname, nn.Parameter(buf, requires_grad=False))
                quant_method = getattr(g.layer, "quant_method", None)
                if isinstance(quant_method, QuantizeMethodBase):
                    if hasattr(g.layer, "_already_called_process_weights_after_loading"):
                        delattr(g.layer, "_already_called_process_weights_after_loading")
                    quant_method.process_weights_after_loading(g.layer)
                if info is not None and info.kernel_tensors is not None:
                    _copy_and_restore_kernel_tensors(g.layer, info)
                if info is not None:
                    info.reset()
            # No non-baked layers (bake requires full coverage), but call finalize
            # to restore torchao state and clear any residual infos.
            from vllm.model_executor.model_loader.reload.layerwise import finalize_layerwise_reload

            finalize_layerwise_reload(model, self.model_runner.model_config)
