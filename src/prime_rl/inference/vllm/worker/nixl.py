"""vLLM worker extension that receives weight updates over NIXL.

Registers every vLLM parameter + weight-scale buffer with NIXL and publishes
per-slot chunked descriptors that match the trainer's write layout. Uses two
``all_gather_obj`` rounds so the inference side has the trainer's layout
before it has to build the chunked descriptors.
"""

from __future__ import annotations

import ctypes as _ctypes
import functools
from typing import TYPE_CHECKING, Any

import torch
from torch.nn import Module
from vllm.distributed.utils import StatelessProcessGroup
from vllm.logger import init_logger

from prime_rl.inference.vllm.worker.weight_transfer import (
    assert_mla_absorbed_weights_match,
    build_expert_map,
    update_mla_absorbed_weights,
)
from prime_rl.trainer.models.slots import LayoutEntry
from prime_rl.utils.mx_rendezvous import (
    discover_spg_coordinator,
    mx_rendezvous_enabled,
    pipeline_replication_enabled,
    publish_as_rollout_source,
)
from prime_rl.utils.nixl_transfer import NixlAgentWrapper, make_agent_name

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker  # noqa: F401

    Worker = Worker  # type: ignore
else:
    Worker = object  # type: ignore

logger = init_logger("vllm.inference.vllm.worker_nixl")


@functools.lru_cache(maxsize=1)
def _cudart():
    cudart = _ctypes.CDLL("libcudart.so")
    cudart.cudaDeviceGetAttribute.argtypes = [_ctypes.POINTER(_ctypes.c_int), _ctypes.c_int, _ctypes.c_int]
    cudart.cudaDeviceGetAttribute.restype = _ctypes.c_int
    cudart.cudaDeviceFlushGPUDirectRDMAWrites.argtypes = [_ctypes.c_int, _ctypes.c_int]
    cudart.cudaDeviceFlushGPUDirectRDMAWrites.restype = _ctypes.c_int
    return cudart


@functools.lru_cache(maxsize=8)
def _gpudirect_rdma_attrs(device: int) -> tuple[int, int]:
    cudart = _cudart()
    value = _ctypes.c_int()
    err = cudart.cudaDeviceGetAttribute(_ctypes.byref(value), 117, device)
    if err != 0:
        raise RuntimeError(f"cudaDeviceGetAttribute(117) failed with error {err}")
    flush_options = value.value
    err = cudart.cudaDeviceGetAttribute(_ctypes.byref(value), 118, device)
    if err != 0:
        raise RuntimeError(f"cudaDeviceGetAttribute(118) failed with error {err}")
    return flush_options, value.value


def _flush_gpudirect_rdma_writes(device: int) -> None:
    """Force GPUDirect RDMA writes visible to the owning CUDA context.

    This is NIXL-specific: NCCL's copy path is stream-ordered, but one-sided
    RDMA lands outside CUDA stream semantics. Even on devices that report
    owner-ordering support, forcing the flush is a cheap runtime falsification
    of the "remote writes are not fully visible yet" hypothesis.
    """
    cudart = _cudart()
    cudaFlushGPUDirectRDMAWritesTargetCurrentDevice = 0
    cudaFlushGPUDirectRDMAWritesToOwner = 100
    err = cudart.cudaDeviceFlushGPUDirectRDMAWrites(
        cudaFlushGPUDirectRDMAWritesTargetCurrentDevice,
        cudaFlushGPUDirectRDMAWritesToOwner,
    )
    if err != 0:
        raise RuntimeError(f"cudaDeviceFlushGPUDirectRDMAWrites failed with error {err}")


_ALLOWED_UNPUBLISHED_CUDA_ATTR_SUFFIXES = (
    "W_UV",
    "W_UK_T",
    "W_K",
    "W_K_scale",
    "W_V",
    "W_V_scale",
)

def _iter_unpublished_cuda_tensor_attrs(model: Module):
    for module_name, module in model.named_modules():
        module_prefix = f"{module_name}." if module_name else ""
        parameters = module._parameters
        buffers = module._buffers
        children = module._modules
        for attr_name, value in vars(module).items():
            if not isinstance(value, torch.Tensor) or not value.is_cuda:
                continue
            if attr_name in parameters or attr_name in buffers or attr_name in children:
                continue
            yield f"{module_prefix}{attr_name}", value


def _assert_live_contiguous_registration(name: str, tensor: torch.Tensor) -> torch.Tensor:
    reg_tensor = tensor.contiguous()
    if reg_tensor.data_ptr() != tensor.data_ptr():
        raise RuntimeError(
            f"NIXL registration would detach non-contiguous tensor {name}: "
            f"shape={tuple(tensor.shape)} stride={tuple(tensor.stride())}"
        )
    return reg_tensor


def _assert_narrow_view(full: torch.Tensor, view: torch.Tensor, name: str) -> None:
    if full.untyped_storage().data_ptr() != view.untyped_storage().data_ptr():
        raise RuntimeError(f"NIXL narrow() lost shared storage for {name}")
    expected_ptr_delta = (view.storage_offset() - full.storage_offset()) * full.element_size()
    actual_ptr_delta = view.data_ptr() - full.data_ptr()
    if expected_ptr_delta != actual_ptr_delta:
        raise RuntimeError(
            f"NIXL narrow() pointer mismatch for {name}: "
            f"expected_delta={expected_ptr_delta} actual_delta={actual_ptr_delta}"
        )


def _capture_tensor_identity(tensor: torch.Tensor) -> tuple[int, int, tuple[int, ...], tuple[int, ...]]:
    return (
        tensor.data_ptr(),
        tensor.untyped_storage().data_ptr(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
    )


class NIXLWeightUpdateWorker(Worker):
    """vLLM worker extension for in-place weight updates over NIXL."""

    def init_nixl_transfer(
        self,
        host: str,
        port: int,
        rank_offset: int,
        trainer_world_size: int,
        inference_world_size: int,
        timeout: int,
        backends: list[str],
    ) -> None:
        local_rank = self.device.index
        global_rank = trainer_world_size + rank_offset + local_rank
        full_world_size = trainer_world_size + inference_world_size

        model_runner = self.model_runner
        model = model_runner.model.runnable if hasattr(model_runner.model, "runnable") else model_runner.model
        assert isinstance(model, Module)
        self._model = model

        self._agent = NixlAgentWrapper(
            name=make_agent_name("inference", global_rank),
            local_rank=local_rank,
            backends=backends,
        )

        named_tensors: dict[str, torch.Tensor] = {}
        unpublished_cuda_attrs = list(_iter_unpublished_cuda_tensor_attrs(model))
        handled_unpublished = [
            (name, tensor)
            for name, tensor in unpublished_cuda_attrs
            if name.endswith(_ALLOWED_UNPUBLISHED_CUDA_ATTR_SUFFIXES)
        ]
        if handled_unpublished:
            logger.info(
                "NIXL unpublished CUDA attrs handled out-of-band: %s",
                [
                    {
                        "name": name,
                        "shape": tuple(tensor.shape),
                        "stride": tuple(tensor.stride()),
                        "contig": tensor.is_contiguous(),
                    }
                    for name, tensor in handled_unpublished
                ],
            )

        # Register each vLLM target tensor as a single NIXL region (one MR on the
        # NIC). Sub-range xfer descriptors at post time resolve to this MR's rkey —
        # overlapping per-chunk MRs confuse mlx5 rkey lookup and trigger local
        # protection errors on WRITE landing.
        for name, param in model.named_parameters():
            t = _assert_live_contiguous_registration(name, param.data)
            named_tensors[name] = t
            self._agent.register_tensor(t)
        for name, buf in model.named_buffers():
            if name in named_tensors or not name.endswith("_weight_scale_inv"):
                continue
            t = _assert_live_contiguous_registration(name, buf)
            named_tensors[name] = t
            self._agent.register_tensor(t)
        self._registered_tensor_identities = {
            name: _capture_tensor_identity(tensor) for name, tensor in named_tensors.items()
        }

        expert_map = {k: v.cpu().tolist() for k, v in build_expert_map(model).items()}

        # ModelExpress overlay: if PRIME_RL_MX_RENDEZVOUS is set, override
        # the orchestrator-provided host/port with one discovered via MX
        # Server. This allows the trainer and inference pods to find each
        # other without static host/port configuration.
        #
        # Also stash the inference rank on self so update_weights_from_path
        # can use it for pipeline-replication publishes after receive.
        self._inference_rank = rank_offset + local_rank
        if mx_rendezvous_enabled():
            endpoint = discover_spg_coordinator(
                role="inference",
                rank=self._inference_rank,
                expected_trainer_ws=trainer_world_size,
                expected_inference_ws=inference_world_size,
                fallback_host=host,
                fallback_port=port,
            )
            host, port = endpoint.host, endpoint.port
            logger.info(
                f"[mx-rendezvous] inference rank={global_rank} using discovered "
                f"SPG coordinator {host}:{port}"
            )

        self._spg = StatelessProcessGroup.create(
            host=host,
            port=port,
            rank=global_rank,
            world_size=full_world_size,
            store_timeout=timeout,
        )

        # Round 1 — pull trainer's flat LayoutEntry list. Agent metadata is
        # deferred to round 2 so the metadata trainer sees already includes
        # every chunk registration.
        round1 = self._spg.all_gather_obj(
            {
                "role": "inference",
                "global_rank": global_rank,
                "expert_map": expert_map,
            }
        )
        layout_entries: list[LayoutEntry] = round1[0]["layout_entries"]

        # For each slot, publish a LIST of serialized 1-entry xfer dlists, each
        # covering one chunk's sub-range within the (already-registered) full
        # tensor. Trainer picks the right index per write.
        descriptors: dict[str, list[bytes]] = {}

        def _publish_chunks(full: torch.Tensor, key: str, chunks: list[torch.Tensor]) -> list[bytes]:
            out: list[bytes] = []
            for i, c in enumerate(chunks):
                _assert_narrow_view(full, c, f"{key}[chunk={i}]")
                dlist = self._agent._agent.get_xfer_descs(
                    [(c.data_ptr(), c.numel() * c.element_size(), c.get_device())],
                    mem_type="cuda",
                )
                out.append(self._agent.serialize_descs(dlist))
            return out

        for entry in layout_entries:
            full = named_tensors[entry.inference_name]
            subview = full.narrow(0, entry.offset_rows, entry.rows)
            sub_rows = entry.rows // entry.num_chunks
            chunks = [subview.narrow(0, i * sub_rows, sub_rows) for i in range(entry.num_chunks)]
            descriptors[entry.slot_key] = _publish_chunks(full, entry.slot_key, chunks)

        for name, tensor in named_tensors.items():
            if ".mlp.experts." not in name:
                continue
            moe_prefix = name.rsplit(".", 1)[0]
            num_local = len(expert_map[moe_prefix])
            chunks = [tensor.narrow(0, e, 1) for e in range(num_local)]
            descriptors[name] = _publish_chunks(tensor, name, chunks)

        # Round 2 — publish agent metadata (now carrying all chunk registrations),
        # serialized descriptors, and expert map. Trainer picks these up.
        round2 = self._spg.all_gather_obj(
            {
                "role": "inference",
                "global_rank": global_rank,
                "agent_name": self._agent.name,
                "agent_metadata": self._agent.get_metadata(),
                "descriptors": descriptors,
                "expert_map": expert_map,
            }
        )
        for peer in round2[:trainer_world_size]:
            self._agent.add_remote(peer["agent_metadata"])
        logger.info(
            f"NIXL transfer ready: rank={global_rank} descriptors={len(descriptors)} trainers={trainer_world_size}"
        )

    @torch.no_grad()
    def update_weights_from_path(self, weight_dir: str | None = None) -> None:
        if not hasattr(self, "_spg"):
            raise RuntimeError("NIXL transfer not initialized — call /init_nixl_transfer first")
        # PRE-WRITE: fully drain any in-flight inference CUDA work and
        # signal to the trainer that this worker is quiescent. Trainer
        # will block at a matching barrier before posting any RDMA.
        torch.cuda.synchronize()
        self._spg.barrier()
        # POST-WRITE: trainer posts writes + drains handles, then hits
        # the end-of-push barrier. When this returns, every RDMA WRITE
        # has been ack'd at this worker's NIC.
        self._spg.barrier()
        flush_options, ordering = _gpudirect_rdma_attrs(self.device.index)
        if not getattr(self, "_logged_gpudirect_caps", False):
            logger.info(
                f"[nixl GPUDirect] device={self.device.index} "
                f"flush_options={flush_options} ordering={ordering}"
            )
            self._logged_gpudirect_caps = True
        _flush_gpudirect_rdma_writes(self.device.index)
        # CUDA cross-stream visibility fence: writes landed via PCIe DMA,
        # not on a CUDA stream; sync so subsequent SM kernels see them.
        torch.cuda.synchronize()

        # Track push count for one-shot diagnostic dumps.
        self._push_counter = getattr(self, "_push_counter", 0) + 1
        self._maybe_dump_inference()

        # Diagnostic: mirror the trainer-side SIG anchors.
        #   G (bf16 gather)   : input_layernorm.weight
        #   F (fp8 gather)    : self_attn.kv_b_proj.weight + scale
        #   E (fp8 expert)    : mlp.experts.w13_weight expert[0]
        named_params = dict(self._model.named_parameters())
        named_bufs = dict(self._model.named_buffers())

        if not getattr(self, "_validated_registered_tensor_identities", False):
            for name, expected in self._registered_tensor_identities.items():
                if name in named_params:
                    live = named_params[name].data
                elif name in named_bufs:
                    live = named_bufs[name].data
                else:
                    raise RuntimeError(f"NIXL published tensor disappeared from live model: {name}")
                actual = _capture_tensor_identity(live)
                if actual != expected:
                    raise RuntimeError(
                        f"NIXL published tensor identity changed after init for {name}: "
                        f"expected={expected} actual={actual}"
                    )
            logger.info(
                "NIXL published tensor identity audit passed for %d tensors",
                len(self._registered_tensor_identities),
            )
            self._validated_registered_tensor_identities = True

        def _lookup(name):
            """vLLM stores FP8 scales sometimes as params, sometimes as buffers."""
            if name in named_params:
                return named_params[name].data, "param"
            if name in named_bufs:
                return named_bufs[name].data, "buf"
            return None, "MISSING"

        g_name = "model.layers.3.input_layernorm.weight"
        f_name = "model.layers.3.self_attn.kv_b_proj.weight"
        f_scale_name = "model.layers.3.self_attn.kv_b_proj.weight_scale_inv"
        e_name = "model.layers.3.mlp.experts.w13_weight"
        e_scale_name = "model.layers.3.mlp.experts.w13_weight_scale_inv"
        e_prefix = "model.layers.3.mlp.experts"
        from prime_rl.inference.vllm.worker.weight_transfer import build_expert_map
        expert_map = build_expert_map(self._model)

        # N anchors for non-layer tensors (iter8 fix verification).
        # For ShardedSlot non-layer tensors (embed, lm_head), log just
        # rows [0:2420] which is exactly trainer rank 0's shard on a
        # 64-rank FSDP setup. Its sum should match trainer's SIG.
        for n_name in (
            "model.embed_tokens.weight",
            "model.norm.weight",
            "lm_head.weight",
        ):
            n_t, n_loc = _lookup(n_name)
            if n_t is None:
                logger.warning(f"[nixl SIG inference] anchor=N MISSING key={n_name}")
                continue
            full_sum = n_t.to(torch.float64).sum().item()
            if n_t.dim() >= 2 and n_t.shape[0] > 2420:
                head_sum = n_t[:2420].to(torch.float64).sum().item()
                logger.info(
                    f"[nixl SIG inference] anchor=N loc={n_loc} key={n_name} "
                    f"full_sum={full_sum:.8f} head2420_sum={head_sum:.8f} "
                    f"shape={tuple(n_t.shape)}"
                )
            else:
                logger.info(
                    f"[nixl SIG inference] anchor=N loc={n_loc} key={n_name} "
                    f"sum={full_sum:.8f} shape={tuple(n_t.shape)}"
                )
        for scale_name in (
            "model.layers.3.self_attn.mla_attn.mla_attn._q_scale",
            "model.layers.3.self_attn.mla_attn.mla_attn._k_scale",
            "model.layers.3.self_attn.mla_attn.mla_attn._v_scale",
            "model.layers.3.self_attn.mla_attn.mla_attn._prob_scale",
        ):
            scale_t, scale_loc = _lookup(scale_name)
            if scale_t is None:
                logger.warning(f"[nixl SIG inference] anchor=A MISSING key={scale_name}")
                continue
            logger.info(
                f"[nixl SIG inference] anchor=A loc={scale_loc} key={scale_name} "
                f"value={scale_t.to(torch.float64).item():.8f} shape={tuple(scale_t.shape)}"
            )
        fused_qkv_name = "model.layers.3.self_attn.fused_qkv_a_proj.weight"
        fused_qkv_scale_name = "model.layers.3.self_attn.fused_qkv_a_proj.weight_scale_inv"
        # F_q covers inference fused_qkv_a_proj[0:2048] → matches q_a_proj (row 0..2047).
        # F_kv covers inference fused_qkv_a_proj[2048:2624] → matches kv_a_proj_with_mqa (row 0..575).
        fq_t, fq_loc = _lookup(fused_qkv_name)
        fq_s, fq_s_loc = _lookup(fused_qkv_scale_name)
        if fq_t is not None:
            q_w_bytes = fq_t[:2048].contiguous().view(torch.uint8).to(torch.int64).sum().item()
            kv_w_bytes = fq_t[2048:2624].contiguous().view(torch.uint8).to(torch.int64).sum().item()
            q_s_sum = fq_s[:16].to(torch.float64).sum().item() if fq_s is not None else -1.0
            kv_s_sum = fq_s[16:21].to(torch.float64).sum().item() if fq_s is not None else -1.0
            logger.info(
                f"[nixl SIG inference] anchor=F_q loc={fq_loc}/{fq_s_loc} key={fused_qkv_name}[:2048] "
                f"w_bytes={q_w_bytes} w_shape={tuple(fq_t[:2048].shape)} scale={q_s_sum:.8f}"
            )
            logger.info(
                f"[nixl SIG inference] anchor=F_kv loc={fq_loc}/{fq_s_loc} key={fused_qkv_name}[2048:2624] "
                f"w_bytes={kv_w_bytes} w_shape={tuple(fq_t[2048:2624].shape)} scale={kv_s_sum:.8f}"
            )
        e_t, e_loc = _lookup(e_name)
        e_s, e_s_loc = _lookup(e_scale_name)
        if e_t is not None and e_prefix in expert_map:
            owned = expert_map[e_prefix].cpu().tolist()
            # Log global experts 0..3 that this worker owns (if any).
            for global_id in (0, 1, 2, 3):
                if global_id in owned:
                    local_idx = owned.index(global_id)
                    w_bytes = e_t[local_idx].view(torch.uint8).to(torch.int64).sum().item()
                    s_sum = e_s[local_idx].to(torch.float64).sum().item() if e_s is not None else -1.0
                    logger.info(
                        f"[nixl SIG inference] anchor=E[E{global_id}] loc={e_loc}/{e_s_loc} "
                        f"key={e_name}[local={local_idx}] "
                        f"w_bytes={w_bytes} scale={s_sum:.8f}"
                    )

        update_mla_absorbed_weights(self._model)
        if not getattr(self, "_validated_mla_absorbed_weights", False):
            checked = assert_mla_absorbed_weights_match(self._model)
            logger.info(f"NIXL MLA refresh audit passed for {checked} modules")
            self._validated_mla_absorbed_weights = True

        # ModelExpress overlay: after every receive completes, optionally
        # publish this rollout as an additional source for the current
        # version so subsequent rollouts can pull from us rather than from
        # the trainer. No-op if PRIME_RL_MX_PIPELINE_REPLICATION != 1.
        if pipeline_replication_enabled() and mx_rendezvous_enabled():
            try:
                publish_as_rollout_source(
                    source_id="",  # inferred from model_name on server
                    rank=getattr(self, "_inference_rank", 0),
                )
            except Exception as exc:  # noqa: BLE001 — non-fatal
                logger.warning(
                    f"[mx-rendezvous] pipeline-replication publish failed (non-fatal): {exc}"
                )

    def _maybe_dump_inference(self) -> None:
        """One-shot dump of every param + relevant buffer to
        ``$NIXL_DUMP_DIR/inference_r{RRR}.pt`` when this push matches
        ``$NIXL_DUMP_PUSH``. Pair with trainer dump + tools/nixl_diff.py.
        """
        import os
        dump_dir = os.environ.get("NIXL_DUMP_DIR")
        target_push = int(os.environ.get("NIXL_DUMP_PUSH", "0"))
        if not dump_dir or self._push_counter != target_push:
            return
        os.makedirs(dump_dir, exist_ok=True)

        LAYER_PREFIX = "model.layers.3."
        NON_LAYER = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}

        def _in_scope(name: str) -> bool:
            if name in NON_LAYER:
                return True
            if name.startswith(LAYER_PREFIX):
                return True
            return False

        out: dict[str, Any] = {"params": {}, "buffers": {}, "expert_map": {}}
        for name, p in self._model.named_parameters():
            if _in_scope(name):
                out["params"][name] = p.data.detach().cpu().clone()
        for name, b in self._model.named_buffers():
            if _in_scope(name):
                out["buffers"][name] = b.data.detach().cpu().clone()

        from prime_rl.inference.vllm.worker.weight_transfer import build_expert_map
        for k, v in build_expert_map(self._model).items():
            if k.startswith(LAYER_PREFIX.rstrip(".")) or LAYER_PREFIX in (k + "."):
                out["expert_map"][k] = v.cpu().tolist()

        # global_rank is captured at init_nixl_transfer; recover via agent name.
        # Agent name format: f"inference-{host}-r{global_rank}"
        rank_str = self._agent.name.split("-r")[-1]
        try:
            global_rank = int(rank_str)
        except ValueError:
            global_rank = -1
        fname = f"{dump_dir}/inference_r{global_rank:03d}.pt"
        import torch as _torch
        _torch.save(out, fname)
        logger.info(
            f"[nixl DUMP] inference rank={global_rank} wrote "
            f"{len(out['params'])} params + {len(out['buffers'])} buffers to {fname}"
        )
