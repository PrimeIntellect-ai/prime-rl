"""End-to-end NIXL weight transfer: tiny GLM MoE DSA trainer -> fake inference.

R=1, I=1. The inference stub mirrors the worker's protocol — registers every
vLLM-kernel-format tensor (expert + non-expert), publishes ``(ptr, nbytes, dev)``
per tensor via a single ``all_gather_obj``, then sits on a barrier while the
trainer writes. After two pushes (second with mutated weights), the stub
reconverts a fresh reference model and compares bitwise.

Run:
    PYTHONPATH=. uv run python tests/integration/nixl/test_nixl_glm_e2e.py
"""

from __future__ import annotations

import os
import socket
import sys
import traceback
from contextlib import closing
from pathlib import Path

import torch
import torch.multiprocessing as mp


def _free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def _fixture_dir() -> Path:
    return Path(os.environ.get("TMPDIR", "/tmp")) / "prime_rl_multi_tiny_glm_moe_dsa"


def _ensure_fixture(path: Path, seed: int = 0) -> Path:
    if path.exists() and (path / "config.json").exists():
        return path
    from tests.fixtures.build_tiny_glm_moe_dsa import build_tiny

    return build_tiny(path, seed=seed, size="multi_tiny")


def _trainer(local_rank: int, port: int, fixture_dir: str, ready_q: mp.Queue) -> None:
    try:
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_WORLD_SIZE"] = "1"
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        torch.manual_seed(42)

        from prime_rl.configs.trainer import NIXLWeightBroadcastConfig
        from prime_rl.trainer.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaForCausalLM
        from prime_rl.trainer.parallel_dims import ParallelDims
        from prime_rl.trainer.rl.broadcast.nixl import NIXLWeightBroadcast, create_nixl_metadata

        model = GlmMoeDsaForCausalLM.from_pretrained(fixture_dir, dtype=torch.bfloat16).to(device).eval()

        parallel_dims = ParallelDims(dp_replicate=1, dp_shard=1, cp=1, pp=1, ep=1, world_size=1)

        config = NIXLWeightBroadcastConfig(
            type="nixl",
            host="localhost",
            port=port,
            timeout=60,
            inference_world_size=1,
        )
        meta = create_nixl_metadata(model, parallel_dims)
        bcast = NIXLWeightBroadcast(Path(fixture_dir), config, meta)
        bcast.push_once(model)
        with torch.no_grad():
            for layer_idx in range(model.config.num_hidden_layers):
                layer = model.model.layers[layer_idx]
                layer.input_layernorm.weight.data.mul_(0.75).add_(0.02)
                layer.self_attn.q_b_proj.weight.data.mul_(0.6).add_(-0.01)
                if layer_idx >= model.config.first_k_dense_replace:
                    experts = layer.mlp.experts
                    for attr in ("w1", "w2", "w3"):
                        getattr(experts, attr).data.mul_(0.5).add_(0.1)
        bcast.push_once(model)
        ready_q.put(("trainer", "ok"))
    except Exception as e:
        ready_q.put(("trainer", f"fail: {e}\n{traceback.format_exc()}"))


def _inference(local_rank: int, port: int, fixture_dir: str, ready_q: mp.Queue) -> None:
    try:
        os.environ["LOCAL_RANK"] = str(local_rank)
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        torch.manual_seed(42)

        from vllm.distributed.utils import StatelessProcessGroup

        from prime_rl.trainer.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaForCausalLM
        from prime_rl.trainer.parallel_dims import ParallelDims
        from prime_rl.utils.nixl_transfer import NixlAgentWrapper, make_agent_name

        ref_model = GlmMoeDsaForCausalLM.from_pretrained(fixture_dir, dtype=torch.bfloat16).to(device)
        cfg = ref_model.config
        parallel_dims_single = ParallelDims(dp_replicate=1, dp_shard=1, cp=1, pp=1, ep=1, world_size=1)

        # Mirror the vLLM kernel-format layout: allocate a FUSED tensor per spec.dst
        # that the trainer will write into. allocate_slots on the reference model
        # (single-rank) gives per-source slots; we fuse them (cat along dim 0) to
        # build one physical buffer per inference param, matching what vLLM would
        # present to the trainer.
        from prime_rl.trainer.models.glm_moe_dsa.converting_glm_moe_dsa import _BASE, _DENSE, _SPARSE

        tensors: dict[str, torch.Tensor] = {}
        from prime_rl.trainer.models.fp8 import BLOCK_SIZE, ceil_div

        ref_sd = ref_model.state_dict()
        for layer_idx in range(cfg.num_hidden_layers):
            prefix = f"model.layers.{layer_idx}"
            is_sparse = f"{prefix}.mlp.router.gate.weight" in ref_sd
            specs = _BASE + (_SPARSE if is_sparse else _DENSE)
            for spec in specs:
                if spec.dst.startswith("mlp.experts."):
                    # Fused expert slot. Use allocate_slots shape from single-rank ref.
                    continue
                src_shapes = [ref_sd[f"{prefix}.{name}"].shape for name in spec.sources]
                dst_shape = list(src_shapes[0])
                dst_shape[spec.cat_dim] = sum(s[spec.cat_dim] for s in src_shapes)
                dtype = torch.float8_e4m3fn if spec.quantize else torch.bfloat16
                tensors[f"{prefix}.{spec.dst}"] = torch.zeros(dst_shape, dtype=dtype, device=device)
                if spec.quantize:
                    scale_shape = tuple(
                        ceil_div(d, BLOCK_SIZE) if i >= len(dst_shape) - 2 else d
                        for i, d in enumerate(dst_shape)
                    )
                    tensors[spec.scale_name(prefix)] = torch.zeros(scale_shape, dtype=torch.float32, device=device)

        # Expert slots: use allocate_slots on a single-rank reference (already fused).
        ref_slots_for_shapes = ref_model.allocate_slots(parallel_dims_single)
        for layer_idx in range(cfg.first_k_dense_replace, cfg.num_hidden_layers):
            for k, t in ref_slots_for_shapes[layer_idx].items():
                if ".mlp.experts." not in k:
                    continue
                tensors[k] = torch.zeros(t.shape, dtype=t.dtype, device=device)

        agent = NixlAgentWrapper(name=make_agent_name("inference", 1), local_rank=local_rank)
        tensor_ptrs: dict[str, tuple[int, int, int]] = {}
        for name, t in tensors.items():
            agent.register_tensor(t)
            tensor_ptrs[name] = (t.data_ptr(), t.numel() * t.element_size(), t.get_device())

        num_experts = cfg.n_routed_experts
        expert_map_per_prefix = {
            f"model.layers.{layer_idx}.mlp.experts": list(range(num_experts))
            for layer_idx in range(cfg.first_k_dense_replace, cfg.num_hidden_layers)
        }

        spg = StatelessProcessGroup.create(host="localhost", port=port, rank=1, world_size=2, store_timeout=60)
        gathered = spg.all_gather_obj(
            {
                "role": "inference",
                "global_rank": 1,
                "agent_name": agent.name,
                "agent_metadata": agent.get_metadata(),
                "tensor_ptrs": tensor_ptrs,
                "expert_map": expert_map_per_prefix,
            }
        )
        agent.add_remote(gathered[0]["agent_metadata"])

        spg.barrier()
        spg.barrier()

        # Verify against per-source/fused reference conversion.
        with torch.no_grad():
            for layer_idx in range(cfg.num_hidden_layers):
                layer = ref_model.model.layers[layer_idx]
                layer.input_layernorm.weight.data.mul_(0.75).add_(0.02)
                layer.self_attn.q_b_proj.weight.data.mul_(0.6).add_(-0.01)
                if layer_idx >= cfg.first_k_dense_replace:
                    experts = layer.mlp.experts
                    for attr in ("w1", "w2", "w3"):
                        getattr(experts, attr).data.mul_(0.5).add_(0.1)
        ref_sd = ref_model.state_dict()

        from prime_rl.trainer.models.glm_moe_dsa.converting_glm_moe_dsa import convert_tt_layer_to_vllm_kernel

        ref_buffers_all: dict[str, torch.Tensor] = {}
        # Use the same fused buffer map we registered for NIXL to hold the expected
        # values — convert_tt_layer_to_vllm_kernel writes in-place to those.
        for k, t in tensors.items():
            ref_buffers_all[k] = torch.zeros(t.shape, dtype=t.dtype, device=device)

        mismatches: list[str] = []
        for layer_idx in range(cfg.num_hidden_layers):
            layer_sd = {
                k: v.to(torch.bfloat16)
                for k, v in ref_sd.items()
                if k.startswith(f"model.layers.{layer_idx}.")
            }
            layer_buffers = {k: v for k, v in ref_buffers_all.items() if k.startswith(f"model.layers.{layer_idx}.")}
            convert_tt_layer_to_vllm_kernel(layer_sd, layer_idx, out_buffers=layer_buffers)
            for name, ref_tensor in layer_buffers.items():
                got = tensors[name]
                if ref_tensor.dtype == torch.float8_e4m3fn:
                    equal = torch.equal(ref_tensor.view(torch.uint8), got.view(torch.uint8))
                else:
                    equal = torch.equal(ref_tensor, got)
                if not equal:
                    mismatches.append(
                        f"{name}: ref.abs_sum={ref_tensor.float().abs().sum().item():.4f} "
                        f"got.abs_sum={got.float().abs().sum().item():.4f}"
                    )

        if mismatches:
            ready_q.put(("inference", f"fail: {len(mismatches)} mismatches:\n  " + "\n  ".join(mismatches[:8])))
        else:
            ready_q.put(("inference", "ok"))
    except Exception as e:
        ready_q.put(("inference", f"fail: {e}\n{traceback.format_exc()}"))


def main() -> int:
    fixture = _ensure_fixture(_fixture_dir())
    port = _free_port()
    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()

    p_trainer = ctx.Process(target=_trainer, args=(0, port, str(fixture), q))
    p_infer = ctx.Process(target=_inference, args=(1, port, str(fixture), q))
    p_trainer.start()
    p_infer.start()
    p_trainer.join(timeout=180)
    p_infer.join(timeout=180)

    results = {}
    while not q.empty():
        role, result = q.get_nowait()
        results[role] = result
    print(f"results: {results}")
    if p_trainer.is_alive() or p_infer.is_alive():
        p_trainer.kill()
        p_infer.kill()
        print("FAIL: processes hung")
        return 1
    t_ok = results.get("trainer", "").startswith("ok")
    i_ok = results.get("inference", "").startswith("ok")
    if not (t_ok and i_ok):
        print("FAIL")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
