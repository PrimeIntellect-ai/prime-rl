"""End-to-end NIXL weight transfer: tiny GLM MoE DSA trainer -> fake inference.

Spawns two processes on two GPUs:
  * Trainer (rank 0): real tiny GLM MoE DSA + real ``NIXLWeightBroadcast``.
  * Inference (rank 1): stub model with the exact vLLM kernel-format parameter
    layout the trainer writes into (``w13_weight`` fp8, ``w2_weight`` fp8,
    ``w13_weight_scale_inv`` fp32, ``w2_weight_scale_inv`` fp32). We mirror
    the relevant bits of ``NIXLWeightUpdateWorker`` (register + rendezvous +
    barrier) without needing the vLLM Worker base class.

Afterwards the inference side runs the same ``convert_tt_layer_to_vllm_kernel``
on its own copy of the trainer's state dict and compares bitwise — any
mismatch points at either the NIXL pipeline or the expert-routing table.

Run directly:

    uv run python tests/integration/nixl/test_nixl_glm_e2e.py
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
    base = Path(os.environ.get("TMPDIR", "/tmp")) / "prime_rl_medium_glm_moe_dsa"
    return base


def _ensure_fixture(path: Path, seed: int = 0) -> Path:
    if path.exists() and (path / "config.json").exists():
        return path
    from tests.fixtures.build_tiny_glm_moe_dsa import build_tiny

    return build_tiny(path, seed=seed, size="medium")


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
        from prime_rl.trainer.rl.broadcast.nixl import NIXLWeightBroadcast

        model = GlmMoeDsaForCausalLM.from_pretrained(fixture_dir, dtype=torch.bfloat16).to(device).eval()

        parallel_dims = ParallelDims(dp_replicate=1, dp_shard=1, cp=1, pp=1, ep=1, world_size=1)

        config = NIXLWeightBroadcastConfig(
            type="nixl",
            host="localhost",
            port=port,
            timeout=60,
            inference_world_size=1,
        )
        bcast = NIXLWeightBroadcast(Path(fixture_dir), config, model, device, parallel_dims)
        # Broadcast once, mutate the model, broadcast again — the realistic training
        # loop reuses the same init across every sync, so stable slots and NIXL xfer
        # handles must be safely reusable.
        bcast.push_once(model)
        with torch.no_grad():
            for layer_idx in range(model.config.first_k_dense_replace, model.config.num_hidden_layers):
                experts = model.model.layers[layer_idx].mlp.experts
                for attr in ("w1", "w2", "w3"):
                    getattr(experts, attr).data.mul_(0.5).add_(0.1)
        bcast.push_once(model)
        ready_q.put(("trainer", "ok"))
    except Exception as e:
        ready_q.put(("trainer", f"fail: {e}\n{traceback.format_exc()}"))


def _build_inference_stub(fixture_dir: str, device: torch.device) -> tuple:
    """Build a torch.nn.Module mirroring vLLM kernel-format layout.

    Returns (model, moe_prefixes, expert_map_per_prefix, num_experts, moe_dim, hidden_dim, first_k_dense, num_layers).
    """
    from prime_rl.trainer.models.fp8 import BLOCK_SIZE, ceil_div
    from prime_rl.trainer.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig

    cfg = GlmMoeDsaConfig.from_pretrained(fixture_dir)
    num_experts = cfg.n_routed_experts
    moe_dim = cfg.moe_intermediate_size
    hidden_dim = cfg.hidden_size
    first_k_dense = cfg.first_k_dense_replace
    num_layers = cfg.num_hidden_layers

    # For I=1 every inference rank owns every expert.
    expert_map = torch.arange(num_experts, dtype=torch.long, device=device)

    model = torch.nn.Module()
    model.layers_mlp_experts = torch.nn.ModuleList()
    moe_prefixes = []

    w13_shape = (num_experts, 2 * moe_dim, hidden_dim)
    w2_shape = (num_experts, hidden_dim, moe_dim)
    s_w13 = (ceil_div(2 * moe_dim, BLOCK_SIZE), ceil_div(hidden_dim, BLOCK_SIZE))
    s_w2 = (ceil_div(hidden_dim, BLOCK_SIZE), ceil_div(moe_dim, BLOCK_SIZE))

    for layer_idx in range(num_layers):
        container = torch.nn.Module()
        if layer_idx >= first_k_dense:
            # Fake FusedMoE with the parameter names vLLM would use.
            container._expert_map = expert_map
            container.w13_weight = torch.nn.Parameter(
                torch.zeros(w13_shape, dtype=torch.float8_e4m3fn, device=device), requires_grad=False
            )
            container.w2_weight = torch.nn.Parameter(
                torch.zeros(w2_shape, dtype=torch.float8_e4m3fn, device=device), requires_grad=False
            )
            # Scales live as buffers (matches vLLM FP8 weight-scale convention).
            container.register_buffer(
                "w13_weight_scale_inv",
                torch.zeros((num_experts, *s_w13), dtype=torch.float32, device=device),
            )
            container.register_buffer(
                "w2_weight_scale_inv",
                torch.zeros((num_experts, *s_w2), dtype=torch.float32, device=device),
            )
            moe_prefixes.append(f"model.layers.{layer_idx}.mlp.experts")
        model.add_module(f"_layer_{layer_idx}", container)

    return model, moe_prefixes, expert_map, cfg


def _inference(local_rank: int, port: int, fixture_dir: str, ready_q: mp.Queue) -> None:
    try:
        os.environ["LOCAL_RANK"] = str(local_rank)
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        torch.manual_seed(42)

        from vllm.distributed.utils import StatelessProcessGroup

        from prime_rl.trainer.models.glm_moe_dsa.converting_glm_moe_dsa import convert_tt_layer_to_vllm_kernel
        from prime_rl.trainer.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaForCausalLM
        from prime_rl.trainer.rl.broadcast.nixl import _allocate_layer_slots
        from prime_rl.utils.nixl_transfer import NixlAgentWrapper, make_agent_name

        # Build stub and register buffers with NIXL.
        stub, moe_prefixes, expert_map_tensor, cfg = _build_inference_stub(fixture_dir, device)
        agent = NixlAgentWrapper(name=make_agent_name("inference", 1), local_rank=local_rank)

        # Register every receivable tensor, publish per-layer-name serialized chunked descs
        # (chunks = num_experts, one chunk per expert). Matches the trainer's chunking.
        descriptors: dict[str, bytes] = {}
        num_experts = cfg.n_routed_experts
        for layer_idx in range(cfg.first_k_dense_replace, cfg.num_hidden_layers):
            layer_mod = getattr(stub, f"_layer_{layer_idx}")
            for attr in ("w13_weight", "w2_weight", "w13_weight_scale_inv", "w2_weight_scale_inv"):
                t = getattr(layer_mod, attr)
                agent.register_tensor(t)
                chunked = agent.chunked_descs(t, num_experts)
                name = f"model.layers.{layer_idx}.mlp.experts.{attr}"
                descriptors[name] = agent.serialize_descs(chunked)

        expert_map_per_prefix = {prefix: expert_map_tensor.cpu().tolist() for prefix in moe_prefixes}

        spg = StatelessProcessGroup.create(host="localhost", port=port, rank=1, world_size=2, store_timeout=60)
        my_info = {
            "role": "inference",
            "global_rank": 1,
            "agent_name": agent.name,
            "agent_metadata": agent.get_metadata(),
            "descriptors": descriptors,
            "expert_map": expert_map_per_prefix,
        }
        peers = spg.all_gather_obj(my_info)
        agent.add_remote(peers[0]["agent_metadata"])

        # Round 1: trainer sends original weights.
        spg.barrier()
        # Round 2: trainer sends mutated weights (x*0.5 + 0.1 on each expert tensor).
        spg.barrier()

        # Reference: same mutation applied client-side so we can compare.
        ref_model = GlmMoeDsaForCausalLM.from_pretrained(fixture_dir, dtype=torch.bfloat16).to(device)
        with torch.no_grad():
            for layer_idx in range(cfg.first_k_dense_replace, cfg.num_hidden_layers):
                experts = ref_model.model.layers[layer_idx].mlp.experts
                for attr in ("w1", "w2", "w3"):
                    getattr(experts, attr).data.mul_(0.5).add_(0.1)
        ref_sd = ref_model.state_dict()

        mismatches: list[str] = []
        for layer_idx in range(cfg.first_k_dense_replace, cfg.num_hidden_layers):
            layer_sd = {k: v.to(torch.bfloat16) for k, v in ref_sd.items() if k.startswith(f"model.layers.{layer_idx}.")}
            reference = _allocate_layer_slots(layer_sd, layer_idx, torch.bfloat16, device)
            convert_tt_layer_to_vllm_kernel(layer_sd, layer_idx, out_buffers=reference)
            layer_mod = getattr(stub, f"_layer_{layer_idx}")
            for attr in ("w13_weight", "w2_weight", "w13_weight_scale_inv", "w2_weight_scale_inv"):
                ref_tensor = reference[f"model.layers.{layer_idx}.mlp.experts.{attr}"].to(device)
                got = getattr(layer_mod, attr)
                if ref_tensor.dtype == torch.float8_e4m3fn:
                    equal = torch.equal(ref_tensor.view(torch.uint8), got.view(torch.uint8))
                else:
                    equal = torch.equal(ref_tensor, got)
                if not equal:
                    mismatches.append(
                        f"{attr}@L{layer_idx}: ref.dtype={ref_tensor.dtype} got.dtype={got.dtype} "
                        f"ref.sum={ref_tensor.float().abs().sum().item():.2f} "
                        f"got.sum={got.float().abs().sum().item():.2f}"
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
