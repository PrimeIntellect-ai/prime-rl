"""Multi-rank NIXL test: R=2 trainer ranks, I=2 inference ranks.

Exercises the real expert routing table — each trainer rank owns a different
half of the experts, each inference rank also owns a different half, so the
write pattern is a 2x2 permutation, not identity. If routing is wrong we'd
see mismatches. 4 GPUs total.

Run:
    uv run python tests/integration/nixl/test_nixl_glm_multi.py
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


R = int(os.environ.get("NIXL_TEST_R", "2"))
I = int(os.environ.get("NIXL_TEST_I", "2"))  # noqa: E741


def _fixture_dir() -> Path:
    # Shared path so srun can reuse the fixture across nodes; /tmp is node-local.
    return Path(os.environ.get("HOME", "/home/matej")) / ".cache" / "prime_rl_nixl" / "glm_moe_dsa_medium"


def _ensure_fixture(path: Path, seed: int = 0) -> Path:
    if path.exists() and (path / "config.json").exists():
        return path
    from tests.fixtures.build_tiny_glm_moe_dsa import build_tiny

    return build_tiny(path, seed=seed, size="medium")


def _trainer(local_rank: int, rank: int, port: int, dist_port: int, fixture_dir: str, ready_q: mp.Queue) -> None:
    try:
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(R)
        os.environ["LOCAL_WORLD_SIZE"] = str(R)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(dist_port)
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

        import torch.distributed as dist

        from prime_rl.configs.trainer import NIXLWeightBroadcastConfig
        from prime_rl.trainer.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaForCausalLM
        from prime_rl.trainer.parallel_dims import ParallelDims
        from prime_rl.trainer.rl.broadcast.nixl import NIXLWeightBroadcast, create_nixl_metadata

        dist.init_process_group(backend="nccl", rank=rank, world_size=R)

        model = GlmMoeDsaForCausalLM.from_pretrained(fixture_dir, dtype=torch.bfloat16).to(device).eval()

        # Simulate EP sharding: rewrite expert tensors to keep only this rank's slice.
        # In a real training run this falls out of DTensor.to_local(); for the test
        # we do it manually so we don't need to set up FSDP/EP on a 2-rank stub.
        num_experts = model.config.n_routed_experts
        experts_per_rank = num_experts // R
        start = rank * experts_per_rank
        end = start + experts_per_rank
        for layer_idx in range(model.config.first_k_dense_replace, model.config.num_hidden_layers):
            experts_mod = model.model.layers[layer_idx].mlp.experts
            for attr in ("w1", "w2", "w3"):
                full = getattr(experts_mod, attr)
                sliced = torch.nn.Parameter(full.data[start:end].contiguous(), requires_grad=False)
                setattr(experts_mod, attr, sliced)

        # EP borrows from dp_shard (see ParallelDims._build_mesh_with_ep): dp_shard=R, ep=R
        # ⇒ dp_shard_mod_ep=1 (no replica), dp_shard_in_ep=R ⇒ EP mesh = all R ranks.
        parallel_dims = ParallelDims(dp_replicate=1, dp_shard=R, cp=1, pp=1, ep=R, world_size=R)
        parallel_dims.build_mesh()

        config = NIXLWeightBroadcastConfig(
            type="nixl", host="localhost", port=port, timeout=120, inference_world_size=I
        )
        meta = create_nixl_metadata(model, parallel_dims)
        bcast = NIXLWeightBroadcast(Path(fixture_dir), config, meta)
        bcast.push_once(model)
        dist.destroy_process_group()
        ready_q.put((f"trainer-{rank}", "ok"))
    except Exception as e:
        ready_q.put((f"trainer-{rank}", f"fail: {e}\n{traceback.format_exc()}"))


def _inference(local_rank: int, inf_rank: int, port: int, fixture_dir: str, ready_q: mp.Queue) -> None:
    try:
        os.environ["LOCAL_RANK"] = str(local_rank)
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

        from vllm.distributed.utils import StatelessProcessGroup

        from prime_rl.trainer.models.fp8 import BLOCK_SIZE, ceil_div
        from prime_rl.trainer.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
        from prime_rl.trainer.models.glm_moe_dsa.converting_glm_moe_dsa import convert_tt_layer_to_vllm_kernel
        from prime_rl.trainer.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaForCausalLM
        from prime_rl.trainer.parallel_dims import ParallelDims
        from prime_rl.utils.nixl_transfer import NixlAgentWrapper, make_agent_name

        cfg = GlmMoeDsaConfig.from_pretrained(fixture_dir)
        num_experts = cfg.n_routed_experts
        moe_dim = cfg.moe_intermediate_size
        hidden_dim = cfg.hidden_size
        first_k_dense = cfg.first_k_dense_replace
        num_layers = cfg.num_hidden_layers

        assert num_experts % I == 0
        experts_per_inf = num_experts // I
        # Each inference rank owns a contiguous half: rank 0 gets [0,experts_per_inf), rank 1 gets the rest.
        owned = list(range(inf_rank * experts_per_inf, (inf_rank + 1) * experts_per_inf))
        # _expert_map entries: local index for owned, -1 for not-owned.
        expert_map_tensor = torch.full((num_experts,), -1, dtype=torch.long, device=device)
        for local_idx, global_idx in enumerate(owned):
            expert_map_tensor[global_idx] = local_idx

        w13_shape = (experts_per_inf, 2 * moe_dim, hidden_dim)
        w2_shape = (experts_per_inf, hidden_dim, moe_dim)
        s_w13 = (ceil_div(2 * moe_dim, BLOCK_SIZE), ceil_div(hidden_dim, BLOCK_SIZE))
        s_w2 = (ceil_div(hidden_dim, BLOCK_SIZE), ceil_div(moe_dim, BLOCK_SIZE))

        agent = NixlAgentWrapper(name=make_agent_name("inference", R + inf_rank), local_rank=local_rank)

        layer_tensors: dict[tuple[int, str], torch.Tensor] = {}
        descriptors: dict[str, bytes] = {}
        for layer_idx in range(first_k_dense, num_layers):
            specs = {
                "w13_weight": torch.zeros(w13_shape, dtype=torch.float8_e4m3fn, device=device),
                "w2_weight": torch.zeros(w2_shape, dtype=torch.float8_e4m3fn, device=device),
                "w13_weight_scale_inv": torch.zeros((experts_per_inf, *s_w13), dtype=torch.float32, device=device),
                "w2_weight_scale_inv": torch.zeros((experts_per_inf, *s_w2), dtype=torch.float32, device=device),
            }
            for attr, t in specs.items():
                agent.register_tensor(t)
                name = f"model.layers.{layer_idx}.mlp.experts.{attr}"
                descriptors[name] = agent.serialize_descs(agent.chunked_descs(t, experts_per_inf))
                layer_tensors[(layer_idx, attr)] = t

        expert_map_per_prefix = {
            f"model.layers.{l}.mlp.experts": owned for l in range(first_k_dense, num_layers)
        }

        global_rank = R + inf_rank
        spg = StatelessProcessGroup.create(
            host="localhost", port=port, rank=global_rank, world_size=R + I, store_timeout=120
        )
        my_info = {
            "role": "inference",
            "global_rank": global_rank,
            "agent_name": agent.name,
            "agent_metadata": agent.get_metadata(),
            "descriptors": descriptors,
            "expert_map": expert_map_per_prefix,
        }
        peers = spg.all_gather_obj(my_info)
        for p in peers[:R]:
            agent.add_remote(p["agent_metadata"])

        spg.barrier()

        ref_model = GlmMoeDsaForCausalLM.from_pretrained(fixture_dir, dtype=torch.bfloat16).to(device)
        ref_sd = ref_model.state_dict()
        ref_slots = ref_model.allocate_slots(
            ParallelDims(dp_replicate=1, dp_shard=1, cp=1, pp=1, ep=1, world_size=1)
        )

        mismatches: list[str] = []
        for layer_idx in range(first_k_dense, num_layers):
            layer_sd = {k: v.to(torch.bfloat16) for k, v in ref_sd.items() if k.startswith(f"model.layers.{layer_idx}.")}
            reference_full = ref_slots[layer_idx]
            convert_tt_layer_to_vllm_kernel(layer_sd, layer_idx, out_buffers=reference_full)
            owned_idx = torch.tensor(owned, device=device)
            for attr in ("w13_weight", "w2_weight", "w13_weight_scale_inv", "w2_weight_scale_inv"):
                ref_full = reference_full[f"model.layers.{layer_idx}.mlp.experts.{attr}"].to(device)
                ref_local = ref_full.index_select(0, owned_idx)
                got = layer_tensors[(layer_idx, attr)]
                if ref_local.dtype == torch.float8_e4m3fn:
                    equal = torch.equal(ref_local.view(torch.uint8), got.view(torch.uint8))
                else:
                    equal = torch.equal(ref_local, got)
                if not equal:
                    mismatches.append(
                        f"{attr}@L{layer_idx}: ref.sum={ref_local.float().abs().sum().item():.2f} "
                        f"got.sum={got.float().abs().sum().item():.2f}"
                    )

        if mismatches:
            ready_q.put((f"inference-{inf_rank}", f"fail: {len(mismatches)} mismatches:\n  " + "\n  ".join(mismatches[:8])))
        else:
            ready_q.put((f"inference-{inf_rank}", "ok"))
    except Exception as e:
        ready_q.put((f"inference-{inf_rank}", f"fail: {e}\n{traceback.format_exc()}"))


def main() -> int:
    fixture = _ensure_fixture(_fixture_dir())
    port = _free_port()
    dist_port = _free_port()
    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()

    procs = []
    for rank in range(R):
        p = ctx.Process(target=_trainer, args=(rank, rank, port, dist_port, str(fixture), q))
        p.start()
        procs.append(p)
    for inf_rank in range(I):
        p = ctx.Process(target=_inference, args=(R + inf_rank, inf_rank, port, str(fixture), q))
        p.start()
        procs.append(p)

    for p in procs:
        p.join(timeout=900)

    results = {}
    while not q.empty():
        role, result = q.get_nowait()
        results[role] = result
    print(f"results: {results}")
    if any(p.is_alive() for p in procs):
        for p in procs:
            p.kill()
        print("FAIL: processes hung")
        return 1
    for role, result in results.items():
        if not result.startswith("ok"):
            print(f"FAIL [{role}]: {result}")
            return 1
    expected = {f"trainer-{r}" for r in range(R)} | {f"inference-{i}" for i in range(I)}
    if set(results.keys()) != expected:
        print(f"FAIL: missing results. got={set(results.keys())} expected={expected}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
