"""Multi-rank NIXL test: R=2 trainer ranks, I=2 inference ranks.

Exercises expert + non-expert routing on 4 GPUs:
  - each trainer rank owns a different half of the experts (EP) and a
    ``Shard(0)`` DTensor slice of every non-expert param (FSDP);
  - each inference rank owns a different half of the experts and a full
    replica of every non-expert param.

Uses ``multi_tiny``, whose dim-0 sizes are multiples of ``2 × BLOCK_SIZE``
so all non-expert slots take the ``per_shard`` path (per-shard FP8 quantize
is bit-exact to full-tensor quantize when shard boundaries land on block
boundaries). One slot that doesn't satisfy that (if any) falls back to the
``gather`` round-robin path.

Run:
    PYTHONPATH=. uv run python tests/integration/nixl/test_nixl_glm_multi.py
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
    return Path(os.environ.get("HOME", "/home/matej")) / ".cache" / "prime_rl_nixl" / "glm_moe_dsa_multi_tiny"


def _ensure_fixture(path: Path, seed: int = 0) -> Path:
    if path.exists() and (path / "config.json").exists():
        return path
    from tests.fixtures.build_tiny_glm_moe_dsa import build_tiny

    return build_tiny(path, seed=seed, size="multi_tiny")


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

        # Simulate EP sharding of expert tensors by slicing in place.
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

        # Non-expert params stay full (each rank holds an identical copy). The
        # trainer's convert step slices dim 0 by rank for per_shard slots, so the
        # inference side still ends up with the full tensor after the writes land.

        # With dp_shard=R and ep=R: dp_shard_mod_ep=1, dp_shard_in_ep=R. EP mesh = all R ranks.
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


def _build_inference_tensors(ref_model, device: torch.device, experts_per_inf: int):
    from prime_rl.trainer.models.fp8 import BLOCK_SIZE, ceil_div
    from prime_rl.trainer.models.glm_moe_dsa.converting_glm_moe_dsa import _BASE, _DENSE, _SPARSE

    cfg = ref_model.config
    ref_sd = ref_model.state_dict()
    tensors: dict[str, torch.Tensor] = {}

    for layer_idx in range(cfg.num_hidden_layers):
        prefix = f"model.layers.{layer_idx}"
        is_sparse = f"{prefix}.mlp.router.gate.weight" in ref_sd
        specs = _BASE + (_SPARSE if is_sparse else _DENSE)
        for spec in specs:
            if spec.dst.startswith("mlp.experts."):
                continue
            src_shapes = [ref_sd[f"{prefix}.{name}"].shape for name in spec.sources]
            dst_shape = list(src_shapes[0])
            dst_shape[spec.cat_dim] = sum(s[spec.cat_dim] for s in src_shapes)
            tensors[f"{prefix}.{spec.dst}"] = torch.zeros(
                dst_shape, dtype=spec.slot_dtype, device=device
            )
            if spec.quantized:
                scale_shape = tuple(
                    ceil_div(d, BLOCK_SIZE) if i >= len(dst_shape) - 2 else d
                    for i, d in enumerate(dst_shape)
                )
                tensors[spec.scale_name(prefix)] = torch.zeros(scale_shape, dtype=torch.float32, device=device)

    # Expert slots shrunk to this inference rank's owned-expert count.
    from prime_rl.trainer.parallel_dims import ParallelDims

    ref_slots_full = ref_model.allocate_slots(ParallelDims(dp_replicate=1, dp_shard=1, cp=1, pp=1, ep=1, world_size=1))
    for layer_idx in range(cfg.first_k_dense_replace, cfg.num_hidden_layers):
        for k, t in ref_slots_full[layer_idx].items():
            if ".mlp.experts." not in k:
                continue
            shape = list(t.shape)
            shape[0] = experts_per_inf
            tensors[k] = torch.zeros(shape, dtype=t.dtype, device=device)

    return tensors


def _inference(local_rank: int, inf_rank: int, port: int, fixture_dir: str, ready_q: mp.Queue) -> None:
    try:
        os.environ["LOCAL_RANK"] = str(local_rank)
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

        from vllm.distributed.utils import StatelessProcessGroup

        from prime_rl.trainer.models.glm_moe_dsa.converting_glm_moe_dsa import convert_tt_layer_to_vllm_kernel
        from prime_rl.trainer.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaForCausalLM
        from prime_rl.utils.nixl_transfer import NixlAgentWrapper, make_agent_name

        ref_model = GlmMoeDsaForCausalLM.from_pretrained(fixture_dir, dtype=torch.bfloat16).to(device)
        cfg = ref_model.config
        num_experts = cfg.n_routed_experts
        assert num_experts % I == 0
        experts_per_inf = num_experts // I
        owned = list(range(inf_rank * experts_per_inf, (inf_rank + 1) * experts_per_inf))

        tensors = _build_inference_tensors(ref_model, device, experts_per_inf)

        agent = NixlAgentWrapper(name=make_agent_name("inference", R + inf_rank), local_rank=local_rank)
        for t in tensors.values():
            agent.register_tensor(t)

        expert_map_per_prefix = {
            f"model.layers.{layer_idx}.mlp.experts": owned
            for layer_idx in range(cfg.first_k_dense_replace, cfg.num_hidden_layers)
        }

        global_rank = R + inf_rank
        spg = StatelessProcessGroup.create(
            host="localhost", port=port, rank=global_rank, world_size=R + I, store_timeout=120
        )
        # Round 1 — pull trainer's layout. Agent metadata deferred to round 2.
        round1 = spg.all_gather_obj(
            {"role": "inference", "global_rank": global_rank, "expert_map": expert_map_per_prefix}
        )
        layout = round1[0]["non_expert_layout"]

        descriptors: dict[str, list[bytes]] = {}

        def _publish_chunks(chunk_list: list[torch.Tensor]) -> list[bytes]:
            out: list[bytes] = []
            for c in chunk_list:
                dlist = agent._agent.get_xfer_descs(
                    [(c.data_ptr(), c.numel() * c.element_size(), c.get_device())],
                    mem_type="cuda",
                )
                out.append(agent.serialize_descs(dlist))
            return out

        for layer_layout in layout.values():
            for slot_key, info in layer_layout.items():
                full = tensors[info["inference_name"]]
                subview = full.narrow(0, info["offset_rows"], info["rows"])
                n_chunks = R if info["handling"] == "per_shard" else 1
                sub_rows = info["rows"] // n_chunks
                chunks = [subview.narrow(0, i * sub_rows, sub_rows) for i in range(n_chunks)]
                descriptors[slot_key] = _publish_chunks(chunks)
        for name, t in tensors.items():
            if ".mlp.experts." not in name:
                continue
            chunks = [t.narrow(0, e, 1) for e in range(experts_per_inf)]
            descriptors[name] = _publish_chunks(chunks)

        # Round 2 — fresh agent_metadata + descriptors + expert_map.
        round2 = spg.all_gather_obj(
            {
                "role": "inference",
                "global_rank": global_rank,
                "agent_name": agent.name,
                "agent_metadata": agent.get_metadata(),
                "descriptors": descriptors,
                "expert_map": expert_map_per_prefix,
            }
        )
        for peer in round2[:R]:
            agent.add_remote(peer["agent_metadata"])

        spg.barrier()

        # Reference: single-shot quantize of the full unsliced model into fused
        # buffers. Expert slots use the FULL expert count (so we can ``index_select``
        # the owned slice for comparison); non-expert slots match inference-side.
        ref_buffers: dict[str, torch.Tensor] = {}
        for name, t in tensors.items():
            if ".mlp.experts." in name:
                shape = list(t.shape)
                shape[0] = num_experts
                ref_buffers[name] = torch.zeros(shape, dtype=t.dtype, device=device)
            else:
                ref_buffers[name] = torch.zeros(t.shape, dtype=t.dtype, device=device)
        ref_sd = ref_model.state_dict()
        mismatches: list[str] = []
        owned_idx = torch.tensor(owned, device=device)
        for layer_idx in range(cfg.num_hidden_layers):
            layer_sd = {
                k: v.to(torch.bfloat16)
                for k, v in ref_sd.items()
                if k.startswith(f"model.layers.{layer_idx}.")
            }
            layer_buffers = {k: v for k, v in ref_buffers.items() if k.startswith(f"model.layers.{layer_idx}.")}
            convert_tt_layer_to_vllm_kernel(layer_sd, layer_idx, out_buffers=layer_buffers)

            for slot_name, ref_tensor in layer_buffers.items():
                got = tensors[slot_name]
                if ".mlp.experts." in slot_name:
                    ref_cmp = ref_tensor.index_select(0, owned_idx)
                else:
                    ref_cmp = ref_tensor
                if ref_cmp.dtype == torch.float8_e4m3fn:
                    equal = torch.equal(ref_cmp.view(torch.uint8), got.view(torch.uint8))
                else:
                    equal = torch.equal(ref_cmp, got)
                if not equal:
                    mismatches.append(
                        f"{slot_name}@L{layer_idx}: "
                        f"ref.abs_sum={ref_cmp.float().abs().sum().item():.4f} "
                        f"got.abs_sum={got.float().abs().sum().item():.4f}"
                    )

        if mismatches:
            ready_q.put(
                (f"inference-{inf_rank}", f"fail: {len(mismatches)} mismatches:\n  " + "\n  ".join(mismatches[:12]))
            )
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
