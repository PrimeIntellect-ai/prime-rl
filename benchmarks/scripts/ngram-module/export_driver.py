"""Record the SFT trainer's weight exports or one training step's gradients for the n-gram A/B."""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors import safe_open
from torch.distributed.tensor import DTensor

import prime_rl._compat  # noqa: F401
import prime_rl.trainer.model as trainer_model
import prime_rl.transports.weights.nccl as nccl_module
import prime_rl.utils.weights as weights_module
from prime_rl.configs.sft import SFTConfig
from prime_rl.trainer.model import forward, setup_model, setup_tokenizer
from prime_rl.trainer.models.layers.lm_head import IGNORE_INDEX
from prime_rl.trainer.moe_runtime import setup_moe_comm_env
from prime_rl.trainer.optim import setup_optimizer
from prime_rl.trainer.parallel_dims import ParallelDims, get_parallel_dims, resolve_ep
from prime_rl.trainer.sft.data import setup_dataloader, setup_dataset
from prime_rl.trainer.utils import clip_grad_norm_, scale_gradients_, setup_torch_distributed
from prime_rl.trainer.world import get_world
from prime_rl.utils.act_offloading import maybe_activation_offloading
from prime_rl.utils.config import cli
from prime_rl.utils.cp import setup_context_parallel, setup_cp_params, shard_for_cp, validate_cp_seq_lens
from prime_rl.utils.logger import get_logger, setup_logger
from prime_rl.utils.vlm import get_layer_prefix
from prime_rl.utils.weights import (
    convert_state_dict_to_hf,
    gather_weights_parallel,
    partition_weights,
    save_state_dict_parallel,
)

SCRIPT_DIR = Path(__file__).resolve().parent
NGRAM_TABLE_WEIGHT_RE = re.compile(r"\.ple\.ple_embedding\.ngram_embedding\.weight$|\.ngram\.tables\.\d+\.weight$")
WRAPPER_PREFIXES = ("_checkpoint_wrapped_module.", "_orig_mod.")


def nixl_transfer_groups(model: torch.nn.Module, state_dict: dict[str, torch.Tensor]) -> dict:
    from prime_rl.transports.weights.nixl.nixl import NIXLWeightSender

    if hasattr(weights_module, "get_consuming_layers"):
        consuming_layers = weights_module.get_consuming_layers(model, state_dict)
        groups = NIXLWeightSender.build_transfer_group_index(state_dict, consuming_layers)
    else:
        groups = NIXLWeightSender.build_transfer_group_index(state_dict)
    return {
        "group_names": list(groups.group_names),
        "key_to_group": {
            key: groups.group_names[NIXLWeightSender.find_transfer_group_index(key, groups)]
            if value.is_floating_point()
            else None
            for key, value in state_dict.items()
        },
    }


def partition_owners(model: torch.nn.Module, state_dict: dict[str, torch.Tensor], world_size: int) -> dict[str, int]:
    if hasattr(weights_module, "get_consuming_layers"):
        consuming_layers = weights_module.get_consuming_layers(model, state_dict)
        return partition_weights(
            state_dict, world_size, torch.bfloat16, get_layer_prefix(model.config), consuming_layers
        )
    return partition_weights(state_dict, world_size, torch.bfloat16)


def canonical_name(name: str) -> str:
    for prefix in WRAPPER_PREFIXES:
        name = name.replace(prefix, "")
    return name


def tensor_sha256(tensor: torch.Tensor) -> str:
    data = tensor.detach().contiguous().cpu().reshape(-1).view(torch.uint8)
    return hashlib.sha256(data.numpy()).hexdigest()


def describe(key: str, value: torch.Tensor) -> dict:
    assert not isinstance(value, DTensor), f"{key} is still a DTensor"
    return {"key": key, "dtype": str(value.dtype), "shape": list(value.shape), "sha256": tensor_sha256(value)}


def source_commit() -> dict:
    repo = Path(trainer_model.__file__).resolve().parents[3]
    head = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    status = subprocess.run(
        ["git", "-C", repo, "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=True
    )
    return {"repo": str(repo), "commit": head.stdout.strip(), "dirty": bool(status.stdout.strip())}


def mesh_layout(parallel_dims: ParallelDims) -> dict:
    mesh = parallel_dims.world_mesh
    return {
        "dp_replicate": parallel_dims.dp_replicate,
        "dp_shard": parallel_dims.dp_shard,
        "cp": parallel_dims.cp,
        "ep": parallel_dims.ep,
        "world_mesh_dims": list(mesh.mesh_dim_names),
        "world_mesh_shape": list(mesh.mesh.shape),
        "head_mesh_size": parallel_dims.get_mesh("head").size(),
        "fsdp_gradient_divide_factor": parallel_dims.fsdp_gradient_divide_factor,
    }


def ngram_table_layouts(model: torch.nn.Module) -> dict:
    layouts = {}
    for name, param in model.named_parameters():
        name = canonical_name(name)
        if not NGRAM_TABLE_WEIGHT_RE.search(name):
            continue
        if isinstance(param, DTensor):
            layouts[name] = {
                "shape": list(param.shape),
                "mesh_dims": list(param.device_mesh.mesh_dim_names),
                "mesh_shape": list(param.device_mesh.mesh.shape),
                "placements": [str(placement) for placement in param.placements],
                "master_local_shape": list(param.to_local().shape),
            }
        else:
            layouts[name] = {"shape": list(param.shape), "mesh_dims": None}
    return layouts


def build_model(sft_args: list[str]) -> tuple[SFTConfig, ParallelDims, torch.nn.Module]:
    config = cli(SFTConfig, args=sft_args)
    setup_logger(config.log.level, json_logging=config.log.json_logging)
    setup_moe_comm_env(config.model)
    setup_torch_distributed(
        timeout=timedelta(seconds=config.dist_timeout_seconds),
        enable_gloo=config.model.offload == "full",
    )
    torch.set_float32_matmul_precision(config.matmul_precision)
    resolve_ep(config.model)
    parallel_dims = get_parallel_dims(config.model)
    model = setup_model(config.model, parallel_dims)
    if parallel_dims.cp_enabled:
        setup_context_parallel(model, config.model, parallel_dims)
    return config, parallel_dims, model


def record_nccl(model: torch.nn.Module) -> dict:
    record = {"integer": None, "packets": []}

    def record_integer(integer, communicator):
        record["integer"] = int(integer)

    def record_state_dict(state_dict, communicator):
        record["packets"].append([describe(key, value) for key, value in state_dict.items()])

    nccl_module.broadcast_integer = record_integer
    nccl_module.broadcast_state_dict = record_state_dict
    broadcaster = nccl_module.NCCLBroadcaster.__new__(nccl_module.NCCLBroadcaster)
    broadcaster.logger = get_logger()
    broadcaster.world = get_world()
    broadcaster.dtype = torch.bfloat16
    broadcaster.communicator = None
    broadcaster.send(model)
    return record


def record_filesystem(model: torch.nn.Module, fs_dir: Path) -> dict | None:
    world = get_world()
    if world.is_master and fs_dir.exists():
        shutil.rmtree(fs_dir)
    dist.barrier()
    state_dict = gather_weights_parallel(model)
    state_dict = convert_state_dict_to_hf(model, state_dict)
    save_state_dict_parallel(state_dict, fs_dir)
    del state_dict
    dist.barrier()
    if not world.is_master:
        return None
    keys = {}
    files = sorted(fs_dir.glob("*.safetensors"))
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                assert key not in keys, f"{key} saved twice"
                keys[key] = describe(key, handle.get_tensor(key))
    return {"dir": str(fs_dir), "num_files": len(files), "keys": keys}


def run_export(model: torch.nn.Module, out_dir: Path) -> dict:
    world = get_world()
    timings = {}
    start = time.perf_counter()
    nccl = record_nccl(model)
    timings["nccl_s"] = time.perf_counter() - start

    start = time.perf_counter()
    filesystem = record_filesystem(model, out_dir / "fs")
    timings["filesystem_s"] = time.perf_counter() - start

    state_dict = model.state_dict()
    return {
        "nccl": nccl if world.is_master else None,
        "filesystem": filesystem,
        "nixl": nixl_transfer_groups(model, state_dict),
        "partition": partition_owners(model, state_dict, world.world_size),
        "timings": timings,
    }


def compute_loss(
    model: torch.nn.Module, config: SFTConfig, parallel_dims: ParallelDims, micro_batch: dict
) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = micro_batch["input_ids"].to("cuda")
    position_ids = micro_batch["position_ids"].to("cuda")
    target_ids = micro_batch["target_ids"].to("cuda")
    loss_mask = micro_batch["loss_mask"].to("cuda")
    seq_lens = micro_batch["seq_lens"].to("cuda")
    seq_lens_are_pre_shard = False
    if parallel_dims.cp_enabled:
        cp_rank = parallel_dims.world_mesh["cp"].get_local_rank()
        validate_cp_seq_lens(micro_batch["seq_lens"], input_ids.shape[-1])
        input_ids, position_ids = setup_cp_params(input_ids, position_ids, cp_rank, parallel_dims.cp)
        seq_lens_are_pre_shard = True
        target_ids = shard_for_cp(target_ids, cp_rank=cp_rank, cp_world_size=parallel_dims.cp)
        loss_mask = shard_for_cp(loss_mask, cp_rank=cp_rank, cp_world_size=parallel_dims.cp)
    token_count = loss_mask.sum(dtype=torch.int64)
    labels = target_ids.masked_fill(~loss_mask, IGNORE_INDEX)
    with maybe_activation_offloading(config.model.ac_offloading, model):
        out = forward(
            model,
            input_ids,
            position_ids,
            seq_lens=seq_lens,
            labels=labels,
            seq_lens_are_pre_shard=seq_lens_are_pre_shard,
        )
    return out["loss"], token_count


def grad_norms(model: torch.nn.Module) -> dict[str, float | None]:
    norms = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            norms[canonical_name(name)] = None
            continue
        norm = torch.linalg.vector_norm(param.grad.to(torch.float64))
        norms[canonical_name(name)] = (norm.full_tensor() if isinstance(norm, DTensor) else norm).item()
    return norms


def ngram_replica_check(model: torch.nn.Module, parallel_dims: ParallelDims) -> list[dict] | None:
    if not parallel_dims.dp_replicate_enabled:
        return None
    world = get_world()
    replicate_group = parallel_dims.world_mesh["dp_replicate"].get_group()
    local_results = []
    for name, param in model.named_parameters():
        name = canonical_name(name)
        if not NGRAM_TABLE_WEIGHT_RE.search(name):
            continue
        local = param.to_local() if isinstance(param, DTensor) else param
        digests = [None] * dist.get_world_size(replicate_group)
        dist.all_gather_object(digests, tensor_sha256(local), group=replicate_group)
        local_results.append(
            {
                "rank": world.rank,
                "table": name,
                "local_shape": list(local.shape),
                "replica_ranks": dist.get_process_group_ranks(replicate_group),
                "replica_digests": digests,
                "equal": len(set(digests)) == 1,
            }
        )
    gathered = [None] * world.world_size
    dist.all_gather_object(gathered, local_results)
    return [result for rank_results in gathered for result in rank_results]


def run_grads(model: torch.nn.Module, config: SFTConfig, parallel_dims: ParallelDims, optimizer_step: bool) -> dict:
    world = get_world()
    assert config.model.offload == "none", "grads mode mirrors the trainer's path without a gradient manager"
    timings = {}
    tokenizer = setup_tokenizer(config.tokenizer)
    dataset = setup_dataset(tokenizer, config.data, config.model.cp)
    dataiter = iter(setup_dataloader(dataset, config.data, config.model.cp))
    total_micro_batches = config.data.batch_size * config.model.cp
    micro_batches_per_step = world.world_size * config.data.micro_batch_size
    assert total_micro_batches % micro_batches_per_step == 0
    grad_accum_steps = total_micro_batches // micro_batches_per_step
    dp_cp_group = parallel_dims.get_mesh("dp_cp").get_group()

    torch.cuda.synchronize()
    start = time.perf_counter()
    step_loss_sum = torch.tensor(0.0, device="cuda")
    step_token_count = torch.tensor(0, dtype=torch.int64, device="cuda")
    batch_digests = []
    for _ in range(grad_accum_steps):
        micro_batch = next(dataiter)
        batch_digests.append(tensor_sha256(micro_batch["input_ids"]))
        loss_sum, token_count = compute_loss(model, config, parallel_dims, micro_batch)
        step_loss_sum += loss_sum.detach()
        step_token_count += token_count
        (loss_sum / grad_accum_steps).backward()
    torch.cuda.synchronize()
    timings["forward_backward_s"] = time.perf_counter() - start

    dist.all_reduce(step_token_count, op=dist.ReduceOp.SUM, group=dp_cp_group)
    global_token_count = step_token_count.item()
    scale_gradients_(None, model, parallel_dims.fsdp_gradient_divide_factor * grad_accum_steps / global_token_count)
    dist.all_reduce(step_loss_sum, op=dist.ReduceOp.SUM, group=dp_cp_group)

    start = time.perf_counter()
    norms = grad_norms(model)
    timings["grad_norms_s"] = time.perf_counter() - start

    total_grad_norm = None
    if config.optim.max_norm is not None:
        total_grad_norm = clip_grad_norm_(None, model, config.optim.max_norm).item()

    replica_check = None
    if optimizer_step:
        optimizer, gradient_manager = setup_optimizer(
            config.optim, list(model.named_parameters()), parallel_dims, offload=config.model.offload, model=model
        )
        assert gradient_manager is None
        optimizer.step()
        optimizer.zero_grad()
        start = time.perf_counter()
        replica_check = ngram_replica_check(model, parallel_dims)
        timings["replica_check_s"] = time.perf_counter() - start

    all_batch_digests = [None] * world.world_size
    dist.all_gather_object(all_batch_digests, batch_digests)
    return {
        "loss": step_loss_sum.item() / global_token_count,
        "global_token_count": global_token_count,
        "grad_accum_steps": grad_accum_steps,
        "batch_input_ids_sha256": all_batch_digests,
        "total_grad_norm": total_grad_norm,
        "grad_norms": norms,
        "optimizer_step": optimizer_step,
        "ngram_replica_check": replica_check,
        "timings": timings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--optimizer-step", action="store_true")
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "configs" / "smoke.toml")
    parser.add_argument("--out-root", type=Path, default=Path("/home/garrett/tmp/ngram-module/drivers"))
    parser.add_argument("--conversion-root", type=Path, default=Path("/home/garrett/tmp/ngram-module/conv"))
    parser.add_argument("mode", choices=["export", "grads"])
    parser.add_argument("arm")
    parser.add_argument("layout")
    parser.add_argument("sft_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    out_dir = args.out_root / args.arm / args.layout
    sft_args = [
        "@",
        str(args.config),
        "--model.conversion-dir",
        str(args.conversion_root / f"{args.arm}-drivers"),
        *args.sft_args,
    ]
    start = time.perf_counter()
    config, parallel_dims, model = build_model(sft_args)
    setup_s = time.perf_counter() - start
    world = get_world()
    logger = get_logger()
    logger.info(f"Built model in {setup_s:.1f}s, mesh {mesh_layout(parallel_dims)}")

    start = time.perf_counter()
    if args.mode == "export":
        result = run_export(model, out_dir)
    else:
        result = run_grads(model, config, parallel_dims, args.optimizer_step)
    result["timings"] |= {"setup_s": setup_s, "mode_s": time.perf_counter() - start}

    replica_failures = [check for check in result.get("ngram_replica_check") or [] if not check["equal"]]
    if world.is_master:
        record = {
            "mode": args.mode,
            "arm": args.arm,
            "layout": args.layout,
            "repeat": args.repeat,
            "world_size": world.world_size,
            "source": source_commit(),
            "sft_args": sft_args,
            "mesh": mesh_layout(parallel_dims),
            "ngram_tables": ngram_table_layouts(model),
            **result,
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{args.mode}_r{args.repeat}.json"
        out_path.write_text(json.dumps(record, indent=1) + "\n")
        logger.success(f"Wrote {out_path} (timings {result['timings']})")
        for check in result.get("ngram_replica_check") or []:
            logger.info(
                f"Replica check rank {check['rank']} {check['table']}: equal={check['equal']} "
                f"replicas={check['replica_ranks']} shape={check['local_shape']}"
            )
    dist.barrier()
    dist.destroy_process_group()
    assert not replica_failures, f"n-gram table shards differ across replicas: {replica_failures}"


if __name__ == "__main__":
    main()
