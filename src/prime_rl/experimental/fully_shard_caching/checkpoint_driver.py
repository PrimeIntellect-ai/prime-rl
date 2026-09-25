"""Torchrun driver that puts the mini model through prime-rl's DCP checkpoint path.

Every claim about the live model is decided inside the spawned process and reported as JSON, since
the gates need an 8-rank world that pytest cannot inspect from the outside.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.checkpoint.state_dict_loader import load as dcp_load
from torch.distributed.checkpoint.state_dict_saver import save as dcp_save
from torch.distributed.tensor import DTensor

from prime_rl.configs.trainer import AdamWConfig
from prime_rl.experimental.fully_shard_caching.mini_model import (
    DEFAULT_NUM_EXPERTS,
    WRAP_MODES,
    MiniModelSpec,
    build_mini_model,
)
from prime_rl.experimental.fully_shard_caching.prepared_tensor import (
    PREPARE_CALLS,
    ShardedPreparedTensor,
    UnshardedPreparedTensor,
)
from prime_rl.trainer.ckpt import AppState
from prime_rl.trainer.model import forward
from prime_rl.trainer.models.layers.moe import GroupedExperts
from prime_rl.trainer.optim import setup_optimizer
from prime_rl.trainer.parallel_dims import ParallelDims
from prime_rl.trainer.utils import clip_grad_norm_, scale_gradients_, setup_torch_distributed
from prime_rl.trainer.world import get_world

PHASES = ("save", "load", "roundtrip")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--wrap", choices=WRAP_MODES, default="fp8")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ep", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-seed", type=int, default=7)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--num-experts", type=int, default=DEFAULT_NUM_EXPERTS)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--eval-microbatches", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args()


@dataclass
class Run:
    args: argparse.Namespace
    model: nn.Module
    optimizer: Any
    optimizer_config: AdamWConfig
    parallel_dims: ParallelDims
    device: torch.device


def local_of(tensor: torch.Tensor) -> torch.Tensor:
    return tensor._local_tensor if isinstance(tensor, DTensor) else tensor


def master_shard(parameter: torch.Tensor) -> torch.Tensor:
    local = local_of(parameter)
    return local._tensor if isinstance(local, ShardedPreparedTensor) else local


def expert_parameters(model: nn.Module):
    for module_name, module in model.named_modules():
        if isinstance(module, GroupedExperts):
            for name, parameter in module.named_parameters(recurse=False):
                yield f"{module_name}.{name}", parameter


def local_types(model: nn.Module) -> dict[str, str]:
    return {fqn: type(local_of(parameter)).__name__ for fqn, parameter in expert_parameters(model)}


def wrapper_objects(model: nn.Module) -> dict[str, torch.Tensor]:
    return {fqn: local_of(parameter) for fqn, parameter in expert_parameters(model)}


def shard_pointers(model: nn.Module) -> dict[str, int]:
    return {fqn: master_shard(parameter).data_ptr() for fqn, parameter in expert_parameters(model)}


def shard_checksums(model: nn.Module) -> dict[str, float]:
    return {fqn: master_shard(parameter).detach().double().sum().item() for fqn, parameter in expert_parameters(model)}


def shard_clones(model: nn.Module) -> dict[str, torch.Tensor]:
    return {fqn: master_shard(parameter).detach().clone() for fqn, parameter in expert_parameters(model)}


def shards_equal(model: nn.Module, reference: dict[str, torch.Tensor]) -> bool:
    return all(torch.equal(reference[fqn], master_shard(parameter)) for fqn, parameter in expert_parameters(model))


def clone_prepared_tensors(model: nn.Module) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, str]]:
    """Unshard each expert unit, copy out the prepared tensors, and reshard."""
    clones: dict[str, dict[str, torch.Tensor]] = {}
    unsharded_types: dict[str, str] = {}
    for module_name, module in model.named_modules():
        if not isinstance(module, GroupedExperts):
            continue
        module.unshard()
        for name, parameter in module.named_parameters(recurse=False):
            fqn = f"{module_name}.{name}"
            local = local_of(parameter)
            unsharded_types[fqn] = type(local).__name__
            if isinstance(local, UnshardedPreparedTensor):
                clones[fqn] = {key: tensor.detach().clone() for key, tensor in local.prepared.items()}
        module.reshard()
    return clones, unsharded_types


def prepared_equal(
    reference: dict[str, dict[str, torch.Tensor]], candidate: dict[str, dict[str, torch.Tensor]]
) -> bool:
    if sorted(reference) != sorted(candidate):
        return False
    return all(
        sorted(reference[fqn]) == sorted(candidate[fqn])
        and all(torch.equal(reference[fqn][key], candidate[fqn][key]) for key in reference[fqn])
        for fqn in reference
    )


def packed_parameter_hook_types(model: nn.Module) -> dict[str, dict[str, str]]:
    """The wrapper's fate under the split the save hook does and the cat the load hook does."""
    types: dict[str, dict[str, str]] = {}
    for module_name, module in model.named_modules():
        if not isinstance(module, GroupedExperts):
            continue
        for spec in getattr(module, "packed_parameter_specs", ()):
            parameter = module.get_parameter(spec.name)
            views = spec.split_into_logical_views(parameter)
            types[f"{module_name}.{spec.name}"] = {
                "parameter": type(local_of(parameter)).__name__,
                "split": type(local_of(views[0])).__name__,
                "cat": type(local_of(spec.pack_logical_tensors(views))).__name__,
            }
    return types


def expert_state_dict_types(model: nn.Module) -> dict[str, str]:
    return {fqn: type(local_of(tensor)).__name__ for fqn, tensor in model.state_dict().items() if ".experts." in fqn}


def optimizer_state_types(model: nn.Module, optimizer: Any) -> dict[str, dict[str, str]]:
    fqn_by_id = {id(parameter): fqn for fqn, parameter in model.named_parameters()}
    return {
        fqn_by_id.get(id(parameter), "unregistered"): {
            key: type(local_of(value)).__name__ for key, value in state.items() if isinstance(value, torch.Tensor)
        }
        for parameter, state in optimizer.state.items()
    }


def fake_batch(seed: int, index: int, seq_len: int, vocab_size: int, device: torch.device):
    generator = torch.Generator(device=device).manual_seed(
        (seed * 1_000_003 + index * 9_973 + dist.get_rank()) % (2**63)
    )
    input_ids = torch.randint(0, vocab_size, (1, seq_len), generator=generator, device=device)
    return input_ids, input_ids.roll(-1, dims=1)


def forward_inputs(run: Run):
    seq_len = run.args.seq_len
    return (
        torch.arange(seq_len, device=run.device).unsqueeze(0),
        torch.tensor([seq_len], device=run.device),
        torch.ones((1, seq_len), device=run.device),
    )


def evaluate(run: Run) -> list[float]:
    position_ids, seq_lens, temperature = forward_inputs(run)
    vocab_size = run.model.config.vocab_size
    losses = []
    with torch.no_grad():
        for index in range(run.args.eval_microbatches):
            input_ids, labels = fake_batch(run.args.data_seed, index, run.args.seq_len, vocab_size, run.device)
            output = forward(
                run.model, input_ids, position_ids, seq_lens=seq_lens, labels=labels, temperature=temperature
            )
            losses.append(-output["logprobs"].mean().item())
    return losses


def training_step(run: Run, step: int) -> None:
    position_ids, seq_lens, temperature = forward_inputs(run)
    vocab_size = run.model.config.vocab_size
    run.optimizer.zero_grad(set_to_none=True)
    for microbatch in range(run.args.grad_accum):
        input_ids, labels = fake_batch(
            run.args.data_seed + 1_000 * (step + 1), microbatch, run.args.seq_len, vocab_size, run.device
        )
        output = forward(run.model, input_ids, position_ids, seq_lens=seq_lens, labels=labels, temperature=temperature)
        loss = -output["logprobs"].mean()
        (loss / run.args.grad_accum).backward()
    scale_gradients_(None, run.model, run.parallel_dims.fsdp_gradient_divide_factor)
    clip_grad_norm_(None, run.model, run.optimizer_config.max_norm, run.parallel_dims.ep_enabled)
    run.optimizer.step()


def save_checkpoint(run: Run) -> None:
    if get_world().is_master:
        run.args.ckpt.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    dcp_save({"app": AppState(run.model, [run.optimizer], None, None)}, checkpoint_id=run.args.ckpt)


def load_checkpoint(run: Run) -> None:
    dcp_load(
        state_dict={"app": AppState(run.model, [run.optimizer], None, None)},
        checkpoint_id=run.args.ckpt,
    )


def run_save(run: Run) -> dict[str, Any]:
    hook_types = packed_parameter_hook_types(run.model)
    training_step(run, step=0)
    save_checkpoint(run)
    return {
        "packed_parameter_hook_types": hook_types,
        "expert_state_dict_types": expert_state_dict_types(run.model),
        "local_types": local_types(run.model),
        "optimizer_state_types": optimizer_state_types(run.model, run.optimizer),
        "eval_losses": evaluate(run),
    }


def run_load(run: Run) -> dict[str, Any]:
    losses_before = evaluate(run)
    types_before = local_types(run.model)
    objects_before = wrapper_objects(run.model)
    pointers_before = shard_pointers(run.model)
    checksums_before = shard_checksums(run.model)

    load_checkpoint(run)

    types_after = local_types(run.model)
    objects_after = wrapper_objects(run.model)
    checksums_after = shard_checksums(run.model)
    PREPARE_CALLS.reset()
    losses_after = evaluate(run)
    return {
        "local_types_before_load": types_before,
        "local_types_after_load": types_after,
        "wrapper_objects_preserved": all(objects_after[fqn] is objects_before[fqn] for fqn in objects_before),
        "shard_storage_preserved": shard_pointers(run.model) == pointers_before,
        "shard_values_changed": all(checksums_after[fqn] != checksums_before[fqn] for fqn in checksums_before),
        "prepare_calls_after_load": PREPARE_CALLS.count,
        "optimizer_state_types": optimizer_state_types(run.model, run.optimizer),
        "eval_losses_before_load": losses_before,
        "eval_losses": losses_after,
    }


def run_roundtrip(run: Run) -> dict[str, Any]:
    training_step(run, step=0)
    save_checkpoint(run)
    saved_shards = shard_clones(run.model)
    saved_prepared, unsharded_types = clone_prepared_tensors(run.model)

    training_step(run, step=1)
    perturbed = not shards_equal(run.model, saved_shards)

    load_checkpoint(run)
    restored_shards = shards_equal(run.model, saved_shards)
    reloaded_prepared, _ = clone_prepared_tensors(run.model)
    return {
        "unsharded_types": unsharded_types,
        "prepared_keys": {fqn: sorted(tensors) for fqn, tensors in saved_prepared.items()},
        "perturbed_before_load": perturbed,
        "master_weights_restored": restored_shards,
        "prepared_restored": prepared_equal(saved_prepared, reloaded_prepared),
        "local_types_after_load": local_types(run.model),
        "optimizer_state_types": optimizer_state_types(run.model, run.optimizer),
    }


PHASE_RUNNERS = {"save": run_save, "load": run_load, "roundtrip": run_roundtrip}


def main() -> None:
    args = parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_float32_matmul_precision("high")
    setup_torch_distributed()
    world = get_world()
    device = torch.device("cuda", world.local_rank)
    torch.manual_seed(args.seed)

    parallel_dims = ParallelDims(
        dp_replicate=1, dp_shard=world.world_size, cp=1, pp=1, ep=args.ep, world_size=world.world_size
    )
    spec = MiniModelSpec(num_hidden_layers=args.layers, wrap=args.wrap, num_experts=args.num_experts, seed=args.seed)
    model = build_mini_model(
        spec,
        parallel_dims,
        expert_reshard_after_forward=True,
        activation_checkpointing=False,
        install_prepared=True,
        fusions=True,
        force_balanced_routing=True,
        compile_layers=False,
        device=device,
    )
    optimizer_config = AdamWConfig(lr=args.lr)
    optimizer, _ = setup_optimizer(optimizer_config, list(model.named_parameters()), parallel_dims)
    run = Run(args, model, optimizer, optimizer_config, parallel_dims, device)

    report = PHASE_RUNNERS[args.phase](run)
    report.update(phase=args.phase, wrap=args.wrap, ep=args.ep, seed=args.seed, world_size=world.world_size)
    if world.is_master:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
