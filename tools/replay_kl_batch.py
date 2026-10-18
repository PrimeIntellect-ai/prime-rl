"""Replay one captured micro batch through the actual trainer, without an optimizer.

uv run torchrun --nproc-per-node 8 tools/replay_kl_batch.py CAPTURE.pt [CHECKPOINT/trainer]
Use the original world size and model settings; every DP replica replays the same batch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("checkpoint", type=Path, nargs="?")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    import torch
    import torch.distributed as dist
    from torch.distributed.checkpoint import load as dcp_load

    import prime_rl._compat  # noqa: F401
    from prime_rl.configs.trainer import ModelConfig
    from prime_rl.multimodal.base import ForwardPolicy
    from prime_rl.trainer.ckpt import AppState
    from prime_rl.trainer.model import forward, setup_model
    from prime_rl.trainer.parallel_dims import get_parallel_dims, resolve_ep
    from prime_rl.trainer.rl.kl_debug import gap_stats, mismatch_mask
    from prime_rl.trainer.rl.loss import (
        selective_log_softmax,
        selective_log_softmax_with_sampling_mask,
        shift_tensor_left,
        shift_tensor_right,
    )
    from prime_rl.trainer.utils import get_dcp_group, setup_torch_distributed
    from prime_rl.trainer.world import get_world
    from prime_rl.utils.cp import gather_for_cp, setup_context_parallel, setup_cp_params, shard_for_cp
    from prime_rl.utils.logger import setup_logger

    capture = torch.load(args.capture, map_location="cpu", weights_only=True)
    manifest = capture["manifest"]
    world = get_world()
    if world.world_size != manifest["world_size"]:
        raise ValueError(f"Replay needs the original world size {manifest['world_size']}, got {world.world_size}")
    config = ModelConfig.model_validate(manifest["model_config"])
    if config.lora is not None or config.offload == "full":
        raise ValueError("This replay tool supports FFT without full optimizer offload")
    version = capture["record"]["trainer_policy_version"]
    if capture["record"]["frozen_weights"]:
        version = manifest["initial_policy_version"]
    if args.checkpoint is None and version != 0:
        raise ValueError(
            f"Policy v{version} requires its retained checkpoint; base weights would be a different experiment"
        )
    if args.checkpoint is not None:
        checkpoint_manifest = json.loads((args.checkpoint.parent / "manifest.json").read_text())
        if checkpoint_manifest["policy_version"] != capture["record"]["trainer_policy_version"]:
            raise ValueError("Checkpoint policy version does not match this capture")
        if checkpoint_manifest["model_config"] != manifest["model_config"]:
            raise ValueError("Checkpoint model configuration does not match this capture")

    setup_logger("info")
    setup_torch_distributed()
    torch.set_float32_matmul_precision(manifest["matmul_precision"])
    resolve_ep(config)
    parallel = get_parallel_dims(config)
    model = setup_model(config, parallel, loading_from_checkpoint_later=args.checkpoint is not None)
    if parallel.cp_enabled:
        setup_context_parallel(model, config, parallel)
    if args.checkpoint is not None:
        dcp_load({"app": AppState(model, [], None, None)}, checkpoint_id=args.checkpoint, process_group=get_dcp_group())
    model.train()

    batch = capture["batch"]
    ids = batch["input_ids"].cuda()
    positions = batch["position_ids"].cuda()
    labels = shift_tensor_left(ids)
    seq_lens = batch["seq_lens"].cuda()
    temperatures = batch["temperatures"].cuda()
    sampling = batch.get("sampling_mask")
    sampling = shift_tensor_left(sampling.cuda(), pad_value=-1) if sampling is not None else None
    routed = batch.get("routed_experts") if manifest["enable_router_replay"] else None
    routed = routed.cuda() if routed is not None else None
    mm_kwargs = {key: value.cuda() for key, value in (batch.get("mm_kwargs") or {}).items()} or None
    mm_types = batch.get("mm_token_type_ids")
    mm_types = mm_types.cuda() if mm_types is not None else None
    policy = ForwardPolicy(**capture["mm_forward_policy"]) if capture["mm_forward_policy"] else None
    cp_group = None
    if parallel.cp_enabled:
        cp_rank = parallel.world_mesh["cp"].get_local_rank()
        cp_group = parallel.world_mesh["cp"].get_group()
        deferred = bool(policy and policy.defer_context_parallelism)
        if not deferred:
            ids, positions = setup_cp_params(
                ids, positions, cp_rank, parallel.cp, cp_group, seq_lens=seq_lens, cp_style=config.cp_style
            )
            if routed is not None:
                routed = shard_for_cp(routed, cp_rank=cp_rank, cp_world_size=parallel.cp)
        labels = shard_for_cp(labels, cp_rank=cp_rank, cp_world_size=parallel.cp)
        temperatures = shard_for_cp(temperatures, cp_rank=cp_rank, cp_world_size=parallel.cp)
        if sampling is not None:
            sampling = shard_for_cp(sampling, cp_rank=cp_rank, cp_world_size=parallel.cp)
    with torch.no_grad():
        output = forward(
            model,
            ids,
            positions,
            labels=labels,
            temperature=temperatures,
            seq_lens=seq_lens,
            seq_lens_are_pre_shard=parallel.cp_enabled,
            routed_experts=routed,
            sampling_mask=sampling,
            mm_kwargs=mm_kwargs,
            mm_forward_policy=policy,
            mm_token_type_ids=mm_types,
        )
        logprobs = output.get("logprobs")
        if logprobs is None:
            scaled = output["logits"] / temperatures.unsqueeze(-1)
            logprobs = (
                selective_log_softmax_with_sampling_mask(scaled, labels, sampling)
                if sampling is not None
                else selective_log_softmax(scaled, labels)
            )
        if parallel.cp_enabled:
            logprobs = gather_for_cp(logprobs, cp_group)
        logprobs = shift_tensor_right(logprobs, pad_value=0).cpu().reshape(-1)
    mask = mismatch_mask(batch)
    threshold = manifest["kl_debug"]["abs_logprob_threshold"]
    report = {
        "capture": str(args.capture),
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "policy_version": version,
        "grad_enabled": False,
        "replay_minus_captured_trainer": gap_stats(
            logprobs[mask], capture["trainer_logprobs"].reshape(-1)[mask], threshold
        ),
        "replay_minus_rollout": gap_stats(logprobs[mask], batch["inference_logprobs"].reshape(-1)[mask], threshold),
    }
    if world.is_master:
        output_path = args.output or args.capture.with_suffix(".replay.json")
        output_path.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
