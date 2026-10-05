"""Benchmark a saved multimodal replay with identical weights and inputs.

The default loss uses the trainer's fused logprob/entropy head and backward
through mean sampled-token logprob. This measures a microbatch, not an optimizer
step or rollout generation. Stage profiling is separate from throughput timing;
its optional pre-gather barrier measures rank skew plus barrier overhead.
"""

import argparse
import ast
import hashlib
import inspect
import json
import math
import os
import socket
import statistics
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoProcessor
from validate_vlm_cp_encoding import activation_errors, prepare_batch

from prime_rl.configs.trainer import ModelConfig
from prime_rl.trainer.model import setup_model
from prime_rl.trainer.models.qwen3_5 import modeling_qwen3_5 as vlm_code
from prime_rl.trainer.parallel_dims import ParallelDims
from prime_rl.utils.act_offloading import maybe_activation_offloading
from prime_rl.utils.cp import gather_for_cp, setup_context_parallel, shard_for_cp
from prime_rl.utils.logger import setup_logger


def load_replay(path, world):
    with np.load(path, allow_pickle=False) as arrays:
        ids = torch.from_numpy(arrays["token_ids"][:-1].copy())[None]
        labels = torch.from_numpy(arrays["token_ids"][1:].copy())[None]
        sampled = torch.from_numpy(arrays["sampled_mask"][1:].copy())[None]
        types = torch.from_numpy(arrays["mm_token_type_ids"][:-1].copy())[None]
        length = ids.shape[1]
        padded = math.ceil(length / (world * 64)) * world * 64
        batch = {
            "input_ids": F.pad(ids, (0, padded - length)).cuda(),
            "mm_token_type_ids": F.pad(types, (0, padded - length)).cuda(),
            "pixel_values": torch.from_numpy(arrays["pixel_values"].copy()).cuda(),
            "image_grid_thw": torch.from_numpy(arrays["image_grid_thw"].copy()).cuda(),
            "seq_lens": torch.tensor([padded], device="cuda"),
        }
        labels = F.pad(labels, (0, padded - length)).cuda()
        sampled = F.pad(sampled, (0, padded - length)).cuda()
    return batch, labels, sampled, length


class VisionStages:
    """Insert timing markers into the actual helper source without changing its operations."""

    def __init__(self, helper, group, align_gather):
        self.group = group
        self.align_gather = align_gather
        self.events = {}
        original = inspect.unwrap(helper)
        self.source = inspect.getsource(original)
        tree = ast.parse(self.source)
        function = tree.body[0]
        function.decorator_list = []
        outer = self

        class MarkStages(ast.NodeTransformer):
            def visit_Assign(self, node):
                names = [target.id for target in node.targets if isinstance(target, ast.Name)]
                stage = {"device": "start", "local": "encode_start", "hidden": "encode_end"}
                if len(names) == 1 and names[0] in stage:
                    return [ast.parse(f"_stage_mark('{stage[names[0]]}')").body[0], node]
                return node

            def visit_If(self, node):
                node = self.generic_visit(node)
                if isinstance(node.test, ast.Name) and node.test.id == "requires_grad":
                    markers = ast.parse("_stage_mark('padding_end')\n_stage_align()\n_stage_mark('gather_start')").body
                    return [*markers, node, ast.parse("_stage_mark('gather_end')").body[0]]
                return node

        tree = ast.fix_missing_locations(MarkStages().visit(tree))
        namespace = dict(original.__globals__, _stage_mark=outer.mark, _stage_align=outer.align)
        exec(compile(tree, "<instrumented CP vision helper>", "exec"), namespace)
        self.encode = torch.compiler.disable(namespace[original.__name__])

    def mark(self, label):
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self.events[label] = event

    def align(self):
        if self.align_gather:
            dist.barrier(group=self.group)

    def __call__(self, *args, **kwargs):
        self.events = {}
        out = self.encode(*args, **kwargs)
        self.mark("end")
        return out

    def milliseconds(self):
        pairs = {
            "partition_and_select": ("start", "encode_start"),
            "local_encode_including_fsdp": ("encode_start", "encode_end"),
            "fsdp_and_input_selection": ("encode_start", "vision_body_start"),
            "vision_body": ("vision_body_start", "vision_body_end"),
            "padding_and_token_counts": ("encode_end", "padding_end"),
            "wait_and_barrier": ("padding_end", "gather_start"),
            "all_gather": ("gather_start", "gather_end"),
            "restore_image_order": ("gather_end", "end"),
            "total": ("start", "end"),
        }
        return {key: self.events[a].elapsed_time(self.events[b]) for key, (a, b) in pairs.items()}


def collect_ranks(value):
    values = [None] * dist.get_world_size()
    dist.all_gather_object(values, value)
    return values


def compile_evidence(requested):
    counters = {key: dict(value) for key, value in torch._dynamo.utils.counters.items()}
    graphs = counters.get("stats", {}).get("unique_graphs", 0)
    return {"requested": requested, "captured_graphs": graphs, "passed": graphs > 0 if requested else None}, counters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("output", type=Path)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--images", nargs="+", type=Path, default=[Path("docs/assets/architecture.png")])
    parser.add_argument("--image-count", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--cp", type=int, default=4)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--capture-dynamic-ops", action="store_true")
    parser.add_argument("--no-ac", action="store_true")
    parser.add_argument("--offload-activations", action="store_true")
    parser.add_argument("--train-vision", action="store_true")
    parser.add_argument("--router-dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--loss", choices=["rl", "ce"], default="rl")
    parser.add_argument("--matmul-precision", choices=["high", "highest"], default="high")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--profile-iterations", type=int, default=5)
    parser.add_argument("--capture-images", action="store_true")
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument("--skip-backward", action="store_true", help="Keep training-mode autograd but skip backward")
    parser.add_argument("--comparison-only", action="store_true")
    args = parser.parse_args()
    assert args.iterations > 0 and args.cp > 0
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(minutes=20))
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world % args.cp == 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    setup_logger("info")
    torch.manual_seed(1234)
    torch.set_float32_matmul_precision(args.matmul_precision)
    if args.capture_dynamic_ops:
        torch._dynamo.config.capture_dynamic_output_shape_ops = True
    config = ModelConfig(
        name=args.model,
        cp=args.cp,
        cp_style="ulysses",
        ep=1,
        moe_router_dtype=args.router_dtype,
        compile={} if args.compile else None,
        ac=None if args.no_ac else {},
        ac_offloading={} if args.offload_activations else None,
        vlm={
            "vision_encoder_attr": "model.visual",
            "language_model_attr": "model.language_model",
            "freeze_vision_encoder": not args.train_vision,
        },
    )
    assert (config.ac is None) == args.no_ac
    dims = ParallelDims(dp_replicate=1, dp_shard=world // args.cp, cp=args.cp, pp=1, ep=1, world_size=world)
    model = setup_model(config, dims)
    compile_setup = {
        "compiled_layers": sum(layer._compiled_call_impl is not None for layer in model.model.language_model.layers),
        "dynamo_disable": str(torch._dynamo.config.disable),
        "suppress_errors": torch._dynamo.config.suppress_errors,
        "capture_dynamic_output_shape_ops": torch._dynamo.config.capture_dynamic_output_shape_ops,
    }
    if rank == 0:
        print("COMPILE_SETUP", json.dumps(compile_setup), flush=True)
    if args.cp > 1:
        setup_context_parallel(model, config, dims)
    context = model.model.cp_context
    if args.fixture:
        batch, labels, sampled, unpadded_length = load_replay(args.fixture, world)
    else:
        processor = AutoProcessor.from_pretrained(args.model)
        paths = [args.images[i % len(args.images)] for i in range(args.image_count)]
        batch, ce_labels = prepare_batch(processor, paths, [args.image_size] * args.image_count, world)
        sampled = ce_labels != -100
        labels = ce_labels.masked_fill(~sampled, 0)
        unpadded_length = batch["input_ids"].shape[1]
    if args.loss == "ce":
        labels = labels.masked_fill(~sampled, -100)
    local_labels = shard_for_cp(labels, context.cp_rank, args.cp)
    image_mask = batch["input_ids"] == model.config.image_token_id
    local_image_mask = shard_for_cp(image_mask, context.cp_rank, args.cp)
    grid = batch["image_grid_thw"].tolist()
    merge = model.config.vision_config.spatial_merge_size
    patches = [math.prod(row) for row in grid]
    buckets = vlm_code._partition_images_for_cp(batch["image_grid_thw"], args.cp)
    records = {
        "config": config.model_dump(mode="json"),
        "compile_setup": compile_setup,
        "hostnames": collect_ranks(socket.gethostname()),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "fla_disable_tensor_cache": os.environ.get("FLA_DISABLE_TENSOR_CACHE", "0"),
        "fixture": str(args.fixture) if args.fixture else "resized repository diagrams",
        "loss": args.loss,
        "forward_only": args.forward_only,
        "skip_backward": args.skip_backward,
        "grid_thw": grid,
        "image_count": len(grid),
        "patches_per_image": patches,
        "vision_tokens_per_image": [p // merge**2 for p in patches],
        "total_vision_tokens": sum(patches) // merge**2,
        "image_indices_per_cp_rank": buckets,
        "sequence_length": labels.shape[1],
        "unpadded_sequence_length": unpadded_length,
        "loss_tokens": int(sampled.sum()),
        "input_sha256": hashlib.sha256(batch["input_ids"].cpu().numpy().tobytes()).hexdigest(),
        "correctness_pytorch_determinism": "warn_only",
        "benchmark_pytorch_determinism": False,
    }
    if rank == 0:
        print(
            "FIXTURE",
            json.dumps({k: records[k] for k in ("image_count", "sequence_length", "loss_tokens")}),
            flush=True,
        )
    vlm_code._TIME_VISION = True
    vlm_code._TIME_VISION_EVERY = 20

    def run(enabled, capture=False):
        vlm_code._PER_RANK_ENCODE = enabled
        model.zero_grad(set_to_none=True)
        captured = {}
        hooks = []
        if capture:
            hooks = [
                model.model.language_model.register_forward_pre_hook(
                    lambda m, a, kw: captured.update(inputs=kw["inputs_embeds"].detach().float().cpu()),
                    with_kwargs=True,
                ),
                model.model.language_model.norm.register_forward_hook(
                    lambda m, a, out: captured.update(hidden=out.detach().float().cpu())
                ),
            ]
        dist.barrier()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        vision_before = vlm_code._vision_timer.total_s
        start = time.perf_counter()
        with torch.set_grad_enabled(not args.forward_only), maybe_activation_offloading(config.ac_offloading):
            output = model(
                **batch,
                labels=local_labels,
                temperature=torch.ones_like(local_labels, dtype=torch.float32) if args.loss == "rl" else None,
                seq_lens_are_pre_shard=True,
            )
            if args.loss == "rl":
                logprobs = output["logprobs"]
                if args.cp > 1:
                    logprobs = gather_for_cp(logprobs, context.cp_group)
                loss = -(logprobs * sampled).sum() / sampled.sum()
            else:
                loss = output["loss"] * args.cp / sampled.sum()
        if not (args.forward_only or args.skip_backward):
            loss.backward()
        torch.cuda.synchronize()
        row = {
            "forward_backward_ms": (time.perf_counter() - start) * 1000,
            "vision_ms": (vlm_code._vision_timer.total_s - vision_before) * 1000,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        }
        value = loss.detach().float()
        dist.all_reduce(value)
        value = float(value / world)
        assert math.isfinite(value)
        for hook in hooks:
            hook.remove()
        return value, captured, row

    torch.use_deterministic_algorithms(True, warn_only=True)
    reference_loss, reference, _ = run(False, capture=True)
    if rank == 0:
        print("DYNAMO_COUNTERS", str(torch._dynamo.utils.counters), flush=True)
        print(
            "BASELINE_FORWARD_PASS" if args.forward_only or args.skip_backward else "BASELINE_BACKWARD_PASS",
            reference_loss,
            flush=True,
        )
    loss, actual, _ = run(True, capture=True)
    errors = activation_errors(actual, reference)
    records["comparison"] = {
        "baseline_loss": reference_loss,
        "per_rank_loss": loss,
        "loss_abs_error": abs(loss - reference_loss),
        "activation_max_abs_errors": errors,
        "exact": loss == reference_loss and all(error == 0 for error in errors.values()),
        "same_weights": True,
        "same_inputs": True,
    }
    if rank == 0:
        print("COMPARISON", json.dumps(records["comparison"]), flush=True)
    assert abs(loss - reference_loss) <= max(1e-4, abs(reference_loss) * 1e-4)
    for key in actual:
        torch.testing.assert_close(actual[key], reference[key], rtol=0.01, atol=0.01)
    if args.capture_images and (args.cp > 1 or rank == 0):
        torch.save(
            {
                "positions": local_image_mask[0].nonzero().flatten().cpu() + context.cp_rank * local_labels.shape[1],
                "inputs": actual["inputs"][local_image_mask.cpu()],
                "hidden": actual["hidden"][local_image_mask.cpu()],
            },
            args.output.with_suffix(f".images-rank{rank}.pt"),
        )
    del actual, reference
    if args.comparison_only:
        records["compile_validation"], records["dynamo_counters"] = compile_evidence(args.compile)
        if rank == 0:
            records["compile_stats"] = dict(torch._dynamo.utils.counters["stats"])
            args.output.write_text(json.dumps(records, indent=2) + "\n")
            print(f"RESULT {args.output}", flush=True)
        dist.destroy_process_group()
        assert not args.compile or records["compile_validation"]["passed"], "Compile requested but no graphs captured"
        return
    model.zero_grad(set_to_none=True)
    torch.use_deterministic_algorithms(False)
    samples = {False: [], True: []}
    for iteration in range(args.iterations + 3):
        for enabled in (False, True) if iteration % 2 == 0 else (True, False):
            _, _, row = run(enabled)
            if iteration >= 3:
                samples[enabled].append(row)
    records["timings"] = {}
    for enabled, values in samples.items():
        rank_values = collect_ranks(values)
        records["timings"]["on" if enabled else "off"] = {
            "per_rank_samples": rank_values,
            "forward_backward_ms": statistics.median(
                max(rows[i]["forward_backward_ms"] for rows in rank_values) for i in range(args.iterations)
            ),
            "vision_ms": statistics.median(
                max(rows[i]["vision_ms"] for rows in rank_values) for i in range(args.iterations)
            ),
            "peak_allocated_gib_per_rank": [max(row["peak_allocated_gib"] for row in rows) for rows in rank_values],
            "peak_reserved_gib_per_rank": [max(row["peak_reserved_gib"] for row in rows) for rows in rank_values],
        }
    if rank == 0:
        records["compile_stats"] = dict(torch._dynamo.utils.counters["stats"])
        records["compile_validation"], records["dynamo_counters"] = compile_evidence(args.compile)
        args.output.write_text(json.dumps(records, indent=2) + "\n")
        print(
            "TIMING",
            json.dumps(
                {k: {a: b for a, b in v.items() if a != "per_rank_samples"} for k, v in records["timings"].items()}
            ),
            flush=True,
        )
    if args.cp > 1 and args.profile_iterations:
        original = vlm_code._encode_images_per_cp_rank
        records["stage_profiles"] = {}
        for align in (False, True):
            stages = VisionStages(original, context.cp_group, align)
            vlm_code._encode_images_per_cp_rank = stages
            hooks = [
                model.model.visual.patch_embed.register_forward_pre_hook(lambda m, a: stages.mark("vision_body_start")),
                model.model.visual.merger.register_forward_hook(lambda m, a, out: stages.mark("vision_body_end")),
            ]
            rows = []
            for _ in range(args.profile_iterations):
                run(True)
                rows.append(stages.milliseconds())
            for hook in hooks:
                hook.remove()
            rank_rows = collect_ranks(rows)
            records["stage_profiles"]["aligned" if align else "natural"] = {
                "per_rank_median_ms": [
                    {key: statistics.median(row[key] for row in values) for key in values[0]} for values in rank_rows
                ],
                "per_rank_samples_ms": rank_rows,
                "helper_source_sha256": hashlib.sha256(stages.source.encode()).hexdigest(),
            }
        vlm_code._encode_images_per_cp_rank = original
    if rank == 0:
        args.output.write_text(json.dumps(records, indent=2) + "\n")
        print(f"RESULT {args.output}", flush=True)
    compile_validation, _ = compile_evidence(args.compile)
    dist.destroy_process_group()
    assert not args.compile or compile_validation["passed"], "Compile requested but no graphs captured"


if __name__ == "__main__":
    main()
