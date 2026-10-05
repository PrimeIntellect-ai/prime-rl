"""Compare redundant and per-rank vision encoding on a pretrained Qwen3.5 VLM.

Run with torchrun on at least two GPUs. Uses PrimeRL's checkpoint loader,
FSDP, Ulysses CP, optional activation checkpointing, and next-token loss. All vision
gradients and samples of every language-model gradient are compared before
an SGD update. PyTorch deterministic operations stabilize forward comparisons;
custom-kernel backward variation is measured by repeating the baseline.
Benchmarks restore ordinary PyTorch operations. JSON results include warmed-up
vision and forward/backward timings; per-rank vision time includes all-gather.

Example:
    uv run torchrun --standalone --nproc_per_node=4 tools/validate_vlm_cp_encoding.py \
        Qwen/Qwen3.5-35B-A3B docs/assets/architecture.png --train-vision \
        --no-ac --output /tmp/vlm-cp-validation.json
"""

import argparse
import json
import math
import os
import statistics
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.distributed.tensor import DTensor
from transformers import AutoProcessor

from prime_rl.configs.trainer import ModelConfig
from prime_rl.trainer.model import setup_model
from prime_rl.trainer.models.qwen3_5 import modeling_qwen3_5 as vlm_code
from prime_rl.trainer.parallel_dims import ParallelDims
from prime_rl.utils.cp import setup_context_parallel, shard_for_cp
from prime_rl.utils.logger import setup_logger


def local_tensor(tensor):
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def prepare_batch(processor, paths, sizes, world):
    images = []
    for path, size in zip(paths, sizes, strict=True):
        with Image.open(path) as image:
            images.append(image.convert("RGB").resize((size, size)))
    messages = [
        {
            "role": "user",
            "content": [*({"type": "image"} for _ in images), {"type": "text", "text": "Describe these diagrams."}],
        },
        {
            "role": "assistant",
            "content": "These diagrams illustrate the components and stages of a machine learning workflow.",
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    batch = processor(text=[text], images=images, return_tensors="pt", return_mm_token_type_ids=True)
    ids = batch["input_ids"]
    length = ids.shape[1]
    # FLA CP and Ulysses require evenly sized shards.
    padded_length = math.ceil(length / (world * 64)) * world * 64
    pad = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    ids = F.pad(ids, (0, padded_length - length), value=pad)
    types = F.pad(batch["mm_token_type_ids"], (0, padded_length - length))
    labels = ids.roll(-1, dims=1)
    labels[:, length - 1 :] = -100
    labels[types.roll(-1, dims=1) != 0] = -100
    return {
        "input_ids": ids.cuda(),
        "mm_token_type_ids": types.cuda(),
        "pixel_values": batch["pixel_values"].cuda(),
        "image_grid_thw": batch["image_grid_thw"].cuda(),
        "seq_lens": torch.tensor([padded_length], device="cuda"),
    }, labels.cuda()


def gradient_snapshot(model):
    gradients = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        assert parameter.grad is not None, f"Missing gradient: {name}"
        gradient = local_tensor(parameter.grad).detach().flatten()
        if ".visual." not in name:
            gradient = gradient[:: max(1, math.ceil(gradient.numel() / 4096))]
        assert torch.isfinite(gradient).all(), f"Non-finite gradient: {name}"
        gradients[name] = gradient.float().cpu()
    return gradients


def gradient_errors(model, reference):
    errors = torch.zeros(2, 3, dtype=torch.float64, device="cuda")
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        assert parameter.grad is not None, f"Missing gradient: {name}"
        gradient = local_tensor(parameter.grad).detach().flatten()
        vision = ".visual." in name
        if not vision:
            gradient = gradient[:: max(1, math.ceil(gradient.numel() / 4096))]
        actual = gradient.float()
        expected = reference[name].to("cuda")
        assert torch.isfinite(actual).all(), f"Non-finite gradient: {name}"
        row = 0 if vision else 1
        errors[row, 0] += (actual - expected).double().square().sum()
        errors[row, 1] += expected.double().square().sum()
        errors[row, 2] += actual.double().square().sum()
    dist.all_reduce(errors)
    result = {}
    for name, row in zip(("vision", "language_samples"), errors.cpu().tolist(), strict=True):
        if row[1] == 0:
            assert row[0] == 0, f"Unexpected {name} gradient"
            result[name] = 0.0
            continue
        relative_error = math.sqrt(row[0] / row[1])
        result[name] = relative_error
    return result


def activation_errors(actual, expected):
    errors = torch.tensor(
        [float((actual[key] - expected[key]).abs().max()) for key in ("inputs", "hidden")], device="cuda"
    )
    dist.all_reduce(errors, op=dist.ReduceOp.MAX)
    return dict(zip(("inputs", "hidden"), errors.tolist(), strict=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Pretrained Hugging Face model ID or snapshot path")
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--train-vision", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--no-ac", action="store_true", help="Disable language-model activation checkpointing")
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    # Stabilize MoE scatter reductions for numerical comparisons. Benchmarks
    # below restore the ordinary kernels used during training.
    # warn_only permits the router's histc of integer expert IDs; these small
    # counts are exact. Operators with deterministic variants still use them.
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(minutes=15))
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world >= 2
    setup_logger("info")
    torch.manual_seed(1234)
    config = ModelConfig(
        name=args.model,
        cp=world,
        cp_style="ulysses",
        ep=1,
        compile={} if args.compile else None,
        ac=None if args.no_ac else {},
        ac_offloading=None,
        vlm={
            "vision_encoder_attr": "model.visual",
            "language_model_attr": "model.language_model",
            "freeze_vision_encoder": not args.train_vision,
        },
    )
    assert (config.ac is None) == args.no_ac
    if rank == 0:
        print(
            f"SETTINGS: cp={world}, train_vision={args.train_vision}, compile={config.compile}, ac={config.ac}",
            flush=True,
        )
    dims = ParallelDims(dp_replicate=1, dp_shard=1, cp=world, pp=1, ep=1, world_size=world)
    model = setup_model(config, dims)
    setup_context_parallel(model, config, dims)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=Path(args.model).exists())
    optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=1e-5, foreach=False)
    vlm_code._TIME_VISION = True
    vlm_code._TIME_VISION_EVERY = 20
    records = {
        "model": args.model,
        "cp": world,
        "train_vision": args.train_vision,
        "compile": args.compile,
        "activation_checkpointing": config.ac is not None,
        "correctness_deterministic_algorithms": True,
        "benchmark_deterministic_algorithms": False,
        "optimization_dtype": config.optimization_dtype,
        "reduce_dtype": config.reduce_dtype,
        "cases": [],
    }

    def run(batch, labels, enabled, capture=False):
        vlm_code._PER_RANK_ENCODE = enabled
        model.zero_grad(set_to_none=True)
        captured = {}
        calls = []
        hooks = []
        if capture:
            hooks.append(
                model.model.language_model.norm.register_forward_hook(
                    lambda module, inputs, output: captured.update(hidden=output.detach().float().cpu())
                )
            )
            hooks.append(
                model.model.language_model.register_forward_pre_hook(
                    lambda module, inputs, kwargs: captured.update(
                        inputs=kwargs["inputs_embeds"].detach().float().cpu()
                    ),
                    with_kwargs=True,
                )
            )
            hooks.append(
                model.model.visual.register_forward_hook(
                    lambda module, inputs, output: calls.append(inputs[0].shape[0])
                )
            )
        local_labels = shard_for_cp(labels, cp_rank=rank, cp_world_size=world)
        token_count = (labels != -100).sum()
        dist.barrier()
        torch.cuda.synchronize()
        vision_before = vlm_code._vision_timer.total_s
        start = time.perf_counter()
        output = model(**batch, labels=local_labels, seq_lens_are_pre_shard=True)
        loss_sum = output["loss"]
        (loss_sum * world / token_count).backward()
        torch.cuda.synchronize()
        timings = torch.tensor(
            [time.perf_counter() - start, vlm_code._vision_timer.total_s - vision_before], device="cuda"
        )
        dist.all_reduce(timings, op=dist.ReduceOp.MAX)
        loss = loss_sum.detach().float()
        dist.all_reduce(loss)
        loss = float(loss / token_count)
        assert math.isfinite(loss)
        for hook in hooks:
            hook.remove()
        if capture:
            assert len(calls) == 1, f"Expected one vision forward on rank {rank}: {calls}"
        return loss, captured, timings.cpu().tolist(), calls

    repeated_paths = [args.images[i % len(args.images)] for i in range(world + 1)]
    cases = [
        ("one_image_empty_ranks", repeated_paths[:1], [args.image_size]),
        ("balanced_images", repeated_paths[:world], [args.image_size] * world),
        ("uneven_images", repeated_paths, [max(64, args.image_size // (1 + i % 3)) for i in range(world + 1)]),
    ]
    benchmark_batch = None
    for case, paths, sizes in cases:
        batch, labels = prepare_batch(processor, paths, sizes, world)
        if rank == 0:
            print(f"VALIDATING {case}: grid={batch['image_grid_thw'].tolist()}, sequence={labels.shape[1]}", flush=True)
        baseline_loss, baseline_hidden, _, _ = run(batch, labels, False, capture=True)
        gradients = gradient_snapshot(model)
        repeat_loss, repeat_hidden, _, _ = run(batch, labels, False, capture=True)
        repeat_errors = activation_errors(repeat_hidden, baseline_hidden)
        repeat_gradient_errors = gradient_errors(model, gradients)
        if rank == 0:
            print(
                f"BASELINE REPEAT: loss={baseline_loss}/{repeat_loss}, "
                f"activations={repeat_errors}, gradients={repeat_gradient_errors}",
                flush=True,
            )
        assert baseline_loss == repeat_loss, "Repeated baseline forward must be deterministic"
        assert all(error == 0 for error in repeat_errors.values())
        assert all(error < 0.05 for error in repeat_gradient_errors.values()), repeat_gradient_errors
        baseline_loss, baseline_hidden = repeat_loss, repeat_hidden
        gradients = gradient_snapshot(model)
        loss, hidden, _, calls = run(batch, labels, True, capture=True)
        errors = activation_errors(hidden, baseline_hidden)
        grad_errors = gradient_errors(model, gradients)
        if rank == 0:
            print(
                f"PER-RANK COMPARISON: loss={baseline_loss}/{loss}, activations={errors}, gradients={grad_errors}",
                flush=True,
            )
        assert abs(loss - baseline_loss) <= max(1e-4, abs(baseline_loss) * 1e-4), (loss, baseline_loss)
        torch.testing.assert_close(hidden["inputs"], baseline_hidden["inputs"], rtol=0.01, atol=0.01)
        torch.testing.assert_close(hidden["hidden"], baseline_hidden["hidden"], rtol=0.01, atol=0.01)
        # Custom attention backward kernels can remain nondeterministic. Bound
        # differences against a measured baseline repeat, with a BF16 floor.
        gradient_tolerances = {name: max(0.02, 2 * noise) for name, noise in repeat_gradient_errors.items()}
        for name, error in grad_errors.items():
            assert error <= gradient_tolerances[name], (name, error, gradient_tolerances[name])
        del gradients
        counts = [None] * world
        dist.all_gather_object(counts, calls[0])
        record = {
            "case": case,
            "grid": batch["image_grid_thw"].tolist(),
            "sequence_length": labels.shape[1],
            "baseline_loss": baseline_loss,
            "per_rank_loss": loss,
            "input_max_abs_error": errors["inputs"],
            "hidden_max_abs_error": errors["hidden"],
            "baseline_repeat_gradient_relative_l2_error": repeat_gradient_errors,
            "gradient_relative_l2_error": grad_errors,
            "gradient_relative_l2_tolerance": gradient_tolerances,
            "encoded_patches_by_rank": counts,
        }
        # Take an actual optimizer step after comparing the same checkpoint weights.
        parameter = next(model.model.language_model.layers[-1].parameters())
        before = local_tensor(parameter).detach().clone()
        optimizer.step()
        changed = int(torch.count_nonzero(local_tensor(parameter).detach() - before))
        record["updated_parameter_elements_local"] = changed
        assert changed > 0, "SGD failed to update the sampled language parameter"
        records["cases"].append(record)
        if rank == 0:
            print(json.dumps(record), flush=True)
        if case == "balanced_images":
            benchmark_batch = (batch, labels)

    torch.use_deterministic_algorithms(False)
    samples = {False: [], True: []}
    for iteration in range(args.iterations + 2):
        for enabled in (False, True) if iteration % 2 == 0 else (True, False):
            _, _, timing, _ = run(*benchmark_batch, enabled)
            if iteration >= 2:
                samples[enabled].append(timing)
    records["timings"] = {}
    for enabled, values in samples.items():
        records["timings"]["per_rank" if enabled else "redundant"] = {
            "forward_backward_median_s": statistics.median(value[0] for value in values),
            "vision_median_s": statistics.median(value[1] for value in values),
            "samples_s": values,
        }
    records["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 1024**3
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(records, indent=2) + "\n")
        print(json.dumps(records["timings"]), flush=True)
        print(f"PASS: pretrained VLM validation written to {args.output}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
