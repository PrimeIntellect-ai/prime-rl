#!/usr/bin/env python3
"""Exercise Prime's layerwise reload path with vLLM online NVFP4 weights."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import time
from functools import partial
from pathlib import Path
from typing import Any, Iterable

DEFAULT_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
EXPERT_WEIGHT_SUFFIXES = (
    ".gate_proj.weight",
    ".up_proj.weight",
    ".down_proj.weight",
)
KERNEL_WEIGHT_NAMES = {"w13_weight", "w2_weight"}
KERNEL_DERIVED_NAMES = {
    "g1_scale_c",
    "gemm1_alpha",
    "gemm1_beta",
    "gemm1_clamp_limit",
}


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def environment_probe() -> dict[str, Any]:
    import torch
    import vllm
    from vllm._custom_ops import scaled_fp4_quant
    from vllm.model_executor.layers.quantization.online.nvfp4 import (
        Nvfp4OnlineMoEMethod,
    )
    from vllm.utils.flashinfer import has_flashinfer_trtllm_fused_moe

    from prime_rl.inference.vllm.worker.weight_transfer import (
        load_weights_checkpoint_layerwise,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    capability = torch.cuda.get_device_capability()
    if capability[0] != 10:
        raise RuntimeError(f"online NVFP4 requires SM100, found SM{capability[0]}{capability[1]}")
    if not has_flashinfer_trtllm_fused_moe():
        raise RuntimeError("FlashInfer TRTLLM fused MoE is unavailable")

    source = torch.arange(256, device="cuda", dtype=torch.bfloat16).reshape(16, 16)
    scale = torch.ones((), device="cuda", dtype=torch.float32)
    packed, block_scales = scaled_fp4_quant(
        source,
        scale,
        is_sf_swizzled_layout=False,
    )
    torch.cuda.synchronize()

    return {
        "torch": _package_version("torch"),
        "torch_cuda": torch.version.cuda,
        "vllm": _package_version("vllm"),
        "vllm_module": str(Path(vllm.__file__).resolve()),
        "flashinfer_python": _package_version("flashinfer-python"),
        "flashinfer_cubin": _package_version("flashinfer-cubin"),
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": list(capability),
        "nvfp4_method": f"{Nvfp4OnlineMoEMethod.__module__}.{Nvfp4OnlineMoEMethod.__name__}",
        "prime_reload_helper": (
            f"{load_weights_checkpoint_layerwise.__module__}.{load_weights_checkpoint_layerwise.__name__}"
        ),
        "scaled_fp4_quant": {
            "packed_shape": list(packed.shape),
            "packed_dtype": str(packed.dtype),
            "block_scale_shape": list(block_scales.shape),
            "block_scale_dtype": str(block_scales.dtype),
        },
        "persistent_cache": {
            "hit": os.environ.get("NVFP4_CACHE_HIT") == "1",
            "key": os.environ.get("NVFP4_CACHE_KEY"),
            "marker": os.environ.get("NVFP4_CACHE_READY_MARKER"),
        },
    }


def _mark_persistent_cache_ready() -> None:
    """Mark the versioned JIT cache reusable after engine warmup succeeds."""
    marker_value = os.environ.get("NVFP4_CACHE_READY_MARKER")
    cache_key = os.environ.get("NVFP4_CACHE_KEY")
    if not marker_value or not cache_key:
        return

    marker = Path(marker_value)
    marker.parent.mkdir(parents=True, exist_ok=True)
    temporary = marker.with_name(f".{marker.name}.{os.getpid()}.tmp")
    temporary.write_text(cache_key + "\n")
    os.replace(temporary, marker)


def _get_worker_model(worker: Any):
    model_runner = worker.model_runner
    get_model = getattr(model_runner, "get_model", None)
    model = get_model() if callable(get_model) else model_runner.model
    return getattr(model, "runnable", model)


def _is_expert_weight(name: str, layer_index: int) -> bool:
    marker = f"model.layers.{layer_index}.mlp.experts."
    return name.startswith(marker) and name.endswith(EXPERT_WEIGHT_SUFFIXES)


def _reload_checkpoint(
    worker: Any,
    *,
    model_path: str,
    mutation_layer: int | None,
) -> dict[str, Any]:
    """Worker-side reload through Prime's production checkpoint-format helper."""
    import torch
    from torch.nn import Module
    from vllm.model_executor.model_loader import DefaultModelLoader, get_model_loader

    from prime_rl.inference.vllm.worker.weight_transfer import (
        load_weights_checkpoint_layerwise,
    )

    model = _get_worker_model(worker)
    if not isinstance(model, Module):
        raise TypeError(f"expected torch Module, got {type(model)!r}")

    model_loader = get_model_loader(worker.load_config)
    if not isinstance(model_loader, DefaultModelLoader):
        raise TypeError(f"expected DefaultModelLoader, got {type(model_loader)!r}")

    source = DefaultModelLoader.Source(
        model_path,
        revision=None,
        prefix="",
        fall_back_to_pt=getattr(model, "fall_back_to_pt_during_load", True),
        allow_patterns_overrides=getattr(model, "allow_patterns_overrides", None),
    )
    source_iterator = model_loader._get_weights_iterator(source)
    stats: dict[str, Any] = {
        "mutated_tensors": 0,
        "mutated_numel": 0,
        "mutated_names": [],
    }

    def transformed_weights() -> Iterable[tuple[str, torch.Tensor]]:
        for name, tensor in source_iterator:
            if mutation_layer is not None and _is_expert_weight(name, mutation_layer):
                stats["mutated_tensors"] += 1
                stats["mutated_numel"] += tensor.numel()
                if len(stats["mutated_names"]) < 12:
                    stats["mutated_names"].append(name)
                # Negation changes every non-zero packed FP4 value without
                # creating the degenerate all-zero activation path.  That
                # keeps this test focused on reload/restore correctness.
                yield name, tensor.neg()
            else:
                yield name, tensor

    started = time.perf_counter()
    load_weights_checkpoint_layerwise(
        model,
        transformed_weights(),
        worker.model_runner.model_config,
        worker.vllm_config,
    )
    torch.cuda.synchronize()
    stats["elapsed_seconds"] = time.perf_counter() - started

    if mutation_layer is not None:
        num_experts = int(worker.model_runner.model_config.hf_text_config.num_experts)
        expected_tensors = num_experts * len(EXPERT_WEIGHT_SUFFIXES)
        if stats["mutated_tensors"] != expected_tensors:
            raise RuntimeError(f"mutated {stats['mutated_tensors']} expert tensors, expected {expected_tensors}")

    return stats


def _sample_tensor(tensor: Any, sample_size: int = 8192) -> dict[str, Any]:
    import torch

    value = tensor.detach()
    if not value.is_contiguous():
        value = value.contiguous()
    raw = value.view(torch.uint8).reshape(-1)
    count = min(sample_size, raw.numel())
    if count:
        indices = torch.linspace(
            0,
            raw.numel() - 1,
            steps=count,
            device=raw.device,
            dtype=torch.float64,
        ).to(torch.long)
        sample = raw[indices].to(device="cpu", dtype=torch.uint8)
        payload = bytes(sample.tolist())
    else:
        payload = b""

    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "nbytes": tensor.numel() * tensor.element_size(),
        "data_ptr": tensor.data_ptr(),
        "sample_bytes": len(payload),
        "sample_nonzero": sum(byte != 0 for byte in payload),
        "sample_sha256": hashlib.sha256(payload).hexdigest(),
        "sample_prefix": list(payload[:16]),
    }


def _probe_model(model: Any, *, layer_index: int) -> dict[str, Any]:
    import torch

    model = getattr(model, "runnable", model)
    marker = f"model.layers.{layer_index}.mlp.experts"
    tensors: dict[str, Any] = {}
    for name, tensor in model.named_parameters():
        leaf = name.rsplit(".", 1)[-1]
        if marker in name and (
            leaf.startswith(("w13_", "w2_")) or leaf in KERNEL_DERIVED_NAMES
        ):
            tensors[name] = tensor
    for name, tensor in model.named_buffers():
        leaf = name.rsplit(".", 1)[-1]
        if marker in name and (
            leaf.startswith(("w13_", "w2_")) or leaf in KERNEL_DERIVED_NAMES
        ):
            tensors.setdefault(name, tensor)

    if not tensors:
        nearby = [name for name, _ in model.named_parameters() if marker in name]
        raise RuntimeError(f"found no NVFP4 kernel tensors under {marker}; nearby={nearby[:20]}")

    quant_methods: dict[str, str] = {}
    kernel_references: dict[str, Any] = {}
    for name, module in model.named_modules():
        if marker not in name:
            continue
        method = getattr(module, "_quant_method", None)
        if method is None:
            method = getattr(module, "quant_method", None)
        if method is not None:
            quant_methods[name] = f"{type(method).__module__}.{type(method).__name__}"
            quant_config = getattr(method, "moe_quant_config", None)
            for attr in (
                "g1_alphas",
                "g2_alphas",
                "a1_gscale",
                "a2_gscale",
                "w1_scale",
                "w2_scale",
            ):
                value = getattr(quant_config, attr, None)
                if isinstance(value, torch.Tensor):
                    kernel_references[f"{name}.moe_quant_config.{attr}"] = _sample_tensor(value)

            moe_kernel = getattr(method, "moe_kernel", None)
            fused_experts = getattr(moe_kernel, "fused_experts", None)
            for attr in KERNEL_DERIVED_NAMES:
                value = getattr(fused_experts, attr, None)
                if isinstance(value, torch.Tensor):
                    kernel_references[f"{name}.moe_kernel.fused_experts.{attr}"] = _sample_tensor(value)

    if not any(value.endswith(".Nvfp4OnlineMoEMethod") for value in quant_methods.values()):
        raise RuntimeError(f"layer {layer_index} is not using Nvfp4OnlineMoEMethod: {quant_methods}")

    return {
        "quant_methods": quant_methods,
        "tensors": {name: _sample_tensor(tensor) for name, tensor in sorted(tensors.items())},
        "kernel_references": dict(sorted(kernel_references.items())),
        "cuda_memory": {
            "allocated": torch.cuda.memory_allocated(),
            "reserved": torch.cuda.memory_reserved(),
            "max_allocated": torch.cuda.max_memory_allocated(),
            "max_reserved": torch.cuda.max_memory_reserved(),
        },
    }


def _generation_signature(llm: Any, prompt: str) -> dict[str, Any]:
    from vllm import SamplingParams

    params = SamplingParams(
        temperature=0.0,
        max_tokens=4,
        logprobs=5,
        seed=0,
    )
    request = llm.generate([prompt], params, use_tqdm=False)[0]
    sequence = request.outputs[0]
    chosen_logprobs: list[float | None] = []
    for token_id, candidates in zip(sequence.token_ids, sequence.logprobs or []):
        candidate = candidates.get(token_id)
        chosen_logprobs.append(None if candidate is None else float(candidate.logprob))
    return {
        "token_ids": list(sequence.token_ids),
        "text": sequence.text,
        "cumulative_logprob": (None if sequence.cumulative_logprob is None else float(sequence.cumulative_logprob)),
        "chosen_logprobs": chosen_logprobs,
    }


def _kernel_leaf(name: str) -> str:
    return name.rsplit(".", 1)[-1]


def _compare_phases(
    baseline: dict[str, Any],
    mutated: dict[str, Any],
    restored: dict[str, Any],
    baseline_generation: dict[str, Any],
    mutated_generation: dict[str, Any],
    restored_generation: dict[str, Any],
) -> dict[str, Any]:
    baseline_tensors = baseline["tensors"]
    mutated_tensors = mutated["tensors"]
    restored_tensors = restored["tensors"]
    if baseline_tensors.keys() != mutated_tensors.keys() or baseline_tensors.keys() != restored_tensors.keys():
        raise RuntimeError("kernel tensor names changed across reload phases")

    pointer_mismatches = [
        name
        for name in baseline_tensors
        if not (
            baseline_tensors[name]["data_ptr"]
            == mutated_tensors[name]["data_ptr"]
            == restored_tensors[name]["data_ptr"]
        )
    ]
    restore_hash_mismatches = [
        name
        for name in baseline_tensors
        if baseline_tensors[name]["sample_sha256"] != restored_tensors[name]["sample_sha256"]
    ]
    changed_kernel_weights = [
        name
        for name in baseline_tensors
        if _kernel_leaf(name) in KERNEL_WEIGHT_NAMES
        and baseline_tensors[name]["sample_sha256"] != mutated_tensors[name]["sample_sha256"]
    ]
    kernel_weights = [name for name in baseline_tensors if _kernel_leaf(name) in KERNEL_WEIGHT_NAMES]

    baseline_logprob = baseline_generation["cumulative_logprob"]
    mutated_logprob = mutated_generation["cumulative_logprob"]
    restored_logprob = restored_generation["cumulative_logprob"]
    mutation_changed_output = baseline_generation["token_ids"] != mutated_generation["token_ids"] or (
        baseline_logprob is not None
        and mutated_logprob is not None
        and not math.isclose(baseline_logprob, mutated_logprob, abs_tol=1e-5)
    )
    restored_output_matches = baseline_generation["token_ids"] == restored_generation["token_ids"]
    if baseline_logprob is not None and restored_logprob is not None:
        restored_output_matches = restored_output_matches and math.isclose(
            baseline_logprob,
            restored_logprob,
            abs_tol=1e-5,
        )

    checks = {
        "kernel_tensor_names_stable": True,
        "kernel_storage_pointers_preserved": not pointer_mismatches,
        "mutated_kernel_weights_changed": bool(kernel_weights) and len(changed_kernel_weights) == len(kernel_weights),
        "restored_kernel_fingerprints_match": not restore_hash_mismatches,
        "mutation_changed_generation": mutation_changed_output,
        "restored_generation_matches": restored_output_matches,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "pointer_mismatches": pointer_mismatches,
        "restore_hash_mismatches": restore_hash_mismatches,
        "changed_kernel_weights": changed_kernel_weights,
    }


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    from vllm import LLM

    probe = environment_probe()
    started = time.perf_counter()
    llm = LLM(
        model=args.model,
        quantization="nvfp4_per_token",
        dtype="bfloat16",
        tensor_parallel_size=1,
        enforce_eager=args.enforce_eager,
        enable_prefix_caching=False,
        max_model_len=args.max_model_len,
        max_num_seqs=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=False,
        disable_log_stats=True,
    )
    load_seconds = time.perf_counter() - started
    _mark_persistent_cache_ready()

    try:
        baseline = llm.apply_model(partial(_probe_model, layer_index=args.layer))[0]
        baseline_generation = _generation_signature(llm, args.prompt)

        mutated_reload = llm.collective_rpc(
            _reload_checkpoint,
            timeout=args.reload_timeout,
            kwargs={"model_path": args.model, "mutation_layer": args.layer},
        )[0]
        mutated = llm.apply_model(partial(_probe_model, layer_index=args.layer))[0]
        mutated_generation = _generation_signature(llm, args.prompt)

        restored_reload = llm.collective_rpc(
            _reload_checkpoint,
            timeout=args.reload_timeout,
            kwargs={"model_path": args.model, "mutation_layer": None},
        )[0]
        restored = llm.apply_model(partial(_probe_model, layer_index=args.layer))[0]
        restored_generation = _generation_signature(llm, args.prompt)

        comparison = _compare_phases(
            baseline,
            mutated,
            restored,
            baseline_generation,
            mutated_generation,
            restored_generation,
        )
        return {
            "status": "passed" if comparison["passed"] else "failed",
            "environment": probe,
            "configuration": {
                "model": args.model,
                "layer": args.layer,
                "quantization": "nvfp4_per_token",
                "enforce_eager": args.enforce_eager,
                "max_model_len": args.max_model_len,
                "gpu_memory_utilization": args.gpu_memory_utilization,
            },
            "initial_load_seconds": load_seconds,
            "reloads": {
                "mutated": mutated_reload,
                "restored": restored_reload,
            },
            "generations": {
                "baseline": baseline_generation,
                "mutated": mutated_generation,
                "restored": restored_generation,
            },
            "kernel_probes": {
                "baseline": baseline,
                "mutated": mutated,
                "restored": restored,
            },
            "comparison": comparison,
        }
    finally:
        shutdown = getattr(llm, "shutdown", None)
        if callable(shutdown):
            shutdown()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-model-len", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--reload-timeout", type=float, default=1800)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--environment-only", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    args = parse_args()
    if args.environment_only:
        result: dict[str, Any] = {"status": "passed", "environment": environment_probe()}
    else:
        result = run_experiment(args)

    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered, flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    if result["status"] != "passed":
        raise RuntimeError("NVFP4 layerwise reload verification failed")


if __name__ == "__main__":
    main()
