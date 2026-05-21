#!/usr/bin/env python3
"""
GPU tests for block masking overlay (T6 + T7).

Run on a GPU workstation after install_block_masking_overlay.sh:
    VLLM_USE_FLASHINFER_SAMPLER=0 uv run python scripts/test_block_masking_gpu.py

T6: Overlay contract test — verifies stock 0.21.0 features + block masking additions.
T7: End-to-end inference — loads Qwen3-0.6B, injects block masking config, verifies
    engine starts and generates correctly with the overlay applied.
"""

import inspect
import os
import sys
import traceback

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
os.environ.setdefault("VLLM_MOE_USE_DEEP_GEMM", "0")


def test_overlay_matches_stock_021():
    """T6: Verify overlay has BOTH stock 0.21.0 features AND block masking additions."""

    # --- Stock vLLM 0.21.0 features ---
    from vllm.v1.core.sched.scheduler import Scheduler

    init_sig = inspect.signature(Scheduler.__init__)
    init_params = list(init_sig.parameters.keys())
    assert "structured_output_manager" in init_params, "Missing structured_output_manager"
    assert "hash_block_size" in init_params, "Missing hash_block_size"
    assert "block_size" in init_params, "Missing block_size"
    assert hasattr(Scheduler, "set_pause_state"), "Missing set_pause_state"

    preempt_sig = inspect.signature(Scheduler._preempt_request)
    assert "timestamp" in preempt_sig.parameters, "Missing timestamp in _preempt_request"

    from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput

    so_fields = set(SchedulerOutput.__dataclass_fields__.keys())
    assert "kv_connector_metadata" in so_fields, "Missing kv_connector_metadata"
    assert "scheduled_spec_decode_tokens" in so_fields, "Missing scheduled_spec_decode_tokens"
    assert "ec_connector_metadata" in so_fields, "Missing ec_connector_metadata"
    assert "new_block_ids_to_zero" in so_fields, "Missing new_block_ids_to_zero"

    crd_fields = set(CachedRequestData.__dataclass_fields__.keys())
    assert "resumed_req_ids" in crd_fields, "Missing resumed_req_ids"
    assert "num_output_tokens" in crd_fields, "Missing num_output_tokens"

    from vllm.v1.core.kv_cache_manager import KVCacheManager

    assert hasattr(KVCacheManager, "allocate_slots"), "Missing allocate_slots"

    from vllm.distributed.kv_events import BlockStored

    assert BlockStored is not None

    from vllm.v1.request import Request

    assert not hasattr(Request, "__slots__"), "Request has __slots__"

    # --- Block masking additions ---
    from vllm.v1.engine import SpanRemovalResult

    assert SpanRemovalResult is not None

    assert hasattr(Scheduler, "mask_token_span"), "Missing mask_token_span"

    assert "kv_copy_operations" in so_fields, "Missing kv_copy_operations"
    assert "block_table_truncations" in so_fields, "Missing block_table_truncations"
    assert "skip_sampling_req_ids" in so_fields, "Missing skip_sampling_req_ids"
    assert "block_masking_barrier" in so_fields, "Missing block_masking_barrier"

    assert hasattr(KVCacheManager, "compact_kv_cache"), "Missing compact_kv_cache"

    from vllm.distributed.kv_events import KVSlotCopy

    assert KVSlotCopy is not None

    from vllm.config import VllmConfig

    assert hasattr(VllmConfig, "block_masking_config"), "VllmConfig missing block_masking_config"

    from vllm.config.block_masking import BlockMaskingConfig

    assert BlockMaskingConfig is not None

    from vllm.v1.core.block_masking import BlockMaskingProcessor, BlockMaskingState

    assert BlockMaskingProcessor is not None
    assert BlockMaskingState is not None

    from vllm.v1.worker.block_table import BlockTable

    assert hasattr(BlockTable, "truncate_row"), "Missing BlockTable.truncate_row"

    from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator

    assert hasattr(
        KVCacheCoordinator, "compact_kv_cache"
    ), "Missing KVCacheCoordinator.compact_kv_cache"

    from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager

    assert hasattr(
        SingleTypeKVCacheManager, "compact_kv_cache"
    ), "Missing SingleTypeKVCacheManager.compact_kv_cache"

    assert "compacted_token_counts" in crd_fields, "Missing compacted_token_counts"

    print("  T6 PASSED: All stock 0.21.0 features and block masking additions verified")


def test_production_plugin_installs_patches():
    """Verify the production vLLM plugin installs runtime block masking patches."""

    from vllm.plugins import load_general_plugins

    load_general_plugins()

    from vllm.config import VllmConfig
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.worker.gpu_input_batch import InputBatch
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    assert hasattr(VllmConfig, "block_masking_config")
    assert getattr(EngineCore, "_prime_rl_block_masking_barrier_patched", False)
    assert getattr(InputBatch, "_prime_rl_block_masking_patched", False)
    assert getattr(GPUModelRunner, "_prime_rl_block_masking_patched", False)
    assert hasattr(GPUModelRunner, "_execute_kv_copy_operations")

    print("  T6b PASSED: prime_rl production plugin installed block masking patches")


def test_end_to_end_inference():
    """T7: Serve Qwen3-0.6B with block masking config, verify inference works."""

    from vllm import LLM, SamplingParams

    MODEL = "Qwen/Qwen3-0.6B"

    print("  Loading model...")
    llm = LLM(
        model=MODEL,
        dtype="float16",
        max_model_len=2048,
        gpu_memory_utilization=0.5,
        enforce_eager=True,
        additional_config={
            "block_masking_config": {
                "enable": True,
                "mask_delimiters": False,
                "keep_last_n_blocks": 0,
                "async_mode": os.environ.get("BLOCK_MASKING_ASYNC_MODE", "safe_sync"),
                "debug": True,
                "block_start_token": "<think>",
                "block_end_token": "</think>",
                "summary_start_token": "<|im_start|>",
                "summary_end_token": "<|im_end|>",
            }
        },
    )
    print("  Engine started with block masking config")

    # Test A: Basic generation
    print("  Running basic generation...")
    outputs = llm.generate(
        ["What is 2 + 3?"],
        SamplingParams(temperature=0.0, max_tokens=128),
    )
    assert len(outputs) == 1
    text = outputs[0].outputs[0].text
    assert len(outputs[0].outputs[0].token_ids) > 0, "Empty token output"
    print(f"  Basic output: {text[:200]}")

    # Test B: Multiple concurrent requests
    print("  Running concurrent requests...")
    prompts = [
        "What is 10+20?",
        "What is 5*6?",
        "What is 100-37?",
        "What is 8/2?",
    ]
    outputs = llm.generate(
        prompts,
        SamplingParams(temperature=0.0, max_tokens=64),
    )
    assert len(outputs) == 4
    for i, out in enumerate(outputs):
        t = out.outputs[0].text
        assert len(out.outputs[0].token_ids) > 0, f"Empty token output for prompt {i}"
        print(f"  [{i}] {t[:80]}")

    # Test C: Prompt with <think> tags (block markers) in prefill
    print("  Running prefill with block markers...")
    prompt_with_think = (
        "<|im_start|>user\nWhat is 15+27?<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>Let me add 15 and 27. 15 + 27 = 42.</think>"
        "The answer is 42.<|im_end|>\n"
        "<|im_start|>user\nNow what is 42 * 2?<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    outputs = llm.generate(
        [prompt_with_think],
        SamplingParams(
            temperature=0.0,
            max_tokens=8,
            min_tokens=8,
            ignore_eos=True,
            skip_special_tokens=False,
        ),
    )
    text = outputs[0].outputs[0].text
    token_count = len(outputs[0].outputs[0].token_ids)
    finish_reason = outputs[0].outputs[0].finish_reason
    assert token_count == 8, f"Expected 8 tokens after prefill block markers, got {token_count}"
    assert finish_reason == "length", f"Expected length finish, got {finish_reason!r}"
    print(f"  Prefill+block output: {text[:200]}")

    print("  T7 PASSED: Engine started, generated, handled block markers")


def main():
    tests = [
        ("T6b: Production plugin contract", test_production_plugin_installs_patches),
        ("T6: Overlay contract", test_overlay_matches_stock_021),
        ("T7: End-to-end inference", test_end_to_end_inference),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        print(f"\n{'='*60}")
        print(f"Running {name}...")
        print(f"{'='*60}")
        try:
            test_fn()
            passed += 1
        except Exception:
            failed += 1
            traceback.print_exc()
            print(f"  {name} FAILED")

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    print(f"{'='*60}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
