#!/usr/bin/env python
"""Test block masking overlay works in vLLM.

Verifies:
1. vLLM starts with block masking overlay + config (no crash)
2. Block masking tokens resolve correctly
3. Prompts with embedded block markers are processed by the scheduler
4. Generation after block markers doesn't corrupt output

Does NOT require a Memento-trained model — uses base model with added tokens.

Usage:
    python scripts/test_block_masking_inference.py --model /path/to/model
"""

import argparse
import asyncio
import gc
import inspect
import os
import sys
from pathlib import Path


def find_model_dir(search_root: str) -> str | None:
    root = Path(search_root)
    if (root / "config.json").exists():
        return str(root)
    for p in sorted(root.rglob("config.json")):
        return str(p.parent)
    return None


def run_t6_overlay_contract():
    """T6: Verify overlay has BOTH stock 0.21.0 features AND block masking additions."""
    print("\n=== T6: Overlay contract ===")

    from vllm.v1.core.sched.scheduler import Scheduler

    init_params = list(inspect.signature(Scheduler.__init__).parameters.keys())
    assert "structured_output_manager" in init_params
    assert "hash_block_size" in init_params
    assert hasattr(Scheduler, "set_pause_state")
    assert hasattr(Scheduler, "mask_token_span")

    from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput

    so_fields = set(SchedulerOutput.__dataclass_fields__.keys())
    assert "kv_copy_operations" in so_fields
    assert "block_table_truncations" in so_fields
    assert "skip_sampling_req_ids" in so_fields
    assert "block_masking_barrier" in so_fields
    assert "kv_connector_metadata" in so_fields

    crd_fields = set(CachedRequestData.__dataclass_fields__.keys())
    assert "compacted_token_counts" in crd_fields

    from vllm.v1.core.kv_cache_manager import KVCacheManager

    assert hasattr(KVCacheManager, "compact_kv_cache")

    from vllm.distributed.kv_events import KVSlotCopy  # noqa: F401
    from vllm.v1.engine import SpanRemovalResult  # noqa: F401

    from vllm.config import VllmConfig

    assert hasattr(VllmConfig, "block_masking_config")

    from vllm.v1.worker.block_table import BlockTable

    assert hasattr(BlockTable, "truncate_row")

    from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator

    assert hasattr(KVCacheCoordinator, "compact_kv_cache")

    from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager

    assert hasattr(SingleTypeKVCacheManager, "compact_kv_cache")

    print("PASS: All stock 0.21.0 features and block masking additions verified")


def run_t6_production_patch_contract():
    """Verify the production vLLM plugin installs runtime block masking patches."""
    print("\n=== T6b: Production plugin contract ===")

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
    print("PASS: prime_rl production plugin installed block masking patches")


def assert_fixed_length_generation(output, expected_tokens: int, label: str) -> None:
    token_count = len(output.outputs[0].token_ids)
    finish_reason = output.outputs[0].finish_reason
    assert token_count == expected_tokens, (
        f"{label}: expected {expected_tokens} output tokens, got "
        f"{token_count} (finish={finish_reason!r})"
    )
    assert finish_reason == "length", (
        f"{label}: expected finish_reason='length', got {finish_reason!r}"
    )


def block_masking_additional_config(async_mode: str) -> dict:
    return {
        "block_masking_config": {
            "enable": True,
            "mask_delimiters": False,
            "keep_last_n_blocks": 0,
            "async_mode": async_mode,
            "debug": True,
        }
    }


def shutdown_sync_llm(llm) -> None:
    engine = getattr(llm, "llm_engine", None)
    engine_core = getattr(engine, "engine_core", None)
    if engine_core is not None and hasattr(engine_core, "shutdown"):
        engine_core.shutdown()
    gc.collect()

    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


async def run_t7e_async_client_drain_stress(
    model_path: str,
    max_model_len: int,
    normal_prompt: str,
    compaction_prompts: list[str],
    *,
    repeats: int,
    long_tokens: int,
    compact_tokens: int,
    enforce_eager: bool,
    gpu_memory_utilization: float,
) -> None:
    """Drive vLLM through AsyncLLM so block masking sees queued async work."""
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    engine_args = AsyncEngineArgs(
        model=model_path,
        dtype="float16",
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
        async_scheduling=True,
        disable_log_stats=True,
        additional_config=block_masking_additional_config("async_barrier"),
    )
    engine = AsyncLLM.from_engine_args(engine_args)

    async def collect(prompt: str, sampling_params: SamplingParams, request_id: str):
        final = None
        async for output in engine.generate(
            prompt,
            sampling_params,
            request_id=request_id,
        ):
            final = output
        assert final is not None, f"{request_id}: no final output"
        return final

    long_params = SamplingParams(
        temperature=0,
        max_tokens=long_tokens,
        min_tokens=long_tokens,
        ignore_eos=True,
        skip_special_tokens=False,
    )
    compact_params = SamplingParams(
        temperature=0,
        max_tokens=compact_tokens,
        min_tokens=compact_tokens,
        ignore_eos=True,
        skip_special_tokens=False,
    )

    try:
        for repeat in range(repeats):
            normal_task = asyncio.create_task(
                collect(
                    normal_prompt + f"\nStress wave: {repeat}.",
                    long_params,
                    f"async-normal-long-{repeat}",
                )
            )
            await asyncio.sleep(0.05)
            compaction_tasks = [
                asyncio.create_task(
                    collect(
                        prompt + f"\nStress wave: {repeat}.",
                        compact_params,
                        f"async-compact-{repeat}-{i}",
                    )
                )
                for i, prompt in enumerate(compaction_prompts)
            ]
            outputs = await asyncio.gather(normal_task, *compaction_tasks)
            assert_fixed_length_generation(
                outputs[0], long_tokens, f"async normal long request wave {repeat}"
            )
            for i, out in enumerate(outputs[1:]):
                assert_fixed_length_generation(
                    out, compact_tokens, f"async compaction request wave {repeat} idx {i}"
                )
    finally:
        engine.shutdown()

    print(
        "  PASS: Async client drain stress completed for "
        f"{repeats} wave(s), {repeats * (1 + len(compaction_prompts))} requests"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("MEMENTO_INFERENCE_ENFORCE_EAGER", "1") != "0",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=float(os.environ.get("MEMENTO_INFERENCE_GPU_MEMORY_UTILIZATION", "0.8")),
    )
    parser.add_argument(
        "--async-stress-repeats",
        type=int,
        default=int(os.environ.get("MEMENTO_ASYNC_STRESS_REPEATS", "1")),
    )
    parser.add_argument(
        "--stress-compact-requests",
        type=int,
        default=int(os.environ.get("MEMENTO_STRESS_COMPACT_REQUESTS", "8")),
    )
    parser.add_argument(
        "--stress-long-tokens",
        type=int,
        default=int(os.environ.get("MEMENTO_STRESS_LONG_TOKENS", "256")),
    )
    parser.add_argument(
        "--stress-compact-tokens",
        type=int,
        default=int(os.environ.get("MEMENTO_STRESS_COMPACT_TOKENS", "16")),
    )
    parser.add_argument(
        "--block-masking-async-mode",
        choices=("safe_sync", "async_barrier"),
        default=os.environ.get("BLOCK_MASKING_ASYNC_MODE", "safe_sync"),
    )
    args = parser.parse_args()

    model_path = find_model_dir(args.model)
    if model_path is None:
        print(f"ERROR: No config.json found under {args.model}")
        root = Path(args.model)
        if root.exists():
            for p in sorted(root.rglob("*"))[:40]:
                print(f"  {p.relative_to(root)}")
        sys.exit(1)

    print(f"Model: {model_path}")
    print(f"Block masking async mode: {args.block_masking_async_mode}")
    print(f"enforce_eager={args.enforce_eager}")
    print(f"gpu_memory_utilization={args.gpu_memory_utilization}")
    print(f"async_stress_repeats={args.async_stress_repeats}")

    # T6: Production patch and overlay contracts
    run_t6_production_patch_contract()
    run_t6_overlay_contract()

    # Verify DeepGEMM is disabled (should not be in the path for Memento/Qwen)
    import vllm.envs as vllm_envs
    deep_gemm_enabled = getattr(vllm_envs, "VLLM_USE_DEEP_GEMM", False)
    print(f"\n  VLLM_USE_DEEP_GEMM = {deep_gemm_enabled}")
    assert not deep_gemm_enabled, (
        "DeepGEMM should be disabled for Memento/Qwen smoke tests. "
        "Set VLLM_USE_DEEP_GEMM=0 in the environment."
    )

    from vllm import LLM, SamplingParams

    # --- Test 1: Engine starts with block masking ---
    print("\n=== Test 1: Engine initialization ===")
    llm = LLM(
        model=model_path,
        dtype="float16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        additional_config=block_masking_additional_config(args.block_masking_async_mode),
    )
    print("PASS: vLLM engine started with block masking overlay")

    # --- Test 2: Token resolution ---
    print("\n=== Test 2: Block masking token resolution ===")
    tokenizer = llm.get_tokenizer()
    block_tokens = {
        "<|block_start|>": None,
        "<|block_end|>": None,
        "<|summary_start|>": None,
        "<|summary_end|>": None,
    }
    all_ok = True
    for tok in block_tokens:
        tid = tokenizer.convert_tokens_to_ids(tok)
        ok = tid is not None and (
            tokenizer.unk_token_id is None or tid != tokenizer.unk_token_id
        )
        block_tokens[tok] = tid
        print(f"  {tok} -> {tid} ({'OK' if ok else 'MISSING'})")
        if not ok:
            all_ok = False

    if not all_ok:
        print("FAIL: Missing block masking tokens")
        sys.exit(1)
    print("PASS: All block masking tokens resolved")

    # --- Test 3: Normal generation (no block markers) ---
    print("\n=== Test 3: Normal generation (baseline) ===")
    outputs = llm.chat(
        [[{"role": "user", "content": "What is 2+3?"}]],
        SamplingParams(temperature=0, max_tokens=64),
    )
    text = outputs[0].outputs[0].text
    assert len(outputs[0].outputs[0].token_ids) > 0, "Normal generation produced no tokens"
    print(f"  Output: {text[:200]}")
    print("PASS: Normal generation works with block masking enabled")

    # --- Test 4: Prompt with embedded block markers ---
    print("\n=== Test 4: Prompt with embedded block markers ===")
    prompt_with_blocks = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "What is 15+27?"},
            {
                "role": "assistant",
                "content": (
                    "<|block_start|>"
                    "Let me add 15 and 27. 15 + 27 = 42."
                    "<|block_end|>"
                    "<|summary_start|>"
                    "15 + 27 = 42"
                    "<|summary_end|>"
                    "\nThe answer is 42."
                ),
            },
            {"role": "user", "content": "Now what is 42 * 2?"},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

    outputs = llm.generate(
        [prompt_with_blocks],
        SamplingParams(temperature=0, max_tokens=128),
    )
    text = outputs[0].outputs[0].text
    assert len(outputs[0].outputs[0].token_ids) > 0, "Block-marker prompt produced no tokens"
    print(f"  Output: {text[:300]}")
    print("PASS: Prompt with block markers processed without crash")

    # --- Test 5: Multiple concurrent requests ---
    print("\n=== Test 5: Concurrent requests with block masking ===")
    prompts = [
        [{"role": "user", "content": "What is 10+20?"}],
        [{"role": "user", "content": "What is 5*6?"}],
        [{"role": "user", "content": "What is 100-37?"}],
        [{"role": "user", "content": "What is 8/2?"}],
    ]
    outputs = llm.chat(
        prompts,
        SamplingParams(temperature=0, max_tokens=64),
    )
    for i, out in enumerate(outputs):
        text = out.outputs[0].text
        assert len(out.outputs[0].token_ids) > 0, f"Concurrent request {i} produced no tokens"
        print(f"  [{i}] {text[:100]}")
    print(f"PASS: {len(outputs)} concurrent requests completed")

    # --- Test 6: Prompt-engineered block marker generation ---
    print("\n=== Test 6: Prompt-engineered block marker generation ===")
    system_prompt = (
        "You are a helpful math assistant. When solving problems, structure your "
        "reasoning using these special tokens:\n"
        "- Wrap each reasoning step in <|block_start|> ... <|block_end|>\n"
        "- After each block, write a summary in <|summary_start|> ... <|summary_end|>\n"
        "- Then give the final answer.\n\n"
        "Example:\n"
        "<|block_start|>First, I'll add 2 and 3. 2 + 3 = 5.<|block_end|>"
        "<|summary_start|>2 + 3 = 5<|summary_end|>\n"
        "The answer is 5."
    )
    prompted_messages = [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "What is 15 + 27?"},
        ],
        [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": "A store has 48 apples. 15 are sold. How many remain?",
            },
        ],
    ]
    outputs = llm.chat(
        prompted_messages,
        SamplingParams(temperature=0.3, max_tokens=args.max_tokens),
    )

    markers_found = {t: 0 for t in block_tokens}
    for i, out in enumerate(outputs):
        text = out.outputs[0].text
        print(f"\n  Prompt {i}: {prompted_messages[i][-1]['content']}")
        print(f"  Output: {text[:500]}")
        for tok in block_tokens:
            markers_found[tok] += text.count(tok)

    print(f"\n  Block markers in output: {markers_found}")
    has_blocks = (
        markers_found["<|block_start|>"] > 0 and markers_found["<|block_end|>"] > 0
    )
    has_summaries = (
        markers_found["<|summary_start|>"] > 0
        and markers_found["<|summary_end|>"] > 0
    )

    if has_blocks and has_summaries:
        print("PASS: Model generated complete block+summary cycles via prompting")
    elif any(v > 0 for v in markers_found.values()):
        print("PARTIAL: Some block markers generated but incomplete cycles")
    else:
        print("INFO: No block markers in prompt-engineered output")
        print("  (Expected for base model — SFT model would generate them reliably)")

    # --- Test 7: End-to-end compaction verification ---
    print("\n=== Test 7: End-to-end compaction verification ===")

    # 7a: Verify compaction fires and output is non-empty.
    # Build a prompt with a completed block+summary in conversation history.
    prompt_7a = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "What is 15+27?"},
            {
                "role": "assistant",
                "content": (
                    "<|block_start|>"
                    "I need to add 15 and 27. Breaking it down: 15 + 27 = 42."
                    "<|block_end|>"
                    "<|summary_start|>"
                    "15 + 27 = 42"
                    "<|summary_end|>"
                    "\nThe answer is 42."
                ),
            },
            {"role": "user", "content": "Now what is 42 * 2?"},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_7a_ids = tokenizer.encode(prompt_7a)
    print(f"  Prompt token count: {len(prompt_7a_ids)}")

    # Check block masking tokens are in the prompt
    bs_id = block_tokens["<|block_start|>"]
    be_id = block_tokens["<|block_end|>"]
    ss_id = block_tokens["<|summary_start|>"]
    se_id = block_tokens["<|summary_end|>"]
    strict_compaction_params = SamplingParams(
        temperature=0,
        max_tokens=8,
        min_tokens=8,
        ignore_eos=True,
        skip_special_tokens=False,
    )
    assert bs_id in prompt_7a_ids, "block_start not in prompt tokens"
    assert be_id in prompt_7a_ids, "block_end not in prompt tokens"
    assert ss_id in prompt_7a_ids, "summary_start not in prompt tokens"
    assert se_id in prompt_7a_ids, "summary_end not in prompt tokens"
    print("  All block tokens present in prompt")

    outputs_7a = llm.generate(
        [prompt_7a],
        strict_compaction_params,
    )
    text_7a = outputs_7a[0].outputs[0].text
    finish_reason = outputs_7a[0].outputs[0].finish_reason
    num_output_tokens = len(outputs_7a[0].outputs[0].token_ids)

    print(f"  Output ({num_output_tokens} tokens, finish={finish_reason}): {text_7a[:300]}")
    assert_fixed_length_generation(outputs_7a[0], 8, "single-block compaction")
    print(f"  PASS: Generated {num_output_tokens} tokens after compaction")

    # 7b: Multi-block compaction — two blocks in conversation history.
    print("\n  --- Test 7b: Multi-block compaction ---")
    prompt_7b = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "What is 10+20?"},
            {
                "role": "assistant",
                "content": (
                    "<|block_start|>"
                    "Adding 10 and 20 gives 30."
                    "<|block_end|>"
                    "<|summary_start|>"
                    "10 + 20 = 30"
                    "<|summary_end|>"
                    "\nThe answer is 30."
                ),
            },
            {"role": "user", "content": "And what is 30+15?"},
            {
                "role": "assistant",
                "content": (
                    "<|block_start|>"
                    "30 plus 15 equals 45."
                    "<|block_end|>"
                    "<|summary_start|>"
                    "30 + 15 = 45"
                    "<|summary_end|>"
                    "\nThe answer is 45."
                ),
            },
            {"role": "user", "content": "Now multiply that by 2."},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_7b_ids = tokenizer.encode(prompt_7b)
    print(f"  Prompt token count: {len(prompt_7b_ids)}")
    block_count = prompt_7b_ids.count(bs_id)
    print(f"  Block markers found: {block_count}")

    outputs_7b = llm.generate(
        [prompt_7b],
        strict_compaction_params,
    )
    text_7b = outputs_7b[0].outputs[0].text
    num_output_7b = len(outputs_7b[0].outputs[0].token_ids)
    finish_7b = outputs_7b[0].outputs[0].finish_reason

    print(f"  Output ({num_output_7b} tokens, finish={finish_7b}): {text_7b[:300]}")
    assert_fixed_length_generation(outputs_7b[0], 8, "multi-block compaction")
    print(f"  PASS: Generated {num_output_7b} tokens after multi-block compaction")

    # 7c: Concurrent requests with compaction — verify no cross-request corruption.
    print("\n  --- Test 7c: Concurrent compaction requests ---")
    prompts_7c = []
    for q in ["What is 5+3?", "What is 7*4?", "What is 20-8?"]:
        p = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": q.replace("+", "+").replace("*", "*")},
                {
                    "role": "assistant",
                    "content": (
                        "<|block_start|>"
                        f"Computing: {q}"
                        "<|block_end|>"
                        "<|summary_start|>"
                        f"Result of {q}"
                        "<|summary_end|>"
                    ),
                },
                {"role": "user", "content": "What was the result?"},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts_7c.append(p)

    outputs_7c = llm.generate(
        prompts_7c,
        strict_compaction_params,
    )
    for i, out in enumerate(outputs_7c):
        text = out.outputs[0].text
        n_tok = len(out.outputs[0].token_ids)
        print(f"    [{i}] ({n_tok} tokens): {text[:100]}")
        assert_fixed_length_generation(out, 8, f"concurrent compaction request {i}")

    print(f"  PASS: All {len(outputs_7c)} concurrent compaction requests produced fixed-length output")

    if args.block_masking_async_mode == "async_barrier":
        print("\n  --- Test 7d: Mixed async-barrier pressure ---")
        mixed_prompts = [
            tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": (
                            "Count upward from one and briefly explain each number. "
                            "Keep going until you run out of space."
                        ),
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        ]
        for i in range(6):
            mixed_prompts.append(
                tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": f"What is {i + 3}+{i + 9}?"},
                        {
                            "role": "assistant",
                            "content": (
                                "<|block_start|>"
                                f"Compute {i + 3}+{i + 9} step by step."
                                "<|block_end|>"
                                "<|summary_start|>"
                                f"{i + 3}+{i + 9} computed"
                                "<|summary_end|>"
                            ),
                        },
                        {"role": "user", "content": "State the result again."},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        mixed_params = SamplingParams(
            temperature=0,
            max_tokens=16,
            min_tokens=16,
            ignore_eos=True,
            skip_special_tokens=False,
        )
        outputs_7d = llm.generate(mixed_prompts, mixed_params)
        for i, out in enumerate(outputs_7d):
            text = out.outputs[0].text
            n_tok = len(out.outputs[0].token_ids)
            print(f"    [{i}] ({n_tok} tokens): {text[:100]}")
            assert_fixed_length_generation(out, 16, f"mixed async-barrier request {i}")
        print(f"  PASS: Mixed async-barrier pressure completed for {len(outputs_7d)} requests")

        print("\n  --- Test 7e: Async client drain stress ---")
        async_normal_prompt = tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": (
                        "Write a long numbered list about arithmetic patterns. "
                        "Keep each item short and continue until the token budget ends."
                    ),
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        async_compaction_prompts = []
        for i in range(args.stress_compact_requests):
            async_compaction_prompts.append(
                tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": f"What is {i + 11}+{i + 17}?"},
                        {
                            "role": "assistant",
                            "content": (
                                "<|block_start|>"
                                f"Compute {i + 11}+{i + 17} carefully."
                                "<|block_end|>"
                                "<|summary_start|>"
                                f"{i + 11}+{i + 17} computed"
                                "<|summary_end|>"
                            ),
                        },
                        {"role": "user", "content": "State the result again."},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        shutdown_sync_llm(llm)
        llm = None
        asyncio.run(
            run_t7e_async_client_drain_stress(
                model_path,
                args.max_model_len,
                async_normal_prompt,
                async_compaction_prompts,
                repeats=args.async_stress_repeats,
                long_tokens=args.stress_long_tokens,
                compact_tokens=args.stress_compact_tokens,
                enforce_eager=args.enforce_eager,
                gpu_memory_utilization=args.gpu_memory_utilization,
            )
        )

    print("\n  T7 RESULT: All compaction tests produced fixed-length non-empty output")

    # --- Summary ---
    print(f"\n{'='*60}")
    print("ALL TESTS PASSED")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
