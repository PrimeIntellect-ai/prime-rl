"""Direct GLM Air inference diagnostics; no trainer, sandbox, or remote logging."""

import argparse
import json
import math
import os
import socket
import subprocess
import time

from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--quantization", default="fp8_per_block")
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--moe-only", action="store_true")
    parser.add_argument("--skip-down-proj", action="store_true")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--nccl-reload", action="store_true")
    parser.add_argument("--cuda-graphs", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    kwargs = {}
    if args.audit or args.reload or args.nccl_reload:
        kwargs["worker_extension_cls"] = "prime_rl.inference.fp8_probe_worker.FP8ProbeWorker"
    if args.moe_only:
        kwargs["quantization_config"] = {"moe": "fp8_per_block"}
    if args.skip_down_proj:
        assert not args.moe_only
        kwargs["quantization_config"] = {
            "ignore": [f"model.layers.{i}.mlp.{part}down_proj" for i in range(46) for part in ("", "shared_experts.")]
        }
    if args.backend != "auto":
        kwargs["kernel_config"] = {
            "moe_backend": args.backend,
            "linear_backend": args.backend,
        }
    print("PROBE_CONFIG", json.dumps(vars(args)), flush=True)
    llm = LLM(
        model="/home/hf-cache/hub/models--zai-org--GLM-4.5-Air/snapshots/a24ceef6ce4f3536971efe9b778bdaa1bab18daa",
        tensor_parallel_size=args.tp,
        enable_expert_parallel=True,
        dtype="bfloat16",
        quantization=("online" if args.moe_only else None if args.quantization == "none" else args.quantization),
        max_model_len=4096,
        max_num_seqs=8,
        max_num_batched_tokens=4096,
        gpu_memory_utilization=0.85,
        enforce_eager=not args.cuda_graphs,
        enable_prefix_caching=False,
        seed=1234,
        **kwargs,
    )
    if args.audit:
        print("PROBE_AUDIT", json.dumps(llm.collective_rpc("fp8_audit")), flush=True)
    questions = [
        "What is 2 + 2? Answer briefly.",
        "Name the capital of France. Answer briefly.",
        "Write a Python function that returns the sum of a list of integers.",
        "Explain why a Python dictionary lookup is usually fast in two sentences.",
        "Return only this exact text: hello world",
        "What is 12 multiplied by 13? Answer briefly.",
        "Complete this sentence: The opposite of hot is",
        "Write a bash command that lists files including hidden files.",
    ]
    tokenizer = llm.get_tokenizer()
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for question in questions
    ]
    params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, logprobs=5)
    results = []
    for round_index in range(args.rounds):
        if args.nccl_reload and round_index:
            assert args.tp == 1
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="1")
            sender = subprocess.Popen(
                [
                    "uv",
                    "run",
                    "--no-sync",
                    "python",
                    "tools/fp8_weight_sender.py",
                    llm.llm_engine.model_config.model,
                    str(port),
                ],
                env=env,
            )
            try:
                print(
                    "PROBE_NCCL_RELOAD", json.dumps(llm.collective_rpc("fp8_receive_kernel", args=(port,))), flush=True
                )
                assert sender.wait(timeout=60) == 0
            finally:
                if sender.poll() is None:
                    sender.terminate()
                    sender.wait(timeout=30)
        if args.reload and round_index:
            print(
                "PROBE_RELOAD",
                json.dumps(llm.collective_rpc("fp8_reload_checkpoint", args=(llm.llm_engine.model_config.model,))),
                flush=True,
            )
        results.extend(llm.generate(prompts, params))
    total, nonfinite = 0, 0
    for question, result in zip(questions * args.rounds, results, strict=True):
        output = result.outputs[0]
        values = [value.logprob for token in output.logprobs for value in token.values()]
        bad = sum(not math.isfinite(value) for value in values)
        total += len(values)
        nonfinite += bad
        print(
            "PROBE_OUTPUT",
            json.dumps(
                {
                    "question": question,
                    "text": output.text,
                    "tokens": len(output.token_ids),
                    "nonfinite_logprobs": bad,
                    "min_logprob": min(values),
                    "max_logprob": max(values),
                    "finish_reason": output.finish_reason,
                },
                allow_nan=False,
            ),
            flush=True,
        )
    print(
        "PROBE_SUMMARY",
        json.dumps(
            {
                "elapsed_seconds": time.monotonic() - started,
                "logprobs_checked": total,
                "nonfinite_logprobs": nonfinite,
                "requests": len(results),
            }
        ),
        flush=True,
    )
    assert nonfinite == 0, "Nonfinite log probabilities"


if __name__ == "__main__":
    main()
