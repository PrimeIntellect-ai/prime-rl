"""Measure the CPU response path, excluding generation and network transport."""

import argparse
import asyncio
import gc
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from fastapi.responses import JSONResponse
from renderers.client import (
    _parse_compact_logprobs,
    _parse_completion_logprobs,
    _parse_score_head,
    parse_generate_response,
)
from vllm.entrypoints.openai.engine.protocol import RequestResponseMetadata
from vllm.entrypoints.scale_out.token_in_token_out.protocol import GenerateRequest
from vllm.logprobs import FlatLogprobs, append_logprobs_for_next_position
from vllm.outputs import CompletionOutput, RequestOutput, SamplingMask
from vllm.sampling_params import SamplingParams

from prime_rl.inference.vllm.serving_tokens import PrimeRlServingTokens


def make_output(tokens, topk, flat):
    rows = FlatLogprobs() if flat else []
    completion_ids = []
    head_ids = list(range(topk + 1))
    head_values = np.asarray([-5.0 - 0.02 * i for i in head_ids], dtype=np.float32).tolist()
    for i in range(tokens):
        sampled = 1000 if i % 2 else i % (topk + 1)
        completion_ids.append(sampled)
        sampled_value = -12.0 if sampled == 1000 else head_values[sampled]
        append_logprobs_for_next_position(
            rows,
            [sampled, *head_ids],
            [sampled_value, *head_values],
            [None] * (topk + 2),
            1,
            topk + 1,
        )
    return RequestOutput(
        request_id="benchmark",
        prompt=None,
        prompt_token_ids=[1, 2],
        prompt_logprobs=None,
        outputs=[
            CompletionOutput(
                index=0,
                text="",
                token_ids=completion_ids,
                cumulative_logprob=None,
                logprobs=rows,
                finish_reason="length",
                sampling_mask=SamplingMask(token_ids=[[1, 2]] * tokens),
                routed_experts=np.zeros((tokens, 1, 2), dtype=np.uint8),
            )
        ],
        finished=True,
        num_cached_tokens=1,
        kv_transfer_params={"check": "preserved"},
    )


async def measure(output, topk, compact):
    handler = object.__new__(PrimeRlServingTokens)
    handler.model_config = SimpleNamespace(enable_return_routed_experts=True)
    handler.enable_prompt_tokens_details = True
    handler.enable_log_outputs = False
    request = GenerateRequest(
        token_ids=[1, 2],
        sampling_params=SamplingParams(
            logprobs=topk + 1,
            extra_args={"prl_compact_logprobs": compact},
            skip_clone=True,
        ),
    )
    metadata = RequestResponseMetadata(request_id="benchmark")

    async def outputs():
        yield output

    gc.collect()
    start = time.perf_counter()
    response = await handler.serve_tokens_full_generator(request, outputs(), "benchmark", "test", metadata)
    formatted = time.perf_counter()
    raw = JSONResponse(response.model_dump()).body
    encoded = time.perf_counter()
    data = parse_generate_response(raw)
    choice = data["choices"][0]
    if compact:
        scores = _parse_compact_logprobs(choice, choice["token_ids"], topk)
        assert choice["logprobs"] is None
    else:
        scores = (_parse_completion_logprobs(choice, choice["token_ids"]), _parse_score_head(choice, topk))
    decoded = time.perf_counter()
    assert request.sampling_params.logprobs == topk + 1
    assert output.outputs[0].logprobs is not None
    assert data["kv_transfer_params"] == {"check": "preserved"}
    assert data["usage"]["prompt_tokens_details"]["cached_tokens"] == 1
    assert choice["sampling_mask"] == [[1, 2]] * len(choice["token_ids"])
    assert choice["routed_experts"]["shape"] == [len(choice["token_ids"]), 1, 2]
    return {
        "format_seconds": formatted - start,
        "json_seconds": encoded - formatted,
        "client_seconds": decoded - encoded,
        "server_seconds": encoded - start,
        "total_seconds": decoded - start,
        "response_bytes": len(raw),
    }, scores


async def main(args):
    results = {}
    reference = None
    for compact in (False, True):
        output = make_output(args.tokens, args.topk, compact)
        measurements = []
        for _ in range(args.repeats):
            measured, scores = await measure(output, args.topk, compact)
            if reference is None:
                reference = scores
            assert scores == reference, "Compact transport changed sampling evidence"
            measurements.append(measured)
        results["compact" if compact else "legacy"] = {
            "median": {key: statistics.median(row[key] for row in measurements) for key in measurements[0]},
            "repeats": measurements,
        }
    results["tokens"] = args.tokens
    results["topk"] = args.topk
    results["exact_score_parity"] = True
    results["scope"] = "Synthetic CPU response path; no GPU generation, live concurrency, or network benchmark."
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(
        json.dumps({name: result["median"] for name, result in results.items() if isinstance(result, dict)}, indent=2)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--topk", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    asyncio.run(main(parser.parse_args()))
