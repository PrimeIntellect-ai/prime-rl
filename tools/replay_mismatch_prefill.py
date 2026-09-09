"""Replay frozen-run tokens through a matching live inference server's prefill path."""

import argparse
import json
import os
from pathlib import Path

import httpx
import torch
import verifiers.v1 as vf

from prime_rl.monitors.file.traces.chunks import chunk_numbers, open_chunk
from prime_rl.trainer.rl.mismatch import mismatch_diagnostics


def records(directory: Path):
    for number in sorted(chunk_numbers(directory)):
        with open_chunk(directory, number) as stream:
            for line in stream:
                if not line.endswith(b"\n"):
                    break
                yield json.loads(line)


def replay(run_dir: Path, endpoint: str, limit: int) -> dict:
    config = json.loads((run_dir / "configs/latest/resolved/trainer.json").read_text())
    if config["optim"]["lr"] != 0:
        raise ValueError("Prefill replay requires frozen weights (trainer.optim.lr=0)")
    model = config["model"]["name"]
    base = run_dir / "monitors/file/traces"
    annotations = {
        (record["trace_id"], branch["index"]): branch["trainer_logprobs"]
        for record in records(base / "annotations/trainer")
        for branch in record.get("branches", [])
        if "trainer_logprobs" in branch
    }
    output = []
    with httpx.Client(
        base_url=endpoint.rstrip("/").removesuffix("/v1"),
        headers={"Authorization": f"Bearer {os.environ.get('VLLM_API_KEY', 'EMPTY')}"},
        timeout=120,
    ) as client:
        for record in records(base / "stream"):
            episode = vf.Episode.model_validate(record)
            for trace in episode.traces:
                for branch in trace.branches:
                    trainer_values = annotations.get((trace.id, branch.index))
                    if trainer_values is None:
                        continue
                    tokens = branch.token_ids
                    response = client.post(
                        "/inference/v1/generate",
                        json={
                            "model": model,
                            "token_ids": tokens,
                            "sampling_params": {
                                "max_tokens": 1,
                                "temperature": 1.0,
                                "top_p": 1.0,
                                "prompt_logprobs": 1,
                            },
                        },
                    )
                    response.raise_for_status()
                    prompt_logprobs = response.json()["prompt_logprobs"]
                    if len(prompt_logprobs) != len(tokens) or len(trainer_values) != len(tokens):
                        raise ValueError("Replay and trainer streams must match the exact token sequence length")
                    selected = [i for i, value in enumerate(trainer_values) if value is not None]
                    prefill = torch.tensor([prompt_logprobs[i][str(tokens[i])]["logprob"] for i in selected])
                    trainer = torch.tensor([trainer_values[i] for i in selected])
                    decode = torch.tensor([branch.logprobs[i] for i in selected])
                    comparisons = {}
                    for name, left, right in (
                        ("trainer_vs_decode", trainer, decode),
                        ("trainer_vs_prefill", trainer, prefill),
                        ("prefill_vs_decode", prefill, decode),
                    ):
                        comparisons[name] = {
                            key: {"mean": value.mean().item(), "max": value.max().item()}
                            for key, value in mismatch_diagnostics(left, right).items()
                        }
                    output.append(
                        {
                            "trace_id": trace.id,
                            "branch": branch.index,
                            "tokens": len(tokens),
                            "sampled_tokens": len(selected),
                            "comparisons": comparisons,
                        }
                    )
                    if len(output) >= limit:
                        return {"model": model, "endpoint": endpoint, "replays": output}
    if not output:
        raise ValueError("No trained traces are available for replay")
    return {"model": model, "endpoint": endpoint, "replays": output}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("endpoint")
    parser.add_argument("limit", type=int)
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("limit must be positive")
    result = replay(args.run_dir, args.endpoint, args.limit)
    (args.run_dir / "prefill-replay.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
