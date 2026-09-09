"""Paired live-router / ID replay / total recall experiment on identical rollouts."""

import json
import os
from contextlib import contextmanager
from pathlib import Path

import torch

from prime_rl.trainer.rl.mismatch import mismatch_diagnostics

REPLAY_MODE = "weights"


def unpack_routing(payload, top_k):
    if payload.dtype != torch.int32 or payload.shape[-1] != 2 * top_k:
        raise ValueError("TRR requires int32 [expert IDs, FP32 weight bits] for every token/layer")
    ids = payload[:, :top_k].to(torch.int64)
    weights = payload[:, top_k:].contiguous().view(torch.float32)
    return ids, weights


@contextmanager
def replay_mode(mode):
    global REPLAY_MODE
    previous = REPLAY_MODE
    REPLAY_MODE = mode
    try:
        yield
    finally:
        REPLAY_MODE = previous


def replay_router(router, x, payload, live_forward):
    if payload is None:
        raise ValueError("TRR enabled but routing payload is missing")
    ids, weights = unpack_routing(payload, router.top_k)
    if REPLAY_MODE == "live":
        return live_forward(router, x, None)
    if REPLAY_MODE == "ids":
        return live_forward(router, x, ids)
    if REPLAY_MODE != "weights":
        raise ValueError(f"Unknown TRR mode {REPLAY_MODE}")
    counts = torch.bincount(ids.flatten(), minlength=router.num_experts)
    return weights, ids, counts, weights.sum()


@torch.no_grad()
def shadow_forwards(forward, model, input_ids, position_ids, **kwargs):
    outputs = {}
    for mode in ("live", "ids"):
        with replay_mode(mode):
            out = forward(model, input_ids, position_ids, **kwargs)
        if out.get("logprobs") is None:
            raise ValueError("TRR comparison requires a head returning selected logprobs")
        outputs[mode] = out["logprobs"].detach().cpu()
    return outputs


@torch.no_grad()
def record_comparison(output_dir, step, micro_step, shadows, trr, inference, mask, micro_batch):
    # Trainer head scores the next token; rollout logprobs are aligned to the token itself.
    mask = mask.cpu()
    inference = inference.cpu()[mask]
    values = {mode: torch.roll(logprobs, 1, dims=1)[mask] for mode, logprobs in shadows.items()}
    values["weights"] = trr.detach().cpu()[mask]
    payload = micro_batch["routed_experts"]
    real = torch.tensor([bool(name) for name in micro_batch["env_names"]])
    layers, width = payload.shape[-2:]
    ids, weights = unpack_routing(payload.reshape(-1, width), width // 2)
    weights = weights.reshape(1, -1, layers, width // 2)[:, real]
    ids = ids.reshape(1, -1, layers, width // 2)[:, real]
    if not torch.isfinite(weights).all() or not (weights >= 0).all():
        raise ValueError("Nonfinite or negative routing weights after rollout transport")
    if not ((weights.sum(-1) - 1).abs() < 2e-6).all():
        raise ValueError("Missing or non-normalized routing payload for a real token/layer")
    if not ((ids >= 0) & (ids < 128)).all():
        raise ValueError("Invalid Qwen3-30B expert ID after transport")
    record = {
        "routing_rows": int(real.sum()) * layers,
        "routing_payload_bytes": payload.numel() * payload.element_size(),
        "step": step,
        "micro_step": micro_step,
        "tokens": int(mask.sum()),
        "trace_ids": micro_batch["trace_ids"],
        "modes": {},
    }
    for mode, logprobs in values.items():
        metrics = mismatch_diagnostics(logprobs, inference)
        record["modes"][mode] = {
            name: {"sum": float(v.double().sum()), "max": float(v.max()) if v.numel() else 0.0}
            for name, v in metrics.items()
        }
    path = Path(output_dir) / "trr-paired"
    path.mkdir(exist_ok=True, parents=True)
    rank = int(os.environ["RANK"])
    with (path / f"rank_{rank}.jsonl").open("a") as handle:
        handle.write(json.dumps(record) + "\n")
