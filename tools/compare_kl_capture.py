"""Inspect a trainer capture, compare vLLM image tensors, or request a fresh-cache prefill.

uv run python tools/compare_kl_capture.py CAPTURE.pt
uv run python tools/compare_kl_capture.py CAPTURE.pt --inference-input INPUT.pt
uv run python tools/compare_kl_capture.py CAPTURE.pt --url http://localhost:8000 --model MODEL --sequence 0 --turn 1
The prefill path requires temperature=1, unrestricted sampling and no penalties.
"""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path

import torch

from prime_rl.configs.trainer import KLDebugConfig
from prime_rl.trainer.rl.kl_debug import gap_stats, mismatch_mask, sequence_reports
from prime_rl.transports.batch.types import MMImageRef, MMRefs
from prime_rl.utils.kl_debug import image_summary


def restore_batch(capture: dict) -> dict:
    batch = dict(capture["batch"])
    batch["mm_refs"] = (
        MMRefs(images=[MMImageRef(**image) for image in capture["images"]]) if capture["images"] else None
    )
    return batch


def prefill_request(capture: dict, sequence: int, turn: int, model: str) -> tuple[dict, list[int], torch.Tensor]:
    batch = restore_batch(capture)
    if not batch.get("kl_debug") or batch["kl_debug"][sequence] is None:
        raise ValueError("Capture has no turn metadata; enable orchestrator.kl_debug")
    info = batch["kl_debug"][sequence]
    matching = [item for item in info["turns"] if item["turn"] == turn]
    if not matching:
        raise ValueError(f"Turn {turn} is absent from sequence {sequence}")
    target = matching[0]
    offset = sum(batch["sequence_lengths"][:sequence])
    length = batch["sequence_lengths"][sequence]
    end = min(target["end"], length)
    if target["start"] >= end:
        raise ValueError("The requested turn was truncated away")
    indices = torch.arange(offset + target["start"], offset + end)
    indices = indices[mismatch_mask(batch)[indices]]
    if not indices.numel():
        raise ValueError("The requested turn has no sampled loss tokens")
    if not torch.all(batch["temperatures"].reshape(-1)[indices] == 1):
        raise ValueError("Fresh prefill comparison requires temperature=1")
    sampling_mask = batch.get("sampling_mask")
    if sampling_mask is not None and torch.any(sampling_mask.reshape(-1, sampling_mask.shape[-1])[indices] >= 0):
        raise ValueError("Fresh prefill comparison requires unrestricted sampling (no top-k/top-p mask)")
    sampling = target.get("sampling") or {}
    for name, defaults in {
        "temperature": (None, 1, 1.0),
        "top_p": (None, 1, 1.0),
        "top_k": (None, -1, 0),
        "min_p": (None, 0, 0.0),
        "presence_penalty": (None, 0, 0.0),
        "frequency_penalty": (None, 0, 0.0),
        "repetition_penalty": (None, 1, 1.0),
    }.items():
        if sampling.get(name) not in defaults:
            raise ValueError(f"Fresh prefill comparison does not reproduce {name}={sampling[name]}")
    expanded = batch["input_ids"].reshape(-1)[offset : offset + end].tolist()
    images = [image for image in capture["images"] if offset <= image["offset"] < offset + end]
    logical = list(expanded)
    # The token endpoint expands one logical image placeholder into the processor's patch span.
    for image in reversed(images):
        start = image["offset"] - offset
        stop = start + image["length"]
        if stop > len(expanded):
            raise ValueError("Captured prefix cuts through an image")
        if len(set(logical[start:stop])) != 1:
            raise ValueError("Expected Qwen's repeated image placeholder tokens")
        logical[start:stop] = logical[start : start + 1]
    body = {
        "model": model,
        "token_ids": logical,
        "return_token_ids": True,
        "cache_salt": f"kl-debug-cold-{uuid.uuid4().hex}",
        "sampling_params": {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "max_tokens": 1, "prompt_logprobs": 1},
    }
    if images:
        body["content_parts"] = [{"type": "image_url", "url": image["url"]} for image in images]
    return body, expanded, indices


def compare_images(capture: dict, inference: dict) -> dict:
    batch = capture["batch"]
    result = {"tensors": {}, "scope": "compare matching image sets/order; a prefix capture may contain fewer images"}
    trainer_images = [image_summary(image["url"]) for image in capture["images"]]
    inference_images = [
        image_summary(part["url"]) for part in inference.get("content_parts") or [] if part.get("type") == "image_url"
    ]
    result["same_image_order"] = [i.get("rgb_sha256") for i in trainer_images] == [
        i.get("rgb_sha256") for i in inference_images
    ]
    items = (inference.get("mm_items") or {}).get("image", [])
    if not items or any(item is None for item in items):
        result["unavailable"] = (
            "Inference capture has no complete processed image tensors (possible processor cache hit)"
        )
        return result
    for key, trainer in (batch.get("mm_kwargs") or {}).items():
        if not all(key in item for item in items):
            result["tensors"][key] = {"unavailable": True}
            continue
        chunks = [item[key] for item in items]
        if key == "image_grid_thw":
            chunks = [tensor.reshape(-1, 3) for tensor in chunks]
        inferred = torch.cat(chunks, dim=0)
        comparison = {"trainer_shape": list(trainer.shape), "inference_shape": list(inferred.shape)}
        if trainer.shape == inferred.shape:
            comparison.update(
                exact=torch.equal(trainer, inferred),
                allclose=torch.allclose(trainer.float(), inferred.float(), rtol=1e-5, atol=1e-6),
                max_abs_difference=(trainer.float() - inferred.float()).abs().max().item() if trainer.numel() else 0,
            )
        result["tensors"][key] = comparison
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--inference-input", type=Path)
    parser.add_argument("--url")
    parser.add_argument("--model")
    parser.add_argument("--api-key-env", help="Optional environment variable holding this server's API key")
    parser.add_argument("--sequence", type=int, default=0)
    parser.add_argument("--turn", type=int, default=1)
    parser.add_argument("--allow-unverified-policy", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    capture = torch.load(args.capture, map_location="cpu", weights_only=True)
    config = KLDebugConfig.model_validate(capture["manifest"]["kl_debug"])
    report = {"sequences": sequence_reports(restore_batch(capture), capture["trainer_logprobs"], config)}
    if args.inference_input:
        report["image_comparison"] = compare_images(
            capture, torch.load(args.inference_input, map_location="cpu", weights_only=True)
        )
    if args.url:
        import httpx

        if not args.model:
            parser.error("--url requires --model naming the checkpoint served by that endpoint")
        body, expanded, indices = prefill_request(capture, args.sequence, args.turn, args.model)
        headers = {"Authorization": f"Bearer {os.environ[args.api_key_env]}"} if args.api_key_env else {}
        base = args.url.rstrip("/").removesuffix("/v1")
        with httpx.Client(headers=headers, timeout=600) as client:
            before = client.get(f"{base}/kl_debug/status")
            if before.status_code != 404:
                before.raise_for_status()
            policy_before = before.json().get("policy") if before.status_code == 200 else None
            expected = capture["record"]["trainer_policy_version"]
            verified = (
                policy_before is not None
                and policy_before.get("status") == "installed"
                and policy_before.get("version") == expected
            )
            if not verified and not args.allow_unverified_policy:
                raise ValueError(
                    "Serving policy is unknown or differs from captured trainer; use matching weights, or explicitly --allow-unverified-policy"
                )
            response = client.post(f"{base}/inference/v1/generate", json=body)
            response.raise_for_status()
            payload = response.json()
            after = client.get(f"{base}/kl_debug/status")
            if after.status_code != 404:
                after.raise_for_status()
            policy_after = after.json().get("policy") if after.status_code == 200 else None
        if policy_before != policy_after:
            raise ValueError(
                "Serving weights changed during the probe; discard this comparison and retry on a frozen server"
            )
        if payload.get("prompt_token_ids") != expanded:
            raise ValueError("vLLM expanded tokens differ from the captured trainer input")
        entries = payload["prompt_logprobs"]
        offset = sum(capture["batch"]["sequence_lengths"][: args.sequence])
        local_indices = indices - offset
        values = []
        for index in local_indices.tolist():
            entry = entries[index]
            token = str(expanded[index])
            if entry is None or token not in entry:
                raise ValueError(f"No prompt logprob for captured token at {index}")
            values.append(entry[token]["logprob"])
        fresh = torch.tensor(values)
        report["fresh_prefill"] = {
            "policy_version_matches": verified,
            "weight_identity": "Caller must serve the capture's checkpoint; matching version numbers alone do not establish weight identity across runs",
            "policy_before": policy_before,
            "policy_after": policy_after,
            "cache_salt": body["cache_salt"],
            "trainer_minus_fresh": gap_stats(
                capture["trainer_logprobs"].reshape(-1)[indices], fresh, config.abs_logprob_threshold
            ),
            "fresh_minus_rollout": gap_stats(
                fresh, capture["batch"]["inference_logprobs"].reshape(-1)[indices], config.abs_logprob_threshold
            ),
        }
    rendered = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(rendered)
    print(rendered)


if __name__ == "__main__":
    main()
