"""Opt-in mismatch localization. All offsets address unshifted sampled-token positions."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from prime_rl.configs.trainer import KLDebugConfig
from prime_rl.utils.kl_debug import append_json, code_provenance, image_summary, save_capture, tensor_summary


def mismatch_mask(batch: dict) -> torch.Tensor:
    mask = batch["loss_mask"].detach().cpu().bool().reshape(-1).clone()
    rl, ref = batch.get("rl_weights"), batch.get("ref_kl_weights")
    if rl is not None or ref is not None:
        sampled = rl.detach().cpu().reshape(-1) != 0 if rl is not None else mask.clone()
        if ref is not None:
            sampled |= ref.detach().cpu().reshape(-1) != 0
        mask &= sampled
    # Never compare the prediction that crosses a packed sequence boundary.
    start = 0
    for length in batch["sequence_lengths"]:
        mask[start] = False
        start += length
    return mask


def gap_stats(trainer: torch.Tensor, rollout: torch.Tensor, threshold: float) -> dict[str, Any]:
    trainer, rollout = trainer.double().cpu(), rollout.double().cpu()
    gap = trainer - rollout
    finite = torch.isfinite(trainer) & torch.isfinite(rollout) & torch.isfinite(gap)
    values = gap[finite]
    result = {
        "count": gap.numel(),
        "nonfinite": int((~finite).sum()),
        "trainer_nonfinite": int((~torch.isfinite(trainer)).sum()),
        "rollout_nonfinite": int((~torch.isfinite(rollout)).sum()),
        "finite_count": values.numel(),
    }
    if values.numel():
        absolute = values.abs()
        result.update(
            signed_mean=values.mean().item(),
            signed_min=values.min().item(),
            signed_max=values.max().item(),
            abs_mean=absolute.mean().item(),
            abs_p95=torch.quantile(absolute, 0.95).item(),
            abs_p99=torch.quantile(absolute, 0.99).item(),
            abs_max=absolute.max().item(),
            above_threshold=int((absolute > threshold).sum()),
            above_threshold_fraction=(absolute > threshold).double().mean().item(),
        )
    return result


def sequence_reports(batch: dict, trainer_logprobs: torch.Tensor, config: KLDebugConfig) -> list[dict]:
    trainer = trainer_logprobs.detach().cpu().reshape(-1)
    rollout = batch["inference_logprobs"].detach().cpu().reshape(-1)
    mask = mismatch_mask(batch)
    if trainer.shape != rollout.shape or trainer.shape != mask.shape:
        raise ValueError("KL debug logprobs and mask must align with the original packed token positions")
    refs = batch.get("mm_refs")
    images = refs.images if refs is not None else []
    metadata = batch.get("kl_debug") or [None] * len(batch["sequence_lengths"])
    reports = []
    offset = 0
    for index, length in enumerate(batch["sequence_lengths"]):
        end = offset + length
        active = mask[offset:end]
        info = metadata[index] or {}
        local_images = [image for image in images if offset <= image.offset < end]
        record = {
            "sequence": index,
            "trace_id": (batch.get("trace_ids") or [None] * len(metadata))[index],
            "branch_index": (batch.get("branch_indices") or [None] * len(metadata))[index],
            "packed_start": offset,
            "sequence_length_with_padding": length,
            "original_tokens": info.get("original_tokens"),
            "episode_policy_span": info.get("episode_policy_span"),
            "policy_span_scope": "episode_dispatch_to_completion_bounds_not_per_token_versions",
            "stats": gap_stats(trainer[offset:end][active], rollout[offset:end][active], config.abs_logprob_threshold),
            "turns": [],
        }
        for turn in info.get("turns", []):
            start, stop = min(turn["start"], length), min(turn["end"], length)
            if start >= stop:
                continue
            indices = torch.arange(offset + start, offset + stop)[mask[offset + start : offset + stop]]
            if not indices.numel():
                continue
            preceding = [image for image in local_images if image.offset + image.length <= indices[0].item()]
            record["turns"].append(
                {
                    **turn,
                    "retained_end": stop,
                    "context_tokens": indices[0].item() - offset,
                    "preceding_images": len(preceding),
                    "tokens_since_last_image": indices[0].item() - preceding[-1].offset - preceding[-1].length
                    if preceding
                    else None,
                    "stats": gap_stats(trainer[indices], rollout[indices], config.abs_logprob_threshold),
                }
            )
        first_after_images = []
        for image_index, image in enumerate(local_images):
            start = image.offset + image.length
            stop = local_images[image_index + 1].offset if image_index + 1 < len(local_images) else end
            indices = torch.arange(start, stop)[mask[start:stop]][: config.first_tokens_after_image]
            first_after_images.append(
                {
                    "image_index": image_index,
                    "stats": gap_stats(trainer[indices], rollout[indices], config.abs_logprob_threshold),
                }
            )
        record["first_tokens_after_images"] = first_after_images
        reports.append(record)
        offset = end
    return reports


class KLDebugRecorder:
    def __init__(self, config: KLDebugConfig, output_dir: Path, rank: int, manifest: dict):
        self.config = config
        self.directory = output_dir / "kl_debug" / f"rank-{rank}-{uuid.uuid4().hex[:8]}"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.manifest = {**manifest, "code": code_provenance()}
        (self.directory / "manifest.json").write_text(json.dumps(self.manifest, indent=2))
        self.baselines = 0
        self.spikes = 0

    def observe(self, batch: dict, output: dict, *, step: int, micro_step: int, processor: Any) -> bool:
        if (step - 1) % self.config.every:
            return False
        reports = sequence_reports(batch, output["logprobs"], self.config)
        spike = any(r["stats"]["nonfinite"] or r["stats"].get("above_threshold", 0) for r in reports)
        record = {
            "step": step,
            "trainer_policy_version": step - 1,
            "frozen_weights": self.config.freeze_weights,
            "micro_step": micro_step,
            "time": time.time(),
            "spike": spike,
            "sequences": reports,
        }
        baseline = not spike and self.baselines < self.config.baseline_captures
        capture = baseline or (spike and self.spikes < self.config.max_captures)
        if capture:
            refs = batch.get("mm_refs")
            images = (
                [{"url": image.url, "offset": image.offset, "length": image.length} for image in refs.images]
                if refs
                else []
            )
            policy = batch.get("mm_forward_policy")
            image_processor = getattr(processor, "image_processor", None)
            processor_config = json.loads(image_processor.to_json_string()) if image_processor is not None else None
            payload = {
                "schema_version": 1,
                "manifest": self.manifest,
                "record": record.copy(),
                "batch": {key: value for key, value in batch.items() if key not in {"mm_refs", "mm_forward_policy"}},
                "images": images,
                "mm_forward_policy": asdict(policy) if policy else None,
                "trainer_logprobs": output["logprobs"].detach(),
                "image_processor_config": processor_config,
            }
            path = self.directory / f"step-{step}-micro-{micro_step}.pt"
            record["capture"] = save_capture(path, payload, self.config.max_capture_mb * 1024**2)
            if "path" in record["capture"]:
                record["images"] = [
                    {**image_summary(image["url"]), "offset": image["offset"], "length": image["length"]}
                    for image in images
                ]
                record["mm_tensors"] = {
                    key: tensor_summary(value) for key, value in (batch.get("mm_kwargs") or {}).items()
                }
                record["image_processor"] = processor_config
                if baseline:
                    self.baselines += 1
                else:
                    self.spikes += 1
            else:
                # Count attempted captures too: a huge batch must not retry serialization every micro step.
                self.baselines += int(baseline)
                self.spikes += int(spike)
        append_json(self.directory / "metrics.jsonl", record)
        return spike
