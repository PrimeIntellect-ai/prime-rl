import json
from types import SimpleNamespace

import pytest
import torch

from prime_rl.configs.trainer import KLDebugConfig
from prime_rl.inference.kl_debug import InferenceKLRecorder, active_request
from prime_rl.multimodal.base import ForwardPolicy
from prime_rl.trainer.rl.kl_debug import KLDebugRecorder, gap_stats, mismatch_mask, sequence_reports
from prime_rl.transports.batch.types import MMImageRef, MMRefs
from prime_rl.utils.kl_debug import image_summary, save_capture
from tools.compare_kl_capture import compare_images, prefill_request

IMAGE = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


@pytest.fixture
def batch():
    # Image sequence followed by a packed text sequence; both have a masked prefix.
    return {
        "input_ids": torch.tensor([[10, 99, 99, 20, 21, 30, 40, 41]]),
        "position_ids": torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2]]),
        "sequence_lengths": [5, 3],
        "seq_lens": torch.tensor([5, 3]),
        "loss_mask": torch.tensor([[False, False, False, True, True, False, True, True]]),
        "inference_logprobs": torch.tensor([[0, 0, 0, -1, -2, 0, -1, -2]], dtype=torch.float32),
        "temperatures": torch.ones(1, 8),
        "env_names": ["image"] * 5 + ["text"] * 3,
        "trace_ids": ["image-trace", "text-trace"],
        "branch_indices": [0, 0],
        "rl_weights": None,
        "ref_kl_weights": None,
        "sampling_mask": None,
        "mm_refs": MMRefs(images=[MMImageRef(IMAGE, 1, 2)]),
        "mm_kwargs": {"pixel_values": torch.ones(2, 3), "image_grid_thw": torch.tensor([[1, 1, 2]])},
        "mm_forward_policy": ForwardPolicy(pass_position_ids=False, requires_mm_token_type_ids=True),
        "mm_token_type_ids": torch.tensor([[0, 1, 1, 0, 0, 0, 0, 0]]),
        "kl_debug": [
            {
                "original_tokens": 5,
                "episode_policy_span": {"start": 1, "end": 3},
                "turns": [{"turn": 1, "start": 3, "end": 5}],
            },
            {"original_tokens": 3, "turns": [{"turn": 1, "start": 1, "end": 3}]},
        ],
    }


def captured(batch):
    return {
        "batch": {key: value for key, value in batch.items() if key not in {"mm_refs", "mm_forward_policy"}},
        "images": [{"url": IMAGE, "offset": 1, "length": 2}],
        "trainer_logprobs": batch["inference_logprobs"].clone(),
        "manifest": {"kl_debug": KLDebugConfig().model_dump(mode="json")},
        "record": {"trainer_policy_version": 0, "frozen_weights": True},
    }


def test_gap_stats_keep_nonfinite_evidence_and_tail():
    values = gap_stats(torch.tensor([0.0, 0.5, 10.0, float("inf"), float("nan")]), torch.zeros(5), 1)
    assert values["count"] == 5 and values["finite_count"] == 3 and values["nonfinite"] == 2
    assert values["above_threshold"] == 1 and values["above_threshold_fraction"] == pytest.approx(1 / 3)
    assert values["abs_p99"] > 9
    json.dumps(values, allow_nan=False)
    empty = gap_stats(torch.tensor([]), torch.tensor([]), 1)
    assert empty["count"] == 0 and "abs_max" not in empty


def test_packed_turn_and_image_offsets_do_not_leak_between_sequences(batch):
    output = batch["inference_logprobs"].clone()
    output[0, 1] = 100  # Image/context positions are never scored.
    output[0, 3] += 4
    output[0, 6] += 0.5
    reports = sequence_reports(batch, output, KLDebugConfig(first_tokens_after_image=1))
    assert reports[0]["stats"]["abs_max"] == 4
    assert reports[0]["first_tokens_after_images"][0]["stats"]["count"] == 1
    assert reports[0]["turns"][0]["preceding_images"] == 1
    assert reports[0]["turns"][0]["tokens_since_last_image"] == 0
    assert reports[1]["stats"]["abs_max"] == 0.5
    assert reports[1]["turns"][0]["context_tokens"] == 1
    assert reports[1]["turns"][0]["preceding_images"] == 0
    assert reports[1]["first_tokens_after_images"] == []
    batch["kl_debug"][0]["turns"].append({"turn": 2, "start": 8, "end": 12})
    assert len(sequence_reports(batch, output, KLDebugConfig())[0]["turns"]) == 1


def test_masks_exclude_ce_and_packing_boundaries_but_keep_reference_tokens(batch):
    batch["loss_mask"][:] = True
    batch["rl_weights"] = torch.zeros(1, 8)
    batch["rl_weights"][0, [0, 3, 5]] = 1
    batch["ref_kl_weights"] = torch.zeros(1, 8)
    batch["ref_kl_weights"][0, 7] = 1
    assert mismatch_mask(batch).nonzero().flatten().tolist() == [3, 7]


def test_capture_limits_roundtrip_and_nonfinite_trigger(tmp_path, batch):
    config = KLDebugConfig(baseline_captures=1, max_captures=1)
    writer = KLDebugRecorder(config, tmp_path, 0, {"kl_debug": config.model_dump(mode="json")})
    output = {"logprobs": batch["inference_logprobs"].clone()}
    assert not writer.observe(batch, output, step=1, micro_step=0, processor=None)
    output["logprobs"][0, 3] = float("nan")
    assert writer.observe(batch, output, step=2, micro_step=0, processor=None)
    assert writer.observe(batch, output, step=3, micro_step=0, processor=None)
    paths = sorted(writer.directory.glob("*.pt"))
    assert len(paths) == 2
    loaded = torch.load(paths[0], weights_only=True)
    assert loaded["images"][0]["url"] == IMAGE
    assert torch.equal(loaded["batch"]["input_ids"], batch["input_ids"])
    records = [json.loads(line) for line in (writer.directory / "metrics.jsonl").read_text().splitlines()]
    assert records[1]["sequences"][0]["stats"]["nonfinite"] == 1
    assert "capture" not in records[2]
    assert "base64" not in (writer.directory / "metrics.jsonl").read_text()
    assert image_summary(IMAGE)["size"] == [1, 1]


def test_oversized_capture_leaves_no_partial_file(tmp_path):
    path = tmp_path / "capture.pt"
    assert save_capture(path, {"tensor": torch.ones(1024)}, 64)["skipped"] == "payload_budget"
    assert not path.exists()
    assert save_capture(path, {"tensor": torch.ones(1)}, 64)["skipped"] == "serialized_budget"
    assert list(tmp_path.iterdir()) == []


def test_fresh_prefill_reconstructs_exact_qwen_prefix_and_rejects_sampling_mismatch(batch):
    capture = captured(batch)
    body, expanded, indices = prefill_request(capture, 0, 1, "qwen")
    assert expanded == [10, 99, 99, 20, 21]
    assert body["token_ids"] == [10, 99, 20, 21]
    assert indices.tolist() == [3, 4]
    assert body["content_parts"] == [{"type": "image_url", "url": IMAGE}]
    text_body, text_expanded, text_indices = prefill_request(capture, 1, 1, "qwen")
    assert text_expanded == [30, 40, 41] and text_indices.tolist() == [6, 7]
    assert "content_parts" not in text_body
    capture["batch"]["temperatures"][0, 3] = 0.7
    with pytest.raises(ValueError, match="temperature=1"):
        prefill_request(capture, 0, 1, "qwen")


def test_image_tensor_comparison_detects_changed_pixels_and_missing_cache_data(batch):
    capture = captured(batch)
    inference = {"content_parts": [{"type": "image_url", "url": IMAGE}], "mm_items": {"image": [batch["mm_kwargs"]]}}
    assert compare_images(capture, inference)["tensors"]["pixel_values"]["exact"]
    inference["mm_items"] = {"image": [{"pixel_values": torch.zeros(2, 3), "image_grid_thw": torch.tensor([1, 1, 2])}]}
    result = compare_images(capture, inference)
    assert result["same_image_order"]
    assert result["tensors"]["pixel_values"]["max_abs_difference"] == 1
    inference["mm_items"] = {"image": [None]}
    assert "unavailable" in compare_images(capture, inference)


def test_inference_capture_concurrency_and_shared_policy_bounds(tmp_path):
    recorder = InferenceKLRecorder(tmp_path, port=8000, max_captures=2, max_capture_mb=1, max_events=20)
    other = InferenceKLRecorder(tmp_path, port=8000, max_captures=0, max_capture_mb=1, max_events=20)
    assert recorder.policy() is None
    other.set_policy(3, status="installed")
    assert recorder.policy()["version"] == 3
    request = SimpleNamespace(
        cache_salt="v3", token_ids=[1, 2], sampling_params=SimpleNamespace(temperature=1), content_parts=None
    )
    first, second = recorder.begin(request), recorder.begin(request)
    assert first["capture_id"] != second["capture_id"]
    for index, state in enumerate((second, first)):
        token = active_request.set(state)
        try:
            recorder.engine_input(f"request-{index}", {"prompt_token_ids": [1, 2]})
        finally:
            active_request.reset(token)
    assert len(list(recorder.directory.glob("input-*.pt"))) == 2
    other.set_policy(None, status="updating")
    assert recorder.policy()["version"] is None
    other.set_policy(4, status="installed")
    recorder.finish(first, SimpleNamespace(choices=[], usage=None))
    last = json.loads(recorder.journal.read_text().splitlines()[-1])
    assert last["policy_at_start"]["version"] == 3 and last["policy_at_end"]["version"] == 4
