"""Tests for deferred multimodal materialization (Phase 16a).

The orchestrator ships lightweight image references (``mm_refs``) and the
trainer materializes pixels from them via ``prime_rl.utils.mm``, reusing the
same materialize/pack code as the orchestrator's flag-off path so parity and
the duplicate-image guarantee hold by construction.
"""

import hashlib
from dataclasses import replace

import msgspec
import pytest
import torch
from renderers.base import MultiModalData

from prime_rl.orchestrator.trajectories import _collect_mm_refs, _pack_mm_kwargs_from_renderer, _reconstruct_mm_pixels
from prime_rl.trainer.batch import _is_multimodal_sample
from prime_rl.transport.types import MicroBatch, MMRefs, TrainingSample
from prime_rl.utils.mm import (
    build_image_messages,
    encode_mm_kwargs,
    pack_mm_kwargs_tensors,
    reconstruct_mm_pixels,
)


def _uri_hash(uri: str) -> str:
    return hashlib.sha256(uri.encode()).hexdigest()[:16]


class _StubRenderer:
    """Mirrors ``materialize_image_pixels``: resolves each URI to a deterministic
    pixel tensor by content hash, decodes each unique hash once, and populates
    every descriptor slot referencing that hash (the duplicate guarantee)."""

    def __init__(self, pixels_by_hash: dict[str, torch.Tensor]):
        self._pixels_by_hash = pixels_by_hash

    def materialize_pixels(self, mm_data: MultiModalData, messages: list) -> MultiModalData:
        image_items = mm_data.mm_items.get("image") or []
        hashes = mm_data.mm_hashes.get("image") or []
        # Decode each referenced URI once (dedup by hash), like the real renderer.
        decoded: dict[str, torch.Tensor] = {}
        for msg in messages:
            for part in msg.get("content", []):
                url = part.get("image_url", {}).get("url", "")
                h = _uri_hash(url)
                if h not in decoded:
                    decoded[h] = self._pixels_by_hash[h]
        new_items = []
        for i, item in enumerate(image_items):
            h = hashes[i]
            new_items.append({"pixel_values": decoded[h], "image_grid_thw": item["image_grid_thw"]})
        return replace(mm_data, mm_items={**mm_data.mm_items, "image": new_items})


def _grid_payload(g: list[int]) -> dict:
    # Wire shape as produced by verifiers' msgpack_encoder (env→orch hop).
    arr = torch.tensor([g], dtype=torch.int64).numpy()
    return {"__torch_tensor__": True, "dtype": "int64", "shape": list(arr.shape), "data": arr.tobytes()}


def _descriptor(uris: list[str], grids: list[list[int]]) -> dict:
    """Descriptor-only mm_data keyed by URI content hash, grids as wire payloads."""
    return {
        "mm_items": {"image": [{"image_grid_thw": _grid_payload(g)} for g in grids]},
        "mm_hashes": {"image": [_uri_hash(u) for u in uris]},
    }


def test_golden_parity_trainer_matches_orchestrator():
    """Trainer-side encode(pack(reconstruct)) is byte-identical to the
    orchestrator's existing _pack_mm_kwargs_from_renderer(_reconstruct_mm_pixels)."""
    uris = ["file:///a.jpg", "file:///b.jpg"]
    grids = [[1, 2, 3], [1, 4, 4]]
    pixels = {_uri_hash(uris[0]): torch.tensor([[1.0, 2.0]]), _uri_hash(uris[1]): torch.tensor([[3.0, 4.0]])}
    renderer = _StubRenderer(pixels)

    descriptor = _descriptor(uris, grids)
    messages = build_image_messages(uris)

    # Trainer path (utils.mm).
    trainer_kwargs = encode_mm_kwargs(pack_mm_kwargs_tensors(reconstruct_mm_pixels(renderer, descriptor, messages)))
    # Orchestrator path (trajectories delegates to the same code).
    orch_kwargs = _pack_mm_kwargs_from_renderer(_reconstruct_mm_pixels(renderer, _descriptor(uris, grids), messages))

    assert trainer_kwargs.keys() == orch_kwargs.keys()
    for key in trainer_kwargs:
        assert trainer_kwargs[key].dtype == orch_kwargs[key].dtype
        assert trainer_kwargs[key].shape == orch_kwargs[key].shape
        assert trainer_kwargs[key].data == orch_kwargs[key].data


def test_duplicate_image_decoded_once_populated_per_slot():
    """A descriptor with the SAME hash in two slots + one URI → both slots get
    identical pixel tensors (decoded once)."""
    uri = "file:///dup.jpg"
    h = _uri_hash(uri)
    pixels = {h: torch.tensor([[7.0, 8.0]])}
    renderer = _StubRenderer(pixels)

    # Two item slots, same hash, one URI.
    descriptor = {
        "mm_items": {
            "image": [{"image_grid_thw": _grid_payload([1, 2, 3])}, {"image_grid_thw": _grid_payload([1, 2, 3])}]
        },
        "mm_hashes": {"image": [h, h]},
    }
    refs = MMRefs(descriptor=descriptor, uris=[uri])

    from prime_rl.utils.mm import materialize_mm_refs

    kwargs = materialize_mm_refs(renderer, refs)
    pv = kwargs["pixel_values"]
    assert pv.shape[0] == 2
    assert torch.equal(pv[0], pv[1])
    assert torch.equal(pv[0], torch.tensor([7.0, 8.0]))


def test_mm_refs_msgpack_round_trip():
    """MMRefs (descriptor + uris) encodes+decodes through msgpack cleanly —
    catches a stray tensor/numpy left in the descriptor."""
    descriptor = _descriptor(["file:///a.jpg"], [[1, 2, 3]])
    refs = MMRefs(descriptor=descriptor, uris=["file:///a.jpg"])

    raw = msgspec.msgpack.encode(refs)
    decoded = msgspec.msgpack.decode(raw, type=MMRefs)

    assert decoded.uris == refs.uris
    assert decoded.descriptor["mm_hashes"] == descriptor["mm_hashes"]


def test_micro_batch_with_mm_refs_is_multimodal():
    """A MicroBatch carrying mm_refs and no mm_kwargs is classified multimodal."""
    refs = MMRefs(descriptor={"mm_items": {}, "mm_hashes": {}}, uris=["file:///a.jpg"])
    mb = MicroBatch(
        input_ids=[1, 2],
        loss_mask=[True, True],
        advantages=[0.0, 0.0],
        inference_logprobs=[0.0, 0.0],
        position_ids=[0, 1],
        temperatures=[1.0, 1.0],
        env_names=["e", "e"],
        mm_refs=refs,
    )
    assert mb.mm_kwargs is None
    assert _is_multimodal_sample(mb)


def test_collect_mm_refs_normalizes_raw_tensor_descriptor():
    """_collect_mm_refs must produce a transport-safe, descriptor-ONLY MMRefs even
    when the union mm_data holds raw torch tensors (in-process path) — pixels
    dropped, grids → list[int], hashes → str, and msgpack-encodable. The bare
    msgspec encoder rejects tensors, so this is the edge the old code missed."""
    # Real Qwen grids are 2-D (1, 3); the flat (3,) shape would mask a
    # shape-preservation bug since _grids_equal compares [[t,h,w]] nested.
    union_mm = {
        "mm_items": {"image": [{"pixel_values": torch.zeros(4, 8), "image_grid_thw": torch.tensor([[1, 2, 3]])}]},
        "mm_hashes": {"image": ["abc123"]},
    }
    trajectory = [
        {"prompt": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "file:///a.jpg"}}]}]}
    ]
    refs = _collect_mm_refs(union_mm, trajectory, [0])

    item = refs.descriptor["mm_items"]["image"][0]
    assert "pixel_values" not in item  # descriptor-only
    assert item["image_grid_thw"] == [[1, 2, 3]]  # (1, 3) nesting preserved, transport-safe lists
    assert refs.descriptor["mm_hashes"]["image"] == ["abc123"]
    assert refs.uris == ["file:///a.jpg"]
    # Must survive the bare msgspec encoder used by the batch sender.
    msgspec.msgpack.decode(msgspec.msgpack.encode(refs), type=MMRefs)


@pytest.mark.parametrize(
    "union_mm",
    [
        {"mm_items": {"video": [{"image_grid_thw": [[1, 2, 3]]}]}, "mm_hashes": {"video": ["v"]}},
        {"mm_items": {}, "mm_hashes": {"video": ["v"]}},  # non-image present only in mm_hashes
    ],
    ids=["in_mm_items", "in_mm_hashes_only"],
)
def test_collect_mm_refs_rejects_non_image_modality(union_mm):
    """Image-only this iteration: a non-image modality in EITHER mm_items or
    mm_hashes must fail loudly, not silently drop to empty uris."""
    with pytest.raises(ValueError, match="image modality only"):
        _collect_mm_refs(union_mm, [], [])


def test_collect_mm_refs_rejects_non_qwen_image_descriptor():
    """Renderer-family guard: an image item without image_grid_thw (e.g. a
    non-Qwen renderer keyed on grid_thws) must fail loudly, not ship a None grid."""
    union_mm = {"mm_items": {"image": [{"grid_thws": [[1, 2, 3]]}]}, "mm_hashes": {"image": ["h"]}}
    with pytest.raises(ValueError, match="Qwen-style image descriptors"):
        _collect_mm_refs(union_mm, [], [])


def _mm_sample(uri: str, env: str = "e", n_prompt: int = 4, n_comp: int = 4):
    """A multimodal TrainingSample carrying deferred mm_refs (post-normalization:
    plain-list grid, str hash keyed by URI content hash so the stub renderer matches)."""
    refs = MMRefs(
        descriptor={"mm_items": {"image": [{"image_grid_thw": [[1, 2, 3]]}]}, "mm_hashes": {"image": [_uri_hash(uri)]}},
        uris=[uri],
    )
    return TrainingSample(
        prompt_ids=list(range(n_prompt)),
        prompt_mask=[False] * n_prompt,
        completion_ids=list(range(n_comp)),
        completion_mask=[True] * n_comp,
        completion_logprobs=[0.0] * n_comp,
        completion_temperatures=[1.0] * n_comp,
        env_name=env,
        advantage=0.0,
        reward=0.0,
        mm_token_type_ids=[1] * (n_prompt + n_comp),
        mm_refs=refs,
    )


def _text_sample(env: str = "e", n_prompt: int = 4, n_comp: int = 4):
    return TrainingSample(
        prompt_ids=list(range(n_prompt)),
        prompt_mask=[False] * n_prompt,
        completion_ids=list(range(n_comp)),
        completion_mask=[True] * n_comp,
        completion_logprobs=[0.0] * n_comp,
        completion_temperatures=[1.0] * n_comp,
        env_name=env,
        advantage=0.0,
        reward=0.0,
    )


def test_multirun_packing_preserves_mm_refs_modality_and_run_tagging():
    """Multi-run: deferred mm_refs samples from 2 runs pack correctly through the
    REAL prepare_batch — each MM sample is its own microbatch carrying its mm_refs,
    tagged to exactly one run via lora_num_tokens, and the modality-separated
    strided distribution keeps every rank on the same modality per step index
    (the FSDP vision-encoder safety property)."""
    from prime_rl.trainer.batch import prepare_batch

    uri0, uri1 = "file:///run0.jpg", "file:///run1.jpg"
    rollouts = [_mm_sample(uri0), _text_sample(), _mm_sample(uri1), _text_sample()]
    idxs = [0, 0, 1, 1]  # run 0 and run 1
    grid = prepare_batch(rollouts, seq_len=64, num_train_workers=2, idxs=idxs, num_loras=2)

    assert len(grid) == 2  # 2 dp ranks
    # FSDP safety: at each step index, both ranks see the SAME modality.
    for step_mbs in zip(*grid):
        modalities = {_is_multimodal_sample(mb) for mb in step_mbs}
        assert len(modalities) == 1, "ranks diverge in modality at a step index → FSDP all-gather would hang"

    # Every MM microbatch carries its mm_refs (not pixels) and is tagged to one run.
    mm_mbs = [mb for rank in grid for mb in rank if _is_multimodal_sample(mb) and mb.mm_refs is not None]
    assert len(mm_mbs) == 2  # one per run (padding microbatches have no mm_refs)
    for mb in mm_mbs:
        assert mb.mm_kwargs is None
        tagged = [i for i, n in enumerate(mb.lora_num_tokens) if n > 0]
        assert len(tagged) == 1 and mb.lora_num_tokens[tagged[0]] == len(mb.input_ids)

    # Run-agnostic materialization: one shared renderer materializes either run's refs.
    renderer = _StubRenderer({_uri_hash(uri0): torch.tensor([[5.0, 6.0]]), _uri_hash(uri1): torch.tensor([[7.0, 8.0]])})
    from prime_rl.utils.mm import materialize_mm_refs

    for mb in mm_mbs:
        kwargs = materialize_mm_refs(renderer, mb.mm_refs)
        assert kwargs is not None and kwargs["pixel_values"].shape[0] == 1


@pytest.mark.parametrize(
    "trainer_defers, trainer_renderer_name, run_defers, run_renderer_name, expected_ok",
    [
        # Run doesn't defer → always fine (ships pixels; trainer handles either way).
        (True, "Qwen3VLRendererConfig", False, "Qwen3RendererConfig", True),
        (False, None, False, None, True),
        # Run defers but trainer doesn't → reject (no renderer to materialize).
        (False, None, True, "Qwen3VLRendererConfig", False),
        # Both defer, same renderer family → ok.
        (True, "Qwen3VLRendererConfig", True, "Qwen3VLRendererConfig", True),
        # Both defer, run uses Auto → ok (resolves against the shared base model).
        (True, "Qwen3VLRendererConfig", True, "AutoRendererConfig", True),
        # Both defer, different renderer family → reject (wrong image processor).
        (True, "Qwen3VLRendererConfig", True, "Qwen3RendererConfig", False),
    ],
)
def test_defer_mm_validation_hook_matrix(
    trainer_defers, trainer_renderer_name, run_defers, run_renderer_name, expected_ok
):
    """Direct coverage of the discovery-time config rejection matrix."""
    from types import SimpleNamespace

    import renderers

    from prime_rl.utils.mm import make_defer_mm_validation_hook

    def _cfg(name):
        return getattr(renderers, name)() if name else None

    hook = make_defer_mm_validation_hook(trainer_defers, _cfg(trainer_renderer_name))
    orch_config = SimpleNamespace(defer_mm_materialization=run_defers, renderer=_cfg(run_renderer_name))
    ok, msg = hook(orch_config)
    assert ok is expected_ok
    assert (msg == "") is expected_ok  # rejection carries a non-empty reason
