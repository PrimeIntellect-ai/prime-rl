"""Tests for deferred multimodal materialization.

The orchestrator ships lightweight image references (``mm_refs``) and the
trainer materializes pixels from them via ``prime_rl.utils.mm``, reusing the
same materialize/pack code for packed and unpacked samples.
"""

import hashlib
from dataclasses import replace

import msgspec
import pytest
import torch
from renderers.base import MultiModalData

from prime_rl.orchestrator.trajectories import _collect_mm_refs
from prime_rl.trainer.batch import _is_multimodal_sample
from prime_rl.transport.types import EncodedTensor, MicroBatch, MMRefs, TrainingSample
from prime_rl.utils.mm import (
    build_image_messages,
    materialize_mm_refs,
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


def test_materialize_mm_refs_matches_reconstruct_then_pack():
    """Trainer-side materialization is exactly reconstructing pixels from refs
    and packing the resulting renderer tensors for model forward."""
    uris = ["file:///a.jpg", "file:///b.jpg"]
    grids = [[1, 2, 3], [1, 4, 4]]
    pixels = {_uri_hash(uris[0]): torch.tensor([[1.0, 2.0]]), _uri_hash(uris[1]): torch.tensor([[3.0, 4.0]])}
    renderer = _StubRenderer(pixels)

    descriptor = _descriptor(uris, grids)
    messages = build_image_messages(uris)
    refs = MMRefs(descriptor=descriptor, uris=uris)

    direct_kwargs = pack_mm_kwargs_tensors(reconstruct_mm_pixels(renderer, descriptor, messages))
    refs_kwargs = materialize_mm_refs(renderer, refs)

    assert direct_kwargs is not None
    assert refs_kwargs is not None
    assert direct_kwargs.keys() == refs_kwargs.keys()
    for key in direct_kwargs:
        torch.testing.assert_close(direct_kwargs[key], refs_kwargs[key])


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


def test_collect_mm_refs_requires_offloaded_file_uri():
    """The refs-only transport is also offload-only: inline image payloads must
    be rewritten to file:// before the orchestrator ships mm_refs."""
    uri = "data:image/jpeg;base64,abc"
    union_mm = {
        "mm_items": {"image": [{"image_grid_thw": [[1, 2, 3]]}]},
        "mm_hashes": {"image": [_uri_hash(uri)]},
    }
    trajectory = [{"prompt": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": uri}}]}]}]

    with pytest.raises(ValueError, match="offloaded file:// image URIs"):
        _collect_mm_refs(union_mm, trajectory, [0])


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


def _encoded_tensor(tensor: torch.Tensor) -> EncodedTensor:
    arr = tensor.detach().cpu().numpy()
    return EncodedTensor(dtype=str(arr.dtype), shape=list(arr.shape), data=arr.tobytes())


def _mm_kwargs_sample(pixel_values: torch.Tensor, env: str = "e", n_prompt: int = 2, n_comp: int = 2):
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
        mm_kwargs={
            "pixel_values": _encoded_tensor(pixel_values),
            "image_grid_thw": _encoded_tensor(torch.tensor([[1, 1, pixel_values.shape[0]]], dtype=torch.int64)),
        },
    )


def test_prepare_batch_packs_mm_refs_when_enabled_preserving_order_and_boundaries():
    """Deferred refs pack by token length, while descriptors/uris concatenate in
    the same order and position_ids keep per-sample resets."""
    from prime_rl.trainer.batch import prepare_batch

    uri = "file:///dup.jpg"
    rollouts = [
        _mm_sample(uri, n_prompt=2, n_comp=2),
        _mm_sample(uri, n_prompt=1, n_comp=3),
        _text_sample(n_prompt=2, n_comp=2),
    ]

    grid = prepare_batch(
        rollouts,
        seq_len=16,
        num_train_workers=1,
        idxs=[0, 0, 0],
        num_loras=1,
        pack_multimodal=True,
    )
    flat = grid[0]
    mm_mbs = [mb for mb in flat if _is_multimodal_sample(mb)]
    text_mbs = [mb for mb in flat if not _is_multimodal_sample(mb)]

    assert len(mm_mbs) == 1
    assert len(text_mbs) == 1
    mb = mm_mbs[0]
    assert mb.mm_refs is not None and mb.mm_kwargs is None
    assert mb.input_ids == [0, 1, 0, 1, 0, 0, 1, 2]
    assert mb.position_ids == [0, 1, 2, 3, 0, 1, 2, 3]
    assert mb.mm_token_type_ids == [1] * len(mb.input_ids)
    assert mb.mm_refs.uris == [uri, uri]
    assert mb.mm_refs.descriptor["mm_hashes"]["image"] == [_uri_hash(uri), _uri_hash(uri)]
    assert len(mb.mm_refs.descriptor["mm_items"]["image"]) == 2
    assert mb.lora_num_tokens == [len(mb.input_ids)]


def test_packed_mm_refs_filesystem_transport_materializes_stitched_tensors(tmp_path):
    """End-to-end trainer mechanics: prepare packed MM refs, write/read them
    through the real filesystem microbatch transport, then materialize them in
    DataLoader into the model kwargs consumed by forward."""
    from types import SimpleNamespace

    from prime_rl.trainer.batch import prepare_batch
    from prime_rl.trainer.rl.data import DataLoader
    from prime_rl.transport.filesystem import FileSystemMicroBatchReceiver, FileSystemMicroBatchSender

    uri0, uri1 = "file:///packed-a.jpg", "file:///packed-b.jpg"
    rollouts = [
        _mm_sample(uri0, n_prompt=2, n_comp=2),
        _mm_sample(uri1, n_prompt=2, n_comp=2),
    ]
    grid = prepare_batch(
        rollouts,
        seq_len=16,
        num_train_workers=2,
        idxs=[0, 0],
        num_loras=1,
        pack_multimodal=True,
    )

    assert len(grid) == 2
    assert len(grid[0]) == len(grid[1]) == 1
    assert any(grid[0][0].loss_mask)
    assert not any(grid[1][0].loss_mask)  # modality-preserving dummy for rank alignment

    sender = FileSystemMicroBatchSender(tmp_path, data_world_size=2, current_step=0)
    sender.send(grid)
    rank0_mb = FileSystemMicroBatchReceiver(tmp_path, data_rank=0, current_step=0).receive()[0]
    rank1_mb = FileSystemMicroBatchReceiver(tmp_path, data_rank=1, current_step=0).receive()[0]

    for mb, has_loss in ((rank0_mb, True), (rank1_mb, False)):
        assert mb.mm_refs is not None and mb.mm_kwargs is None
        assert mb.mm_refs.uris == [uri0, uri1]
        assert mb.mm_refs.descriptor["mm_hashes"]["image"] == [_uri_hash(uri0), _uri_hash(uri1)]
        assert len(mb.mm_refs.descriptor["mm_items"]["image"]) == 2
        assert mb.position_ids == [0, 1, 2, 3, 0, 1, 2, 3]
        assert any(mb.loss_mask) is has_loss

    renderer = _StubRenderer(
        {
            _uri_hash(uri0): torch.tensor([[10.0, 11.0]], dtype=torch.float32),
            _uri_hash(uri1): torch.tensor([[20.0, 21.0]], dtype=torch.float32),
        }
    )

    loader = DataLoader.__new__(DataLoader)
    loader.multi_run_manager = SimpleNamespace(max_runs=1)
    loader._renderer = renderer
    loader.last_mm_materialize_time = 0.0
    loader.last_mm_images_materialized = 0

    rank0 = DataLoader._micro_batch_to_tensor(loader, rank0_mb)
    rank1 = DataLoader._micro_batch_to_tensor(loader, rank1_mb)

    for tensor_batch, has_loss in ((rank0, True), (rank1, False)):
        torch.testing.assert_close(tensor_batch["position_ids"], torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]]))
        torch.testing.assert_close(tensor_batch["mm_token_type_ids"], torch.ones((1, 8), dtype=torch.long))
        assert bool(tensor_batch["loss_mask"].any().item()) is has_loss
        assert tensor_batch["mm_kwargs"] is not None
        torch.testing.assert_close(
            tensor_batch["mm_kwargs"]["pixel_values"],
            torch.tensor([[10.0, 11.0], [20.0, 21.0]], dtype=torch.float32),
        )
        torch.testing.assert_close(
            tensor_batch["mm_kwargs"]["image_grid_thw"],
            torch.tensor([[1, 2, 3], [1, 2, 3]], dtype=torch.int64),
        )

    assert loader.last_mm_images_materialized == 4  # both aligned ranks materialized two image refs


def test_prepare_batch_does_not_pack_mm_refs_with_text_or_other_lora():
    from prime_rl.trainer.batch import prepare_batch

    rollouts = [
        _mm_sample("file:///run0.jpg", n_prompt=2, n_comp=2),
        _mm_sample("file:///run1.jpg", n_prompt=2, n_comp=2),
        _text_sample(n_prompt=2, n_comp=2),
    ]
    grid = prepare_batch(
        rollouts,
        seq_len=16,
        num_train_workers=1,
        idxs=[0, 1, 0],
        num_loras=2,
        pack_multimodal=True,
    )

    mm_mbs = [mb for mb in grid[0] if _is_multimodal_sample(mb)]
    text_mbs = [mb for mb in grid[0] if not _is_multimodal_sample(mb)]
    assert len(mm_mbs) == 2
    assert len(text_mbs) == 1
    assert [mb.lora_num_tokens for mb in mm_mbs] == [[4, 0], [0, 4]]


def test_prepare_batch_rejects_eager_mm_kwargs():
    from prime_rl.trainer.batch import prepare_batch

    rollouts = [
        _mm_sample("file:///refs.jpg", n_prompt=2, n_comp=2),
        _mm_kwargs_sample(torch.tensor([[1.0, 2.0]], dtype=torch.float32)),
    ]
    with pytest.raises(ValueError, match="Eager multimodal mm_kwargs transport is unsupported"):
        prepare_batch(
            rollouts,
            seq_len=16,
            num_train_workers=1,
            idxs=[0, 0],
            num_loras=1,
            pack_multimodal=True,
        )


def test_data_loader_rejects_eager_mm_kwargs_transport():
    from types import SimpleNamespace

    from prime_rl.trainer.rl.data import DataLoader

    mb = MicroBatch(
        input_ids=[1, 2],
        loss_mask=[True, True],
        advantages=[0.0, 0.0],
        inference_logprobs=[0.0, 0.0],
        position_ids=[0, 1],
        temperatures=[1.0, 1.0],
        env_names=["e", "e"],
        mm_token_type_ids=[1, 0],
        mm_kwargs={"pixel_values": _encoded_tensor(torch.tensor([[1.0, 2.0]], dtype=torch.float32))},
    )
    loader = DataLoader.__new__(DataLoader)
    loader.multi_run_manager = SimpleNamespace(max_runs=1)

    with pytest.raises(ValueError, match="Eager multimodal mm_kwargs transport is unsupported"):
        DataLoader._micro_batch_to_tensor(loader, mb)


def test_prepare_batch_rejects_truncated_multimodal_sample():
    from prime_rl.trainer.batch import prepare_batch

    with pytest.raises(ValueError, match="Cannot truncate multimodal"):
        prepare_batch(
            [_mm_sample("file:///too-long.jpg", n_prompt=2, n_comp=2)],
            seq_len=3,
            num_train_workers=1,
            idxs=[0],
            num_loras=1,
            pack_multimodal=True,
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
    "trainer_defers, trainer_renderer_name, run_defers, run_renderer_name, run_vlm, expected_ok",
    [
        # Text-only/non-VLM runs can keep defer disabled.
        (True, "Qwen3VLRendererConfig", False, "Qwen3RendererConfig", False, True),
        (False, None, False, None, False, True),
        # VLM runs must ship refs; eager pixels are no longer supported.
        (True, "Qwen3VLRendererConfig", False, "Qwen3RendererConfig", True, False),
        # Run defers but trainer doesn't → reject (no renderer to materialize).
        (False, None, True, "Qwen3VLRendererConfig", True, False),
        # Both defer, same renderer family → ok.
        (True, "Qwen3VLRendererConfig", True, "Qwen3VLRendererConfig", True, True),
        # Both defer, run uses Auto → ok (resolves against the shared base model).
        (True, "Qwen3VLRendererConfig", True, "AutoRendererConfig", True, True),
        # Both defer, different renderer family → reject (wrong image processor).
        (True, "Qwen3VLRendererConfig", True, "Qwen3RendererConfig", True, False),
    ],
)
def test_defer_mm_validation_hook_matrix(
    trainer_defers, trainer_renderer_name, run_defers, run_renderer_name, run_vlm, expected_ok
):
    """Direct coverage of the discovery-time config rejection matrix."""
    from types import SimpleNamespace

    import renderers

    from prime_rl.utils.mm import make_defer_mm_validation_hook

    def _cfg(name):
        return getattr(renderers, name)() if name else None

    hook = make_defer_mm_validation_hook(trainer_defers, _cfg(trainer_renderer_name))
    orch_config = SimpleNamespace(
        defer_mm_materialization=run_defers,
        renderer=_cfg(run_renderer_name),
        student=SimpleNamespace(model=SimpleNamespace(vlm=object() if run_vlm else None)),
    )
    ok, msg = hook(orch_config)
    assert ok is expected_ok
    assert (msg == "") is expected_ok  # rejection carries a non-empty reason
