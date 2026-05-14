"""Phase 4 rubric tests.

Plan §"phase4-rubric" allows exactly one test: ``compute_clip_cosine_similarity``
on synthetic same-color vs different-color images. The metric helpers
(``xml_parse_success`` / ``render_success`` / ``code_non_empty``) are pure dict
reads, so we deliberately don't test them — there's nothing meaningful to
assert that isn't tautological.
"""

from __future__ import annotations

from pathlib import Path

import pytest

PIL = pytest.importorskip("PIL")
torch = pytest.importorskip("torch")
open_clip = pytest.importorskip("open_clip")


def _make_solid_png(path: Path, color: tuple[int, int, int], size: int = 224) -> None:
    PIL.Image.new("RGB", (size, size), color).save(path)


@pytest.fixture(scope="module")
def clip_runtime():
    """Load CLIP once on CPU (cheap and CI-friendly) and reuse across tests."""
    device = "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="openai",
        device=device,
        force_quick_gelu=True,
    )
    model.eval()
    return model, preprocess, device


@pytest.mark.slow
def test_clip_cosine_similarity_same_color_high_diff_color_lower(tmp_path, clip_runtime):
    """Same-color image pair scores higher than different-color pair, both in [-1, 1]."""
    model, preprocess, device = clip_runtime
    from blendergym.services.score.clip_scorer import compute_clip_cosine_similarity

    red_a = tmp_path / "red_a.png"
    red_b = tmp_path / "red_b.png"
    blue = tmp_path / "blue.png"
    _make_solid_png(red_a, (255, 0, 0))
    _make_solid_png(red_b, (255, 0, 0))
    _make_solid_png(blue, (0, 0, 255))

    same = compute_clip_cosine_similarity(red_a, red_b, model=model, preprocess=preprocess, device=device)
    diff = compute_clip_cosine_similarity(red_a, blue, model=model, preprocess=preprocess, device=device)

    # Cosine of unit-norm vectors lies in [-1, 1] mathematically, but fp32
    # accumulation can overshoot by up to ~1e-6 — allow a small slack.
    eps = 1e-3
    assert -1.0 - eps <= diff <= 1.0 + eps
    assert -1.0 - eps <= same <= 1.0 + eps
    # Any sensible image embedding should rank "red vs red" above "red vs blue".
    assert same > diff, f"same-color similarity {same:.4f} should exceed cross-color {diff:.4f}"
    # Identical solid-color images should embed essentially identically.
    assert same > 0.95, f"identical solid images should score very high, got {same:.4f}"
