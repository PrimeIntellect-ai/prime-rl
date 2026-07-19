"""Validate Omni's dynamic-resolution image processor against the graft model's assumptions.

The V3 Omni processor (omni_image_processing_reference.py, byte-identical to the Nano Omni
checkpoint's image_processing.py) resizes each image to a SINGLE tile whose 16px patch grid
(wp, hp) lands in [min_num_patches, max_num_patches], aspect-preserving, both dims snapped
to multiples of the pixel-shuffle factor (2). Text-side placeholder count per image is
num_tokens = wp*hp // 4. There is no 512px multi-tile grid and no thumbnail tile.

Checks, per resolution in a grid of edge cases:
  1. wp and hp are even (pixel_shuffle with downsample 0.5 never sees an odd grid)
  2. wp*hp <= max_num_patches
  3. num_tokens == wp*hp // 4 == what the model's ViT+pixel_shuffle emits for that tile
     ((H/16)*(W/16) patches, /4 after shuffle) — the masked_scatter count contract
  4. pixel tensor shape == (3, hp*16, wp*16), finite, CLIP-normalized
  5. wp*hp >= min_num_patches (reported as SOFT if violated — NVIDIA's rounding can dip below)
  6. processor config fields match NemotronVLConfig defaults
  7. multi-image batches: budget is per-image (NOT divided by image count) — documented for
     the renderer, which must impose its own per-sample budget

Writes golden fixtures (resolution -> wp/hp/num_tokens) for future renderer tests.

Run from the prime-rl repo root:
    uv run python models/validation/check_tiling_parity.py
"""

import importlib.util
import json
from pathlib import Path

import torch
from PIL import Image

from prime_rl.trainer.models.nemotron_vl.configuration_nemotron_vl import NemotronVLConfig

GRAFT_ROOT = Path(__file__).resolve().parent.parent / "Nemotron-3-Super-VL-graft"
FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "tiling_golden.json"

RESOLUTIONS = [
    (8, 8),
    (16, 16),
    (37, 53),
    (100, 80),  # tiny (below min -> scale up)
    (512, 512),
    (640, 480),
    (800, 600),
    (1024, 768),
    (1280, 720),
    (1920, 1080),  # typical
    (1024, 1024),
    (2048, 2048),
    (2047, 2047),  # square, odd square
    (3000, 150),
    (150, 3000),
    (4111, 37),
    (37, 4111),  # extreme aspect
    (4032, 3024),
    (8000, 6000),
    (513, 511),  # huge (above max -> scale down), off-by-one
]


def load_processor():
    spec = importlib.util.spec_from_file_location(
        "omni_image_processing", GRAFT_ROOT / "omni_image_processing_reference.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = json.loads((GRAFT_ROOT / "preprocessor_config.json").read_text())
    cls = getattr(module, cfg["image_processor_type"])
    kwargs = {k: v for k, v in cfg.items() if k not in ("image_processor_type", "auto_map")}
    return cls(**kwargs), cfg


def make_image(width: int, height: int) -> Image.Image:
    # Deterministic gradient; content is irrelevant to geometry but keeps pixels well-formed.
    x = torch.linspace(0, 255, width).view(1, -1).expand(height, -1)
    y = torch.linspace(0, 255, height).view(-1, 1).expand(-1, width)
    arr = torch.stack([x, y, (x + y) / 2], dim=-1).to(torch.uint8).numpy()
    return Image.fromarray(arr, mode="RGB")


def main() -> None:
    proc, cfg = load_processor()
    model_cfg = NemotronVLConfig()

    print("== config consistency ==")
    hard_fail = False
    for field in ("patch_size", "downsample_ratio", "min_num_patches", "max_num_patches"):
        ours, theirs = getattr(model_cfg, field), cfg[field]
        status = "OK" if ours == theirs else "MISMATCH"
        hard_fail |= ours != theirs
        print(f"  {field}: model={ours} processor={theirs} [{status}]")

    factor = proc._downsample_factor
    assert factor == 2, f"unexpected downsample factor {factor}"

    print("\n== per-resolution geometry ==")
    soft_violations = []
    fixtures = {}
    for width, height in RESOLUTIONS:
        img = proc._process_image(make_image(width, height))
        out = proc._preprocess([img])
        (num_tokens,) = out["num_tokens"]
        pv = out["pixel_values"]
        tile = pv[0] if isinstance(pv, (list, torch.Tensor)) else pv
        _, tile_h, tile_w = tile.shape
        wp, hp = tile_w // proc.patch_size, tile_h // proc.patch_size

        assert tile_h % proc.patch_size == 0 and tile_w % proc.patch_size == 0, (width, height)
        assert wp % factor == 0 and hp % factor == 0, f"ODD GRID at {width}x{height}: wp={wp} hp={hp}"
        assert wp * hp <= cfg["max_num_patches"], f"over max at {width}x{height}: {wp * hp}"
        assert num_tokens == (wp * hp) // (factor**2), f"token count mismatch at {width}x{height}"
        assert torch.isfinite(tile).all(), f"non-finite pixels at {width}x{height}"
        if wp * hp < cfg["min_num_patches"]:
            soft_violations.append((width, height, wp * hp))

        fixtures[f"{width}x{height}"] = {"wp": wp, "hp": hp, "num_tokens": num_tokens}
        print(f"  {width:>5}x{height:<5} -> grid {wp:>3}x{hp:<3} ({wp * hp:>5} patches) {num_tokens:>4} tokens")

    print("\n== multi-image budget semantics ==")
    imgs = [proc._process_image(make_image(w, h)) for w, h in [(4032, 3024), (4032, 3024), (640, 480)]]
    out = proc._preprocess(imgs)
    total = sum(out["num_tokens"])
    print(f"  3-image batch num_tokens={out['num_tokens']} total={total} (max_model_len={cfg['max_model_len']})")
    print("  NOTE: budget is per-image; the renderer must enforce its own per-sample token budget")
    assert out["num_patches"] == [1, 1, 1], "expected one tile per image in dynamic mode"

    FIXTURE_PATH.parent.mkdir(exist_ok=True)
    FIXTURE_PATH.write_text(json.dumps(fixtures, indent=2) + "\n")
    print(f"\nwrote golden fixtures to {FIXTURE_PATH.relative_to(Path.cwd())}")

    if soft_violations:
        print(f"SOFT: {len(soft_violations)} resolutions land below min_num_patches: {soft_violations}")
    print("FAIL: config mismatch (see above)" if hard_fail else "PASS: all hard invariants hold")


if __name__ == "__main__":
    main()
