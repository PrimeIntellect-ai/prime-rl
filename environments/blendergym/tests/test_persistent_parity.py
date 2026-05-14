"""Shadow parity test: one-shot run_blender vs PersistentBlender.render.

Run on a GPU machine with Blender available to validate that
PersistentBlender produces identical results to the original one-shot
subprocess approach.

Usage:
    python tests/test_persistent_parity.py \
        --blender-bin /path/to/blender \
        --blend /path/to/test.blend \
        --code 'import bpy; ...'

Compares:
    - render1.png existence and pixel diff (allow <= 1% OIDN variance)
    - exit status (success/failure agreement)
    - stderr keywords (Cycles fallback, OPTIX JIT)
    - duration delta
"""

from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import sys
import tempfile
from pathlib import Path

# Ensure blendergym package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from blendergym.render import run_blender
from blendergym.services.render.persistent_blender import (
    BlenderPool,
    RenderRequest,
)

# Pixel diff comparison (requires Pillow, optional numpy)
try:
    import numpy as np
    from PIL import Image

    def pixel_diff_pct(path_a: Path, path_b: Path) -> float:
        """Return mean absolute pixel difference as percentage of 255."""
        a = np.asarray(Image.open(path_a).convert("RGB"), dtype=np.float32)
        b = np.asarray(Image.open(path_b).convert("RGB"), dtype=np.float32)
        if a.shape != b.shape:
            return 100.0
        return float(np.mean(np.abs(a - b)) / 255.0 * 100.0)

except ImportError:
    def pixel_diff_pct(path_a: Path, path_b: Path) -> float:
        return -1.0  # numpy not available


async def run_parity_test(
    blender_bin: str,
    blend_file: str,
    code: str,
    gpu_id: int = 0,
) -> dict:
    """Run one-shot and persistent render, compare results."""
    tmpdir = Path(tempfile.mkdtemp(prefix="blendergym_parity_"))
    oneshot_dir = tmpdir / "oneshot"
    persistent_dir = tmpdir / "persistent"
    oneshot_dir.mkdir()
    persistent_dir.mkdir()

    # --- One-shot ---
    os.environ.setdefault("BLENDERGYM_RENDER_RESOLUTION", "256")
    os.environ.setdefault("BLENDERGYM_CYCLES_SAMPLES", "8")
    os.environ.setdefault("BLENDERGYM_CYCLES_DENOISER", "OPENIMAGEDENOISE")
    os.environ.setdefault("BLENDERGYM_CYCLES_COMPUTE_DEVICE", "OPTIX")

    print("[parity] Running one-shot render...")
    oneshot = run_blender(
        blend_file=blend_file,
        code=code,
        output_dir=str(oneshot_dir),
        blender_bin=blender_bin,
        gpu_id=gpu_id,
    )
    print(f"  success={oneshot.success} duration={oneshot.duration_s:.2f}s")

    # --- Persistent ---
    print("[parity] Running persistent render...")
    pool = BlenderPool(gpu_id, Path(blender_bin), pool_size=1)
    try:
        await pool.wait_ready(timeout=60)
        req = RenderRequest(
            blend_file=blend_file,
            code=code,
            output_dir=str(persistent_dir),
        )
        persistent = await pool.render(req)
        print(f"  success={persistent.success} duration={persistent.duration_s:.2f}s")
    finally:
        pool.shutdown()

    # --- Compare ---
    results = {
        "oneshot_success": oneshot.success,
        "persistent_success": persistent.success,
        "status_match": oneshot.success == persistent.success,
        "oneshot_duration_s": oneshot.duration_s,
        "persistent_duration_s": persistent.duration_s,
    }

    oneshot_img = oneshot_dir / "render1.png"
    persistent_img = persistent_dir / "render1.png"

    if oneshot_img.exists() and persistent_img.exists():
        diff = pixel_diff_pct(oneshot_img, persistent_img)
        results["pixel_diff_pct"] = diff
        results["pixel_ok"] = diff < 1.0 if diff >= 0 else None
        print(f"  pixel_diff={diff:.4f}%")
    elif not oneshot_img.exists() and not persistent_img.exists():
        results["pixel_diff_pct"] = 0.0
        results["pixel_ok"] = True
    else:
        results["pixel_diff_pct"] = 100.0
        results["pixel_ok"] = False

    if oneshot.stderr:
        for kw in ["fallback", "OPTIX", "cannot set"]:
            if kw.lower() in oneshot.stderr.lower():
                if persistent.stderr and kw.lower() in persistent.stderr.lower():
                    results[f"stderr_{kw}_match"] = True
                else:
                    results[f"stderr_{kw}_match"] = False

    shutil.rmtree(tmpdir, ignore_errors=True)
    return results


def main():
    parser = argparse.ArgumentParser(description="PersistentBlender parity test")
    parser.add_argument("--blender-bin", required=True)
    parser.add_argument("--blend", required=True, help="Path to .blend file")
    parser.add_argument(
        "--code",
        default="import bpy\n",
        help="Python code to exec (default: noop)",
    )
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    results = asyncio.run(
        run_parity_test(args.blender_bin, args.blend, args.code, args.gpu)
    )

    print("\n=== Parity Results ===")
    all_ok = True
    for k, v in results.items():
        status = "PASS" if v is True or (isinstance(v, float) and v < 1.0) else ""
        if v is False:
            status = "FAIL"
            all_ok = False
        print(f"  {k}: {v} {status}")

    if all_ok:
        print("\n[parity] ALL CHECKS PASSED")
    else:
        print("\n[parity] SOME CHECKS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
