"""Blender background script: exec a code file and render Camera1.

Adapted from VIGA's ``pipeline_render_script.py``. The first version of the
BlenderGym verifiers env only renders ``Camera1`` (single-view) — Camera2 is
intentionally dropped to keep token + GPU costs down (plan §"第一版砍掉的复杂度").

Invocation pattern (as launched by :mod:`blendergym.render`)::

    blender --background <BLEND_FILE> --python pipeline_render_script.py \\
        -- <CODE_FILE> <OUTPUT_DIR>

The ``--`` separator is a Blender CLI convention; everything after it is
forwarded verbatim to ``sys.argv`` (with ``argv[0]`` still being the Blender
binary path). We therefore index from ``argv[6]``: ``[blender, --background,
<blend>, --python, <script>, --, <code_file>, <output_dir>]``.
"""

from __future__ import annotations

import os
import sys

import bpy


def _enable_gpu_cycles(resolution: int = 512) -> None:
    """Configure Cycles via env vars (resolution / samples / denoiser / device).

    Reads four optional environment variables:

    * ``BLENDERGYM_RENDER_RESOLUTION`` (default ``"512"``) — square render size.
    * ``BLENDERGYM_CYCLES_SAMPLES`` (default ``"16"``) — Cycles samples per pixel.
    * ``BLENDERGYM_CYCLES_DENOISER`` (default ``"OPENIMAGEDENOISE"``) — Cycles
      denoiser. Infinigen's bundled Blender 4.2 only ships ``OPENIMAGEDENOISE``;
      ``OPTIX`` raises ``TypeError`` and falls back to OIDN, then to disabled.
    * ``BLENDERGYM_CYCLES_COMPUTE_DEVICE`` (default ``"OPTIX"``) — Cycles
      compute backend. ``OPTIX`` falls back to ``CUDA`` if the runtime lacks
      OptiX libraries.

    The defaults intentionally match the BlenderGym RL training config so that
    ``blender ... --python pipeline_render_script.py`` works standalone for
    micro-benchmarks (see ``scripts/bench_render.py``).
    """
    resolution = int(os.environ.get("BLENDERGYM_RENDER_RESOLUTION", str(resolution)))
    samples = int(os.environ.get("BLENDERGYM_CYCLES_SAMPLES", "16"))
    denoiser = os.environ.get("BLENDERGYM_CYCLES_DENOISER", "OPENIMAGEDENOISE")
    compute = os.environ.get("BLENDERGYM_CYCLES_COMPUTE_DEVICE", "OPTIX")

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.cycles.samples = samples
    scene.render.image_settings.color_mode = "RGB"

    prefs = bpy.context.preferences.addons["cycles"].preferences
    try:
        prefs.compute_device_type = compute
    except TypeError as e:
        sys.stderr.write(
            f"[blendergym] cannot set compute_device_type {compute}: {e}; "
            "falling back to CUDA\n"
        )
        prefs.compute_device_type = "CUDA"
    prefs.get_devices()
    for device in prefs.devices:
        if device.type == "GPU":
            device.use = True
    scene.cycles.device = "GPU"

    # Three-tier fallback: requested → OPENIMAGEDENOISE → disabled. Infinigen
    # Blender 4.2 only supports OPENIMAGEDENOISE; "OPTIX" raises TypeError.
    scene.cycles.use_denoising = True
    try:
        scene.cycles.denoiser = denoiser
    except (TypeError, AttributeError) as e:
        sys.stderr.write(
            f"[blendergym] cannot set denoiser {denoiser}: {e}; "
            "falling back to OPENIMAGEDENOISE\n"
        )
        try:
            scene.cycles.denoiser = "OPENIMAGEDENOISE"
        except (TypeError, AttributeError):
            scene.cycles.use_denoising = False

    effective_denoiser = (
        scene.cycles.denoiser if scene.cycles.use_denoising else "OFF"
    )
    sys.stderr.write(
        f"[blendergym] cycles samples={samples} "
        f"denoiser={effective_denoiser} "
        f"compute={prefs.compute_device_type}\n"
    )


def _exec_user_code(code_fpath: str) -> None:
    """Execute the model-generated Blender code in the loaded scene context."""
    with open(code_fpath, "r", encoding="utf-8") as f:
        code = f.read()
    exec(compile(code, code_fpath, "exec"), {"__name__": "__main__", "bpy": bpy})


def _render_camera1(output_dir: str) -> None:
    """Render Camera1 to ``<output_dir>/render1.png``. No-op if Camera1 missing."""
    if "Camera1" not in bpy.data.objects:
        # Make the failure mode explicit in stderr — caller treats non-zero
        # exit as render failure.
        sys.stderr.write("[blendergym] ERROR: scene has no 'Camera1' object\n")
        sys.exit(2)

    scene = bpy.context.scene
    scene.camera = bpy.data.objects["Camera1"]
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = os.path.join(output_dir, "render1.png")
    bpy.ops.render.render(write_still=True)


def main() -> None:
    code_fpath = sys.argv[6]
    output_dir = sys.argv[7]
    os.makedirs(output_dir, exist_ok=True)

    _enable_gpu_cycles()
    _exec_user_code(code_fpath)
    _render_camera1(output_dir)


if __name__ == "__main__":
    main()
