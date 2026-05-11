# BlenderGym PyPI bpy PoC

## TL;DR

| Daemon path | Install | OPTIX | .blend load | Recommended |
| --- | --- | --- | --- | --- |
| **A** Infinigen Blender bin + socket | n/a (already vendored) | OK (production) | OK (production) | **YES** |
| **B** PyPI bpy in main 3.12 venv | FAIL (no cp312 wheel) | n/a | n/a | no |
| **B'** PyPI bpy in isolated 3.13 venv + IPC | OK | **FAIL** (OptiX 7804, CUDA fallback only) | OK (70ms) | no |

Recommendation: go with **path A** for the next daemon implementation plan.

## Evidence

### Path B: install in main `~=3.12` venv

```bash
uv add bpy
```

```text
No solution found ... bpy has no wheels with a matching Python version tag
(e.g., cp312). Wheels are available for bpy (v5.1.1) with the following Python
ABI tag: cp313.
```

PyPI `bpy` only ships `cp313` linux_x86_64 wheels. The repo is pinned to
`requires-python = "~=3.12.0"` (`pyproject.toml`). Path B is dead without
either a project-wide Python upgrade or a `bpy` cp312 wheel.

### Path B': isolated `cp313` venv

```bash
uv venv --python 3.13 /tmp/bpy-poc-venv
VIRTUAL_ENV=/tmp/bpy-poc-venv uv pip install bpy
# bpy==5.1.1 + numpy + cython etc., 380MB download
```

OPTIX device probe:

```text
00:02.170  cycles  | WARNING OptiX initialization failed with error code 7804
bpy version: 5.1.1
python: 3.13.13

--- OPTIX ---
compute_device_type OK = OPTIX
devices:
  ('NVIDIA H20', 'CUDA', True)   # 8x
  ('Intel Xeon CPU Max 9468', 'CPU', False)

--- CUDA fallback ---
compute_device_type OK = CUDA
devices:
  ('NVIDIA H20', 'CUDA', True)   # 8x
  ('Intel Xeon CPU Max 9468', 'CPU', False)
```

Even after `compute_device_type='OPTIX'`, no device has `type='OPTIX'` — the
runtime silently fell back to `CUDA`. Error 7804 is
`OPTIX_ERROR_FUNCTION_TABLE_SIZE_MISMATCH`: the OPTIX header version inside
PyPI `bpy` 5.1.1 does not match the OPTIX runtime exposed by the system NVIDIA
driver (560.35.03).

`.blend` load probe (`data/blendergym/placement1/blender_file.blend`):

```text
open_mainfile took 0.07s
object count: 14
first 10 objects: ['Camera1', 'Camera2', 'Camera3', 'Camera4',
                   'Chair.001', 'Chair.002', 'Chair.003', 'Chair.004',
                   'Dining-table', 'Dinning Table']
cameras: ['Camera', 'Camera.001', 'Camera.002', 'Camera.003']
Camera1 in objects: True
material count: 8
mesh count: 7
image count: 9
scene resolution: 512 x 512
```

So the Infinigen-customized `.blend` files load cleanly in vanilla PyPI bpy.
Path B' is technically functional, but limited to CUDA-only Cycles.

### Path A: Infinigen Blender 4.2.0 (production today)

From `outputs/blendergym_v3_real/.../turn_0/blender.log`:

```text
$ /data/.../infinigen/blender/blender --background <blend> ...
Blender 4.2.0 (hash a51f293548ad built 2024-07-16 06:27:02)
[blendergym] cycles samples=16 denoiser=OPENIMAGEDENOISE compute=OPTIX
Time: 00:03.35 (Saving: 00:00.21)
```

OPTIX is healthy in the Infinigen-bundled binary on this exact host. The
8-GPU H20 cluster is rendering at 16 spp 512x512 in ~3.4s today.

## Why path B' is not worth pursuing

- OPTIX is the dominant Cycles raycast accelerator on H20: roughly 2-3x
  faster than CUDA for the kind of indoor scenes BlenderGym uses. Switching
  the entire render pipeline from OPTIX (path A baseline) to CUDA (path B'
  ceiling) directly contradicts the daemon-isation goal of cutting per-render
  wall time.
- Restoring OPTIX in PyPI bpy would mean either (a) shipping a custom OPTIX
  runtime that matches `bpy`'s header version, or (b) downgrading bpy to a
  version that matches the host driver's OPTIX. Both are unbounded debugging
  rabbit holes that the next plan should not own.
- Path A keeps the proven Infinigen binary, the proven OPTIX support, and
  the proven `.blend` compatibility. Daemon-isation shrinks to "turn the
  existing fork-per-task subprocess into a long-lived background process and
  drive it through a socket protocol." No Python-version migration, no wheel
  rebuilds, no upstream driver coordination.

## Cleanup

- `/tmp/bpy-poc-venv` is harmless and can stay until the next reboot, or be
  removed with `rm -rf /tmp/bpy-poc-venv`.
- `pyproject.toml` and `uv.lock` were not touched.
