# MXFP8 EP all-to-all on B300: blocked at the kernel smoke check

Node `primeintellect-tur-r2n5`: 8x NVIDIA B300 SXM6, compute capability 10.3, driver 580.173.02, CUDA 13.0.
Commits: parent `0c95d3964` (#4068 tip), branch `938ad25d6` (#4086 tip), both matching origin on 2026-10-14.
Envs built with `uv sync --all-extras --all-packages`, submodules as recorded, `--check` clean.

`smoke.sh` fails on both arms before any training run:

```
{'flash_moe': 'flash_moe requires sm_100a, this device is sm_103', 'indexed_attention': 'available'}
KeyError: "unknown kernel 'mxfp8_moe', have flash_moe, indexed_attention"
```

Two independent problems:

1. `moe_runtime.py` and `distributed/collectives.py` call `prime_kernels.load("mxfp8_moe")`, but the pinned
   wheel (`v0.1.0-6f48176`, same pin on `main`) has no kernel by that name. The MXFP8 MoE runtime exists only
   on unmerged prime-kernels branches (`feat/mxfp8-moe-runtime`, `feat/sm103-mxfp8`). The failure does not
   depend on the GPU, so MXFP8 compute and transport are unusable on `main` everywhere.
2. The wheel's one MoE kernel, `flash_moe`, is built only for `sm_100a` and reports unavailable on B300 (`sm_103`).

No arms were run. Skipped: the model download, the dry runs, the 5-step run, and the 11-arm matrix.
