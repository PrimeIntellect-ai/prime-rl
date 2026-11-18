"""The accelerator the trainer runs on: CUDA on NVIDIA GPUs, XPU on Intel GPUs.

Chosen by the accelerator the installed torch build targets: an XPU build selects XPU even before a
device is visible; CUDA and CPU-only builds keep the CUDA defaults. ``device_module`` is
``torch.cuda`` or ``torch.xpu``. Both expose the same device selection, synchronization, memory
statistics and stream API, so call sites use it in place of ``torch.cuda``.
"""

import torch

_accelerator = torch.accelerator.current_accelerator()

DEVICE_TYPE: str = "xpu" if _accelerator is not None and _accelerator.type == "xpu" else "cuda"
"""Device type string for tensors, ``torch.device`` and device meshes."""

device_module = getattr(torch, DEVICE_TYPE)
"""``torch.cuda`` or ``torch.xpu``."""

DIST_BACKEND: str = "xccl" if DEVICE_TYPE == "xpu" else "nccl"
"""torch.distributed backend for accelerator tensors."""

VISIBLE_DEVICES_ENV: str = "ZE_AFFINITY_MASK" if DEVICE_TYPE == "xpu" else "CUDA_VISIBLE_DEVICES"
"""Environment variable that restricts which physical GPUs a process sees."""
