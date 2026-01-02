#!/usr/bin/env python3
"""Print the keys, shapes, and total numels of a safetensor file."""

from safetensors import safe_open
from safetensors.torch import save_file
path = "outputs/run_meow/broadcasts/step_10/adapter_model.safetensors"
#path = "trial_broadcast/adapter_model.safetensors"

subset = {}
total_numel = 0
with safe_open(path, framework="pt") as f:
    for key in sorted(f.keys()):
        tensor = f.get_tensor(key)
        numel = tensor.numel()
        total_numel += numel
        print(f"{key}: {list(tensor.shape)} ({numel:,} params)")
        if key.startswith("model.layers.1"):
            break
print(f"\nTotal: {total_numel:,} parameters")
