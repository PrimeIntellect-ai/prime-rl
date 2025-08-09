import os

import psutil
import torch
from torch import nn


def get_memory_usage() -> tuple[float, float]:
    """Get current process memory usage in bytes."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss, mem_info.vms  # RSS: physical memory, VMS: virtual memory


# Use CPU instead of CUDA
model = nn.Linear(1_000, 1_000_000, bias=False)
a = next(model.parameters()).data

print("a = torch.randn(100_000_000) on CPU")
rss, vms = get_memory_usage()
print(f"RSS (physical): {rss / 1024**3:.2f} GB, VMS (virtual): {vms / 1024**3:.2f} GB")

a.resize_(0)

print("\na.resize_(0)")
rss, vms = get_memory_usage()
print(f"RSS (physical): {rss / 1024**3:.2f} GB, VMS (virtual): {vms / 1024**3:.2f} GB")

a.untyped_storage().resize_(0)

print("\na.untyped_storage().resize_(0)")
rss, vms = get_memory_usage()
print(f"RSS (physical): {rss / 1024**3:.2f} GB, VMS (virtual): {vms / 1024**3:.2f} GB")

b = torch.randn(1_000)

print("\nb = torch.randn(1_000) on CPU")
rss, vms = get_memory_usage()
print(f"RSS (physical): {rss / 1024**3:.2f} GB, VMS (virtual): {vms / 1024**3:.2f} GB")

# Force garbage collection to free memory
import gc

gc.collect()

print("\ngc.collect() (garbage collection)")
rss, vms = get_memory_usage()
print(f"RSS (physical): {rss / 1024**3:.2f} GB, VMS (virtual): {vms / 1024**3:.2f} GB")
