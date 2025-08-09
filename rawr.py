import os

import psutil
import torch
from torch import nn
from transformers import AutoModelForCausalLM


def get_memory_usage() -> tuple[float, float]:
    """Get current process memory usage in bytes."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss, mem_info.vms  # RSS: physical memory, VMS: virtual memory

class MeowModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin0 = nn.Linear(1_000, 1_000_000, bias=False)
        self.lin1 = nn.Linear(1_000_000, 1_000, bias=False)

# Use CPU instead of CUDA
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

print("model.from_pretrained('gpt-2') on CPU")
rss, vms = get_memory_usage()
print(f"RSS (physical): {rss / 1024**3:.2f} GB, VMS (virtual): {vms / 1024**3:.2f} GB")

for name, param in model.named_parameters():
    print(name, param.data.shape)
    param.data.untyped_storage().resize_(0)

print("\na.resize_(0)")
rss, vms = get_memory_usage()
print(f"RSS (physical): {rss / 1024**3:.2f} GB, VMS (virtual): {vms / 1024**3:.2f} GB")
