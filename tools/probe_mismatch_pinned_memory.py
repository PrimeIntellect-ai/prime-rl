"""Measure pinned allocation overhead for GLM's fused expert weight sizes."""

import argparse
import json
import os
from pathlib import Path

import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", type=Path)
args = parser.parse_args()

sizes = [128 * 2 * 1408 * 4096 * 2, 128 * 1408 * 4096 * 2]
torch.cuda.init()
before = torch.cuda.memory.host_memory_stats()
tensors = [torch.empty(size, dtype=torch.uint8, pin_memory=True) for size in sizes]
after = torch.cuda.memory.host_memory_stats()
result = {
    "torch_version": torch.__version__,
    "allocator_config": os.environ.get("PYTORCH_ALLOC_CONF"),
    "requested_bytes": sum(tensor.nbytes for tensor in tensors),
    "allocated_bytes": after["allocated_bytes.current"] - before["allocated_bytes.current"],
    "active_bytes": after["active_bytes.current"] - before["active_bytes.current"],
    "tensor_bytes": sizes,
}
args.output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
