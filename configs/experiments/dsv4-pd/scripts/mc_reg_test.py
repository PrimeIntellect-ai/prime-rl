"""Register a CUDA tensor with Mooncake's RDMA transfer engine (no nvidia_peermem on these nodes)."""
import os
import socket

import torch
from mooncake.engine import TransferEngine

x = torch.empty(1 << 30, dtype=torch.uint8, device="cuda")
h = torch.empty(1 << 26, dtype=torch.uint8).pin_memory()
te = TransferEngine()
ip = socket.gethostbyname(socket.gethostname())
assert te.initialize(ip, "P2PHANDSHAKE", "rdma", os.environ.get("MC_DEVICES", "")) == 0
print("host  register:", te.register_memory(h.data_ptr(), h.numel()))
print("cuda  register:", te.register_memory(x.data_ptr(), x.numel()))
print("cuda+off register:", te.register_memory(x.data_ptr() + 4096, 1 << 20))
