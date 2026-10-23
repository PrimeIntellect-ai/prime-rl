import pytest
import torch

# The kernels' shared tiles fit only the datacenter Hopper and Blackwell SMs. The opt-in limit does
# not follow capability order, so this enumerates rather than compares: sm120 allows well under half
# of what sm90 does.
requires_datacenter_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (9, 10),
    reason="the fused sparse attention kernels need the shared memory of a datacenter Hopper or Blackwell GPU",
)
