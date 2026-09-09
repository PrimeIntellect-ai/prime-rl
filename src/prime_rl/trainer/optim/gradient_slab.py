"""Page-aligned FP32 gradient storage whose consumed pages can be reclaimed."""

import mmap

import torch


class ReclaimableGradientSlab:
    alignment = mmap.PAGESIZE // torch.float32.itemsize

    def __init__(self, numel: int):
        if numel <= 0 or numel % self.alignment:
            raise ValueError("Gradient slab size must be a positive whole number of pages")
        self.mapping = mmap.mmap(-1, numel * torch.float32.itemsize, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
        self.tensor = torch.frombuffer(self.mapping, dtype=torch.float32, count=numel)

    def release(self, offset: int, numel: int) -> None:
        if offset < 0 or numel < 0 or offset + numel > self.tensor.numel():
            raise ValueError("Gradient page range is outside its slab")
        if offset % self.alignment or numel % self.alignment:
            raise ValueError("Gradient release range must contain whole pages")
        if numel:
            self.mapping.madvise(mmap.MADV_DONTNEED, offset * torch.float32.itemsize, numel * torch.float32.itemsize)
