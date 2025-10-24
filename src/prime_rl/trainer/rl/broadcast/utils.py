import pickle

import torch


def tensor_string_description(tensor: torch.Tensor) -> bytes:
    return pickle.dumps((tensor.shape))


def init_tensor_from_string_description(description: bytes, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    shape = pickle.loads(description)
    return torch.empty(shape, dtype=dtype, device=device)
