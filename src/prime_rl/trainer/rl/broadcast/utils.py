import pickle

import torch


def tensor_string_description(tensor: torch.Tensor) -> bytes:
    return pickle.dumps((tensor.dtype, tensor.shape))


def init_tensor_from_string_description(description: bytes, device: torch.device) -> torch.Tensor:
    dtype, shape = pickle.loads(description)
    return torch.empty(shape, dtype=dtype, device=device)
