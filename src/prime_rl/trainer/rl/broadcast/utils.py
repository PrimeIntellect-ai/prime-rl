import pickle

import torch


def tensor_string_description(tensor: torch.Tensor) -> bytes:
    return pickle.dumps((tensor.shape, tensor.dtype))


def init_tensor_from_string_description(description: bytes, device: torch.device) -> torch.Tensor:
    shape, dtype = pickle.loads(description)
    return torch.empty(shape, dtype=dtype, device=device)
