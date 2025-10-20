import hashlib

import torch
from torch.distributed.tensor import DTensor
from torch.optim.optimizer import Optimizer

TENSOR_SIG_SAMPLE_SIZE = 1000


def get_tensor_signature(a: torch.Tensor | torch.nn.Parameter) -> str:
    while isinstance(a, torch.nn.Parameter):
        a = a.data

    if isinstance(a, DTensor):
        a = a.to_local()

    if a.device == torch.device("meta"):
        return f"{a.dtype}{a.shape}<meta>"

    if a.numel() < TENSOR_SIG_SAMPLE_SIZE:
        b = a.as_strided(size=(a.numel(),), stride=(1,))
    else:
        step_size = a.numel() // TENSOR_SIG_SAMPLE_SIZE
        b = a.as_strided(size=(TENSOR_SIG_SAMPLE_SIZE,), stride=(step_size,))
    element_str = "".join([f"{x:.3e}" for x in b])
    element_hash = hashlib.md5(element_str.encode("utf-8")).hexdigest()
    return f"{a.dtype}{a.shape}{a.stride()}<{element_hash}>"


def get_signature(state_dict, compress: bool = True) -> str:
    if compress:
        return hashlib.md5(str(state_dict).encode("utf-8")).hexdigest()
    else:
        return "\n".join(f"{name}: {sig}" for name, sig in state_dict.items())


def get_optimizer_signature(optimizer: Optimizer, compress: bool = True) -> str:
    def unwrap_tensor(state_dict: dict) -> dict:
        new_dict = {}
        for key, value in state_dict.items():
            if isinstance(value, dict):
                new_dict[key] = unwrap_tensor(value)
            elif isinstance(value, torch.Tensor):
                new_dict[key] = get_tensor_signature(value)
            else:
                new_dict[key] = str(value)
        return new_dict

    state_dict_sig = unwrap_tensor(optimizer.state_dict())
    return get_signature(state_dict_sig, compress)


def get_module_signature(module: torch.nn.Module, compress: bool = True) -> str:
    param_sig = {name: get_tensor_signature(param) for name, param in module.named_parameters()}
    buffer_sig = {name: get_tensor_signature(buffer) for name, buffer in module.named_buffers()}
    state_dict_sig = {**param_sig, **buffer_sig}
    return get_signature(state_dict_sig, compress)
