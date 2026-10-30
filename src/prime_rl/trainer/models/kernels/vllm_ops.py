"""vLLM's compiled ops, loaded without importing the `vllm` package.

Importing any `vllm` module runs `vllm.env_override`, which sets `TORCHINDUCTOR_COMPILE_THREADS=1` and
patches Inductor's scheduler for the whole process. Loading vLLM's stable-ABI library directly only
registers its ops, as `torch.ops._C.<name>`.
"""

import functools
import importlib.util
import os

import torch

from prime_rl.utils.logger import get_logger

VLLM_OPS_LIBRARY = "_C_stable_libtorch.abi3.so"


@functools.cache
def _load_vllm_ops_library() -> str | None:
    """Register the ops in vLLM's library once; why that failed, or `None` if it loaded."""
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        return "vLLM is not installed"
    path = os.path.join(spec.submodule_search_locations[0], VLLM_OPS_LIBRARY)
    try:
        torch.ops.load_library(path)
    except OSError as error:
        return f"{path} did not load: {error}"
    return None


def op_signature(schema: torch.FunctionSchema) -> str:
    """`schema`'s argument types and names and its return types, without alias annotations or defaults."""
    arguments = ", ".join(f"{argument.type} {argument.name}" for argument in schema.arguments)
    return f"({arguments}) -> ({', '.join(str(ret.type) for ret in schema.returns)})"


@functools.cache
def vllm_op(name: str, signature: str) -> torch._ops.OpOverload | None:
    """vLLM's compiled `torch.ops._C.<name>` if it has `signature` (see `op_signature`), else `None`.

    Returns `None`, and logs why once, when vLLM is not installed, its library does not load, it has no
    such op, or the op's signature differs, so a caller can fall back to its own implementation.
    """
    problem = _load_vllm_ops_library()
    if problem is None:
        packet = getattr(torch.ops._C, name, None)
        if packet is None:
            problem = "vLLM has no such op"
        elif (found := op_signature(packet.default._schema)) != signature:
            problem = f"its signature changed from {signature} to {found}"
        else:
            return packet.default
    get_logger().warning(f"vLLM's compiled op _C::{name} is unavailable, falling back: {problem}")
    return None
