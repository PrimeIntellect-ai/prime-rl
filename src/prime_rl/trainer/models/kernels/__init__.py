import importlib.util

import torch


def cudnn_flashmla_error(heads: int, kv_group: int, dim: int) -> str | None:
    """Why the ``cudnn_flashmla`` sparse-attention backend cannot run this shape here, or ``None`` if it can."""
    if kv_group != 1:
        return f"FlashMLA and the cuDNN backward serve a single KV head, but this shape has {kv_group}"
    if heads not in (64, 128):
        return f"FlashMLA's sparse prefill serves 64 or 128 query heads, but this shape has {heads}"
    if dim != 512:
        return f"FlashMLA's sparse prefill and the cuDNN backward need head_dim 512, but it is {dim}"
    if not torch.cuda.is_available():
        return "the cudnn_flashmla backend needs a CUDA GPU"
    capability = torch.cuda.get_device_capability()
    if capability[0] != 9 and capability not in ((10, 0), (10, 3)):
        return f"the cudnn_flashmla backend runs on SM90, SM100 and SM103, but this GPU is SM{capability[0]}{capability[1]}"
    if importlib.util.find_spec("flash_mla") is None:
        return (
            "`flash-mla` is not installed; install it with `uv sync --extra gpu` (the published wheel is x86_64 only)"
        )
    return None
