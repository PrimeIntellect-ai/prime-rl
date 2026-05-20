from functools import lru_cache

import torch
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

_COMPILED_CREATE_BLOCK_MASK = torch.compile(create_block_mask)


def _device_cache_key(device: torch.device | str) -> tuple[str, int | None]:
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and torch_device.index is None:
        return torch_device.type, torch.cuda.current_device()
    return torch_device.type, torch_device.index


@lru_cache(maxsize=128)
def _build_tree_block_mask_cached(
    node_of_token_values: tuple[int, ...],
    is_ancestor_node_values: tuple[bool, ...],
    num_nodes: int,
    seq_len: int,
    device_type: str,
    device_index: int | None,
    block_size: int,
) -> BlockMask:
    device = torch.device(device_type if device_index is None else f"{device_type}:{device_index}")
    num_tokens = len(node_of_token_values)
    node_of_token = torch.tensor(node_of_token_values, dtype=torch.long, device=device)
    is_ancestor_node = torch.tensor(is_ancestor_node_values, dtype=torch.bool, device=device).view(num_nodes, num_nodes)

    def tree_mask_mod(batch_idx, head_idx, q_idx, kv_idx):
        q_valid = q_idx < num_tokens
        kv_valid = kv_idx < num_tokens
        safe_q_idx = q_idx.clamp(max=num_tokens - 1)
        safe_kv_idx = kv_idx.clamp(max=num_tokens - 1)
        q_node = node_of_token[safe_q_idx]
        kv_node = node_of_token[safe_kv_idx]
        causal = kv_idx <= q_idx
        ancestor = is_ancestor_node[q_node, kv_node]
        return q_valid & kv_valid & causal & ancestor

    return _COMPILED_CREATE_BLOCK_MASK(
        tree_mask_mod,
        B=None,
        H=None,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device=device,
        BLOCK_SIZE=block_size,
    )


def build_tree_block_mask(
    node_of_token: torch.LongTensor,
    is_ancestor_node: torch.BoolTensor,
    seq_len: int | None = None,
    device: torch.device | str | None = None,
    block_size: int = 128,
) -> BlockMask:
    """Build a FlexAttention BlockMask for DFS-packed tree attention."""
    if node_of_token.ndim != 1:
        raise ValueError("node_of_token must be a 1D tensor")
    if node_of_token.numel() == 0:
        raise ValueError("node_of_token must not be empty")
    if is_ancestor_node.ndim != 2 or is_ancestor_node.shape[0] != is_ancestor_node.shape[1]:
        raise ValueError("is_ancestor_node must be a square 2D tensor")
    if is_ancestor_node.shape[0] <= int(node_of_token.max().item()):
        raise ValueError("is_ancestor_node is too small for node_of_token")

    seq_len = node_of_token.numel() if seq_len is None else seq_len
    if seq_len < node_of_token.numel():
        raise ValueError("seq_len must be at least the packed token count")
    cache_device = node_of_token.device if device is None else torch.device(device)
    device_type, device_index = _device_cache_key(cache_device)
    node_values = tuple(int(v) for v in node_of_token.detach().cpu().tolist())
    ancestor_values = tuple(bool(v) for v in is_ancestor_node.detach().cpu().flatten().tolist())
    return _build_tree_block_mask_cached(
        node_values,
        ancestor_values,
        int(is_ancestor_node.shape[0]),
        int(seq_len),
        device_type,
        device_index,
        int(block_size),
    )
