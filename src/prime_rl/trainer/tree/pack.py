from dataclasses import dataclass

import torch

from prime_rl.trainer.tree.tree import Tree


@dataclass(frozen=True)
class PackedTree:
    input_ids: torch.LongTensor
    position_ids: torch.LongTensor
    attn_mask: torch.BoolTensor
    loss_mask: torch.BoolTensor
    loss_weights: torch.Tensor
    prev_map: torch.LongTensor
    node_of_token: torch.LongTensor
    is_ancestor_node: torch.BoolTensor
    node_token_range: list[tuple[int, int]]
    K: int


def _dfs_preorder(tree: Tree, node_idx: int = 0) -> list[int]:
    order = [node_idx]
    for child_idx in tree.children(node_idx):
        order.extend(_dfs_preorder(tree, child_idx))
    return order


def _node_depths(tree: Tree) -> list[int]:
    depths = [0] * len(tree.nodes)
    for idx, node in enumerate(tree.nodes[1:], start=1):
        parent = node.parent
        depths[idx] = depths[parent] + len(tree.nodes[parent].token_ids)
    return depths


def _leaf_counts(tree: Tree) -> list[int]:
    counts = [0] * len(tree.nodes)
    for node_idx in reversed(range(len(tree.nodes))):
        children = tree.children(node_idx)
        counts[node_idx] = 1 if not children else sum(counts[child_idx] for child_idx in children)
    return counts


def _ancestor_sets(tree: Tree) -> list[set[int]]:
    ancestors: list[set[int]] = []
    for idx, node in enumerate(tree.nodes):
        if idx == 0:
            ancestors.append({0})
        else:
            ancestors.append(ancestors[node.parent] | {idx})
    return ancestors


def pack_tree(
    tree: Tree,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> PackedTree:
    num_tokens = sum(len(node.token_ids) for node in tree.nodes)
    input_ids = torch.empty(num_tokens, dtype=torch.long, device=device)
    position_ids = torch.empty(num_tokens, dtype=torch.long, device=device)
    loss_mask = torch.empty(num_tokens, dtype=torch.bool, device=device)
    loss_weights = torch.empty(num_tokens, dtype=dtype, device=device)
    prev_map = torch.empty(num_tokens, dtype=torch.long, device=device)
    node_of_token = torch.empty(num_tokens, dtype=torch.long, device=device)
    node_token_range = [(0, 0)] * len(tree.nodes)

    depths = _node_depths(tree)
    leaf_counts = _leaf_counts(tree)
    K = leaf_counts[0]
    if K < 1:
        raise ValueError("Tree must have at least one leaf")

    cursor = 0
    for node_idx in _dfs_preorder(tree):
        node = tree.nodes[node_idx]
        node_len = len(node.token_ids)
        start, end = cursor, cursor + node_len
        node_token_range[node_idx] = (start, end)

        input_ids[start:end] = torch.tensor(node.token_ids, dtype=torch.long, device=device)
        position_ids[start:end] = torch.arange(depths[node_idx], depths[node_idx] + node_len, device=device)
        loss_mask[start:end] = torch.tensor(node.loss_mask, dtype=torch.bool, device=device)
        loss_weights[start:end] = leaf_counts[node_idx] / K
        node_of_token[start:end] = node_idx
        cursor = end

    prev_map[0] = -1
    for node_idx, node in enumerate(tree.nodes):
        start, end = node_token_range[node_idx]
        if node_idx != 0:
            _, parent_end = node_token_range[node.parent]
            prev_map[start] = parent_end - 1
        for token_idx in range(start + 1, end):
            prev_map[token_idx] = token_idx - 1

    loss_weights = loss_weights * loss_mask.to(dtype)

    ancestors = _ancestor_sets(tree)
    is_ancestor_node = torch.zeros((len(tree.nodes), len(tree.nodes)), dtype=torch.bool, device=device)
    for node_idx, ancestor_nodes in enumerate(ancestors):
        is_ancestor_node[node_idx, list(ancestor_nodes)] = True

    attn_mask = torch.zeros((num_tokens, num_tokens), dtype=torch.bool, device=device)
    for query_node_idx in range(len(tree.nodes)):
        query_start, query_end = node_token_range[query_node_idx]
        for key_node_idx in torch.nonzero(is_ancestor_node[query_node_idx], as_tuple=False).flatten().tolist():
            key_start, key_end = node_token_range[key_node_idx]
            if key_node_idx == query_node_idx:
                for query_pos in range(query_start, query_end):
                    attn_mask[query_pos, key_start : query_pos + 1] = True
            else:
                attn_mask[query_start:query_end, key_start:key_end] = True

    if prev_map[0].item() != -1:
        raise AssertionError("prev_map[0] must be -1")
    if num_tokens > 1 and not torch.all(prev_map[1:] >= 0).item():
        raise AssertionError("prev_map must be non-negative after the first token")
    if not torch.all(loss_weights[~loss_mask] == 0).item():
        raise AssertionError("loss_weights must be zero wherever loss_mask is false")

    return PackedTree(
        input_ids=input_ids,
        position_ids=position_ids,
        attn_mask=attn_mask,
        loss_mask=loss_mask,
        loss_weights=loss_weights,
        prev_map=prev_map,
        node_of_token=node_of_token,
        is_ancestor_node=is_ancestor_node,
        node_token_range=node_token_range,
        K=K,
    )
