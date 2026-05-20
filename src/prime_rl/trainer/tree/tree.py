from dataclasses import dataclass


@dataclass(frozen=True)
class TreeNode:
    parent: int
    token_ids: list[int]
    loss_mask: list[bool]
    advantage: float = 0.0

    def __post_init__(self):
        if len(self.token_ids) == 0:
            raise ValueError("Tree nodes must contain at least one token")
        if len(self.token_ids) != len(self.loss_mask):
            raise ValueError("Tree node token_ids and loss_mask must have the same length")


@dataclass(frozen=True)
class Tree:
    nodes: list[TreeNode]

    def __post_init__(self):
        if len(self.nodes) == 0:
            raise ValueError("Tree must contain at least one node")
        if self.nodes[0].parent != -1:
            raise ValueError("Tree root must have parent -1")
        for idx, node in enumerate(self.nodes[1:], start=1):
            if node.parent < 0 or node.parent >= idx:
                raise ValueError("Tree nodes must be topologically sorted with parent indices before children")

    def children(self, i: int) -> list[int]:
        self._validate_node_idx(i)
        return [idx for idx, node in enumerate(self.nodes) if node.parent == i]

    def leaves(self) -> list[int]:
        parents = {node.parent for node in self.nodes[1:]}
        return [idx for idx in range(len(self.nodes)) if idx not in parents]

    def root_path(self, leaf_idx: int) -> list[int]:
        self._validate_node_idx(leaf_idx)
        if leaf_idx not in self.leaves():
            raise ValueError(f"Node {leaf_idx} is not a leaf")

        path = []
        node_idx = leaf_idx
        while node_idx != -1:
            path.append(node_idx)
            node_idx = self.nodes[node_idx].parent
        return list(reversed(path))

    def _validate_node_idx(self, i: int) -> None:
        if i < 0 or i >= len(self.nodes):
            raise IndexError(f"Tree node index out of range: {i}")
