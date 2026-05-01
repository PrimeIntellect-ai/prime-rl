"""Tree Training v1 support for SFT-only caterpillar experiments.

This package implements the tree data model, DFS packer, ancestor-only SDPA mask, and
weighted NLL used by the synthetic SFT tree path. v1 is intentionally narrow: one
materialized tree per micro-batch, caterpillar-style test data, and HF SDPA models.
The `[N, N]` attention mask is simple but has the expected O(N^2) memory and runtime
cost, so large-tree training needs a kernel or block-sparse replacement first.

`tests/unit/train/tree/test_tree_equivalence.py` is the executable correctness spec:
it checks that packed-tree logits, loss, and gradients match independent per-branch
training.
"""

from prime_rl.trainer.tree.caterpillar import build_caterpillar
from prime_rl.trainer.tree.loss import tree_nll_loss
from prime_rl.trainer.tree.pack import PackedTree, pack_tree
from prime_rl.trainer.tree.tree import Tree, TreeNode

__all__ = [
    "PackedTree",
    "Tree",
    "TreeNode",
    "build_caterpillar",
    "pack_tree",
    "tree_nll_loss",
]
