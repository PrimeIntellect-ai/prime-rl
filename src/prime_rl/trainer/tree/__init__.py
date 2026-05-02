"""Tree Training support for SFT-only caterpillar experiments.

This package implements the tree data model, DFS packer, ancestor-only attention
masks, and weighted NLL used by the SFT tree path. The v1 scope is intentionally
narrow: one tree per micro-batch, caterpillar-style data, and HF dense models.
`attn=sdpa` is the canonical correctness reference and uses a materialized `[N, N]`
mask. `attn=flex_attention` is the v1.1 fast path and builds a FlexAttention
BlockMask from the same packed tree helpers, avoiding the SDPA path's O(N^2)
masked-kernel cost on large trees.

`tests/unit/train/tree/test_tree_equivalence.py` is the executable correctness spec:
it checks that packed-tree logits, loss, and gradients match independent per-branch
training. `tests/perf/test_tree_attention_speedup.py` is the speedup regression spec
for the FlexAttention path.
"""

from prime_rl.trainer.tree.caterpillar import build_caterpillar
from prime_rl.trainer.tree.flex_mask import build_tree_block_mask
from prime_rl.trainer.tree.loss import tree_nll_loss
from prime_rl.trainer.tree.pack import PackedTree, pack_tree
from prime_rl.trainer.tree.tree import Tree, TreeNode

__all__ = [
    "PackedTree",
    "Tree",
    "TreeNode",
    "build_caterpillar",
    "build_tree_block_mask",
    "pack_tree",
    "tree_nll_loss",
]
