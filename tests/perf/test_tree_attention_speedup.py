import statistics
import time

import pytest
import torch
import torch.nn.functional as F

from prime_rl.trainer.tree import build_caterpillar, build_tree_block_mask, pack_tree, tree_nll_loss
from tests.unit.train.tree._tiny_model import TinyTransformer, causal_mask

pytestmark = [pytest.mark.gpu, pytest.mark.perf]


def _ids(start: int, length: int, vocab_size: int = 512) -> list[int]:
    return [((start + offset) % (vocab_size - 1)) + 1 for offset in range(length)]


def _build_tree_fixture(device: torch.device):
    tree = build_caterpillar(
        turns=[
            (
                _ids(0, 2048),
                _ids(2048, 1024),
                _ids(3072, 1024),
            )
        ]
    )
    return tree, pack_tree(tree, device=device)


def _time_steps(fn, warmup: int = 3, steps: int = 20) -> float:
    times = []
    for idx in range(warmup + steps):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        if idx >= warmup:
            times.append(time.perf_counter() - start)
    return statistics.median(times)


def _tree_step(model: TinyTransformer, packed, attn_mask):
    model.zero_grad(set_to_none=True)
    input_ids = packed.input_ids.unsqueeze(0)
    position_ids = packed.position_ids.unsqueeze(0)
    logits = model(input_ids, position_ids, attn_mask)
    loss = tree_nll_loss(
        logits,
        input_ids,
        packed.prev_map.unsqueeze(0),
        packed.loss_mask.unsqueeze(0),
        packed.loss_weights.unsqueeze(0),
    )
    loss.backward()


def _branch_step(model: TinyTransformer, tree):
    model.zero_grad(set_to_none=True)
    loss = torch.zeros((), device="cuda")
    for leaf_idx in tree.leaves():
        path = tree.root_path(leaf_idx)
        ids = [token for node_idx in path for token in tree.nodes[node_idx].token_ids]
        masks = [mask for node_idx in path for mask in tree.nodes[node_idx].loss_mask]
        input_ids = torch.tensor(ids[:-1], dtype=torch.long, device="cuda").unsqueeze(0)
        target_ids = torch.tensor(ids[1:], dtype=torch.long, device="cuda")
        position_ids = torch.arange(len(ids) - 1, dtype=torch.long, device="cuda").unsqueeze(0)
        logits = model(input_ids, position_ids, causal_mask(len(ids) - 1, device=torch.device("cuda")))
        ce = F.cross_entropy(logits[0], target_ids, reduction="none")
        loss = loss + (ce * torch.tensor(masks[1:], dtype=torch.float32, device="cuda")).sum() / len(tree.leaves())
    loss.backward()


@pytest.fixture
def perf_fixture():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for tree attention performance tests")
    device = torch.device("cuda")
    tree, packed = _build_tree_fixture(device)
    torch.manual_seed(0)
    model = TinyTransformer(
        vocab_size=512,
        hidden_size=64,
        num_heads=4,
        num_layers=1,
        max_position_embeddings=4096,
    ).to(device)
    model.train()
    return tree, packed, model


def test_tree_flex_beats_tree_sdpa_at_4k(perf_fixture):
    _, packed, model = perf_fixture
    sdpa_mask = packed.attn_mask.unsqueeze(0).unsqueeze(0)
    flex_mask = build_tree_block_mask(
        packed.node_of_token,
        packed.is_ancestor_node,
        seq_len=packed.input_ids.numel(),
        device=torch.device("cuda"),
    )

    sdpa_time = _time_steps(lambda: _tree_step(model, packed, sdpa_mask))
    flex_time = _time_steps(lambda: _tree_step(model, packed, flex_mask))

    assert flex_time <= 0.5 * sdpa_time


def test_tree_flex_beats_per_branch_baseline_at_4k(perf_fixture):
    tree, packed, model = perf_fixture
    flex_mask = build_tree_block_mask(
        packed.node_of_token,
        packed.is_ancestor_node,
        seq_len=packed.input_ids.numel(),
        device=torch.device("cuda"),
    )

    branch_time = _time_steps(lambda: _branch_step(model, tree))
    flex_time = _time_steps(lambda: _tree_step(model, packed, flex_mask))

    assert flex_time <= branch_time / 1.5
