import torch
import torch.nn.functional as F

from prime_rl.trainer.tree import build_caterpillar, pack_tree, tree_nll_loss
from tests.unit.train.tree._tiny_model import TinyTransformer, causal_mask


def _concat(items: list[list[int]]) -> list[int]:
    return [token for item in items for token in item]


def _rand_ids(length: int, generator: torch.Generator) -> list[int]:
    return torch.randint(1, 80, (length,), generator=generator).tolist()


def test_tree_training_matches_per_branch_baseline_fp64():
    generator = torch.Generator().manual_seed(0)
    tree = build_caterpillar(
        turns=[
            (_rand_ids(5, generator), _rand_ids(8, generator), _rand_ids(6, generator)),
            (_rand_ids(4, generator), _rand_ids(7, generator), _rand_ids(6, generator)),
            (_rand_ids(4, generator), _rand_ids(9, generator), _rand_ids(5, generator)),
        ]
    )
    packed = pack_tree(tree, dtype=torch.float64)

    torch.manual_seed(1)
    model = TinyTransformer(vocab_size=128).double()
    model.eval()

    baseline_loss = torch.zeros((), dtype=torch.float64)
    cached_logits: dict[tuple[int, int], torch.Tensor] = {}
    for leaf_idx in tree.leaves():
        path = tree.root_path(leaf_idx)
        ids = _concat([tree.nodes[node_idx].token_ids for node_idx in path])
        masks = _concat([tree.nodes[node_idx].loss_mask for node_idx in path])
        input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
        position_ids = torch.arange(len(ids), dtype=torch.long).unsqueeze(0)
        logits = model(input_ids, position_ids, causal_mask(len(ids)))
        ce = F.cross_entropy(logits[0, :-1], input_ids[0, 1:], reduction="none")
        baseline_loss = baseline_loss + (ce * torch.tensor(masks[1:], dtype=torch.float64)).sum() / packed.K

        cursor = 0
        for node_idx in path:
            node_len = len(tree.nodes[node_idx].token_ids)
            for offset in range(node_len):
                key = (node_idx, offset)
                value = logits[0, cursor + offset].detach()
                if key in cached_logits:
                    torch.testing.assert_close(cached_logits[key], value, rtol=0, atol=1e-12)
                else:
                    cached_logits[key] = value
            cursor += node_len

    baseline_loss.backward()
    baseline_grads = {name: param.grad.detach().clone() for name, param in model.named_parameters()}
    model.zero_grad(set_to_none=True)

    tree_logits = model(
        packed.input_ids.unsqueeze(0),
        packed.position_ids.unsqueeze(0),
        packed.attn_mask.unsqueeze(0).unsqueeze(0),
    )
    tree_loss = tree_nll_loss(
        tree_logits,
        packed.input_ids.unsqueeze(0),
        packed.prev_map.unsqueeze(0),
        packed.loss_mask.unsqueeze(0),
        packed.loss_weights.unsqueeze(0),
    )
    tree_loss.backward()
    tree_grads = {name: param.grad.detach().clone() for name, param in model.named_parameters()}

    for node_idx, (start, end) in enumerate(packed.node_token_range):
        for offset in range(end - start):
            torch.testing.assert_close(
                tree_logits[0, start + offset], cached_logits[(node_idx, offset)], rtol=0, atol=1e-12
            )

    torch.testing.assert_close(tree_loss, baseline_loss, rtol=0, atol=1e-12)

    for name, baseline_grad in baseline_grads.items():
        diff = (tree_grads[name] - baseline_grad).abs().max().item()
        denom = max(baseline_grad.abs().max().item(), 1e-30)
        assert diff / denom < 1e-10, f"{name}: rel diff {diff / denom:.2e}"
