import pytest
import torch

from prime_rl.trainer.tree import build_caterpillar, build_tree_block_mask, pack_tree, tree_nll_loss
from tests.unit.train.tree._tiny_model import TinyTransformer

pytestmark = [pytest.mark.gpu]


def _rand_ids(length: int, generator: torch.Generator) -> list[int]:
    return torch.randint(1, 80, (length,), generator=generator).tolist()


def _relative_max(actual: torch.Tensor, expected: torch.Tensor) -> float:
    diff = (actual - expected).abs().max().item()
    denom = max(expected.abs().max().item(), 1e-12)
    return diff / denom


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for FlexAttention")
def test_flex_attention_matches_sdpa_path_fp32():
    generator = torch.Generator().manual_seed(0)
    tree = build_caterpillar(
        turns=[
            (_rand_ids(5, generator), _rand_ids(8, generator), _rand_ids(6, generator)),
            (_rand_ids(4, generator), _rand_ids(7, generator), _rand_ids(6, generator)),
        ]
    )
    device = torch.device("cuda")
    packed = pack_tree(tree, device=device)

    torch.manual_seed(1)
    model = TinyTransformer(
        vocab_size=128,
        hidden_size=32,
        num_heads=4,
        num_layers=1,
        max_position_embeddings=128,
    ).to(device)
    model.eval()

    input_ids = packed.input_ids.unsqueeze(0)
    position_ids = packed.position_ids.unsqueeze(0)
    prev_map = packed.prev_map.unsqueeze(0)
    loss_mask = packed.loss_mask.unsqueeze(0)
    loss_weights = packed.loss_weights.unsqueeze(0)

    sdpa_logits = model(input_ids, position_ids, packed.attn_mask.unsqueeze(0).unsqueeze(0))
    sdpa_loss = tree_nll_loss(sdpa_logits, input_ids, prev_map, loss_mask, loss_weights)
    sdpa_loss.backward()
    sdpa_grads = {name: param.grad.detach().clone() for name, param in model.named_parameters()}
    model.zero_grad(set_to_none=True)

    block_mask = build_tree_block_mask(
        packed.node_of_token,
        packed.is_ancestor_node,
        seq_len=packed.input_ids.numel(),
        device=device,
    )
    flex_logits = model(input_ids, position_ids, block_mask)
    flex_loss = tree_nll_loss(flex_logits, input_ids, prev_map, loss_mask, loss_weights)
    flex_loss.backward()
    flex_grads = {name: param.grad.detach().clone() for name, param in model.named_parameters()}

    assert _relative_max(flex_logits, sdpa_logits) < 1e-4
    assert _relative_max(flex_loss, sdpa_loss) < 1e-4
    for name, sdpa_grad in sdpa_grads.items():
        assert _relative_max(flex_grads[name], sdpa_grad) < 1e-4, name
