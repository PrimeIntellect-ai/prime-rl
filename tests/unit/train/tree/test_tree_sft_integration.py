import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from prime_rl.configs.sft import CaterpillarFakeDataConfig, SFTConfig
from prime_rl.configs.trainer import ModelConfig
from prime_rl.trainer.model import forward
from prime_rl.trainer.sft.data import setup_dataloader, setup_dataset
from prime_rl.trainer.tree import build_caterpillar, tree_nll_loss
from tests.unit.train.tree._tiny_model import TinyTransformer, causal_mask

pytestmark = [pytest.mark.gpu]


class _Tokenizer:
    vocab_size = 128


class _TinyLmWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TinyTransformer(vocab_size=_Tokenizer.vocab_size)

    def forward(self, input_ids, position_ids=None, attention_mask=None, **kwargs):
        return {"logits": self.model(input_ids, position_ids, attention_mask)}


def _rand_len(bounds: tuple[int, int], generator: torch.Generator) -> int:
    low, high = bounds
    return int(torch.randint(low, high + 1, (1,), generator=generator).item())


def _rand_ids(length: int, generator: torch.Generator) -> list[int]:
    return torch.randint(0, _Tokenizer.vocab_size, (length,), generator=generator).tolist()


def _build_tree(config: CaterpillarFakeDataConfig, sample_idx: int):
    generator = torch.Generator().manual_seed(config.seed + sample_idx)
    turns = []
    for _ in range(config.num_turns):
        turns.append(
            (
                _rand_ids(_rand_len(config.user_len, generator), generator),
                _rand_ids(_rand_len(config.think_len, generator), generator),
                _rand_ids(_rand_len(config.response_len, generator), generator),
            )
        )
    return build_caterpillar(turns, train_response=config.train_response, train_think=config.train_think)


def _branch_average_loss(model: _TinyLmWrapper, tree) -> torch.Tensor:
    loss = torch.zeros((), device="cuda")
    for leaf_idx in tree.leaves():
        path = tree.root_path(leaf_idx)
        ids = [token for node_idx in path for token in tree.nodes[node_idx].token_ids]
        masks = [mask for node_idx in path for mask in tree.nodes[node_idx].loss_mask]
        input_ids = torch.tensor(ids, dtype=torch.long, device="cuda").unsqueeze(0)
        position_ids = torch.arange(len(ids), dtype=torch.long, device="cuda").unsqueeze(0)
        logits = model.model(input_ids, position_ids, causal_mask(len(ids), device=torch.device("cuda")))
        ce = F.cross_entropy(logits[0, :-1], input_ids[0, 1:], reduction="none")
        loss = loss + (ce * torch.tensor(masks[1:], dtype=ce.dtype, device="cuda")).sum() / len(tree.leaves())
    return loss


def test_tree_sft_data_and_loss_match_branch_baseline_for_first_three_steps():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for SFT dataloader integration")

    data_config = CaterpillarFakeDataConfig(
        batch_size=1,
        seq_len=64,
        num_turns=3,
        user_len=(2, 3),
        think_len=(3, 4),
        response_len=(3, 4),
        seed=123,
    )
    SFTConfig(model=ModelConfig(attn="sdpa", cp=1, ep=1, impl="hf"), data=data_config, loss_impl="torch")

    dataset = setup_dataset(_Tokenizer(), data_config)
    dataiter = iter(setup_dataloader(dataset, data_config))

    torch.manual_seed(5)
    model = _TinyLmWrapper().cuda()
    model.eval()

    for sample_idx in range(3):
        micro_batch = next(dataiter)
        out = forward(
            model,
            micro_batch["input_ids"],
            micro_batch["position_ids"],
            attn_mask=micro_batch["attn_mask"],
        )
        tree_loss = tree_nll_loss(
            out["logits"],
            micro_batch["input_ids"],
            micro_batch["prev_map"],
            micro_batch["loss_mask"],
            micro_batch["loss_weights"],
        )
        branch_loss = _branch_average_loss(model, _build_tree(data_config, sample_idx))
        torch.testing.assert_close(tree_loss, branch_loss, rtol=1e-5, atol=1e-5)
