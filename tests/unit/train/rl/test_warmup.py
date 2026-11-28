from types import SimpleNamespace

import pytest
import torch

from prime_rl.trainer.models.layers.moe import GroupedExperts, MoE, TokenChoiceTopKRouter
from prime_rl.trainer.rl.warmup import synthetic_micro_batch, warmup_micro_batches, warmup_routed_experts


@pytest.mark.parametrize("seq_len,multiple", [(131072, 4), (4096, 1), (65536, 256), (100, 32)])
def test_warmup_micro_batches(seq_len: int, multiple: int) -> None:
    full, near_full, long = warmup_micro_batches(seq_len, multiple)
    assert sum(full) == -(-seq_len // multiple) * multiple
    assert sum(near_full) == sum(long) == sum(full) - multiple
    assert max(full) < sum(full) // 8
    assert max(long) == sum(long) // 2
    assert all(doc_len > 0 for doc_len in full + near_full + long)


def test_warmup_routed_experts_uses_layer_visits() -> None:
    router = TokenChoiceTopKRouter(
        dim=4, num_experts=8, top_k=2, score_func="sigmoid", route_norm=True, route_scale=1.0
    )
    moe = MoE(
        router=router,
        experts=GroupedExperts(dim=4, hidden_dim=8, num_experts=8),
        shared_expert=None,
        score_before_experts=False,
        load_balance_coeff=None,
    )
    model = SimpleNamespace(
        model=SimpleNamespace(
            layers=[SimpleNamespace(mlp=moe)],
            config=SimpleNamespace(num_layer_visits=48),
        )
    )

    routed_experts = warmup_routed_experts(model, num_tokens=5, generator=torch.Generator().manual_seed(0))

    assert routed_experts is not None
    assert routed_experts.shape == (1, 5, 48, 2)


@pytest.mark.parametrize("score_centering", [False, True])
def test_synthetic_micro_batch(score_centering: bool) -> None:
    model = SimpleNamespace(config=SimpleNamespace(vocab_size=1000))
    loss = SimpleNamespace(type="icepop", score_centering=score_centering, score_centering_topk=4)
    config = SimpleNamespace(enable_router_replay=False, loss=loss)
    doc_lens = [5, 3, 8]
    micro_batch = synthetic_micro_batch(doc_lens, model, config, torch.Generator().manual_seed(0))

    assert micro_batch["input_ids"].shape == (1, 16)
    assert int(micro_batch["input_ids"].max()) < 1000
    assert micro_batch["position_ids"][0].tolist() == [*range(5), *range(3), *range(8)]
    assert micro_batch["seq_lens"].tolist() == doc_lens
    assert not micro_batch["loss_mask"].any()
    if not score_centering:
        assert micro_batch["support_offsets"] is None
        return
    offsets = micro_batch["support_offsets"]
    assert offsets.tolist() == list(range(0, 16 * 4 + 1, 4))
    rows = micro_batch["support_ids"].view(16, 4)
    assert all(len(set(row.tolist())) == 4 for row in rows)
    assert int(rows.max()) < 1000
    assert (micro_batch["support_logprobs"] <= 0).all()
    assert not micro_batch["support_replayed"].any()
