import torch

from prime_rl.trainer.rl.data import drop_micro_batches

SUPPORT = ("support_ids", "support_logprobs", "support_offsets", "support_replayed")


def micro_batch() -> dict:
    return {
        "loss_mask": torch.ones(4, dtype=torch.bool),
        "rl_weights": torch.ones(4),
        "ce_weights": None,
        "ref_kl_weights": None,
        "support_ids": torch.arange(6),
        "support_logprobs": torch.zeros(6),
        "support_offsets": torch.tensor([0, 2, 3, 5, 6]),
        "support_replayed": torch.zeros(4, dtype=torch.bool),
    }


def test_dropped_micro_batch_loses_its_sampler_support():
    """A rank whose payload fetch failed has no sampler support for a dropped micro batch while its context-parallel
    peers do. Clearing it on every rank keeps all of them off the ragged support all-gather."""
    kept, dropped = micro_batch(), micro_batch()
    drop_micro_batches([kept, dropped], [1])

    assert not dropped["loss_mask"].any()
    assert not dropped["rl_weights"].any()
    assert all(dropped[name] is None for name in SUPPORT)

    assert kept["loss_mask"].all()
    assert all(kept[name] is not None for name in SUPPORT)
