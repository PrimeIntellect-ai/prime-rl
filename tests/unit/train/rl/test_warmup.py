from types import SimpleNamespace

import pytest
import torch

from prime_rl.trainer.rl.warmup import synthetic_micro_batch, warmup_micro_batches


@pytest.mark.parametrize("seq_len,multiple", [(131072, 4), (4096, 1), (65536, 256), (100, 32)])
def test_warmup_micro_batches(seq_len: int, multiple: int) -> None:
    full, near_full, long = warmup_micro_batches(seq_len, multiple)
    assert sum(full) == -(-seq_len // multiple) * multiple
    assert sum(near_full) == sum(long) == sum(full) - multiple
    assert max(full) < sum(full) // 8
    assert max(long) == sum(long) // 2
    assert all(doc_len > 0 for doc_len in full + near_full + long)


@pytest.mark.parametrize("score_centering", [False, True])
def test_synthetic_micro_batch(score_centering: bool) -> None:
    model = SimpleNamespace(config=SimpleNamespace(vocab_size=1000))
    loss = SimpleNamespace(captures_sampler_head=score_centering, score_centering_k=4)
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
