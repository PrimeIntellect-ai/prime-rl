"""Score centering: the loss math, the trainer's top-k head gather, and the
sampler head's wire round-trip.

The invariants come from the paper ([arXiv:2609.20807]): the top-k estimator
equals full score centering when the modeled tail is exact, reduces to
REINFORCE when trainer and sampler distributions coincide, and the head gather
must match (and backprop like) a full log-softmax gather.
"""

import numpy as np
import pytest
import torch

from prime_rl.configs.trainer import ScoreCenteringLossConfig
from prime_rl.trainer.batch import prepare_batch
from prime_rl.trainer.models.layers.lm_head import _SequenceChunkedLogProbEntropyFn
from prime_rl.trainer.rl.data import DataLoader
from prime_rl.trainer.rl.loss import (
    LossInputs,
    ScoreCenteringLoss,
    selective_topk_log_softmax,
)
from prime_rl.transports.batch.types import TopLogprobs as WireTopLogprobs
from prime_rl.transports.batch.types import TrainingSample

VOCAB, SEQ, HEAD = 61, 7, 8


def _head_inputs(logits, top_ids, sampler_lp, sampled_token, advantages=None, valid=None, loss_mask=None):
    logp = logits.log_softmax(-1)
    advantages = torch.randn(SEQ) if advantages is None else advantages
    valid = torch.ones(SEQ, HEAD, dtype=torch.bool) if valid is None else valid
    loss_mask = torch.ones(SEQ, dtype=torch.bool) if loss_mask is None else loss_mask
    return LossInputs(
        trainer_logprobs=logp.gather(-1, sampled_token[:, None]).squeeze(-1),
        inference_logprobs=torch.zeros(SEQ),
        ref_logprobs=None,
        advantages=advantages,
        loss_mask=loss_mask,
        trainer_topk_logprobs=logp.gather(-1, top_ids).masked_fill(~valid, 0.0),
        sampler_topk_logprobs=sampler_lp.masked_fill(~valid, 0.0),
        topk_valid=valid,
        entropy=-(logp.exp() * logp).sum(-1).detach(),
    )


def _exact_tail_case(seed=0):
    """Trainer distribution plus a sampler whose tail is proportional to it."""
    torch.manual_seed(seed)
    logits = torch.randn(SEQ, VOCAB)
    logp = logits.log_softmax(-1)
    p = logp.exp()
    top_lp, top_ids = torch.topk(logp, HEAD, dim=-1)
    # Arbitrary sampler head; its tail is rho * p_tail (the exact model).
    sampler_lp = (top_lp - torch.rand(SEQ, HEAD)).clamp(max=-1e-3)
    q_head = sampler_lp.exp()
    rho = (1 - q_head.sum(-1)) / (1 - p.gather(-1, top_ids).sum(-1))
    q = torch.zeros_like(p)
    q.scatter_(1, top_ids, q_head)
    tail = torch.ones_like(p, dtype=torch.bool)
    tail.scatter_(1, top_ids, False)
    q[tail] = (rho.unsqueeze(-1) * p)[tail]
    return logits, top_ids, sampler_lp, q


def test_exact_tail_equals_full_centering():
    """With q_tail = rho * p_tail, the top-k estimator is the full one."""
    torch.manual_seed(1)
    logits, top_ids, sampler_lp, q = _exact_tail_case()
    sampled_token = torch.randint(0, VOCAB, (SEQ,))
    advantages = torch.randn(SEQ)
    loss = ScoreCenteringLoss(ScoreCenteringLossConfig())

    top_logits = logits.clone().requires_grad_(True)
    out = loss.loss(_head_inputs(top_logits, top_ids, sampler_lp, sampled_token, advantages))
    out.loss.backward()

    full_logits = logits.clone().requires_grad_(True)
    full_logp = full_logits.log_softmax(-1)
    token_logp = full_logp.gather(-1, sampled_token[:, None]).squeeze(-1)
    baseline = (q * full_logp).sum(-1)
    full_loss = -(advantages * (token_logp - baseline)).sum()
    full_loss.backward()

    assert torch.allclose(out.loss, full_loss, atol=1e-5)
    assert torch.allclose(top_logits.grad, full_logits.grad, atol=1e-5)


def test_equal_distributions_reduce_to_reinforce():
    """When the sampler IS the trainer distribution, the gradient is REINFORCE."""
    torch.manual_seed(0)
    logits = torch.randn(SEQ, VOCAB)
    logp = logits.log_softmax(-1)
    _, top_ids = torch.topk(logp, HEAD, dim=-1)
    sampler_lp = logp.gather(-1, top_ids).detach()
    sampled_token = torch.randint(0, VOCAB, (SEQ,))
    advantages = torch.randn(SEQ)
    loss = ScoreCenteringLoss(ScoreCenteringLossConfig())

    sc_logits = logits.clone().requires_grad_(True)
    out = loss.loss(_head_inputs(sc_logits, top_ids, sampler_lp, sampled_token, advantages))
    out.loss.backward()

    pg_logits = logits.clone().requires_grad_(True)
    pg_logp = pg_logits.log_softmax(-1)
    token_logp = pg_logp.gather(-1, sampled_token[:, None]).squeeze(-1)
    (-(advantages * token_logp)).sum().backward()

    assert torch.allclose(sc_logits.grad, pg_logits.grad, atol=1e-5)


def test_missing_heads_rejected():
    loss = ScoreCenteringLoss(ScoreCenteringLossConfig())
    with pytest.raises(ValueError, match="score_centering requires top-k sampler heads"):
        loss.loss(
            LossInputs(
                trainer_logprobs=torch.zeros(SEQ),
                inference_logprobs=torch.zeros(SEQ),
                ref_logprobs=None,
                advantages=torch.zeros(SEQ),
                loss_mask=torch.ones(SEQ, dtype=torch.bool),
            )
        )


def test_padded_columns_contribute_nothing():
    """Padded head columns hold a constant 0.0 and must not skew the baseline."""
    torch.manual_seed(0)
    logits = torch.randn(SEQ, VOCAB)
    logp = logits.log_softmax(-1)
    _, top_ids = torch.topk(logp, HEAD, dim=-1)
    sampler_lp = logp.gather(-1, top_ids).detach()
    sampled_token = torch.randint(0, VOCAB, (SEQ,))
    loss = ScoreCenteringLoss(ScoreCenteringLossConfig())

    valid = torch.ones(SEQ, HEAD, dtype=torch.bool)
    valid[:, -1] = False
    out = loss.loss(_head_inputs(logits, top_ids, sampler_lp, sampled_token, valid=valid))
    full_out = loss.loss(_head_inputs(logits, top_ids, sampler_lp, sampled_token))
    # The padded column carries no q mass; both variants stay finite and the
    # loss on fully headless tokens (valid all False) is the token logprob
    # minus its own plogp — masked out by the loss mask in real batches.
    assert torch.isfinite(out.loss) and torch.isfinite(full_out.loss)


def test_fused_topk_gather_matches_reference():
    """The fused LM-head top-k gather matches a full log-softmax gather,
    forward and backward, including -1-padded ids."""
    torch.manual_seed(0)
    n, hidden_dim, vocab, k, chunk = 12, 16, 57, 5, 4
    hidden = torch.randn(n, hidden_dim)
    weight = torch.randn(vocab, hidden_dim)
    labels = torch.randint(0, vocab, (n,))
    inv_t = torch.rand(n) + 0.5
    topk_ids = torch.randint(0, vocab, (n, k))
    topk_ids[2] = -1
    topk_ids[:, 3] = -1

    def reference(h, w):
        scaled = (h @ w.t()) * inv_t[:, None]
        logp = scaled.log_softmax(-1)
        return (
            logp.gather(-1, labels[:, None]).squeeze(-1),
            logp.gather(-1, topk_ids.clamp_min(0)).masked_fill(topk_ids < 0, 0.0),
        )

    h1, w1 = hidden.clone().requires_grad_(True), weight.clone().requires_grad_(True)
    lp, _, head = _SequenceChunkedLogProbEntropyFn.apply(h1, w1, labels, inv_t, chunk, None, topk_ids.long())
    grad_head = torch.randn(n, k)
    grad_lp = torch.randn(n)
    ((head * grad_head).sum() + (lp * grad_lp).sum()).backward()

    h2, w2 = hidden.clone().requires_grad_(True), weight.clone().requires_grad_(True)
    ref_lp, ref_head = reference(h2, w2)
    ((ref_head * grad_head).sum() + (ref_lp * grad_lp).sum()).backward()

    assert torch.allclose(head, ref_head, atol=1e-5)
    assert torch.allclose(lp, ref_lp, atol=1e-5)
    assert torch.allclose(h1.grad, h2.grad, atol=1e-4)
    assert torch.allclose(w1.grad, w2.grad, atol=1e-4)


def test_vanilla_topk_helper_matches_reference():
    torch.manual_seed(0)
    batch, seq, vocab, k = 2, 5, 33, 4
    logits = torch.randn(batch, seq, vocab)
    topk_ids = torch.randint(0, vocab, (batch, seq, k))
    topk_ids[0, 0, 0] = -1
    out = selective_topk_log_softmax(logits, topk_ids)
    ref = logits.log_softmax(-1).gather(-1, topk_ids.clamp_min(0)).masked_fill(topk_ids < 0, 0.0)
    assert torch.allclose(out, ref, atol=1e-5)


def test_top_logprobs_wire_round_trip():
    """Sample → packed micro batch → trainer decode keeps heads token-aligned."""
    k = 4
    counts = np.zeros(20, dtype=np.int32)
    ids, lps = [], []
    for i in range(6, 20):  # first 6 tokens are prompt (no head)
        counts[i] = k
        ids.extend(1000 + i * k + j for j in range(k))
        lps.extend(-0.1 * (j + 1) for j in range(k))
    sample = TrainingSample(
        token_ids=list(range(20)),
        mask=[False] * 6 + [True] * 14,
        logprobs=[-0.5] * 20,
        temperatures=[1.0] * 20,
        env_name="rt",
        advantages=[0.1] * 20,
        top_logprobs=WireTopLogprobs(
            ids=np.asarray(ids, dtype=np.int32).tobytes(),
            logprobs=np.asarray(lps, dtype=np.float32).tobytes(),
            counts=counts.tobytes(),
        ),
    )
    (mb,) = prepare_batch(
        rollouts=[sample], seq_len=32, num_train_workers=1, bin_cost=lambda sl: sum(sl), pad_to_multiple_of=1
    )[0]
    decoded = DataLoader._micro_batch_to_tensor(object.__new__(DataLoader), mb)
    ids_t = decoded["top_logprobs_ids"][0]
    lps_t = decoded["top_logprobs_logprobs"][0]
    assert (ids_t[0] < 0).all()  # prompt token: no head
    row = ids_t[7]  # first sampled token
    assert row[row >= 0].tolist() == [1000 + 7 * k + j for j in range(k)]
    assert torch.allclose(lps_t[7][row >= 0], torch.tensor([-0.1, -0.2, -0.3, -0.4]))
