import torch
from prime_rl.trainer.rl.score_centering import score_correction, chunked_score_logprobs

torch.manual_seed(17)
for k in (2, 7):
    for eps in (None, 0.3):
        logits = torch.tensor([[2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0]], requires_grad=True)
        logp = logits.log_softmax(-1)
        q = torch.tensor([[0.04, 0.72, 0.12, 0.06, 0.03, 0.02, 0.01]])
        q_head, ids = q.topk(k)
        correction = score_correction(logp, ids, q_head.log(), eps)
        p = logp.exp().detach()
        p_head = p.gather(-1, ids)
        rho = (1 - q_head.sum(-1)) / (1 - p_head.sum(-1)).clamp_min(1e-6)
        qhat = rho[:, None] * p
        qhat.scatter_(-1, ids, q_head)
        weight = (
            torch.ones_like(p) if eps is None else torch.where((p - qhat).abs() <= eps, p / qhat.clamp_min(1e-30), 0.0)
        )
        expected = ((qhat * weight).detach() * logp).sum(-1)
        (actual_grad,) = torch.autograd.grad(correction.sum(), logits, retain_graph=True)
        (expected_grad,) = torch.autograd.grad(expected.sum(), logits, retain_graph=True)
        torch.testing.assert_close(actual_grad, expected_grad, atol=3e-7, rtol=2e-5)
        (centered_grad,) = torch.autograd.grad((expected - correction).sum(), logits)
        assert centered_grad.abs().max() < 3e-7
        print("constant-reward drift canceled", k, eps)

h = torch.randn(1, 5, 4, requires_grad=True)
w = torch.randn(7, 4, requires_grad=True)
target = torch.tensor([[0, 1, 2, 3, 4]])
temp = torch.ones_like(target, dtype=torch.float)
q = torch.randn(1, 5, 7).softmax(-1)
qhead, ids = q.topk(3)
a = chunked_score_logprobs(h, w, target, temp, ids, qhead.log(), 2, 0.3)
logp = (h @ w.t()).float().log_softmax(-1)
b = (
    logp.gather(-1, target[..., None]).squeeze(-1),
    -(logp.exp() * logp).sum(-1),
    score_correction(logp, ids, qhead.log(), 0.3),
)
for x, y in zip(a, b):
    torch.testing.assert_close(x, y)
ga = torch.autograd.grad((a[0] + a[2]).sum(), (h, w), retain_graph=True)
gb = torch.autograd.grad((b[0] + b[2]).sum(), (h, w))
for x, y in zip(ga, gb):
    torch.testing.assert_close(x, y, atol=1e-6, rtol=1e-5)
print("chunked outputs and gradients match dense reference")
from prime_rl.configs.trainer import IPOLossConfig
from prime_rl.trainer.rl.loss import IPOLoss, LossInputs

for enabled in (False, True):
    logits = torch.randn(5, 7, requires_grad=True)
    logp = logits.log_softmax(-1)
    q = torch.randn(5, 7).softmax(-1)
    qhead, ids = q.topk(4)
    correction = score_correction(logp, ids, qhead.log(), 0.3)
    adv = torch.tensor([-1.0, 0.5, 0.0, 2.0, -0.25])
    mask = torch.tensor([True, True, False, True, True])
    weights = torch.tensor([1.0, 0.5, 0.75, 2.0, 1.0])
    actual = IPOLoss(IPOLossConfig(eps=0.3, kl_tau=0.001, score_centering=enabled)).loss(
        LossInputs(
            logp[:, 0], q[:, 0].log(), None, adv, mask,
            loss_weights=weights, score_correction=correction,
        )
    ).loss
    ratio_log = logp[:, 0] - q[:, 0].log()
    accepted = (logp[:, 0].exp() - q[:, 0]).abs() <= 0.3
    reference = -adv * ratio_log.exp() * accepted + 0.001 * ratio_log.square()
    if enabled:
        p = logp.detach().exp()
        phead = p.gather(-1, ids)
        rho = (1 - qhead.sum(-1)).clamp_min(1e-6) / (1 - phead.sum(-1)).clamp_min(1e-6)
        qhat = rho[:, None] * p
        qhat.scatter_(-1, ids, qhead)
        keep = (p - qhat).abs() <= 0.3
        expected_score = ((p * keep).detach() * logp).sum(-1)
        reference = reference + adv * expected_score
    reference = (reference * mask * weights).sum()
    actual_grad, = torch.autograd.grad(actual, logits, retain_graph=True)
    reference_grad, = torch.autograd.grad(reference, logits)
    torch.testing.assert_close(actual_grad, reference_grad, atol=1e-6, rtol=1e-5)
    print("IPO loss gradients match independent weighted-score reference", enabled)

logp = torch.tensor([[0.0, 0.1, -0.1]], requires_grad=True).log_softmax(-1)
q = torch.tensor([[0.32, 0.35, 0.33]])
qhead, ids = q.topk(3)
correction = score_correction(logp, ids, qhead.log(), 0.3)
assert correction.item() == 0.0
assert torch.autograd.grad(correction.sum(), logp)[0].abs().max() == 0
print("IPO correction is zero when all tokens pass its mask")

from prime_rl.configs.trainer import ScoreCenteringLossConfig
from prime_rl.trainer.rl.loss import ScoreCenteringLoss

logits = torch.randn(5, 7, requires_grad=True)
logp = logits.log_softmax(-1)
q = torch.randn(5, 7).softmax(-1)
qh, ids = q.topk(3)
correction = score_correction(logp, ids, qh.log(), None)
adv = torch.randn(5)
mask = torch.tensor([True, True, False, True, False])
actual = ScoreCenteringLoss(ScoreCenteringLossConfig()).loss(
    LossInputs(logp[:, 0], q[:, 0].log(), None, adv, mask, score_correction=correction)
).loss
ph = logp.gather(-1, ids).exp()
rho = (1-qh.sum(-1)).clamp_min(1e-6)/(1-ph.sum(-1)).clamp_min(1e-6)
reference = (-adv * (logp[:, 0] - ((qh-rho[:, None]*ph).detach()*logp.gather(-1, ids)).sum(-1)))[mask].sum()
torch.testing.assert_close(actual, reference)
grad_actual, = torch.autograd.grad(actual, logits, retain_graph=True)
grad_reference, = torch.autograd.grad(reference, logits)
torch.testing.assert_close(grad_actual, grad_reference)
print("standalone loss matches official PyTorch reference")
