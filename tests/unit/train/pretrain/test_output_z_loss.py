import torch
import torch.nn.functional as F

from prime_rl.trainer.models.layers.lm_head import IGNORE_INDEX, FusedOutputLinear


def test_fused_lm_head_z_loss_matches_autograd_reference():
    torch.manual_seed(0)
    hidden, weight = torch.randn(10, 8), torch.randn(37, 8)
    labels = torch.randint(0, 37, (10,))
    labels[::3] = IGNORE_INDEX
    valid = labels != IGNORE_INDEX
    z_coeff = 0.1

    ref_hidden, ref_weight = hidden.clone().requires_grad_(True), weight.clone().requires_grad_(True)
    logits = ref_hidden @ ref_weight.T
    ce = F.cross_entropy(logits, labels, ignore_index=IGNORE_INDEX, reduction="sum")
    z = logits.logsumexp(dim=-1)[valid].square().sum()
    (0.5 * (ce + z_coeff * z)).backward()

    lm_head = FusedOutputLinear(8, 37, chunk_size=4)
    lm_head.weight = torch.nn.Parameter(weight.clone())
    lm_head.z_coeff = z_coeff
    hidden = hidden.clone().requires_grad_(True)
    out = lm_head(hidden[None], labels[None])
    (0.5 * out["loss"]).backward()

    torch.testing.assert_close(out["loss"], ce.detach())
    torch.testing.assert_close(out["z_loss"], z.detach())
    torch.testing.assert_close(hidden.grad, ref_hidden.grad)
    torch.testing.assert_close(lm_head.weight.grad, ref_weight.grad)
