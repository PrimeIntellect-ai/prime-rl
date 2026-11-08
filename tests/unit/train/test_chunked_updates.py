import torch

from prime_rl.trainer.optim import subset_step
from prime_rl.trainer.optim.subset_step import UpdateHooks


class HookedAdamW(UpdateHooks, torch.optim.AdamW):
    pass


def test_chunked_step_matches_plain_step(monkeypatch):
    monkeypatch.setattr(subset_step, "HOOK_NUMEL", 10)
    params = [torch.nn.Parameter(torch.randn(numel)) for numel in (4, 8, 16, 2)]
    reference = [torch.nn.Parameter(param.detach().clone()) for param in params]
    for param in params + reference:
        param.grad = torch.ones_like(param)
    params[3].grad = reference[3].grad = None
    optimizer, plain = HookedAdamW(params, lr=0.1), torch.optim.AdamW(reference, lr=0.1)
    chunks = []

    def hook(chunk):
        chunks.append([param.numel() for param in chunk])
        return lambda: None

    optimizer.update_hook = hook
    for _ in range(2):
        optimizer.step()
        plain.step()

    assert chunks == [[4, 8], [16]] * 2
    assert all(torch.equal(param, ref) for param, ref in zip(params, reference))
    assert [optimizer.state[param]["step"] for param in params[:3]] == [
        plain.state[ref]["step"] for ref in reference[:3]
    ]
    assert optimizer.param_groups[0]["params"] == params
    assert optimizer.update_hook is hook
