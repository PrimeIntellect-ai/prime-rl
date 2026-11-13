import torch

from prime_rl.trainer.optim.base import ChunkedUpdates


class CountingSGD(torch.optim.SGD):
    """Counts its steps per param group, like Muon."""

    def step(self, closure=None):
        for group in self.param_groups:
            group["step"] = group.get("step", 0) + 1
        return super().step(closure)


class ChunkedSGD(ChunkedUpdates, CountingSGD):
    CHUNK_NUMEL = 10


def test_chunked_step_matches_plain_step():
    params = [torch.nn.Parameter(torch.randn(numel)) for numel in (4, 8, 16, 2)]
    reference = [torch.nn.Parameter(param.detach().clone()) for param in params]
    for param in params + reference:
        param.grad = torch.ones_like(param)
    params[3].grad = reference[3].grad = None
    optimizer, plain = ChunkedSGD(params, lr=0.1), CountingSGD(reference, lr=0.1)
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
    assert optimizer.param_groups[0]["step"] == plain.param_groups[0]["step"] == 2
    assert optimizer.param_groups[0]["params"] == params
