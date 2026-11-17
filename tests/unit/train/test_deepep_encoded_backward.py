import importlib
import sys
from types import ModuleType, SimpleNamespace

import torch


def test_dispatch_backward_without_a_row_gradient_combines_zero_rows(monkeypatch):
    # Encoded (NVFP4) rows need no gradient while the routing weights do: autograd passes None for the rows
    stub = ModuleType("deep_ep")
    stub.Buffer = object
    monkeypatch.setitem(sys.modules, "deep_ep", stub)
    monkeypatch.delitem(sys.modules, "prime_rl.trainer.distributed.deepep", raising=False)
    deepep = importlib.import_module("prime_rl.trainer.distributed.deepep")
    seen = {}

    def fake_backward_op(grad_x, grad_topk_weights, handle_id, num_tokens):
        seen["grad_x"] = grad_x
        return torch.zeros(num_tokens, grad_x.shape[1]), grad_topk_weights

    monkeypatch.setattr(deepep, "_dispatch_backward_op", fake_backward_op)
    ctx = SimpleNamespace(
        saved_tensors=(torch.zeros(1, dtype=torch.int64),),
        num_tokens=5,
        recv_x=(torch.Size([7, 24]), torch.bfloat16, torch.device("cpu")),
    )
    grad_weights = torch.ones(7, 2)
    _, _, grad_topk_weights, _ = deepep._dispatch_backward(ctx, None, grad_weights)
    assert torch.equal(seen["grad_x"], torch.zeros(7, 24, dtype=torch.bfloat16))
    assert torch.equal(grad_topk_weights, grad_weights)
