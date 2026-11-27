from types import SimpleNamespace

import torch
from torch import nn

from prime_rl.inference.vllm.worker.weight_transfer import WeightRPCMixin


def test_fingerprint_covers_buffers_on_request():
    model = nn.Linear(2, 2)
    model.register_buffer("scale", torch.tensor(1.0))
    worker = WeightRPCMixin()
    worker.model_runner = SimpleNamespace(get_model=lambda: model, kv_caches=[])

    weights = worker.get_weight_fingerprint()
    assert {name: entry["kind"] for name, entry in weights.items()} == {"weight": "parameter", "bias": "parameter"}

    everything = worker.get_weight_fingerprint(buffers=True)
    assert everything["scale"]["kind"] == "buffer" and everything["scale"]["shape"] == []
    assert {name: everything[name] for name in weights} == weights
