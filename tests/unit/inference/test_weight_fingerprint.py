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
    assert {
        name: {key: everything[name][key] for key in ("kind", "shape", "dtype", "data_ptr", "sha256")}
        for name in weights
    } == {name: {key: value for key, value in entry.items() if key != "nonzero"} for name, entry in weights.items()}


def test_audit_fingerprints_fp8_and_preserves_shared_storage_metadata():
    model = nn.Linear(2, 2)
    model.register_buffer("scales", torch.tensor([125, 127], dtype=torch.uint8).view(torch.float8_e8m0fnu))
    model.register_buffer("weight_view", model.weight.detach().T)
    worker = WeightRPCMixin()
    worker.model_runner = SimpleNamespace(get_model=lambda: model, kv_caches=[])

    entries = worker.get_weight_fingerprint(buffers=True)
    assert entries["scales"]["dtype"] == "torch.float8_e8m0fnu"
    assert entries["weight_view"]["stride"] == [1, 2]
    assert entries["weight"]["storage_ptr"] == entries["weight_view"]["storage_ptr"]


def test_audit_checks_both_qwen_hyperconnection_projections_and_skips_their_padding():
    model = nn.Module()
    model.hc = nn.Module()
    model.hc.pad_size = 2
    model.hc.input_mix_weight_down_block_inject = nn.Linear(2, 4, bias=False)
    worker = WeightRPCMixin()
    worker.model_runner = SimpleNamespace(get_model=lambda: model, kv_caches=[])
    name = "hc.input_mix_weight_down_block_inject.weight"
    weight = model.hc.input_mix_weight_down_block_inject.weight
    reference = worker.get_weight_fingerprint(buffers=True)[name]
    with torch.no_grad():
        weight[2:].add_(10)
        assert worker.get_weight_fingerprint(buffers=True)[name] == reference
        weight[1].add_(10)
    assert worker.get_weight_fingerprint(buffers=True)[name]["sha256"] != reference["sha256"]
