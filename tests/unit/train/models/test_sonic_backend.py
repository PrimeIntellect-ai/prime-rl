import torch

from prime_rl.trainer.models.layers.sonic_backend import check_sonic_runtime


def test_sonic_runtime_requires_grouped_experts() -> None:
    runtime_info = check_sonic_runtime(
        hidden_size=1024,
        intermediate_size=512,
        top_k=2,
        input_dtype=torch.bfloat16,
        score_before_experts=False,
        has_grouped_experts=False,
        device=torch.device("cuda"),
    )

    assert not runtime_info.is_supported
    assert runtime_info.code == "unsupported_experts_module"


def test_sonic_runtime_rejects_score_before_experts() -> None:
    runtime_info = check_sonic_runtime(
        hidden_size=1024,
        intermediate_size=512,
        top_k=2,
        input_dtype=torch.bfloat16,
        score_before_experts=True,
        has_grouped_experts=True,
        device=torch.device("cuda"),
    )

    assert not runtime_info.is_supported
    assert runtime_info.code == "score_before_experts_not_supported"


def test_sonic_runtime_rejects_unsupported_dtype() -> None:
    runtime_info = check_sonic_runtime(
        hidden_size=1024,
        intermediate_size=512,
        top_k=2,
        input_dtype=torch.float32,
        score_before_experts=False,
        has_grouped_experts=True,
        device=torch.device("cuda"),
    )

    assert not runtime_info.is_supported
    assert runtime_info.code == "unsupported_dtype"


def test_sonic_runtime_accepts_supported_config(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (9, 0))
    monkeypatch.setattr(torch.version, "cuda", "12.9", raising=False)

    import prime_rl.trainer.models.layers.sonic_backend as sonic_backend

    monkeypatch.setattr(sonic_backend, "_module_missing", lambda _module_name: False)

    runtime_info = check_sonic_runtime(
        hidden_size=1024,
        intermediate_size=512,
        top_k=2,
        input_dtype=torch.bfloat16,
        score_before_experts=False,
        has_grouped_experts=True,
        device=torch.device("cuda"),
    )

    assert runtime_info.is_supported
    assert runtime_info.code == "ok"
