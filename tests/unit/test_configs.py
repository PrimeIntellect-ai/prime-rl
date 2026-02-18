import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from prime_rl.inference.config import InferenceConfig
from prime_rl.orchestrator.config import OrchestratorConfig
from prime_rl.rl import RLConfig
from prime_rl.trainer.rl.config import RLTrainerConfig
from prime_rl.trainer.sft.config import SFTTrainerConfig
from prime_rl.utils.pydantic_config import parse_argv

# All config config classes
CONFIG_CLASSES = [RLConfig, RLTrainerConfig, SFTTrainerConfig, OrchestratorConfig, InferenceConfig]


def get_config_files() -> list[Path]:
    """Any TOML file inside `configs/` or `examples/`"""
    config_files = list(Path("configs").rglob("*.toml"))
    example_files = list(Path("examples").rglob("*.toml"))

    return config_files + example_files


def pd_disagg_overrides(**overrides) -> dict[str, object]:
    config: dict[str, object] = {
        "enabled": True,
        "prefill_gpu_ids": [0],
        "decode_gpu_ids": [1],
    }
    config.update(overrides)
    return config


_INFERENCE_DEFAULT = object()


def build_rl_config(
    *,
    inference: InferenceConfig | None | object = _INFERENCE_DEFAULT,
    pd_disagg: dict[str, object] | None = None,
    **overrides,
) -> RLConfig:
    resolved_inference = InferenceConfig() if inference is _INFERENCE_DEFAULT else inference
    return RLConfig(
        trainer=RLTrainerConfig(),
        orchestrator=OrchestratorConfig(),
        inference=resolved_inference,
        pd_disagg=pd_disagg_overrides(**(pd_disagg or {})),
        **overrides,
    )


@pytest.mark.parametrize("config_file", get_config_files(), ids=lambda x: x.as_posix())
def test_load_configs(config_file: Path, monkeypatch: pytest.MonkeyPatch):
    """Tests that all config files can be loaded by at least one config class."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["dummy.py", "@", config_file.as_posix()],
        raising=False,
    )
    could_parse = []
    for config_cls in CONFIG_CLASSES:
        try:
            parse_argv(config_cls)
            could_parse.append(True)
        except ValidationError:
            could_parse.append(False)
    assert any(could_parse), f"No config class could be parsed from {config_file}"


def test_rl_config_pd_disagg_sets_proxy_client_base_url():
    config = build_rl_config(
        pd_disagg={
            "host": "127.0.0.1",
            "proxy_port": 9000,
            "prefill_gpu_ids": [2],
            "decode_gpu_ids": [3],
        }
    )
    assert config.orchestrator.client.base_url == ["http://127.0.0.1:9000/v1"]


def test_rl_config_pd_disagg_requires_inference():
    with pytest.raises(ValueError, match="requires an \\[inference\\] config"):
        build_rl_config(
            inference=None,
            pd_disagg={
                "prefill_gpu_ids": [0],
                "decode_gpu_ids": [1],
            },
        )


def test_rl_config_pd_disagg_disallows_overlapping_gpus():
    with pytest.raises(ValueError, match="must not overlap"):
        build_rl_config(
            pd_disagg={
                "prefill_gpu_ids": [0],
                "decode_gpu_ids": [0],
            },
        )


def test_rl_config_pd_disagg_allows_multiple_workers_when_gpu_lists_match_tp():
    config = build_rl_config(
        inference=InferenceConfig(parallel={"tp": 2}),
        pd_disagg={
            "host": "127.0.0.1",
            "proxy_port": 9000,
            "prefill_gpu_ids": [0, 1, 2, 3],
            "decode_gpu_ids": [4, 5],
        },
    )
    assert config.orchestrator.client.base_url == ["http://127.0.0.1:9000/v1"]


def test_rl_config_pd_disagg_requires_gpu_ids_to_be_multiple_of_role_tp():
    with pytest.raises(ValueError, match="must be a multiple of prefill_tp"):
        build_rl_config(
            inference=InferenceConfig(parallel={"tp": 2}),
            pd_disagg={
                "prefill_gpu_ids": [0, 1, 2],
                "decode_gpu_ids": [3, 4],
            },
        )


def test_rl_config_pd_disagg_supports_role_specific_tp():
    config = build_rl_config(
        inference=InferenceConfig(parallel={"tp": 1}),
        pd_disagg={
            "host": "127.0.0.1",
            "proxy_port": 9000,
            "prefill_port": 8100,
            "decode_port": 8200,
            "prefill_kv_port": 14579,
            "decode_kv_port": 14679,
            "prefill_tp": 1,
            "decode_tp": 2,
            "prefill_gpu_ids": [0, 1],
            "decode_gpu_ids": [2, 3, 4, 5],
        },
    )
    assert config.orchestrator.client.base_url == ["http://127.0.0.1:9000/v1"]


def test_rl_config_pd_disagg_disallows_overlapping_ports_when_auto_ports_disabled():
    with pytest.raises(ValueError, match="port ranges must all be distinct"):
        build_rl_config(
            inference=InferenceConfig(parallel={"tp": 2}),
            pd_disagg={
                "auto_ports": False,
                "prefill_gpu_ids": [0, 1, 2, 3],
                "decode_gpu_ids": [4, 5],
            },
        )


def test_rl_config_pd_disagg_allows_nccl_weight_broadcast():
    config = build_rl_config(
        inference=InferenceConfig(parallel={"tp": 2}),
        weight_broadcast={"type": "nccl"},
        pd_disagg={
            "prefill_gpu_ids": [0, 1, 2, 3],
            "decode_gpu_ids": [4, 5],
        },
    )
    assert config.trainer.weight_broadcast.type == "nccl"
    assert config.orchestrator.weight_broadcast.type == "nccl"
    assert config.trainer.weight_broadcast.inference_world_size == 6


def test_rl_config_pd_disagg_nccl_requires_matching_role_tp():
    with pytest.raises(ValueError, match="requires matching prefill_tp and decode_tp"):
        build_rl_config(
            inference=InferenceConfig(parallel={"tp": 1}),
            weight_broadcast={"type": "nccl"},
            pd_disagg={
                "prefill_tp": 1,
                "decode_tp": 2,
                "prefill_gpu_ids": [0],
                "decode_gpu_ids": [2, 3],
            },
        )


def test_inference_config_expert_parallel_fields_translate_to_vllm_namespace():
    config = InferenceConfig(
        enable_expert_parallel=True,
        all2all_backend="pplx",
        enable_eplb=True,
    )
    vllm_config = config.to_vllm()
    assert vllm_config.enable_expert_parallel is True
    assert vllm_config.all2all_backend == "pplx"
    assert vllm_config.enable_eplb is True


def test_rl_config_pd_disagg_accepts_role_specific_all2all_overrides():
    config = build_rl_config(
        inference=InferenceConfig(enable_expert_parallel=True, all2all_backend="allgather_reducescatter"),
        pd_disagg={
            "host": "127.0.0.1",
            "proxy_port": 9000,
            "prefill_gpu_ids": [0],
            "decode_gpu_ids": [1],
            "prefill_all2all_backend": "deepep_high_throughput",
            "decode_all2all_backend": "deepep_low_latency",
        },
    )
    assert config.pd_disagg.prefill_all2all_backend == "deepep_high_throughput"
    assert config.pd_disagg.decode_all2all_backend == "deepep_low_latency"
