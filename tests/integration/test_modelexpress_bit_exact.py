"""Compare every MX install with a fresh vLLM load of the same checkpoint."""

import copy
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer, PretrainedConfig

from prime_rl.utils.weights import convert_state_dict_to_hf
from tests.unit.train.models.test_vllm_parity import (
    CASES,
    ENGINE_ARGS,
    TOKENIZER,
    Case,
    build_trainer_model,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.slow,
    # NIXL's UCX backend needs a network transport, and UCX finds TCP interfaces only under /sys/class/net, which
    # sandboxed containers such as Modal's gVisor lack
    pytest.mark.skipif(not Path("/sys/class/net").exists(), reason="UCX finds no TCP interface: no /sys/class/net"),
    pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs a trainer GPU and an inference GPU"),
]

TOOL = Path(__file__).parents[2] / "tools" / "check_weight_update_bit_exact.py"
TIMEOUT = 4800

MX_CAPTURE_FIX = "ai-dynamo/modelexpress#1089 (linear-attention views, concatenated sources, epilogue replay)"
KNOWN_FAILURES = {
    "qwen3_5": MX_CAPTURE_FIX,
    "qwen3_5_moe": MX_CAPTURE_FIX,
    "qwen3_8_flash_next": MX_CAPTURE_FIX,
    "nemotron_h": MX_CAPTURE_FIX,
    "gpt_oss": "#4804 (traceable interleave) and ai-dynamo/modelexpress#1089 (concatenated sources)",
    "inkling2": "#4804 (traceable interleave) and ai-dynamo/modelexpress#1089 (concatenated sources)",
    "afmoe": "#4804 (fp32 router bias), #4814 (vLLM reload patches) and ai-dynamo/modelexpress#1089 (aliased bias)",
    "laguna": "#4804 (fp32 router bias) and #4814 (interim vLLM reload patches)",
    "minimax_m2": "#4804 (fp32 router bias) and #4814 (interim vLLM reload patches)",
}
"""Cases that fail today, with what fixes them. They run as strict xfails, so a fix shows up as a failure until its
entry is removed."""

# The engine cold-loads the checkpoint, so the parity test's dummy load and tokenizer-free engine do not apply
ENGINE = {key: value for key, value in ENGINE_ARGS.items() if key not in ("load_format", "skip_tokenizer_init")}
# vLLM checks that a cold load fills every parameter, including the alignment padding of these models' fused linears
# (Inkling 2's attention projection, Qwen3.8 Flash Next's hyper-connection projections), which no checkpoint holds
UNCHECKED_COLD_LOAD = {"inkling2", "qwen3_8_flash_next"}
# Checkpoint tensors the trainer rounds to bf16 when it loads them, although it publishes them in fp32: the checkpoint
# holds bf16 values, so that it equals the trainer's initial weights
BF16_ON_LOAD = {"inkling2": "model.llm.attn_res."}

# These expectations belong to the fixtures, so changing a production dtype declaration cannot
# also change the oracle. Unlisted tensors travel in BF16.
ROUTER_BIAS = r"mlp\.router\.selection_bias$"
WIRE_FP32 = {
    **dict.fromkeys(("afmoe", "glm4_moe", "glm_moe_dsa", "laguna", "minimax_m2"), ROUTER_BIAS),
    **dict.fromkeys(("qwen3_5", "qwen3_5_moe"), r"linear_attn\.A_log$"),
    "deepseek_v4": r"attn_hc|ffn_hc|hc_head|sinks|position_bias|selection_bias|norm",
    "glm5_next": rf"hc\.(fn|base|scale)$|self_attn\.decay\.(A_log|dt_bias)$|{ROUTER_BIAS}",
    "inkling2": rf"attn_res\.|{ROUTER_BIAS}",
    "nemotron_h": rf"mamba\.(A_log|D)$|{ROUTER_BIAS}",
}


def on_sm100() -> bool:
    return torch.cuda.get_device_capability()[0] == 10


def serveable_config(config: PretrainedConfig) -> PretrainedConfig:
    """The case's config, with what this GPU's kernels require."""
    config = copy.deepcopy(config)
    if on_sm100() and config.architectures[0] == "DeepseekV4ForCausalLM":
        # The SM100 sparse decode kernel takes index_topk in multiples of 64
        config.index_topk = -(-config.index_topk // 64) * 64
    return config


def engine_args(case: Case, config: PretrainedConfig) -> dict:
    engine = {**ENGINE, **case.engine}
    if on_sm100() and config.architectures[0] == "GlmMoeDsaForCausalLM" and config.qk_nope_head_dim not in (128, 192):
        # vLLM serves GLM-5 with an fp8 KV cache on SM100, which only FlashInfer's sparse MLA kernel takes, and only at
        # the production head dims
        engine.setdefault("kv_cache_dtype", "bfloat16")
    if case.name in UNCHECKED_COLD_LOAD:
        engine["model_loader_extra_config"] = {"enable_weights_track": False}
    return engine


def write_checkpoint(case: Case, config: PretrainedConfig, model_dir: Path) -> None:
    """Write the reference input with fixture-defined wire precision."""
    config.save_pretrained(model_dir)
    model = build_trainer_model(model_dir, case)
    keep_fp32 = re.compile(WIRE_FP32.get(case.name, r"(?!)")).search
    parameters = {
        name: tensor.detach().to(torch.float32 if keep_fp32(name) else torch.bfloat16)
        for name, tensor in model.named_parameters(remove_duplicate=False)
        if tensor.is_floating_point()
    }
    state_dict = convert_state_dict_to_hf(
        model, {name: parameters.get(name, tensor) for name, tensor in model.state_dict().items()}
    )
    state_dict = {name: tensor.contiguous().cpu() for name, tensor in state_dict.items()}
    if case.name in BF16_ON_LOAD:
        for name, tensor in state_dict.items():
            if name.startswith(BF16_ON_LOAD[case.name]):
                state_dict[name] = tensor.bfloat16()
    save_file(state_dict, model_dir / "model.safetensors", metadata={"format": "pt"})
    AutoTokenizer.from_pretrained(TOKENIZER).save_pretrained(model_dir)
    del model, state_dict
    torch.cuda.empty_cache()


def case_param(case: Case):
    marks = []
    if case.name in KNOWN_FAILURES and os.environ.get("PRIME_RL_MX_STRICT") != "1":
        marks.append(pytest.mark.xfail(reason=f"Needs {KNOWN_FAILURES[case.name]}", strict=True))
    return pytest.param(case, id=case.name, marks=marks)


@pytest.mark.parametrize("case", [case_param(case) for case in CASES])
def test_modelexpress_update_is_bit_exact(case: Case, tmp_path: Path):
    run_check(case, tmp_path, trainer_gpus="0", inference_gpus="1")


@pytest.mark.skipif(torch.cuda.device_count() < 4 or not on_sm100(), reason="needs four Blackwell GPUs")
def test_modelexpress_distributed_quantized_update_is_bit_exact(tmp_path: Path, monkeypatch):
    case = next(case for case in CASES if case.name == "deepseek_v4")
    monkeypatch.setenv("VLLM_USE_DEEP_GEMM_E8M0", "0")
    run_check(
        case,
        tmp_path,
        trainer_gpus="0,1",
        inference_gpus="2,3",
        model_override={"ep": 2},
        engine_override={
            "enforce_eager": False,
            "data_parallel_size": 2,
            "enable_expert_parallel": True,
            "quantization": "mxfp8",
            "quantization_config": {"ignore": ["*.wo_a", "*.indexer.*"]},
            "kv_cache_dtype": "fp8",
            "block_size": 256,
        },
        extra_args=["--staging-buffer-bytes", str(256 * 2**20), "--staging-buffers-count", "2", "--forward"],
    )


def run_check(
    case: Case,
    tmp_path: Path,
    *,
    trainer_gpus: str,
    inference_gpus: str,
    model_override: dict | None = None,
    engine_override: dict | None = None,
    extra_args: list[str] | None = None,
):
    model_dir, run_dir = tmp_path / "model", tmp_path / "run"
    config = serveable_config(case.config)
    write_checkpoint(case, config, model_dir)
    command = [
        sys.executable,
        TOOL.as_posix(),
        model_dir.as_posix(),
        run_dir.as_posix(),
        "--transport",
        "modelexpress",
        "--vllm",
        json.dumps({**engine_args(case, config), **(engine_override or {})}),
        "--model",
        json.dumps({**case.model, **(model_override or {})}),
        "--trainer-gpus",
        trainer_gpus,
        "--inference-gpus",
        inference_gpus,
        "--wire-fp32",
        WIRE_FP32.get(case.name, r"(?!)"),
        *(extra_args or []),
    ]
    result = subprocess.run(command, env=os.environ, capture_output=True, text=True, timeout=TIMEOUT)
    report_path = run_dir / "report.json"
    assert report_path.exists(), f"No report (exit code {result.returncode}):\n{result.stdout[-5000:]}"
    report = json.loads(report_path.read_text())
    assert not report["failures"], "\n".join(report["failures"])
    assert result.returncode == 0, result.stdout[-5000:]
