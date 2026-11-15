"""Distributed invariance of the trainer on two GPUs.

One torchrun launch runs every case in-process. A case builds a tiny random-init model with the trainer's own
`setup_model` on a parallel layout (FSDP, CP, EP) with a set of model features, runs forward and backward on fixed
packed Reverse-Text-SFT rows, and compares the loss, grad norm and every parameter's gradient with a plain unsharded
copy of the same weights that runs all rows on one GPU.

A case is one line in `CASES`; adding a model is one entry in `TINY_MODELS`. To run the cases without pytest:

    uv run torchrun --nproc-per-node=2 tests/unit/train/test_distributed_invariance.py <output_dir>
"""

import json
import math
import sys
import time
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from renderers import AutoRendererConfig
from torch.distributed.tensor import DTensor
from transformers import AutoModelForCausalLM, Glm4MoeConfig, LlamaConfig, PretrainedConfig, Qwen3Config, Qwen3MoeConfig

from prime_rl.configs.sft import SFTDataConfig
from prime_rl.configs.trainer import CompileConfig, ModelConfig, TokenizerConfig
from prime_rl.trainer.data import setup_dataloader
from prime_rl.trainer.model import (
    forward,
    get_model,
    load_dcp_from_hf,
    resolve_auto_attn,
    setup_model,
    setup_tokenizer,
)
from prime_rl.trainer.models.fusions import apply_model_fusions
from prime_rl.trainer.models.layers.lm_head import IGNORE_INDEX, inject_prime_lm_head
from prime_rl.trainer.parallel_dims import get_parallel_dims
from prime_rl.trainer.sft.data import setup_sft_dataset
from prime_rl.trainer.utils import clip_grad_norm_, scale_gradients_, setup_torch_distributed
from prime_rl.trainer.world import get_world
from prime_rl.utils.act_offloading import maybe_activation_offloading
from prime_rl.utils.cp import setup_context_parallel, setup_cp_params, shard_for_cp
from prime_rl.utils.logger import setup_logger

pytestmark = [pytest.mark.gpu]

TOKENIZER = "Qwen/Qwen3-0.6B"
NUM_ROWS = 4
ROW_LEN = 512
TIMEOUT = 5 * 60

TINY_DIMS = dict(
    # Every model reads the same rows, tokenized with TOKENIZER
    vocab_size=151936,
    hidden_size=256,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=64,
    max_position_embeddings=4096,
)
MOE_DIMS = dict(moe_intermediate_size=128, num_experts_per_tok=2, tie_word_embeddings=False)
TINY_MODELS: dict[str, PretrainedConfig] = {
    "qwen3": Qwen3Config(**TINY_DIMS, intermediate_size=512, tie_word_embeddings=True),
    "llama": LlamaConfig(**TINY_DIMS, intermediate_size=512, tie_word_embeddings=False),
    "qwen3_moe": Qwen3MoeConfig(**TINY_DIMS, **MOE_DIMS, num_experts=8),
    "glm4_moe": Glm4MoeConfig(
        **TINY_DIMS, **MOE_DIMS, n_routed_experts=8, n_shared_experts=1, first_k_dense_replace=1, intermediate_size=512
    ),
}


@dataclass(frozen=True)
class Case:
    name: str
    model: str
    cp: int = 1
    ep: int = 1
    compile: bool = False
    """torch.compile costs seconds per case, so few cases turn it on."""
    overrides: dict = field(default_factory=dict)
    """ModelConfig fields on top of the trainer defaults (full AC with activation offloading, fusions)."""


CASES = [
    Case("qwen3-dp2", "qwen3"),
    Case("qwen3-cp2", "qwen3", cp=2),
    Case("qwen3-cp2-compile", "qwen3", cp=2, compile=True),
    Case("qwen3-dp2-no_ac_no_fusions", "qwen3", overrides={"ac": None, "fusions": {"enabled": []}}),
    Case("llama-cp2", "llama", cp=2),
    Case("qwen3_moe-dp2", "qwen3_moe"),
    Case("qwen3_moe-ep2", "qwen3_moe", ep=2),
    Case("qwen3_moe-cp2_ep2", "qwen3_moe", cp=2, ep=2),
    Case("qwen3_moe-ep2-compile", "qwen3_moe", ep=2, compile=True),
    Case("glm4_moe-cp2_ep2", "glm4_moe", cp=2, ep=2),
]

# Relative errors against the unsharded reference, about 5x the largest seen. "grad" is the worst parameter's.
TOLERANCES = {"loss": 1e-5, "grad_norm": 1e-3, "grad": 2e-2}
# torch.compile fuses the bf16 ops differently than the eager reference
COMPILE_TOLERANCES = {"loss": 1e-4, "grad_norm": 3e-3, "grad": 1e-1}


def model_config(case: Case, models_dir: Path) -> ModelConfig:
    # SFT does not freeze the router, and the ModelConfig default "auto" is resolved by the trainer config
    config = ModelConfig(
        name=(models_dir / case.model).as_posix(),
        cp=case.cp,
        ep=case.ep,
        compile=CompileConfig() if case.compile else None,
        freeze_moe_router=False,
        **case.overrides,
    )
    resolve_auto_attn(config)
    return config


def save_tiny_models(models_dir: Path) -> None:
    """HF checkpoints, which the trainer converts to its own format on load."""
    for name, config in TINY_MODELS.items():
        torch.manual_seed(0)
        AutoModelForCausalLM.from_config(config).save_pretrained(models_dir / name)


def load_rows(tokenizer) -> list[dict]:
    """The first rows of Reverse-Text-SFT, packed as the SFT trainer packs them, identical on every rank. Rows of one
    length keep torch.compile from recompiling for a second shape."""
    config = SFTDataConfig(seq_len=ROW_LEN, shuffle=False, pad_to_seq_len=True)
    dataset = setup_sft_dataset(
        tokenizer, config, non_dp_size=get_world().world_size, renderer_config=AutoRendererConfig()
    )
    return list(islice(setup_dataloader(dataset, config, cp=2), NUM_ROWS))


def compute_loss(model, row: dict, cp_rank: int, cp_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The SFT trainer's `compute_loss`: summed cross-entropy and token count of this rank's share of a row."""
    input_ids, position_ids, target_ids, loss_mask = (
        row[key].cuda() for key in ("input_ids", "position_ids", "target_ids", "loss_mask")
    )
    if cp_size > 1:
        input_ids, position_ids = setup_cp_params(input_ids, position_ids, cp_rank, cp_size)
        target_ids = shard_for_cp(target_ids, cp_rank=cp_rank, cp_world_size=cp_size)
        loss_mask = shard_for_cp(loss_mask, cp_rank=cp_rank, cp_world_size=cp_size)
    labels = target_ids.masked_fill(~loss_mask, IGNORE_INDEX)
    out = forward(
        model,
        input_ids,
        position_ids,
        seq_lens=row["seq_lens"].cuda(),
        labels=labels,
        seq_lens_are_pre_shard=cp_size > 1,
    )
    return out["loss"], loss_mask.sum()


def unwrapped_name(name: str) -> str:
    return name.replace("_checkpoint_wrapped_module.", "")


def run_reference(config: ModelConfig, rows: list[dict]) -> dict:
    """All rows through an unsharded model in bf16 (FSDP's compute dtype), accumulating gradients in fp32."""
    model = get_model(config, device=torch.device("meta"), dtype=torch.float32)
    if config.fusions.enabled:
        apply_model_fusions(model, config.fusions.enabled)
    inject_prime_lm_head(model, chunk_size=config.fused_lm_head_token_chunk_size)
    # Moving off the meta device unties the embeddings, which FSDP keeps tied in the trainer
    model.to_empty(device="cuda")
    model.tie_weights()
    load_dcp_from_hf(model, config)
    for param in model.parameters():
        param.data = param.data.bfloat16()
    grads = {name: torch.zeros_like(param, dtype=torch.float32) for name, param in model.named_parameters()}
    loss_sum, num_tokens = 0.0, 0
    for row in rows:
        row_loss, row_tokens = compute_loss(model, row, cp_rank=0, cp_size=1)
        row_loss.backward()
        loss_sum += row_loss.item()
        num_tokens += row_tokens.item()
        for name, param in model.named_parameters():
            grads[name] += param.grad.float()
            param.grad = None
    grads = {name: grad / num_tokens for name, grad in grads.items()}
    grad_norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(grad) for grad in grads.values()]))
    return {"loss": loss_sum / num_tokens, "grad_norm": grad_norm.item(), "grads": grads}


def run_case(config: ModelConfig, parallel_dims, rows: list[dict]) -> dict:
    """One SFT trainer step up to the optimizer: forward and backward over this rank's rows, then the gradient
    scaling and norm of the trainer loop."""
    model = setup_model(config, parallel_dims)
    if parallel_dims.cp_enabled:
        setup_context_parallel(model, config, parallel_dims)
    dp_mesh = parallel_dims.get_mesh("dp")
    cp_rank = parallel_dims.world_mesh["cp"].get_local_rank() if parallel_dims.cp_enabled else 0
    loss_sum = torch.zeros((), device="cuda")
    num_tokens = torch.zeros((), dtype=torch.int64, device="cuda")
    for row in rows[dp_mesh.get_local_rank() :: dp_mesh.size()]:
        with maybe_activation_offloading(config.ac_offloading, model):
            row_loss, row_tokens = compute_loss(model, row, cp_rank, parallel_dims.cp)
        row_loss.backward()
        loss_sum += row_loss.detach()
        num_tokens += row_tokens
    dp_cp_group = parallel_dims.get_mesh("dp_cp").get_group()
    dist.all_reduce(loss_sum, group=dp_cp_group)
    dist.all_reduce(num_tokens, group=dp_cp_group)
    scale_gradients_(None, model, parallel_dims.fsdp_gradient_divide_factor / num_tokens.item())
    grad_norm = clip_grad_norm_(None, model, max_norm=math.inf)
    grads = {
        unwrapped_name(name): param.grad.full_tensor() if isinstance(param.grad, DTensor) else param.grad
        for name, param in model.named_parameters()
    }
    return {"loss": (loss_sum / num_tokens).item(), "grad_norm": grad_norm.item(), "grads": grads}


def relative_errors(result: dict, reference: dict) -> dict:
    assert result["grads"].keys() == reference["grads"].keys()
    grad_errors = {
        name: (torch.linalg.vector_norm(result["grads"][name] - ref) / torch.linalg.vector_norm(ref)).item()
        for name, ref in reference["grads"].items()
    }
    worst = max(grad_errors, key=grad_errors.get)
    return {
        "loss": abs(result["loss"] / reference["loss"] - 1),
        "grad_norm": abs(result["grad_norm"] / reference["grad_norm"] - 1),
        "grad": grad_errors[worst],
        "worst_grad": worst,
    }


def main(output_dir: Path) -> None:
    setup_logger("warning")
    setup_torch_distributed()
    models_dir = output_dir / "models"
    if get_world().is_master:
        save_tiny_models(models_dir)
    dist.barrier()
    rows = load_rows(setup_tokenizer(TokenizerConfig(name=TOKENIZER)))

    configs = {case.name: model_config(case, models_dir) for case in CASES}
    reference_keys = {case.name: (case.model, tuple(configs[case.name].fusions.enabled)) for case in CASES}
    # Before any CP case: CP setup patches the attention classes for the rest of the process
    references = {}
    for case in CASES:
        if reference_keys[case.name] not in references:
            references[reference_keys[case.name]] = run_reference(configs[case.name], rows)

    results, layouts = {}, {}
    for case in sorted(CASES, key=lambda case: case.cp):
        start = time.perf_counter()
        # Compiled graphs of the previous case's model must not leak into this one, as in a fresh trainer process
        torch._dynamo.reset()
        config = configs[case.name]
        if (case.cp, case.ep) not in layouts:
            layouts[case.cp, case.ep] = get_parallel_dims(config)
        result = run_case(config, layouts[case.cp, case.ep], rows)
        results[case.name] = relative_errors(result, references[reference_keys[case.name]])
        results[case.name]["seconds"] = time.perf_counter() - start
        torch.cuda.empty_cache()

    if get_world().is_master:
        print(json.dumps(results, indent=2))
        (output_dir / "results.json").write_text(json.dumps(results, indent=2))
    # No destroy_process_group: tearing down the process groups of every layout takes seconds


@pytest.fixture(scope="module")
def results(run_process, tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict]:
    if torch.cuda.device_count() < 2:
        pytest.skip("needs 2 GPUs")
    output_dir = tmp_path_factory.mktemp("distributed-invariance")
    cmd = ["uv", "run", "torchrun", "--standalone", "--nproc-per-node=2", __file__, output_dir.as_posix()]
    assert run_process(cmd, timeout=TIMEOUT).returncode == 0
    return json.loads((output_dir / "results.json").read_text())


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_matches_unsharded_reference(results: dict[str, dict], case: Case):
    errors = results[case.name]
    tolerances = COMPILE_TOLERANCES if case.compile else TOLERANCES
    for metric, tolerance in tolerances.items():
        assert errors[metric] <= tolerance, f"{metric} relative error {errors[metric]:.1e} > {tolerance:.0e} ({errors})"


if __name__ == "__main__":
    main(Path(sys.argv[1]))
