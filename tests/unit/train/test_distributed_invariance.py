"""Distributed invariance: the trainer computes the same loss and gradients however the model is split over 2 GPUs.

Each case loads a tiny seeded checkpoint through the trainer's own path, builds the model with `setup_model` on its
layout (FSDP, CP, EP, activation checkpointing and offloading, fusions, torch.compile), and runs forward and backward
on 4 packed Reverse-Text-SFT rows. One `torchrun` launch runs every case of a suite; each case reports as its own test.

- Reference: the same checkpoint unsharded on one GPU, bf16 compute with fp32 gradient accumulation.
- Routing replay: MoE cases replay the reference's expert picks, as RL's router replay does. Near-tied router scores
  would otherwise pick other experts under CP/EP numerics. `test_vllm_parity.py` checks the picks themselves.
- Tolerances: relative error of the loss (1e-5), the grad norm (1e-3) and the worst parameter gradient (2e-2), or
  1e-4 / 3e-3 / 1e-1 with torch.compile: about 5x the largest difference observed. A gradient under 0.1% of the total
  grad norm is measured against 0.1% of the total; a gradient on only one side fails. `Case.tolerances` raises a
  tolerance for a measured reason.
- Suites: `CASES` run in the tiny GPU subset, `FULL_SUITE_CASES` (archs whose kernels take minutes to compile) in the
  full GPU suite. A case is one line there; a new arch also needs a tiny config in `tiny_models.py`, in `TINY_MODELS`
  (built by HF) or `TINY_PRIME_RL_MODELS` (built by the prime-rl model class).

Run a suite without pytest (it writes `results.json`):

    uv run torchrun --nproc-per-node=2 tests/unit/train/test_distributed_invariance.py <output_dir> tiny|full
"""

import json
import math
import re
import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from tiny_models import TOKENIZER, Case, compute_loss, load_rows, model_config, save_tiny_model
from torch.distributed.tensor import DTensor

from prime_rl.configs.trainer import ModelConfig, TokenizerConfig
from prime_rl.trainer.model import get_model, load_dcp_from_hf, setup_model, setup_tokenizer
from prime_rl.trainer.models.fusions import apply_model_fusions
from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from prime_rl.trainer.models.layers.moe import TokenChoiceTopKRouter
from prime_rl.trainer.parallel_dims import get_parallel_dims
from prime_rl.trainer.utils import clip_grad_norm_, scale_gradients_, setup_torch_distributed
from prime_rl.trainer.vlm import get_language_model
from prime_rl.trainer.world import get_world
from prime_rl.utils.act_offloading import maybe_activation_offloading
from prime_rl.utils.cp import setup_context_parallel
from prime_rl.utils.logger import setup_logger

pytestmark = [pytest.mark.gpu]

CASES = [
    Case("qwen3-dp2", "qwen3"),
    Case("qwen3-cp2", "qwen3", cp=2),
    Case("qwen3-cp2-compile_fullgraph", "qwen3", cp=2, compile=True, fullgraph=True),
    Case("qwen3-dp2-no_ac_no_fusions", "qwen3", overrides={"ac": None, "fusions": {"enabled": []}}),
    Case("llama-cp2", "llama", cp=2),
    Case("qwen3_moe-dp2", "qwen3_moe"),
    Case("qwen3_moe-ep2", "qwen3_moe", ep=2),
    Case("qwen3_moe-cp2_ep2", "qwen3_moe", cp=2, ep=2),
    Case("glm4_moe-cp2_ep2", "glm4_moe", cp=2, ep=2),
]
# GatedDeltaNet passes its recurrent state across the CP cut. The document that straddles the cut then differs by up
# to ~3e-3 per token on the second rank (bf16, measured on the layer alone; the other documents match exactly), which
# moves the loss by ~1e-5 (1.4e-5 measured for Qwen3.5 MoE with cp2 and identical routing). Attention CP moves it by
# ~1e-8, which the default loss tolerance is calibrated on.
GATED_DELTA_NET_CP_TOLERANCES = {"loss": 1e-4}
# Too slow to compile in a fresh process for the tiny subset: torch.compile of the MoE layers takes ~25 s, and the
# newer archs' kernels take minutes (the cuDNN sparse-attention backward, DeepGEMM, TileLang, FLA's Triton autotuning)
FULL_SUITE_CASES = [
    Case("qwen3_moe-ep2-compile", "qwen3_moe", ep=2, compile=True),
    Case("qwen3_5_moe-ep2", "qwen3_5_moe", ep=2),
    Case("qwen3_5_moe-cp2_ep2", "qwen3_5_moe", cp=2, ep=2, tolerances=GATED_DELTA_NET_CP_TOLERANCES),
    Case("qwen3_8_flash_next-ep2", "qwen3_8_flash_next", ep=2),
    Case("qwen3_8_flash_next-cp2_ep2", "qwen3_8_flash_next", cp=2, ep=2, tolerances=GATED_DELTA_NET_CP_TOLERANCES),
    Case("deepseek_v4-ep2", "deepseek_v4", ep=2),
    Case("deepseek_v4-cp2_ep2", "deepseek_v4", cp=2, ep=2),
    Case("glm_moe_dsa-ep2", "glm_moe_dsa", ep=2),
    Case("glm_moe_dsa-cp2_ep2", "glm_moe_dsa", cp=2, ep=2),
    Case("glm5_next-cp2_ep2", "glm5_next", cp=2, ep=2),
]
SUITES = {"tiny": CASES, "full": FULL_SUITE_CASES}
TIMEOUTS = {"tiny": 5 * 60, "full": 20 * 60}

# Relative errors against the unsharded reference, about 5x the largest seen. "grad" is the worst parameter's.
TOLERANCES = {"loss": 1e-5, "grad_norm": 1e-3, "grad": 2e-2}
# torch.compile fuses the bf16 ops differently than the eager reference
COMPILE_TOLERANCES = {"loss": 1e-4, "grad_norm": 3e-3, "grad": 1e-1}


def record_routing(model: torch.nn.Module) -> Callable[[], torch.Tensor | None]:
    """Hooks every MoE router of `model`. The returned function gives the experts the routers picked in the last
    forward, as the `[1, seq, layers, top_k]` tensor that `routed_experts` takes, or None for a dense model."""
    picked: dict[int, torch.Tensor] = {}
    for name, module in model.named_modules():
        if isinstance(module, TokenChoiceTopKRouter):
            layer = int(re.search(r"layers\.(\d+)\.", name).group(1))
            module.register_forward_hook(lambda module, args, output, layer=layer: picked.__setitem__(layer, output[1]))
    num_layers = len(get_language_model(model).layers)

    def routing() -> torch.Tensor | None:
        if not picked:
            return None
        num_tokens, top_k = next(iter(picked.values())).shape
        routed_experts = torch.zeros(1, num_tokens, num_layers, top_k, dtype=torch.long)
        for layer, experts in picked.items():
            routed_experts[0, :, layer] = experts.cpu()
        return routed_experts

    return routing


def unwrapped_name(name: str) -> str:
    return name.replace("_checkpoint_wrapped_module.", "")


def run_reference(config: ModelConfig, rows: list[dict]) -> dict:
    """All rows through an unsharded model in bf16 (FSDP's compute dtype), accumulating gradients in fp32. Parameters
    that get no gradient, like the indexers, which reach the loss only through top-k indices, have no entry. Also
    returns the experts each row routed to, which the cases replay."""
    model = get_model(config, dtype=torch.float32)
    if config.fusions.enabled:
        apply_model_fusions(model, config.fusions.enabled)
    inject_prime_lm_head(model, chunk_size=config.fused_lm_head_token_chunk_size)
    # Moving off the meta device unties the embeddings, which FSDP keeps tied in the trainer
    model.to_empty(device="cuda")
    model.tie_weights()
    load_dcp_from_hf(model, config)
    fp32_params = {param for module in model.fp32_modules() for param in module.parameters()}
    for param in model.parameters():
        if param not in fp32_params:
            param.data = param.data.bfloat16()
    routing = record_routing(model)
    grads: dict[str, torch.Tensor] = {}
    routes = []
    loss_sum, num_tokens = 0.0, 0
    for row in rows:
        row_loss, row_tokens = compute_loss(model, row, cp_rank=0, cp_size=1)
        routes.append(routing())
        row_loss.backward()
        loss_sum += row_loss.item()
        num_tokens += row_tokens.item()
        for name, param in model.named_parameters():
            if param.grad is not None:
                grads.setdefault(name, torch.zeros_like(param, dtype=torch.float32)).add_(param.grad.float())
                param.grad = None
    grads = {name: grad / num_tokens for name, grad in grads.items()}
    grad_norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(grad) for grad in grads.values()]))
    return {"loss": loss_sum / num_tokens, "grad_norm": grad_norm.item(), "grads": grads, "routes": routes}


def run_case(config: ModelConfig, parallel_dims, rows: list[dict], routes: list[torch.Tensor | None]) -> dict:
    """One SFT trainer step up to the optimizer: forward and backward over this rank's rows, then the gradient
    scaling and norm of the trainer loop. MoE models replay the reference's routing of each row."""
    model = setup_model(config, parallel_dims)
    if parallel_dims.cp_enabled:
        setup_context_parallel(model, config, parallel_dims)
    dp_mesh = parallel_dims.get_mesh("dp")
    cp_rank = parallel_dims.world_mesh["cp"].get_local_rank() if parallel_dims.cp_enabled else 0
    loss_sum = torch.zeros((), device="cuda")
    num_tokens = torch.zeros((), dtype=torch.int64, device="cuda")
    for index in range(dp_mesh.get_local_rank(), len(rows), dp_mesh.size()):
        with maybe_activation_offloading(config.ac_offloading, model):
            row_loss, row_tokens = compute_loss(model, rows[index], cp_rank, parallel_dims.cp, routes[index])
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
        if param.grad is not None
    }
    return {"loss": (loss_sum / num_tokens).item(), "grad_norm": grad_norm.item(), "grads": grads}


def isolated(run, *args) -> dict:
    """The result of `run`, or the error it raised, so that one failing arch does not hide the other cases."""
    try:
        return run(*args)
    except Exception:
        traceback.print_exc()
        return {"error": traceback.format_exc()}


# A parameter whose gradient nearly cancels over the tokens has a tiny gradient norm, so its relative error measures
# rounding and the backward's own nondeterminism rather than the sharding. Its error is measured against this share of
# the total grad norm instead. DeepSeek-V4's mHC gate parameters (shares 1e-7 to 1e-6) differ by up to 1.8e-2 between
# two runs of the same unsharded reference, and by at most 1.4e-6 of the total grad norm between case and reference.
MIN_GRAD_SHARE = 1e-3


def grad_error(grad: torch.Tensor, ref: torch.Tensor, floor: float) -> float:
    """The relative error of `grad` against `ref`, dividing by at least `floor`. A gradient that is exactly zero on
    only one side, as from a layer only one side exercises, is an infinite error."""
    if not grad.any() or not ref.any():
        return 0.0 if not grad.any() and not ref.any() else math.inf
    return torch.linalg.vector_norm(grad - ref).item() / max(torch.linalg.vector_norm(ref).item(), floor)


def relative_errors(result: dict, reference: dict) -> dict:
    floor = MIN_GRAD_SHARE * reference["grad_norm"]
    grad_errors = {
        name: grad_error(result["grads"][name], ref, floor)
        for name, ref in reference["grads"].items()
        if name in result["grads"]
    }
    worst = max(grad_errors, key=grad_errors.get)
    return {
        "loss": abs(result["loss"] / reference["loss"] - 1),
        "grad_norm": abs(result["grad_norm"] / reference["grad_norm"] - 1),
        "grad": grad_errors[worst],
        "worst_grad": worst,
        "worst_grad_share": (torch.linalg.vector_norm(reference["grads"][worst]) / reference["grad_norm"]).item(),
        "one_sided_grads": sorted(result["grads"].keys() ^ reference["grads"].keys()),
    }


def main(output_dir: Path, cases: list[Case]) -> None:
    setup_logger("warning")
    setup_torch_distributed()
    models_dir = output_dir / "models"
    built = {name: isolated(save_tiny_model, name, models_dir) for name in sorted({case.model for case in cases})}
    dist.barrier()
    rows = load_rows(setup_tokenizer(TokenizerConfig(name=TOKENIZER), model_name=TOKENIZER))

    configs = {case.name: model_config(case, models_dir) for case in cases}
    reference_keys = {case.name: (case.model, tuple(configs[case.name].fusions.enabled)) for case in cases}
    # Before any CP case: CP setup patches the attention classes for the rest of the process
    references = {}
    for case in cases:
        if reference_keys[case.name] not in references:
            build = built[case.model]
            references[reference_keys[case.name]] = (
                dict(build) if "error" in build else isolated(run_reference, configs[case.name], rows)
            )

    results, layouts = {}, {}
    for case in sorted(cases, key=lambda case: case.cp):
        start = time.perf_counter()
        # Compiled graphs of the previous case's model must not leak into this one, as in a fresh trainer process
        torch._dynamo.reset()
        config = configs[case.name]
        if (case.cp, case.ep) not in layouts:
            layouts[case.cp, case.ep] = get_parallel_dims(config)
        reference = references[reference_keys[case.name]]
        result = (
            dict(reference)
            if "error" in reference
            else isolated(run_case, config, layouts[case.cp, case.ep], rows, reference["routes"])
        )
        results[case.name] = result if "error" in result else relative_errors(result, reference)
        results[case.name]["seconds"] = time.perf_counter() - start
        torch.cuda.empty_cache()

    if get_world().is_master:
        print(json.dumps(results, indent=2))
        (output_dir / "results.json").write_text(json.dumps(results, indent=2))
    # No destroy_process_group: tearing down the process groups of every layout takes seconds


def launch(suite: str, run_process, tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict]:
    if torch.cuda.device_count() < 2:
        pytest.skip("needs 2 GPUs")
    output_dir = tmp_path_factory.mktemp(f"distributed-invariance-{suite}")
    cmd = ["uv", "run", "torchrun", "--standalone", "--nproc-per-node=2", __file__, output_dir.as_posix(), suite]
    assert run_process(cmd, timeout=TIMEOUTS[suite]).returncode == 0
    return json.loads((output_dir / "results.json").read_text())


@pytest.fixture(scope="module")
def tiny_results(run_process, tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict]:
    return launch("tiny", run_process, tmp_path_factory)


@pytest.fixture(scope="module")
def full_suite_results(run_process, tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict]:
    return launch("full", run_process, tmp_path_factory)


def check(errors: dict, case: Case) -> None:
    assert "error" not in errors, errors.get("error")
    assert not errors["one_sided_grads"], f"gradients on only one side: {errors['one_sided_grads']}"
    tolerances = {**(COMPILE_TOLERANCES if case.compile else TOLERANCES), **case.tolerances}
    for metric, tolerance in tolerances.items():
        assert errors[metric] <= tolerance, f"{metric} relative error {errors[metric]:.1e} > {tolerance:.0e} ({errors})"


@pytest.mark.tiny
@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_matches_unsharded_reference(tiny_results: dict[str, dict], case: Case):
    check(tiny_results[case.name], case)


@pytest.mark.parametrize("case", FULL_SUITE_CASES, ids=[case.name for case in FULL_SUITE_CASES])
def test_full_suite_matches_unsharded_reference(full_suite_results: dict[str, dict], case: Case):
    check(full_suite_results[case.name], case)


if __name__ == "__main__":
    main(Path(sys.argv[1]), SUITES[sys.argv[2]])
