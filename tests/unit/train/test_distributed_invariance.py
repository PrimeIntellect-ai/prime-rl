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
  full GPU suite. A case is one line there; a new arch also needs a tiny config in `TINY_MODELS` (built by HF) or
  `TINY_PRIME_RL_MODELS` (built by the prime-rl model class).

Run a suite without pytest (it writes `results.json`):

    uv run torchrun --nproc-per-node=2 tests/unit/train/test_distributed_invariance.py <output_dir> tiny|full
"""

import json
import math
import re
import sys
import time
import traceback
from collections import defaultdict
from collections.abc import Callable
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
from prime_rl.trainer.models import get_custom_causal_lm_cls
from prime_rl.trainer.models.deepseek_v4 import DeepseekV4Config
from prime_rl.trainer.models.fusions import apply_model_fusions
from prime_rl.trainer.models.glm_moe_dsa import GlmMoeDsaConfig
from prime_rl.trainer.models.inkling2 import Inkling2LoopTextConfig
from prime_rl.trainer.models.layers.lm_head import IGNORE_INDEX, inject_prime_lm_head
from prime_rl.trainer.models.layers.moe import TokenChoiceTopKRouter
from prime_rl.trainer.models.qwen3_5 import Qwen3_5MoeTextConfig
from prime_rl.trainer.models.qwen3_8_flash_next import Qwen3_8FlashNextTextConfig
from prime_rl.trainer.parallel_dims import get_parallel_dims
from prime_rl.trainer.sft.data import setup_sft_dataset
from prime_rl.trainer.utils import clip_grad_norm_, scale_gradients_, setup_torch_distributed
from prime_rl.trainer.vlm import get_language_model
from prime_rl.trainer.world import get_world
from prime_rl.utils.act_offloading import maybe_activation_offloading
from prime_rl.utils.cp import setup_context_parallel, setup_cp_params, shard_for_cp
from prime_rl.utils.logger import setup_logger
from prime_rl.utils.weights import save_state_dict

pytestmark = [pytest.mark.gpu]

TOKENIZER = "Qwen/Qwen3-0.6B"
NUM_ROWS = 4
ROW_LEN = 512

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
# Every layer kind of each arch with small widths and 8 experts. Head counts and head dims stay at the published
# values where kernels require them: FlashMLA serves 64 or 128 query heads of dim 512, and Qwen's mrope sections cover
# a quarter of a 256-dim head. DeepSeek-V4 and GLM-5 build with the TileLang backend, whose constructor does not check
# the GPU; the trainer picks `model.dsa_backend` itself.
QWEN_MROPE = {
    "mrope_interleaved": True,
    "mrope_section": [11, 11, 10],
    "rope_type": "default",
    "rope_theta": 10000000,
    "partial_rotary_factor": 0.25,
}
QWEN_NEXT_DIMS = dict(
    vocab_size=151936,
    hidden_size=256,
    num_hidden_layers=2,
    layer_types=["linear_attention", "full_attention"],
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=256,
    linear_num_key_heads=4,
    linear_key_head_dim=128,
    linear_value_head_dim=128,
    num_experts=8,
    num_experts_per_tok=2,
    moe_intermediate_size=128,
    shared_expert_intermediate_size=128,
    rope_parameters=QWEN_MROPE,
    max_position_embeddings=4096,
    bos_token_id=151643,
    eos_token_id=151645,
)
TINY_PRIME_RL_MODELS: dict[str, PretrainedConfig] = {
    "qwen3_5_moe": Qwen3_5MoeTextConfig(**QWEN_NEXT_DIMS, linear_num_value_heads=8, tie_word_embeddings=False),
    "qwen3_8_flash_next": Qwen3_8FlashNextTextConfig(
        **QWEN_NEXT_DIMS,
        linear_num_value_heads=12,
        # A budget below the documents' lengths, so the indexer drops keys
        indexer_budget=64,
        hc_lowrank=64,
        ple_layer_ids=[1],
        ple_embed_dim=256,
        ngram_vocab_size_base=4096,
        split_ngram_parts=4,
    ),
    "deepseek_v4": DeepseekV4Config(
        vocab_size=151936,
        hidden_size=256,
        num_hidden_layers=3,
        layer_types=["sliding_attention", "compressed_sparse_attention", "heavily_compressed_attention"],
        # The packed Reverse-Text-SFT documents are ~90-100 tokens, and a document shorter than the compress rate gets
        # no compressed entry, so the published rate of 128 would leave the heavily compressed attention unused
        compress_rates={"compressed_sparse_attention": 4, "heavily_compressed_attention": 16},
        q_lora_rank=256,
        o_lora_rank=128,
        index_topk=64,
        n_routed_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        num_hash_layers=1,
        max_position_embeddings=4096,
        dsa_backend="tilelang",
    ),
    "glm_moe_dsa": GlmMoeDsaConfig(
        vocab_size=151936,
        hidden_size=256,
        num_hidden_layers=2,
        first_k_dense_replace=1,
        intermediate_size=512,
        moe_intermediate_size=128,
        q_lora_rank=256,
        index_topk=128,
        n_routed_experts=8,
        num_experts_per_tok=2,
        pad_token_id=None,
        max_position_embeddings=4096,
        dsa_backend="tilelang",
    ),
    # Layers 2 to 5 run three times and read the previous pass from its two blocks. The core holds sliding-window
    # layers that start a block (2, 4) and one that does not (5), and a global layer (3).
    "inkling2_loop": Inkling2LoopTextConfig(
        vocab_size=151936,
        hidden_size=256,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        swa_num_attention_heads=4,
        swa_head_dim=64,
        d_rel=16,
        rel_extent=128,
        sliding_window_size=128,
        local_layer_ids=[0, 1, 2, 4, 5, 6],
        n_layers_per_attn_res_block=2,
        log_scaling_n_floor=64,
        n_routed_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        loop_config=dict(core_start=2, core_end=6, num_passes=3, gate_rank=16),
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
    tolerances: dict = field(default_factory=dict)
    """Tolerances on top of the defaults below, each measured for this case."""


CASES = [
    Case("qwen3-dp2", "qwen3"),
    Case("qwen3-cp2", "qwen3", cp=2),
    Case("qwen3-cp2-compile", "qwen3", cp=2, compile=True),
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
    Case("inkling2_loop-ep2", "inkling2_loop", ep=2),
    Case("inkling2_loop-ep2-compile", "inkling2_loop", ep=2, compile=True),
]
SUITES = {"tiny": CASES, "full": FULL_SUITE_CASES}
TIMEOUTS = {"tiny": 5 * 60, "full": 20 * 60}

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


def save_tiny_model(name: str, models_dir: Path) -> dict:
    """An HF checkpoint, which the trainer converts to its own format on load. Every rank builds the model, so that a
    build that fails on this GPU fails on all ranks alike, and the master saves it."""
    torch.manual_seed(0)
    if name in TINY_MODELS:
        model = AutoModelForCausalLM.from_config(TINY_MODELS[name])
        if get_world().is_master:
            model.save_pretrained(models_dir / name)
    else:
        config = TINY_PRIME_RL_MODELS[name]
        model = get_custom_causal_lm_cls(config)._from_config(config)
        init_weights(model)
        if get_world().is_master:
            save_state_dict(model.convert_to_hf(model.state_dict()), models_dir / name)
            config.save_pretrained(models_dir / name)
    return {}


@torch.no_grad()
def init_weights(model: torch.nn.Module) -> None:
    """Seeded values for every floating parameter, since `_from_config` leaves some of them uninitialized, and a
    DeepSeek-V4 hash routing table with distinct experts per token."""
    for name, param in model.named_parameters():
        if param.ndim == 1 and name.endswith("norm.weight"):
            param.fill_(1.0)
        else:
            param.normal_(0.0, 0.02)
    for module in model.modules():
        if hasattr(module, "tid2eid"):
            experts = torch.rand(module.tid2eid.shape[0], module.num_experts).argsort(dim=1)
            module.tid2eid.copy_(experts[:, : module.top_k])


def load_rows(tokenizer) -> list[dict]:
    """The first rows of Reverse-Text-SFT, packed as the SFT trainer packs them, identical on every rank. Rows of one
    length keep torch.compile from recompiling for a second shape."""
    config = SFTDataConfig(
        source=[{"name": "reverse-text", "dataset": "PrimeIntellect/Reverse-Text-SFT"}],
        seq_len=ROW_LEN,
        shuffle=False,
        pad_to_seq_len=True,
    )
    dataset = setup_sft_dataset(tokenizer, config, dp_rank=0, dp_size=1, renderer_config=AutoRendererConfig())
    return list(islice(setup_dataloader(dataset, config, cp=2), NUM_ROWS))


def compute_loss(
    model, row: dict, cp_rank: int, cp_size: int, routed_experts: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """The SFT trainer's `compute_loss`: summed cross-entropy and token count of this rank's share of a row.
    `routed_experts` replays an MoE routing, as RL's router replay does."""
    input_ids, position_ids, target_ids, loss_mask = (
        row[key].cuda() for key in ("input_ids", "position_ids", "target_ids", "loss_mask")
    )
    if routed_experts is not None:
        routed_experts = routed_experts.cuda()
    if cp_size > 1:
        input_ids, position_ids = setup_cp_params(input_ids, position_ids, cp_rank, cp_size)
        target_ids = shard_for_cp(target_ids, cp_rank=cp_rank, cp_world_size=cp_size)
        loss_mask = shard_for_cp(loss_mask, cp_rank=cp_rank, cp_world_size=cp_size)
        if routed_experts is not None:
            routed_experts = shard_for_cp(routed_experts, cp_rank=cp_rank, cp_world_size=cp_size)
    labels = target_ids.masked_fill(~loss_mask, IGNORE_INDEX)
    out = forward(
        model,
        input_ids,
        position_ids,
        seq_lens=row["seq_lens"].cuda(),
        labels=labels,
        routed_experts=routed_experts,
        seq_lens_are_pre_shard=cp_size > 1,
    )
    return out["loss"], loss_mask.sum()


def record_routing(model: torch.nn.Module) -> Callable[[], torch.Tensor | None]:
    """Hooks every MoE router of `model`. The returned function gives the experts the routers picked in the last
    forward, as the `[1, seq, layer visits, top_k]` tensor that `routed_experts` takes, or None for a dense model.
    A looped model runs some layers several times and routes anew each time."""
    language_model = get_language_model(model)
    visits_of_layer = getattr(language_model.config, "layer_visit_ids", lambda layer: (layer,))
    num_visits = getattr(language_model.config, "num_layer_visits", len(language_model.layers))
    picked: dict[int, torch.Tensor] = {}
    calls: dict[int, int] = defaultdict(int)

    def record(layer: int):
        def hook(module, args, output):
            picked[visits_of_layer(layer)[calls[layer]]] = output[1]
            calls[layer] += 1

        return hook

    for name, module in model.named_modules():
        if isinstance(module, TokenChoiceTopKRouter):
            module.register_forward_hook(record(int(re.search(r"layers\.(\d+)\.", name).group(1))))

    def routing() -> torch.Tensor | None:
        calls.clear()
        if not picked:
            return None
        num_tokens, top_k = next(iter(picked.values())).shape
        routed_experts = torch.zeros(1, num_tokens, num_visits, top_k, dtype=torch.long)
        for visit, experts in picked.items():
            routed_experts[0, :, visit] = experts.cpu()
        return routed_experts

    return routing


def unwrapped_name(name: str) -> str:
    return name.replace("_checkpoint_wrapped_module.", "")


def run_reference(config: ModelConfig, rows: list[dict]) -> dict:
    """All rows through an unsharded model in bf16 (FSDP's compute dtype), accumulating gradients in fp32. Parameters
    that get no gradient, like the indexers, which reach the loss only through top-k indices, have no entry. Also
    returns the experts each row routed to, which the cases replay."""
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
