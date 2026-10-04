"""Two-GPU parallel-layout equivalence tests for every PrimeRL architecture.

Each case builds the model sharded the way the trainer shards it (expert parallelism, activation
checkpointing, torch.compile, FSDP2, context parallelism), loads the same weights into a single-GPU
reference, and checks one forward/backward on a packed batch: logits, loss and every parameter gradient
must match the reference. Cases cross the parallel layouts (FSDP, CP ring/ulysses, EP, EP+CP) with the
runtime modes (full AC, no AC, selective AC, compile). Run on a 2-GPU node (e.g. 2x H200):

    uv run torchrun --nproc-per-node=2 -m pytest tests/unit/train/models/test_parallel_equivalence.py -v
    # skip the compile cases
    uv run torchrun --nproc-per-node=2 -m pytest tests/unit/train/models/test_parallel_equivalence.py -m "not slow"
"""

import functools
import gc
import os
from dataclasses import dataclass
from typing import Literal

import pytest
import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict
from torch.distributed.tensor import DTensor

from prime_rl.configs.trainer import ActivationCheckpointConfig, CompileConfig, ModelConfig
from prime_rl.trainer.model import apply_ac, apply_compile, forward, setup_fsdp
from prime_rl.trainer.models.afmoe.modeling_afmoe import AfmoeFlashAttention
from prime_rl.trainer.models.gpt_oss.attention import GptOssAttention
from prime_rl.trainer.models.layers.attn import FlashAttention
from prime_rl.trainer.models.layers.lm_head import IGNORE_INDEX, cross_entropy_sum
from prime_rl.trainer.models.layers.moe import MoE
from prime_rl.trainer.moe_runtime import configure_moe_runtime
from prime_rl.trainer.parallel_dims import ParallelDims
from prime_rl.trainer.utils import setup_torch_distributed
from prime_rl.utils.cp import setup_context_parallel, setup_cp_params, shard_for_cp
from prime_rl.utils.weights import resolve_fqn
from tests.unit.train.models.arch_configs import ARCH_CONFIGS, build_model, reference_state_dict

# Deterministic cuBLAS needs a fixed workspace, configured before CUDA initializes.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

WORLD_SIZE = 2
SEQ_LEN = 256
# Several documents per packed row so per-document boundaries (and the CP shard cut) are exercised.
DOC_LENS = [96, 112, 48]

# Relative L2 error against the single-GPU reference. Eager layouts reproduce the reference logits exactly;
# gradients differ only by the bf16 rounding of the FSDP reduce-scatter.
LOGITS_RTOL = 2e-2
LOSS_RTOL = 5e-3
GRAD_RTOL = 5e-2
# Inductor fuses ops with different rounding (notably ring attention's log-sum-exp merge, and the sparse
# indexer scores), and discrete top-k choices (MoE routing, sparse attention) amplify it: up to ~10% on the
# logits and ~40% on router gradients of the tiny test models, while the summed loss stays within 1e-4.
COMPILE_LOGITS_RTOL = 0.15
COMPILE_LOSS_RTOL = 5e-4
COMPILE_GRAD_RTOL = 0.5
# FLA's context-parallel linear attention hands the recurrent state across the shard cut, which rounds
# differently than keeping it inside one kernel: ~1e-3 relative per layer on the ranks past the cut (below
# bf16's rounding unit), ~1% at the logits, amplified to ~5% by MoE routing. The summed loss stays within 3e-4.
LINEAR_ATTENTION_ARCHS = ("qwen3_5", "qwen3_5_moe", "qwen3_5_vlm", "qwen3_8_flash_next")
LINEAR_ATTENTION_CP_LOGITS_RTOL = 0.08
LINEAR_ATTENTION_CP_LOSS_RTOL = 5e-4
LINEAR_ATTENTION_CP_GRAD_RTOL = 0.25


@dataclass(frozen=True)
class Parallel:
    cp: int = 1
    cp_style: str = "ring"
    ep: int = 1


@dataclass(frozen=True)
class Runtime:
    ac: Literal["none", "full", "selective"] = "full"
    compile: bool = False


PARALLELS = {
    "fsdp": Parallel(),
    "cp2-ring": Parallel(cp=2, cp_style="ring"),
    "cp2-ulysses": Parallel(cp=2, cp_style="ulysses"),
    "ep2": Parallel(ep=2),
    "ep2-cp2-ring": Parallel(cp=2, cp_style="ring", ep=2),
    "ep2-cp2-ulysses": Parallel(cp=2, cp_style="ulysses", ep=2),
}
RUNTIMES = {
    "ac": Runtime(ac="full"),
    "noac": Runtime(ac="none"),
    "selac": Runtime(ac="selective"),
    "compile": Runtime(ac="full", compile=True),
}
# Compiling costs tens of seconds per case, so it only runs on the layouts without expert parallelism.
COMPILE_PARALLELS = ("fsdp", "cp2-ring", "cp2-ulysses")

# Archs whose MoE combine accumulates with float atomics, and whose tiny test config then flips discrete
# top-k choices (the sparse indexer keeps 2 entries): two identical eager forwards differ by ~7%, so they
# run with deterministic kernels.
DETERMINISTIC_ARCHS = ("deepseek_v4",)

# Strict, so a fix shows up as an unexpected pass.
KNOWN_FAILURES: dict[tuple[str, str], str] = {
    ("deepseek_v4", "compile"): (
        "compiled DeepSeek-V4 hyper-connections: the attn_hc.scale gradient is ~7x off the eager reference "
        "(eager layouts match exactly)"
    ),
}


def _cases() -> list:
    cases = []
    for arch in ARCH_CONFIGS:
        for parallel in PARALLELS:
            for runtime in RUNTIMES:
                if RUNTIMES[runtime].compile and parallel not in COMPILE_PARALLELS:
                    continue
                marks = [pytest.mark.slow] if RUNTIMES[runtime].compile else []
                if (arch, runtime) in KNOWN_FAILURES:
                    marks.append(pytest.mark.xfail(reason=KNOWN_FAILURES[arch, runtime], strict=True))
                cases.append(pytest.param(arch, parallel, runtime, marks=marks, id=f"{arch}-{parallel}-{runtime}"))
    return cases


@pytest.fixture(scope="module")
def distributed():
    if int(os.environ.get("WORLD_SIZE", 1)) != WORLD_SIZE:
        pytest.skip(f"run with torchrun --nproc-per-node={WORLD_SIZE}")
    setup_torch_distributed()
    yield
    dist.destroy_process_group()


@pytest.fixture(autouse=True)
def reset_global_state():
    """Context parallelism rebinds attention methods on the classes; undo it between cases. Also drop
    compiled graphs, so one case's recompiles never push another past dynamo's recompile limit."""
    patched = [
        (FlashAttention, "_compute_attention"),
        (AfmoeFlashAttention, "_compute_attention"),
        (GptOssAttention, "compute_attention"),
    ]
    originals = [(cls, name, cls.__dict__[name]) for cls, name in patched]
    yield
    for cls, name, original in originals:
        setattr(cls, name, original)
    torch._dynamo.reset()
    torch.use_deterministic_algorithms(False)
    gc.collect()
    torch.cuda.empty_cache()


def _packed_batch(vocab_size: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """The same packed row on every rank: input ids, per-document position ids, labels, seq_lens."""
    generator = torch.Generator().manual_seed(0)
    input_ids = torch.randint(0, vocab_size, (1, SEQ_LEN), generator=generator)
    position_ids = torch.cat([torch.arange(length) for length in DOC_LENS]).unsqueeze(0)
    labels = torch.roll(input_ids, shifts=-1, dims=1)
    # No next-token target across a document boundary.
    labels[0, torch.cumsum(torch.tensor(DOC_LENS), 0) - 1] = IGNORE_INDEX
    return input_ids.cuda(), position_ids.cuda(), labels.cuda(), torch.tensor(DOC_LENS).cuda()


def _relative_error(actual: Tensor, expected: Tensor) -> float:
    """Relative L2 error, maxed over ranks so every rank takes the same assertion branch (a one-sided
    failure would desync the collectives of every later case)."""
    error = (actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-12)
    dist.all_reduce(error, op=dist.ReduceOp.MAX)
    return error.item()


def _run_reference(arch: str, state_dict: dict[str, Tensor], batch) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    """Unsharded forward/backward on this rank's GPU: logits, summed loss, per-parameter gradients."""
    input_ids, position_ids, labels, seq_lens = batch
    # Same precision as the sharded model: fp32-built (so fp32 buffers) with bf16 parameters, which is
    # what FSDP's mixed-precision policy computes with.
    model = build_model(arch, device="cuda", dtype=torch.float32)
    model.load_state_dict(state_dict)
    for param in model.parameters():
        param.data = param.data.to(torch.bfloat16)
    logits = forward(model, input_ids, position_ids, seq_lens=seq_lens)["logits"]
    loss = cross_entropy_sum(logits, labels)
    loss.backward()
    grads = {name: param.grad.float() for name, param in model.named_parameters() if param.grad is not None}
    return logits.detach(), loss.detach(), grads


@functools.cache
def _parallel_dims(parallel: Parallel) -> ParallelDims:
    """One set of device meshes per layout: each mesh owns NCCL communicators that are never freed."""
    return ParallelDims(dp_replicate=1, dp_shard=-1, cp=parallel.cp, pp=1, ep=parallel.ep, world_size=WORLD_SIZE)


def _build_parallel(arch: str, parallel: Parallel, runtime: Runtime, state_dict: dict[str, Tensor]):
    """Shard the model in the trainer's order (EP -> AC -> compile -> FSDP), then load the reference weights."""
    # The models are built with ARCH_CONFIGS' default flash_attention_2; CP must patch the same kernel.
    config = ModelConfig(
        attn="flash_attention_2",
        cp=parallel.cp,
        cp_style=parallel.cp_style,
        ep=parallel.ep,
        ac=None if runtime.ac == "none" else ActivationCheckpointConfig(mode=runtime.ac),
        ac_offloading=None,
        compile=CompileConfig() if runtime.compile else None,
    )
    parallel_dims = _parallel_dims(parallel)
    # FP32 master weights hold the bf16 reference values exactly; FSDP computes in bf16.
    model = build_model(arch, device="meta", dtype=torch.float32)
    configure_moe_runtime(model, config, parallel_dims)
    if config.ac is not None:
        apply_ac(model, config.ac)
    if config.compile is not None:
        apply_compile(model, config.compile)
    setup_fsdp(model, config, parallel_dims)
    model.to_empty(device="cuda")
    model.init_buffers_post_meta()
    set_model_state_dict(model, state_dict, options=StateDictOptions(full_state_dict=True))
    if parallel_dims.cp_enabled:
        setup_context_parallel(model, config, parallel_dims)
    return model, parallel_dims


@pytest.mark.gpu
@pytest.mark.parametrize(("arch", "parallel_name", "runtime_name"), _cases())
def test_parallel_matches_single_gpu(arch: str, parallel_name: str, runtime_name: str, distributed):
    parallel = PARALLELS[parallel_name]
    case = f"{parallel_name}-{runtime_name}"
    reference_model = build_model(arch, device="meta")
    model_config = reference_model.config
    if parallel.cp > 1:
        supported = type(reference_model).cp_support(model_config).styles
        if parallel.cp_style not in supported:
            pytest.skip(f"{arch} does not support cp_style={parallel.cp_style!r}")
    if parallel.ep > 1 and not any(isinstance(module, MoE) for module in reference_model.modules()):
        pytest.skip(f"{arch} has no MoE layers")

    if arch in DETERMINISTIC_ARCHS:
        torch.use_deterministic_algorithms(True, warn_only=True)
    state_dict = reference_state_dict(arch)
    text_config = getattr(model_config, "text_config", model_config)
    batch = _packed_batch(text_config.vocab_size)
    ref_logits, ref_loss, ref_grads = _run_reference(arch, state_dict, batch)

    model, parallel_dims = _build_parallel(arch, parallel, RUNTIMES[runtime_name], state_dict)
    input_ids, position_ids, labels, seq_lens = batch
    if parallel_dims.cp_enabled:
        cp_mesh = parallel_dims.world_mesh["cp"]
        cp_rank, cp_group = cp_mesh.get_local_rank(), cp_mesh.get_group()
        input_ids, position_ids = setup_cp_params(
            input_ids, position_ids, cp_rank, parallel.cp, cp_group, seq_lens=seq_lens, cp_style=parallel.cp_style
        )
        labels = shard_for_cp(labels, cp_rank=cp_rank, cp_world_size=parallel.cp)
        ref_logits = shard_for_cp(ref_logits, cp_rank=cp_rank, cp_world_size=parallel.cp)

    logits = forward(
        model, input_ids, position_ids, seq_lens=seq_lens, seq_lens_are_pre_shard=parallel_dims.cp_enabled
    )["logits"]
    loss = cross_entropy_sum(logits, labels)
    # FSDP averages gradients over all ranks; every data-parallel replica sees the full batch, so
    # this scale makes the reduced gradient the gradient of the full-sequence summed loss.
    data_parallel_replicas = WORLD_SIZE // parallel.cp
    (loss * parallel_dims.fsdp_gradient_divide_factor / data_parallel_replicas).backward()

    logits_error = _relative_error(logits, ref_logits)
    full_loss = loss.detach().clone()
    if parallel_dims.cp_enabled:
        dist.all_reduce(full_loss, group=parallel_dims.world_mesh["cp"].get_group())
    loss_error = _relative_error(full_loss, ref_loss)

    grad_errors = {}
    for name, param in model.named_parameters():
        name = resolve_fqn(model, name)
        if param.grad is None:
            assert name not in ref_grads, f"{arch} {case}: {name} has no gradient but the reference does"
            continue
        grad = param.grad.full_tensor() if isinstance(param.grad, DTensor) else param.grad
        assert name in ref_grads, f"{arch} {case}: {name} has a gradient but the reference does not"
        grad_errors[name] = _relative_error(grad, ref_grads[name])
    worst = sorted(grad_errors.items(), key=lambda item: item[1], reverse=True)[:5]
    if dist.get_rank() == 0:
        print(f"[errors] {arch} {case}: logits={logits_error:.2e} loss={loss_error:.2e} grad={worst[0][1]:.2e}")

    logits_rtol, loss_rtol, grad_rtol = LOGITS_RTOL, LOSS_RTOL, GRAD_RTOL
    if arch in LINEAR_ATTENTION_ARCHS and parallel.cp > 1:
        logits_rtol, loss_rtol, grad_rtol = (
            LINEAR_ATTENTION_CP_LOGITS_RTOL,
            LINEAR_ATTENTION_CP_LOSS_RTOL,
            LINEAR_ATTENTION_CP_GRAD_RTOL,
        )
    if RUNTIMES[runtime_name].compile:
        logits_rtol = max(logits_rtol, COMPILE_LOGITS_RTOL)
        loss_rtol = max(loss_rtol, COMPILE_LOSS_RTOL)
        grad_rtol = max(grad_rtol, COMPILE_GRAD_RTOL)
    assert logits_error < logits_rtol, f"{arch} {case}: logits relative error {logits_error:.2e}"
    assert loss_error < loss_rtol, f"{arch} {case}: loss {full_loss.item()} vs {ref_loss.item()}"
    assert worst[0][1] < grad_rtol, f"{arch} {case}: worst gradient relative errors {worst}"
