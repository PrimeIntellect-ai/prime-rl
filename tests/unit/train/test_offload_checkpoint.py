"""Offloading and checkpointing of the trainer on two GPUs.

Question: does each of the trainer's offloading strategies train the same weights as training without it, and does a
checkpoint restore the exact training state under each of them?

The strategies:

| Strategy | Config | On the CPU | Math against no offloading |
|---|---|---|---|
| Optimizer offload (default) | `model.offload = "optimizer"` | optimizer states between steps | same, AdamW unfused |
| Full offload | `model.offload = "full"` | FP32 masters, optimizer states, gradient sums | BF16 gradients, CPU AdamW |
| Activation offloading (default) | `model.ac_offloading` | tensors saved for the backward | same |
| Async checkpointing | `ckpt.async_save` | a copy of the saved state, written in the background | same |

Optimizer offload (`CPUOffloadOptimizer`) runs AdamW's per-parameter implementation and no offloading its fused kernel;
the two differ by an ulp. Full offload (`FullCPUOffloadOptimizer`) keeps BF16 weights on the GPU and steps each
chunk of parameters with a C++ AdamW or SignSGD on the CPU as soon as the chunk's last gradient of the step arrives:
during the last backward, as RL and SFT train, or after it, on SFT validation steps. The trainer does not use FSDP's
own CPU offload (`OffloadPolicy`).

How it works:

- One launch, many cases, as in `test_distributed_invariance.py`, whose tiny models (Qwen3 with tied embeddings,
  Qwen3-MoE), Reverse-Text-SFT rows and SFT loss this suite reuses.
- A run builds the model, optimizer and scheduler with the trainer's setup functions and trains 3 steps of the SFT
  trainer's loop: `run_micro_batches` over 4 rows per step (2 micro batches per rank on dp2 and ep2, 4 on cp2), the
  gradient scaling, clipping at 1.0, the optimizer step and the scheduler step. The cosine schedule changes the
  learning rate every step, so a resume that loses the scheduler or the learning rate shows. MoE runs replay one fixed
  routing, recorded from the initial weights, so that runs whose weights differ slightly still route every token alike.
- The state of a run is what its checkpoint saves, `AppState.state_dict()`: the model and optimizer state by parameter
  name, the param groups, the scheduler and the progress counters. Every tensor is gathered to the full tensor and
  hashed with sha256.

Cases:

- Offload: a strategy against the same layout without it, 3 steps each from the same weights, compared bitwise. The
  reference for optimizer offload runs AdamW unfused as well. Full offload's math differs: it rounds each micro batch's
  reduced gradient to BF16 before summing them, runs its own AdamW kernel and never clips. Its case therefore compares
  each parameter's update (final minus initial FP32 weight) and each optimizer state tensor by relative error, against
  a reference without clipping, and the step counters, learning rates, scheduler and progress bitwise. Parameters whose
  micro-batch gradients nearly cancel move the most: the worst update errors measured are 3.6e-2 (a norm weight) and
  3.4e-2 (the MoE embedding), and the worst state error 6.9e-3, on 2x RTX PRO 6000 and 2x H100. The tolerances are
  about 3x those.
- Resume: train 2 steps and save, then train step 3 with the same objects. Fresh objects load the checkpoint and train
  step 3. The loaded state must hash the same as the saved state, and both step-3 states the same as the straight
  3-step run. With `async_save`, the write runs while step 3 trains.
- Load: a checkpoint saved with one strategy or layout, loaded with another. The loaded training state (FP32 weights,
  optimizer states and step counters, learning rates, scheduler, progress) must equal the saved one bitwise, and the
  next step must run.
- One GPU, full suite only: a one-GPU launch loads checkpoints that the two-GPU launch saved and compares the hashes.

Bitwise comparisons rely on deterministic training: two runs of one setup in one process are bitwise identical for
these models and layouts in eager mode, with FA2 on SM120 and FA3 on H100.

Known bugs are strict xfails, so their fix has to remove the mark:

- A full offload resume keeps the configured learning rate: `AppState.load_state_dict` restores the param groups of
  `checkpoint_optimizer()`, which full offload builds as a copy. The other full offload resumes use a constant learning
  rate, which this bug leaves alone.
- Loading a checkpoint saved without offloading or with full offload into optimizer offload copies the saved AdamW's
  `fused=True` into the live param groups. The next step runs the fused kernel on the step counters, which optimizer
  offload keeps on the CPU, and fails.
- Full offload checkpoints hold the BF16 GPU weights under `model` and the FP32 masters only in the optimizer state, so
  loading one without full offload loads BF16-rounded weights. A checkpoint without masters does not load into full
  offload at all.

Adding a case is one line in `CASES` or `FULL_SUITE_CASES`, and a new strategy one `Setup` field. To run a suite
without pytest (it writes `results.json` to the output dir):

    uv run torchrun --nproc-per-node=2 tests/unit/train/test_offload_checkpoint.py <output_dir> tiny|full
"""

import hashlib
import json
import math
import shutil
import sys
import time
import traceback
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import pytest
import torch
import torch.distributed as dist
from test_distributed_invariance import TOKENIZER, compute_loss, load_rows, record_routing, save_tiny_model
from torch.distributed.checkpoint.api import CheckpointException
from torch.distributed.checkpoint.state_dict import get_model_state_dict
from torch.distributed.tensor import DTensor

from prime_rl.configs.trainer import (
    ActivationOffloadingConfig,
    AdamWConfig,
    CheckpointConfig,
    ConstantSchedulerConfig,
    CosineSchedulerConfig,
    ModelConfig,
    MuonConfig,
    OffloadMode,
    SignSGDConfig,
    TokenizerConfig,
)
from prime_rl.trainer.ckpt import AppState, CheckpointManager, Progress, setup_model_files
from prime_rl.trainer.micro_batches import run_micro_batches
from prime_rl.trainer.model import get_full_offload_dtype_policy, resolve_auto_attn, setup_model, setup_tokenizer
from prime_rl.trainer.optim import setup_optimizer
from prime_rl.trainer.optim.offload import FullCPUOffloadOptimizer
from prime_rl.trainer.parallel_dims import ParallelDims, get_parallel_dims
from prime_rl.trainer.perf import FlopCounter
from prime_rl.trainer.scheduler import setup_scheduler
from prime_rl.trainer.utils import clip_grad_norm_, prepare_gradient_offload, scale_gradients_, setup_torch_distributed
from prime_rl.trainer.world import get_world
from prime_rl.utils.act_offloading import maybe_activation_offloading
from prime_rl.utils.cp import setup_context_parallel
from prime_rl.utils.logger import setup_logger

pytestmark = [pytest.mark.gpu]

NUM_STEPS = 3
SAVE_STEP = 2
ROWS_PER_STEP = 4
MASTER_WEIGHT = FullCPUOffloadOptimizer._MASTER_WEIGHT_STATE
OPTIMIZERS = {"adamw": AdamWConfig, "muon": MuonConfig, "sign_sgd": SignSGDConfig}
# The cosine schedule changes the learning rate every step
SCHEDULES = {"cosine": CosineSchedulerConfig(warmup_steps=0), "constant": ConstantSchedulerConfig()}
HYPERPARAMETERS = ("lr", "initial_lr", "betas", "eps", "weight_decay")


@dataclass(frozen=True)
class Setup:
    """How a run trains: the model, its layout, the offloading and the optimizer."""

    model: str
    cp: int = 1
    ep: int = 1
    offload: OffloadMode = "none"
    ac_offloading: bool = False
    optimizer: Literal["adamw", "muon", "sign_sgd"] = "adamw"
    fused_adamw: bool = True
    """False runs AdamW unfused without offloading, as optimizer offload runs it, for a bitwise reference."""
    max_norm: float | None = 1.0
    """Full offload turns clipping off itself."""
    step_during_backward: bool = True
    """Full offload: False steps the CPU optimizer after the backward, as SFT validation steps do."""
    lr_schedule: Literal["cosine", "constant"] = "cosine"
    freeze_moe_router: bool = False
    """RL's default."""
    async_save: bool = False

    def model_config(self, models_dir: Path) -> ModelConfig:
        config = ModelConfig(
            name=(models_dir / self.model).as_posix(),
            cp=self.cp,
            ep=self.ep,
            compile=None,
            offload=self.offload,
            ac_offloading=ActivationOffloadingConfig() if self.ac_offloading else None,
            freeze_moe_router=self.freeze_moe_router,
        )
        resolve_auto_attn(config)
        return config


@dataclass(frozen=True)
class Offload:
    """Train `setup` and `reference` for NUM_STEPS steps from the same weights and compare their states."""

    name: str
    setup: Setup
    reference: Setup
    tolerances: dict | None = None
    """None compares bitwise. Otherwise the relative errors allowed on each parameter's update and on each optimizer
    state tensor."""


@dataclass(frozen=True)
class Resume:
    """Save after SAVE_STEP steps, resume from the checkpoint and train to NUM_STEPS."""

    name: str
    setup: Setup
    bug: str | None = None


@dataclass(frozen=True)
class Load:
    """Load the checkpoint `saved` writes after SAVE_STEP steps into a run of `setup`."""

    name: str
    saved: Setup
    setup: Setup
    bug: str | None = None


# About 3x the worst relative errors measured, see the module docstring
FULL_OFFLOAD_TOLERANCES = {"update": 1e-1, "state": 2e-2}
# Known bugs, see the module docstring. The other full offload resumes keep a constant learning rate, which the first
# bug leaves alone.
FULL_OFFLOAD_LR_BUG = (
    "full offload resumes keep the configured lr: AppState.load_state_dict restores a copy's param groups"
)
FUSED_FLAG_BUG = (
    "the saved AdamW's fused=True reaches optimizer offload's param groups, whose step counters are on the CPU"
)
BF16_WEIGHTS_BUG = (
    "full offload checkpoints hold BF16 weights under model and the FP32 masters only in the optimizer state"
)
NO_MASTERS_BUG = "a checkpoint without full offload's FP32 masters does not load into full offload"

DENSE = Setup("qwen3")
MOE = Setup("qwen3_moe")
CASES = [
    Offload("qwen3_moe-ep2-ac_offloading", replace(MOE, ep=2, ac_offloading=True), replace(MOE, ep=2)),
    Offload("qwen3-dp2-optimizer", replace(DENSE, offload="optimizer"), replace(DENSE, fused_adamw=False)),
    Offload("qwen3-dp2-full", replace(DENSE, offload="full"), replace(DENSE, max_norm=None), FULL_OFFLOAD_TOLERANCES),
    Offload(
        "qwen3-dp2-full-step_after_backward",
        replace(DENSE, offload="full", step_during_backward=False),
        replace(DENSE, offload="full"),
    ),
    Resume(
        "resume-qwen3-dp2-optimizer-ac_offloading-async",
        replace(DENSE, offload="optimizer", ac_offloading=True, async_save=True),
    ),
]
FULL_SUITE_CASES = [
    Offload("qwen3-dp2-ac_offloading", replace(DENSE, ac_offloading=True), DENSE),
    Offload("qwen3-cp2-ac_offloading", replace(DENSE, cp=2, ac_offloading=True), replace(DENSE, cp=2)),
    Offload("qwen3_moe-dp2-ac_offloading", replace(MOE, ac_offloading=True), MOE),
    Offload(
        "qwen3-dp2-muon-optimizer",
        replace(DENSE, optimizer="muon", offload="optimizer"),
        replace(DENSE, optimizer="muon"),
    ),
    Offload(
        "qwen3_moe-ep2-muon-optimizer",
        replace(MOE, ep=2, optimizer="muon", offload="optimizer"),
        replace(MOE, ep=2, optimizer="muon"),
    ),
    Offload(
        "qwen3_moe-ep2-full-step_after_backward",
        replace(MOE, ep=2, offload="full", step_during_backward=False),
        replace(MOE, ep=2, offload="full"),
    ),
    *(
        Offload(f"{name}-optimizer", replace(setup, offload="optimizer"), replace(setup, fused_adamw=False))
        for name, setup in [
            ("qwen3-cp2", replace(DENSE, cp=2)),
            ("qwen3_moe-dp2", MOE),
            ("qwen3_moe-ep2", replace(MOE, ep=2)),
        ]
    ),
    *(
        Offload(f"{name}-full", replace(setup, offload="full"), replace(setup, max_norm=None), FULL_OFFLOAD_TOLERANCES)
        for name, setup in [
            ("qwen3-cp2", replace(DENSE, cp=2)),
            ("qwen3_moe-dp2", MOE),
            ("qwen3_moe-ep2", replace(MOE, ep=2)),
        ]
    ),
    Resume("resume-qwen3-dp2-none", DENSE),
    Resume("resume-qwen3-dp2-none-async", replace(DENSE, async_save=True)),
    Resume("resume-qwen3-dp2-optimizer", replace(DENSE, offload="optimizer")),
    Resume("resume-qwen3-dp2-full", replace(DENSE, offload="full"), bug=FULL_OFFLOAD_LR_BUG),
    Resume("resume-qwen3-dp2-full-constant_lr", replace(DENSE, offload="full", lr_schedule="constant")),
    Resume(
        "resume-qwen3-dp2-full-async-constant_lr",
        replace(DENSE, offload="full", lr_schedule="constant", async_save=True),
    ),
    Resume(
        "resume-qwen3-dp2-sign_sgd-full-constant_lr",
        replace(DENSE, optimizer="sign_sgd", offload="full", lr_schedule="constant"),
    ),
    Resume("resume-qwen3-dp2-muon-optimizer", replace(DENSE, optimizer="muon", offload="optimizer")),
    Resume("resume-qwen3-cp2-optimizer", replace(DENSE, cp=2, offload="optimizer")),
    Resume("resume-qwen3_moe-ep2-optimizer-ac_offloading", replace(MOE, ep=2, offload="optimizer", ac_offloading=True)),
    Resume(
        "resume-qwen3_moe-ep2-full-frozen_router-constant_lr",
        replace(MOE, ep=2, offload="full", lr_schedule="constant", freeze_moe_router=True),
    ),
    Load("load-qwen3-dp2-optimizer-to-none", replace(DENSE, offload="optimizer"), DENSE),
    Load("load-qwen3-dp2-none-to-optimizer", DENSE, replace(DENSE, offload="optimizer"), bug=FUSED_FLAG_BUG),
    Load("load-qwen3-dp2-full-to-none", replace(DENSE, offload="full"), DENSE, bug=BF16_WEIGHTS_BUG),
    Load(
        "load-qwen3-dp2-full-to-optimizer",
        replace(DENSE, offload="full"),
        replace(DENSE, offload="optimizer"),
        bug=f"{BF16_WEIGHTS_BUG}; {FUSED_FLAG_BUG}",
    ),
    Load("load-qwen3-dp2-none-to-full", DENSE, replace(DENSE, offload="full"), bug=NO_MASTERS_BUG),
    Load(
        "load-qwen3-dp2-optimizer-to-full",
        replace(DENSE, offload="optimizer"),
        replace(DENSE, offload="full"),
        bug=NO_MASTERS_BUG,
    ),
    Load(
        "load-qwen3-dp2-to-cp2-optimizer",
        replace(DENSE, offload="optimizer"),
        replace(DENSE, cp=2, offload="optimizer"),
    ),
    Load(
        "load-qwen3_moe-ep2-to-dp2-optimizer",
        replace(MOE, ep=2, offload="optimizer"),
        replace(MOE, offload="optimizer"),
    ),
    Load(
        "load-qwen3_moe-dp2-to-ep2-full-constant_lr",
        replace(MOE, offload="full", lr_schedule="constant"),
        replace(MOE, ep=2, offload="full", lr_schedule="constant"),
    ),
]
# The full suite's checkpoints that a one-GPU launch loads
DP1_CASES = [
    "resume-qwen3-dp2-none",
    "resume-qwen3-dp2-optimizer",
    "resume-qwen3-dp2-full-constant_lr",
    "resume-qwen3_moe-ep2-optimizer-ac_offloading",
]
SUITES = {"tiny": CASES, "full": FULL_SUITE_CASES}
TIMEOUTS = {"tiny": 5 * 60, "full": 20 * 60, "dp1": 5 * 60}


class Env:
    """What the cases of a launch share: the tiny models and rows, the device meshes, and the runs and checkpoints
    already made."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.models_dir = output_dir / "models"
        self.tokenizer = setup_tokenizer(TokenizerConfig(name=TOKENIZER), model_name=TOKENIZER)
        self.layouts: dict[tuple[int, int], ParallelDims] = {}
        self.rows: dict[str, list[dict]] = {}
        self.initial_weights: dict[str, dict] = {}
        self.hashes: dict[Setup, dict] = {}
        """The hashed state after NUM_STEPS steps without a checkpoint, per setup."""
        self.tensors: dict[Setup, dict] = {}
        """The training state's tensors after NUM_STEPS steps, for the setups compared with tolerances."""
        self.keep_tensors: set[Setup] = set()
        self.saved: dict[Setup, tuple[CheckpointManager, dict]] = {}
        """A checkpoint after SAVE_STEP steps and the hashed state it saved, per setup."""

    def prepare(self, models: set[str]) -> None:
        for name in sorted(models):
            save_tiny_model(name, self.models_dir)
        dist.barrier()
        rows = load_rows(self.tokenizer, NUM_STEPS * ROWS_PER_STEP)
        for name in sorted(models):
            self.initial_weights[name], routes = self.initial(name, rows)
            self.rows[name] = rows if routes is None else [{**row, "routed_experts": r} for row, r in zip(rows, routes)]

    def parallel_dims(self, config: ModelConfig) -> ParallelDims:
        # Device meshes are made once per layout, as in a trainer process
        if (config.cp, config.ep) not in self.layouts:
            self.layouts[config.cp, config.ep] = get_parallel_dims(config)
        return self.layouts[config.cp, config.ep]

    @torch.no_grad()
    def initial(self, name: str, rows: list[dict]) -> tuple[dict, list[torch.Tensor] | None]:
        """The model's initial FP32 weights, and for an MoE model the experts its routers pick for every row, which
        every run replays."""
        config = Setup(name).model_config(self.models_dir)
        model = setup_model(config, self.parallel_dims(config))
        weights = {key: full_tensor(value) for key, value in get_model_state_dict(model).items()}
        routing = record_routing(model)
        routes = []
        for row in rows:
            compute_loss(model, row, cp_rank=0, cp_size=1)
            routes.append(routing())
        return weights, None if routes[0] is None else routes

    def train(self, setup: Setup) -> None:
        """Train `setup` for NUM_STEPS steps without a checkpoint, once."""
        setup = replace(setup, async_save=False)
        if setup in self.hashes:
            return
        run = Run(setup, self)
        run.train(NUM_STEPS)
        state = checkpoint_state(run)
        run.close()
        self.hashes[setup] = hashes(state)
        if setup in self.keep_tensors:
            self.tensors[setup] = training_state(state)

    def checkpoint(self, setup: Setup) -> tuple[CheckpointManager, dict]:
        if setup not in self.saved:
            run = Run(setup, self)
            manager = self.ckpt_manager(setup, f"saved-{len(self.saved)}")
            saved = run.train(SAVE_STEP, manager, save_at=SAVE_STEP)
            manager.wait()
            run.close()
            self.saved[setup] = manager, saved
        return self.saved[setup]

    def ckpt_manager(self, setup: Setup, name: str) -> CheckpointManager:
        model_files = setup_model_files(setup.model_config(self.models_dir), self.tokenizer, processor=None)
        config = CheckpointConfig(async_save=setup.async_save)
        return CheckpointManager(self.output_dir / "checkpoints" / name, config, model_files)


class Run:
    """A trainer's model, optimizer, scheduler and progress for one setup, trained with the SFT trainer's loop."""

    def __init__(self, setup: Setup, env: Env, from_checkpoint: bool = False):
        self.setup = setup
        self.config = setup.model_config(env.models_dir)
        self.parallel_dims = env.parallel_dims(self.config)
        self.model = setup_model(self.config, self.parallel_dims, loading_from_checkpoint_later=from_checkpoint)
        if self.parallel_dims.cp_enabled:
            setup_context_parallel(self.model, self.config, self.parallel_dims)
        self.optim_config = OPTIMIZERS[setup.optimizer](lr=1e-4, weight_decay=0.01, max_norm=setup.max_norm)
        self.optimizer, self.gradient_manager = setup_optimizer(
            self.optim_config,
            list(self.model.named_parameters()),
            self.parallel_dims,
            offload=setup.offload,
            model=self.model,
            full_offload_dtype_policy=get_full_offload_dtype_policy(self.model) if setup.offload == "full" else None,
        )
        if not setup.fused_adamw:
            for group in self.optimizer.param_groups:
                group["fused"] = False
        schedule = SCHEDULES[setup.lr_schedule]
        self.scheduler = setup_scheduler(self.optimizer, schedule, NUM_STEPS, self.optim_config.lr)
        self.progress = Progress()
        self.rows = env.rows.get(setup.model)
        self.perf_counter = SimpleNamespace(flop_counter=FlopCounter(self.model))

    def train(self, until: int, manager: CheckpointManager | None = None, save_at: int | None = None) -> dict | None:
        """Train up to step `until` as the trainer's loop does. Saving after step `save_at` returns the hashed state
        it saved."""
        saved = None
        while True:
            self.train_step(manager)
            if self.progress.step == save_at:
                saved = hashes(checkpoint_state(self))
                manager.save(self.progress.step, self.model, [self.optimizer], self.scheduler, self.progress)
            if self.progress.step == until:
                return saved
            self.progress.step += 1

    def train_step(self, manager: CheckpointManager | None) -> None:
        """One step of the SFT trainer's loop (`prime_rl.trainer.train`) on this step's rows."""
        parallel_dims = self.parallel_dims
        rows = self.rows[(self.progress.step - 1) * ROWS_PER_STEP : self.progress.step * ROWS_PER_STEP]
        dp_mesh = parallel_dims.get_mesh("dp")
        micro_batches = rows[dp_mesh.get_local_rank() :: dp_mesh.size()]
        cp_rank = parallel_dims.world_mesh["cp"].get_local_rank() if parallel_dims.cp_enabled else 0
        if self.gradient_manager is not None:
            tokens = torch.tensor(sum(int(row["loss_mask"].sum()) for row in micro_batches), device="cuda")
            dist.all_reduce(tokens, group=parallel_dims.get_mesh("dp_cp").get_group())
            grad_scale = parallel_dims.fsdp_gradient_divide_factor / (tokens.item() // parallel_dims.cp)
            prepare_gradient_offload(
                self.gradient_manager, grad_scale, overlap_optimizer=self.setup.step_during_backward
            )
        loss_tokens = torch.zeros((), dtype=torch.float64, device="cuda")

        def summed_loss(row: dict) -> torch.Tensor:
            with maybe_activation_offloading(self.config.ac_offloading, self.model):
                loss, row_tokens = compute_loss(self.model, row, cp_rank, parallel_dims.cp, row.get("routed_experts"))
            loss_tokens.add_(row_tokens)
            return loss

        stats = run_micro_batches(
            self.model,
            micro_batches,
            summed_loss,
            len(micro_batches),
            parallel_dims,
            self.perf_counter,
            self.gradient_manager,
            sums={"loss_tokens": loss_tokens},
        )
        if self.gradient_manager is None:
            factor = parallel_dims.fsdp_gradient_divide_factor / stats["loss_tokens"]
            scale_gradients_(None, self.model, factor.float())
        if self.optim_config.max_norm is not None:
            clip_grad_norm_(self.gradient_manager, self.model, self.optim_config.max_norm)
        if manager is not None:
            manager.wait_for_staging()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()
        self.progress.total_tokens += round(stats["tokens"].item() / parallel_dims.cp)
        self.progress.total_samples += len(rows)

    def load(self, manager: CheckpointManager) -> None:
        manager.load(SAVE_STEP, self.model, [self.optimizer], self.scheduler, self.progress)

    def close(self) -> None:
        if self.gradient_manager is not None:
            self.gradient_manager.close()


def full_tensor(value: torch.Tensor) -> torch.Tensor:
    """A CPU copy of the full tensor. Offloaded DTensors keep their shard on the CPU, so the gather runs on the GPU."""
    if isinstance(value, DTensor):
        value = DTensor(value._local_tensor.cuda(), value._spec, requires_grad=False).full_tensor()
    return value.detach().to("cpu", copy=True)


def flatten(value, prefix: str = "") -> dict:
    """The leaves of nested dicts and lists of dicts by path, tensors as full CPU copies. Every rank walks the same
    keys in the same order, since the gathers are collectives."""
    if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
        value = dict(enumerate(value))
    if not isinstance(value, dict):
        return {prefix: full_tensor(value) if isinstance(value, torch.Tensor) else value}
    leaves = {}
    for key in sorted(value, key=str):
        leaves |= flatten(value[key], f"{prefix}/{key}" if prefix else str(key))
    return leaves


def checkpoint_state(run: Run) -> dict:
    """What a checkpoint of `run` saves (`AppState.state_dict()`), flattened."""
    return flatten(AppState(run.model, [run.optimizer], run.scheduler, run.progress).state_dict())


def sha256(value) -> str:
    digest = hashlib.sha256()
    if isinstance(value, torch.Tensor):
        digest.update(f"{value.dtype}{tuple(value.shape)}".encode())
        digest.update(value.contiguous().view(-1).view(torch.uint8).numpy())
    else:
        digest.update(repr(value).encode())
    return digest.hexdigest()


def hashes(state: dict) -> dict[str, str]:
    return {key: sha256(value) for key, value in state.items()}


def differing(state: dict[str, str], reference: dict[str, str], keys: list[str] | None = None) -> list[str]:
    """The entries whose hashes differ, or that only one side has."""
    keys = sorted(state.keys() | reference.keys()) if keys is None else keys
    return [key for key in keys if key not in state or key not in reference or state[key] != reference[key]]


def training_state(state: dict) -> dict:
    """What a resume must restore under any offloading: the FP32 weights of the trained parameters (full offload's
    masters, else the model's parameters), the optimizer states and step counters, the param groups'
    hyperparameters, the scheduler and the progress counters. Works on states and on their hashes."""
    optimizer_states = {key: value for key, value in state.items() if key.startswith("optimizers/state/")}
    trained = {key.removeprefix("optimizers/state/").rsplit("/", 1)[0] for key in optimizer_states}
    masters = {fqn: state.get(f"optimizers/state/{fqn}/{MASTER_WEIGHT}") for fqn in trained}
    training = {
        f"weights/{fqn}": master if master is not None else state[f"model/{fqn}"] for fqn, master in masters.items()
    }
    training |= {key: value for key, value in optimizer_states.items() if not key.endswith(f"/{MASTER_WEIGHT}")}
    for key, value in state.items():
        if key.startswith("optimizers/param_groups/") and key.rsplit("/", 1)[1] in HYPERPARAMETERS:
            training[key] = value
        elif key.startswith(("scheduler/", "progress/")):
            training[key] = value
    return training


def relative_error(value: torch.Tensor, reference: torch.Tensor) -> float:
    error, norm = (torch.linalg.vector_norm(tensor.double()).item() for tensor in (value - reference, reference))
    return error / norm if norm > 0 else (0.0 if error == 0 else math.inf)


def tolerance_errors(state: dict, reference: dict, initial_weights: dict) -> dict:
    """Relative errors of each parameter's update and each optimizer state tensor; the other entries of the training
    state (step counters, learning rates, scheduler, progress) are compared bitwise."""
    keys = sorted(state.keys() & reference.keys())
    worst = {"update": ("", 0.0), "state": ("", 0.0)}
    exact = []
    for key in keys:
        if key.startswith("weights/"):
            initial = initial_weights[key.removeprefix("weights/")]
            kind, error = "update", relative_error(state[key] - initial, reference[key] - initial)
        elif isinstance(state[key], torch.Tensor) and state[key].is_floating_point() and not key.endswith("/step"):
            kind, error = "state", relative_error(state[key], reference[key])
        else:
            exact.append(key)
            continue
        if error >= worst[kind][1]:
            worst[kind] = (key, error)
    return {
        "differing": differing(
            hashes({key: state[key] for key in exact}), hashes({key: reference[key] for key in exact})
        ),
        "one_sided": sorted(state.keys() ^ reference.keys()),
        "worst": worst,
    }


def run_offload(case: Offload, env: Env) -> dict:
    setup, reference = (replace(s, async_save=False) for s in (case.setup, case.reference))
    env.train(setup)
    env.train(reference)
    if case.tolerances is None:
        return {"differing": differing(env.hashes[setup], env.hashes[reference])}
    return tolerance_errors(env.tensors[setup], env.tensors[reference], env.initial_weights[setup.model])


def run_resume(case: Resume, env: Env) -> dict:
    env.train(case.setup)
    straight = env.hashes[replace(case.setup, async_save=False)]
    manager = env.ckpt_manager(case.setup, case.name)
    run = Run(case.setup, env)
    saved = run.train(NUM_STEPS, manager, save_at=SAVE_STEP)
    manager.wait()
    env.saved[case.setup] = manager, saved
    continued = differing(hashes(checkpoint_state(run)), straight)
    run.close()
    resumed = Run(case.setup, env, from_checkpoint=True)
    resumed.load(manager)
    loaded = differing(hashes(checkpoint_state(resumed)), saved)
    resumed.progress.step += 1
    resumed.train(NUM_STEPS)
    final = differing(hashes(checkpoint_state(resumed)), straight)
    resumed.close()
    return {"loaded": loaded, "continued": continued, "resumed": final, "saved": saved}


def run_load(case: Load, env: Env) -> dict:
    manager, saved = env.checkpoint(case.saved)
    run = Run(case.setup, env, from_checkpoint=True)
    run.load(manager)
    loaded, saved = training_state(hashes(checkpoint_state(run))), training_state(saved)
    result = {"differing": differing(loaded, saved, sorted(loaded.keys() & saved.keys()))}
    # The next step must run
    run.progress.step += 1
    run.train(NUM_STEPS)
    run.close()
    return result


RUNNERS = {Offload: run_offload, Resume: run_resume, Load: run_load}


def isolated(run, *args) -> dict:
    """The result of `run`, or the error it raised. DCP raises `CheckpointException`, which is no `Exception`."""
    try:
        return run(*args)
    except (Exception, CheckpointException):
        traceback.print_exc()
        return {"error": traceback.format_exc()}


def setups(case) -> list[Setup]:
    return [value for value in vars(case).values() if isinstance(value, Setup)]


def main(output_dir: Path, cases: list) -> None:
    setup_logger("warning")
    setup_torch_distributed(enable_gloo=True)
    env = Env(output_dir)
    env.prepare({setup.model for case in cases for setup in setups(case)})
    # CP setup patches the attention classes for the rest of the process
    cases = sorted(cases, key=lambda case: max(setup.cp for setup in setups(case)))
    results = {}
    for index, case in enumerate(cases):
        # Only the setups that later cases compare with tolerances keep their tensors
        env.keep_tensors = {
            replace(setup, async_save=False)
            for later in cases[index:]
            if isinstance(later, Offload) and later.tolerances is not None
            for setup in setups(later)
        }
        env.tensors = {setup: tensors for setup, tensors in env.tensors.items() if setup in env.keep_tensors}
        start = time.perf_counter()
        results[case.name] = isolated(RUNNERS[type(case)], case, env)
        results[case.name]["seconds"] = time.perf_counter() - start
        torch.cuda.empty_cache()
        # Checkpoints take ~1 GB each: keep only those that later cases or the one-GPU launch load
        sources = {later.saved for later in cases[index + 1 :] if isinstance(later, Load)}
        for setup, (manager, _) in list(env.saved.items()):
            if setup not in sources and manager.ckpt_dir.parent.name not in DP1_CASES:
                del env.saved[setup]
                if get_world().is_master:
                    shutil.rmtree(manager.ckpt_dir.parent)
    if get_world().is_master:
        summary = {
            name: {key: value for key, value in result.items() if key != "saved"} for name, result in results.items()
        }
        print(json.dumps(summary, indent=2))
        (output_dir / "results.json").write_text(json.dumps(results, indent=2))


def load_on_one_gpu(output_dir: Path, two_gpu_dir: Path) -> None:
    """Loads the checkpoints of the full suite's DP1_CASES on one GPU and compares the loaded states with the saved
    ones."""
    setup_logger("warning")
    setup_torch_distributed(enable_gloo=True)
    env = Env(two_gpu_dir)
    saved = json.loads((two_gpu_dir / "results.json").read_text())
    results = {}
    for name in DP1_CASES:
        case = next(case for case in FULL_SUITE_CASES if case.name == name)

        def load(case: Resume = case) -> dict:
            setup = replace(case.setup, cp=1, ep=1, async_save=False)
            run = Run(setup, env, from_checkpoint=True)
            run.load(env.ckpt_manager(setup, case.name))
            loaded = hashes(checkpoint_state(run))
            run.close()
            return {"differing": differing(loaded, saved[case.name]["saved"])}

        results[name] = isolated(load)
    (output_dir / "results.json").write_text(json.dumps(results, indent=2))


def launch(run_process, output_dir: Path, suite: str, nproc: int = 2, *args: str) -> dict[str, dict]:
    cmd = ["uv", "run", "torchrun", "--standalone", f"--nproc-per-node={nproc}", __file__, output_dir.as_posix(), suite]
    assert run_process([*cmd, *args], timeout=TIMEOUTS[suite]).returncode == 0
    return json.loads((output_dir / "results.json").read_text())


@pytest.fixture(scope="module")
def tiny_results(run_process, tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict]:
    if torch.cuda.device_count() < 2:
        pytest.skip("needs 2 GPUs")
    output_dir = tmp_path_factory.mktemp("offload-checkpoint-tiny")
    results = launch(run_process, output_dir, "tiny")
    for directory in ("checkpoints", "models"):
        shutil.rmtree(output_dir / directory, ignore_errors=True)
    return results


@pytest.fixture(scope="module")
def full_suite_dir(run_process, tmp_path_factory: pytest.TempPathFactory):
    if torch.cuda.device_count() < 2:
        pytest.skip("needs 2 GPUs")
    output_dir = tmp_path_factory.mktemp("offload-checkpoint-full")
    launch(run_process, output_dir, "full")
    yield output_dir
    for directory in ("checkpoints", "models"):
        shutil.rmtree(output_dir / directory, ignore_errors=True)


@pytest.fixture(scope="module")
def full_suite_results(full_suite_dir: Path) -> dict[str, dict]:
    return json.loads((full_suite_dir / "results.json").read_text())


@pytest.fixture(scope="module")
def dp1_results(run_process, full_suite_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict]:
    output_dir = tmp_path_factory.mktemp("offload-checkpoint-dp1")
    return launch(run_process, output_dir, "dp1", 1, full_suite_dir.as_posix())


def check(result: dict, case) -> None:
    assert "error" not in result, result.get("error")
    if isinstance(case, Resume):
        for check_name in ("loaded", "continued", "resumed"):
            assert not result[check_name], f"{check_name} state differs from the reference: {result[check_name][:10]}"
        return
    assert not result["differing"], f"differs from the reference: {result['differing'][:10]}"
    if isinstance(case, Offload) and case.tolerances is not None:
        assert not result["one_sided"], f"entries on only one side: {result['one_sided'][:10]}"
        for kind, tolerance in case.tolerances.items():
            key, error = result["worst"][kind]
            assert error <= tolerance, f"{kind} relative error {error:.1e} > {tolerance:.0e} ({key})"


def params(cases: list) -> list:
    marks = lambda case: pytest.mark.xfail(strict=True, reason=case.bug) if getattr(case, "bug", None) else ()
    return [pytest.param(case, id=case.name, marks=marks(case)) for case in cases]


@pytest.mark.tiny
@pytest.mark.parametrize("case", params(CASES))
def test_offload_and_checkpoint(tiny_results: dict[str, dict], case):
    check(tiny_results[case.name], case)


@pytest.mark.parametrize("case", params(FULL_SUITE_CASES))
def test_full_suite_offload_and_checkpoint(full_suite_results: dict[str, dict], case):
    check(full_suite_results[case.name], case)


@pytest.mark.parametrize("name", DP1_CASES)
def test_one_gpu_loads_two_gpu_checkpoint(dp1_results: dict[str, dict], name: str):
    assert "error" not in dp1_results[name], dp1_results[name].get("error")
    assert not dp1_results[name]["differing"], f"differs from the saved state: {dp1_results[name]['differing'][:10]}"


if __name__ == "__main__":
    if sys.argv[2] == "dp1":
        load_on_one_gpu(Path(sys.argv[1]), Path(sys.argv[3]))
    else:
        main(Path(sys.argv[1]), SUITES[sys.argv[2]])
