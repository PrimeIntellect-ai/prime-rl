"""Offloading and checkpointing of the trainer on two GPUs: each offloading strategy (optimizer offload, full offload,
activation offloading, async checkpointing) must train the same weights as no offloading, and checkpoints must
restore the exact training state under each.

One `torchrun` launch runs every case of a suite. A run builds the model, optimizer and scheduler with the trainer's
setup functions and trains 3 SFT steps on the tiny models and rows of `test_distributed_invariance.py`. Its state is
what a checkpoint saves (`AppState.state_dict()`), one sha256 per gathered tensor. Training is deterministic, so states
compare bitwise, except full offload against no offloading (BF16 gradient sums, its own CPU AdamW, no clipping): each
weight update and optimizer state is compared by relative error.

- Offload: a strategy against the same setup without it, both trained 3 steps from the same weights.
- Resume: save after step 2, train step 3 on, and train it again from the checkpoint. The loaded state must equal the
  saved one, and both step-3 states the straight 3-step run.
- Load: a checkpoint of one strategy or layout, loaded into another, must restore the training state (FP32 weights,
  optimizer states, lr, progress). The one-GPU suite loads checkpoints of the full suite on one GPU.
- Reconfigure: a checkpoint loaded into a run whose lr and weight decay differ: the run's values must apply, with a
  warning, and the rest of the training state must load.

`CASES` run in the tiny GPU subset, the other suites in the full GPU suite. A case is one entry in its suite, a new
strategy a `Setup` field. Run a suite without pytest (it writes `results-<suite>.json`; `one_gpu` runs with one process
in the output dir of `full`):

    uv run torchrun --nproc-per-node=2 tests/unit/train/test_offload_checkpoint.py <output_dir> tiny|full
"""

import hashlib
import json
import math
import shutil
import sys
import time
import traceback
from collections.abc import Iterator
from dataclasses import dataclass, replace
from itertools import permutations
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
from prime_rl.trainer.parallel_dims import ParallelDims, get_parallel_dims
from prime_rl.trainer.perf import FlopCounter
from prime_rl.trainer.scheduler import setup_scheduler
from prime_rl.trainer.utils import clip_grad_norm_, prepare_gradient_offload, scale_gradients_, setup_torch_distributed
from prime_rl.trainer.world import get_world
from prime_rl.utils.act_offloading import maybe_activation_offloading
from prime_rl.utils.cp import setup_context_parallel
from prime_rl.utils.logger import get_logger, setup_logger

pytestmark = [pytest.mark.gpu]

NUM_STEPS = 3
SAVE_STEP = 2
ROWS_PER_STEP = 4
OPTIMIZERS = {"adamw": AdamWConfig, "muon": MuonConfig, "sign_sgd": SignSGDConfig}
# The cosine schedule changes the lr every step, so a resume at the wrong position of the schedule shows
SCHEDULES = {"cosine": CosineSchedulerConfig(warmup_steps=0), "constant": ConstantSchedulerConfig()}
HYPERPARAMETERS = ("lr", "initial_lr", "betas", "eps", "weight_decay")
# Full offload against no offloading: about 3x the worst relative errors measured, 3.6e-2 for a norm weight's update
# and 6.9e-3 for an optimizer state
TOLERANCES = {"update": 1e-1, "state": 2e-2}


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
    """False runs AdamW unfused, as optimizer offload does, for a bitwise reference."""
    max_norm: float | None = 1.0
    step_during_backward: bool = True
    """Full offload: False steps the CPU optimizer after the backward, as on SFT validation steps."""
    lr_schedule: Literal["cosine", "constant"] = "cosine"
    lr: float = 1e-4
    weight_decay: float = 0.01
    freeze_moe_router: bool = False
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
    """Train `setup` and `reference` 3 steps from the same weights and compare their states."""

    name: str
    setup: Setup
    reference: Setup


@dataclass(frozen=True)
class Resume:
    """Save after step 2 and train on, then load the checkpoint into fresh objects and train step 3."""

    name: str
    setup: Setup


@dataclass(frozen=True)
class Load:
    """Load the checkpoint that `saved` writes after step 2 into a run of `setup` and train step 3."""

    name: str
    saved: Setup
    setup: Setup


@dataclass(frozen=True)
class Reconfigure:
    """Load the checkpoint that `saved` writes after step 2 into a run of `setup`, whose lr and weight decay differ,
    and train step 3."""

    name: str
    saved: Setup
    setup: Setup


DENSE, MOE = Setup("qwen3"), Setup("qwen3_moe")
DENSE_CP2, MOE_EP2 = replace(DENSE, cp=2), replace(MOE, ep=2)
OPTIMIZER, FULL = replace(DENSE, offload="optimizer"), replace(DENSE, offload="full")
# Optimizer offload runs AdamW unfused and full offload never clips, so their references do the same
CASES = [
    Offload("qwen3_moe-ep2-ac_offloading", replace(MOE_EP2, ac_offloading=True), MOE_EP2),
    Offload("qwen3-dp2-optimizer", OPTIMIZER, replace(DENSE, fused_adamw=False)),
    Offload("qwen3-dp2-full", FULL, replace(DENSE, max_norm=None)),
    Offload("qwen3-dp2-full-step_after_backward", replace(FULL, step_during_backward=False), FULL),
    Resume("resume-qwen3-dp2-optimizer-ac_offloading-async", replace(OPTIMIZER, ac_offloading=True, async_save=True)),
]
FULL_SUITE_CASES = [
    Offload("qwen3-cp2-ac_offloading", replace(DENSE_CP2, ac_offloading=True), DENSE_CP2),
    Offload("qwen3-cp2-full", replace(DENSE_CP2, offload="full"), replace(DENSE_CP2, max_norm=None)),
    Offload("qwen3_moe-ep2-optimizer", replace(MOE_EP2, offload="optimizer"), replace(MOE_EP2, fused_adamw=False)),
    Offload(
        "qwen3_moe-ep2-muon-optimizer",
        replace(MOE_EP2, offload="optimizer", optimizer="muon"),
        replace(MOE_EP2, optimizer="muon"),
    ),
    Offload("qwen3_moe-ep2-full", replace(MOE_EP2, offload="full"), replace(MOE_EP2, max_norm=None)),
    Offload(
        "qwen3_moe-ep2-full-step_after_backward",
        replace(MOE_EP2, offload="full", step_during_backward=False),
        replace(MOE_EP2, offload="full"),
    ),
    Resume("resume-qwen3-dp2-none", DENSE),
    Resume("resume-qwen3-dp2-none-async", replace(DENSE, async_save=True)),
    Resume("resume-qwen3-dp2-optimizer", OPTIMIZER),
    Resume("resume-qwen3-dp2-muon-optimizer", replace(OPTIMIZER, optimizer="muon")),
    Resume("resume-qwen3-dp2-full", FULL),
    Resume("resume-qwen3-dp2-full-constant_lr", replace(FULL, lr_schedule="constant")),
    Resume("resume-qwen3-dp2-full-async-constant_lr", replace(FULL, lr_schedule="constant", async_save=True)),
    Resume("resume-qwen3-dp2-sign_sgd-full-constant_lr", replace(FULL, optimizer="sign_sgd", lr_schedule="constant")),
    Resume("resume-qwen3_moe-ep2-optimizer-ac_offloading", replace(MOE_EP2, offload="optimizer", ac_offloading=True)),
    # RL freezes the MoE routers by default
    Resume(
        "resume-qwen3_moe-ep2-full-frozen_router-constant_lr",
        replace(MOE_EP2, offload="full", lr_schedule="constant", freeze_moe_router=True),
    ),
    *(
        Load(f"load-qwen3-dp2-{saved}-to-{loaded}", replace(DENSE, offload=saved), replace(DENSE, offload=loaded))
        for saved, loaded in permutations(("none", "optimizer", "full"), 2)
    ),
    Load(
        "load-qwen3_moe-ep2-to-dp2-optimizer", replace(MOE_EP2, offload="optimizer"), replace(MOE, offload="optimizer")
    ),
    Load(
        "load-qwen3_moe-dp2-to-ep2-full-constant_lr",
        replace(MOE, offload="full", lr_schedule="constant"),
        replace(MOE_EP2, offload="full", lr_schedule="constant"),
    ),
]
# Checkpoints of the full suite, loaded on one GPU
ONE_GPU_CASES = [
    Load(f"load-{name}-to-one_gpu", setup, replace(setup, ep=1))
    for name, setup in [
        ("qwen3-dp2-none", DENSE),
        ("qwen3-dp2-optimizer", OPTIMIZER),
        ("qwen3-dp2-full-constant_lr", replace(FULL, lr_schedule="constant")),
        ("qwen3_moe-ep2-optimizer-ac_offloading", replace(MOE_EP2, offload="optimizer", ac_offloading=True)),
    ]
] + [
    Reconfigure(
        "reconfigure-qwen3-dp2-full-constant_lr-to-one_gpu",
        replace(FULL, lr_schedule="constant"),
        replace(FULL, lr_schedule="constant", lr=2e-4, weight_decay=0.02),
    )
]
SUITES = {"tiny": CASES, "full": FULL_SUITE_CASES, "one_gpu": ONE_GPU_CASES}
TIMEOUTS = {"tiny": 5 * 60, "full": 20 * 60, "one_gpu": 5 * 60}


class Env:
    """What the cases of a launch share: the tiny models and rows, the device meshes and the runs already made."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.models_dir = output_dir / "models"
        self.tokenizer = setup_tokenizer(TokenizerConfig(name=TOKENIZER), model_name=TOKENIZER)
        self.layouts: dict[tuple[int, int], ParallelDims] = {}
        self.rows: dict[str, list[dict]] = {}
        self.initial_weights: dict[str, dict] = {}
        self.hashes: dict[Setup, dict] = {}

    @torch.no_grad()
    def prepare(self, models: set[str]) -> None:
        """Saves the tiny models and records their initial FP32 weights and, for an MoE model, the experts its routers
        pick for every row. Every run replays that routing, so runs whose weights differ slightly route alike."""
        for name in sorted(models):
            save_tiny_model(name, self.models_dir)
        dist.barrier()
        rows = load_rows(self.tokenizer, NUM_STEPS * ROWS_PER_STEP)
        for name in sorted(models):
            config = Setup(name).model_config(self.models_dir)
            model = setup_model(config, self.parallel_dims(config))
            self.initial_weights[name] = {key: full_tensor(value) for key, value in get_model_state_dict(model).items()}
            routing = record_routing(model)
            self.rows[name] = []
            for row in rows:
                compute_loss(model, row, cp_rank=0, cp_size=1)
                self.rows[name].append({**row, "routed_experts": routing()})

    def parallel_dims(self, config: ModelConfig) -> ParallelDims:
        # Device meshes are made once per layout, as in a trainer process
        if (config.cp, config.ep) not in self.layouts:
            self.layouts[config.cp, config.ep] = get_parallel_dims(config)
        return self.layouts[config.cp, config.ep]

    def train(self, setup: Setup) -> dict:
        """The flattened state of `setup` after NUM_STEPS steps without a checkpoint. Keeps its hashes."""
        run = Run(setup, self)
        run.train(NUM_STEPS)
        state = checkpoint_state(run)
        run.close()
        self.hashes[setup] = hashes(state)
        return state

    def straight(self, setup: Setup) -> dict[str, str]:
        """The hashed state of `setup` after NUM_STEPS steps without a checkpoint, trained once."""
        setup = replace(setup, async_save=False)
        if setup not in self.hashes:
            self.train(setup)
        return self.hashes[setup]

    def ckpt_dir(self, setup: Setup) -> Path:
        return self.output_dir / "checkpoints" / sha256(setup)[:16]

    def ckpt_manager(self, setup: Setup) -> CheckpointManager:
        model_files = setup_model_files(setup.model_config(self.models_dir), self.tokenizer, processor=None)
        return CheckpointManager(self.ckpt_dir(setup), CheckpointConfig(async_save=setup.async_save), model_files)

    def save(self, run: "Run", until: int) -> tuple[CheckpointManager, dict]:
        """Train `run` up to step `until`, saving after SAVE_STEP. Returns the checkpoint and the hashed state it saved,
        which is also written next to it for later loads."""
        manager = self.ckpt_manager(run.setup)
        saved = run.train(until, manager)
        manager.wait()
        if get_world().is_master:
            (self.ckpt_dir(run.setup) / "saved.json").write_text(json.dumps(saved))
        return manager, saved

    def checkpoint(self, setup: Setup) -> tuple[CheckpointManager, dict]:
        """The checkpoint of `setup` after SAVE_STEP steps and the hashed state it saved, made once per output dir."""
        saved = self.ckpt_dir(setup) / "saved.json"
        if saved.exists():
            return self.ckpt_manager(setup), json.loads(saved.read_text())
        assert get_world().world_size == 2, f"the full suite's launch saved no checkpoint of {setup}"
        run = Run(setup, self)
        checkpoint = self.save(run, SAVE_STEP)
        run.close()
        return checkpoint


class Run:
    """A trainer's model, optimizer, scheduler and progress for one setup, trained with the SFT trainer's loop."""

    def __init__(self, setup: Setup, env: Env, from_checkpoint: bool = False):
        self.setup = setup
        self.config = setup.model_config(env.models_dir)
        self.parallel_dims = env.parallel_dims(self.config)
        self.model = setup_model(self.config, self.parallel_dims, loading_from_checkpoint_later=from_checkpoint)
        if self.parallel_dims.cp_enabled:
            setup_context_parallel(self.model, self.config, self.parallel_dims)
        self.optim_config = OPTIMIZERS[setup.optimizer](
            lr=setup.lr, weight_decay=setup.weight_decay, max_norm=setup.max_norm
        )
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
        self.scheduler = setup_scheduler(
            self.optimizer,
            SCHEDULES[setup.lr_schedule],
            NUM_STEPS,
            self.optim_config.lr,
            checkpoint_step=SAVE_STEP if from_checkpoint else None,
        )
        self.progress = Progress()
        self.rows = env.rows[setup.model]
        self.perf_counter = SimpleNamespace(flop_counter=FlopCounter(self.model))

    def train(self, until: int, manager: CheckpointManager | None = None) -> dict | None:
        """Train up to step `until` as the trainer's loop does. With a `manager`, saves after SAVE_STEP and returns
        the hashed state it saved."""
        saved = None
        while True:
            self.train_step(manager)
            if manager is not None and self.progress.step == SAVE_STEP:
                saved = hashes(checkpoint_state(self))
                manager.save(SAVE_STEP, self.model, [self.optimizer], self.progress)
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
                loss, row_tokens = compute_loss(self.model, row, cp_rank, parallel_dims.cp, row["routed_experts"])
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

    def load(self, manager: CheckpointManager) -> dict[str, str]:
        """Load the checkpoint and go on at the next step, as the trainer resumes. Returns the hashed loaded state."""
        manager.load(SAVE_STEP, self.model, [self.optimizer], self.progress)
        loaded = hashes(checkpoint_state(self))
        self.progress.step += 1
        return loaded

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
    return flatten(AppState(run.model, [run.optimizer], progress=run.progress).state_dict())


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


def differing(state: dict[str, str], reference: dict[str, str], label: str = "") -> list[str]:
    """The entries whose hashes differ, or that only one side has."""
    return [label + key for key in sorted(state.keys() | reference.keys()) if state.get(key) != reference.get(key)]


def training_state(state: dict) -> dict:
    """What a load must restore under any offloading: the FP32 weights of the trained parameters, the optimizer states
    and step counters, the param groups' hyperparameters and the progress counters. Works on states and on their
    hashes."""
    training = {}
    for key, value in state.items():
        if key.startswith("optimizers/state/"):
            training[key] = value
            fqn = key.removeprefix("optimizers/state/").rsplit("/", 1)[0]
            training.setdefault(f"weights/{fqn}", state[f"model/{fqn}"])
        elif key.startswith("progress/"):
            training[key] = value
        elif key.startswith("optimizers/param_groups/") and key.rsplit("/", 1)[1] in HYPERPARAMETERS:
            training[key] = value
    return training


def relative_error(value: torch.Tensor, reference: torch.Tensor) -> float:
    error, norm = (torch.linalg.vector_norm(tensor.double()).item() for tensor in (value - reference, reference))
    return error / norm if norm > 0 else (0.0 if error == 0 else math.inf)


def tolerance_errors(state: dict, reference: dict, initial_weights: dict) -> dict:
    """The worst relative errors of the weight updates and of the optimizer state tensors, which must stay within the
    tolerances. The other entries (step counters, learning rates, progress) must be equal."""
    errors = {"update": {}, "state": {}}
    unequal = sorted(state.keys() ^ reference.keys())
    for key in sorted(state.keys() & reference.keys()):
        value, ref = state[key], reference[key]
        if key.startswith("weights/"):
            initial = initial_weights[key.removeprefix("weights/")]
            errors["update"][key] = relative_error(value - initial, ref - initial)
        elif isinstance(value, torch.Tensor) and value.is_floating_point() and not key.endswith("/step"):
            errors["state"][key] = relative_error(value, ref)
        elif sha256(value) != sha256(ref):
            unequal.append(key)
    worst = {kind: max(kind_errors.items(), key=lambda item: item[1]) for kind, kind_errors in errors.items()}
    unequal += [f"{key}: {kind} error {error:.1e}" for kind, (key, error) in worst.items() if error > TOLERANCES[kind]]
    return {"differing": unequal, "worst": worst}


def run_offload(case: Offload, env: Env) -> dict:
    # Full offload's math differs from that of the other strategies
    if (case.setup.offload == "full") != (case.reference.offload == "full"):
        state, reference = (training_state(env.train(setup)) for setup in (case.setup, case.reference))
        return tolerance_errors(state, reference, env.initial_weights[case.setup.model])
    return {"differing": differing(env.straight(case.setup), env.straight(case.reference))}


def run_resume(case: Resume, env: Env) -> dict:
    straight = env.straight(case.setup)
    run = Run(case.setup, env)
    manager, saved = env.save(run, NUM_STEPS)
    continued = hashes(checkpoint_state(run))
    run.close()
    resumed = Run(case.setup, env, from_checkpoint=True)
    loaded = resumed.load(manager)
    resumed.train(NUM_STEPS)
    final = hashes(checkpoint_state(resumed))
    resumed.close()
    return {
        "differing": differing(loaded, saved, "loaded ")
        + differing(continued, straight, "continued ")
        + differing(final, straight, "resumed ")
    }


def run_load(case: Load, env: Env) -> dict:
    manager, saved = env.checkpoint(case.saved)
    run = Run(case.setup, env, from_checkpoint=True)
    loaded = run.load(manager)
    # The next step must run
    run.train(NUM_STEPS)
    run.close()
    return {"differing": differing(training_state(loaded), training_state(saved))}


def run_reconfigure(case: Reconfigure, env: Env) -> dict:
    manager, saved = env.checkpoint(case.saved)
    run = Run(case.setup, env, from_checkpoint=True)
    warnings = []
    sink = get_logger().add(lambda message: warnings.append(message.record["message"]), level="WARNING")
    loaded = run.load(manager)
    get_logger().remove(sink)
    live = [(group["lr"], group["weight_decay"]) for group in run.optimizer.param_groups]
    run.train(NUM_STEPS)
    run.close()
    changed = {"lr": (case.saved.lr, case.setup.lr), "weight_decay": (case.saved.weight_decay, case.setup.weight_decay)}
    changed["initial_lr"] = changed["lr"]
    unchanged_state = [
        key for key in differing(training_state(loaded), training_state(saved)) if key.rsplit("/", 1)[1] not in changed
    ]
    group_warnings = [warning for warning in warnings if "group 0:" in warning]
    missing = [
        f"group {index}: {key} {old} -> {new}"
        for index in range(len(live))
        for key, (old, new) in changed.items()
        if not any(f"group {index}: {key} {old} -> {new}" in warning for warning in group_warnings)
    ]
    return {
        "differing": unchanged_state
        + [f"live (lr, weight_decay) {value}" for value in live if value != (case.setup.lr, case.setup.weight_decay)]
        + [f"warnings {warnings} lack {line}" for line in missing]
        + ([f"{len(group_warnings)} param group warnings"] if len(group_warnings) != 1 else [])
    }


RUNNERS = {Offload: run_offload, Resume: run_resume, Load: run_load, Reconfigure: run_reconfigure}


def isolated(run, *args) -> dict:
    """The result of `run`, or the error it raised. DCP raises `CheckpointException`, which is no `Exception`."""
    try:
        return run(*args)
    except (Exception, CheckpointException):
        traceback.print_exc()
        return {"error": traceback.format_exc()}


def setups(case) -> list[Setup]:
    return [value for value in vars(case).values() if isinstance(value, Setup)]


def main(output_dir: Path, suite: str) -> None:
    setup_logger("warning")
    setup_torch_distributed(enable_gloo=True)
    env = Env(output_dir)
    env.prepare({setup.model for case in SUITES[suite] for setup in setups(case)})
    # CP setup patches the attention classes for the rest of the process
    cases = sorted(SUITES[suite], key=lambda case: max(setup.cp for setup in setups(case)))
    results = {}
    for index, case in enumerate(cases):
        start = time.perf_counter()
        results[case.name] = isolated(RUNNERS[type(case)], case, env)
        results[case.name]["seconds"] = time.perf_counter() - start
        torch.cuda.empty_cache()
        # Checkpoints take ~1 GB each: keep those that later cases or the one-GPU launch load
        kept = {
            env.ckpt_dir(later.saved)
            for later in cases[index + 1 :] + ONE_GPU_CASES
            if isinstance(later, (Load, Reconfigure))
        }
        if get_world().is_master:
            for directory in (output_dir / "checkpoints").glob("*"):
                if directory not in kept:
                    shutil.rmtree(directory)
    if get_world().is_master:
        print(json.dumps(results, indent=2))
        (output_dir / f"results-{suite}.json").write_text(json.dumps(results, indent=2))


def launch(run_process, output_dir: Path, suite: str) -> dict[str, dict]:
    nproc = 1 if suite == "one_gpu" else 2
    cmd = ["uv", "run", "torchrun", "--standalone", f"--nproc-per-node={nproc}", __file__, output_dir.as_posix(), suite]
    assert run_process(cmd, timeout=TIMEOUTS[suite]).returncode == 0
    return json.loads((output_dir / f"results-{suite}.json").read_text())


@pytest.fixture(scope="module")
def suite_dir(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """The output dir of every launch: the one-GPU launch loads checkpoints of the full suite's launch."""
    if torch.cuda.device_count() < 2:
        pytest.skip("needs 2 GPUs")
    output_dir = tmp_path_factory.mktemp("offload-checkpoint")
    yield output_dir
    shutil.rmtree(output_dir)


@pytest.fixture(scope="module")
def tiny_results(run_process, suite_dir: Path) -> dict[str, dict]:
    return launch(run_process, suite_dir, "tiny")


@pytest.fixture(scope="module")
def full_suite_results(run_process, suite_dir: Path) -> dict[str, dict]:
    return launch(run_process, suite_dir, "full")


@pytest.fixture(scope="module")
def one_gpu_results(run_process, suite_dir: Path, full_suite_results: dict[str, dict]) -> dict[str, dict]:
    return launch(run_process, suite_dir, "one_gpu")


def check(result: dict) -> None:
    assert "error" not in result, result.get("error")
    assert not result["differing"], f"differs from the reference: {result['differing'][:10]}"


@pytest.mark.tiny
@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_offload_and_checkpoint(tiny_results: dict[str, dict], case):
    check(tiny_results[case.name])


@pytest.mark.parametrize("case", FULL_SUITE_CASES, ids=[case.name for case in FULL_SUITE_CASES])
def test_full_suite_offload_and_checkpoint(full_suite_results: dict[str, dict], case):
    check(full_suite_results[case.name])


@pytest.mark.parametrize("case", ONE_GPU_CASES, ids=[case.name for case in ONE_GPU_CASES])
def test_one_gpu_loads_two_gpu_checkpoint(one_gpu_results: dict[str, dict], case):
    check(one_gpu_results[case.name])


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2])
