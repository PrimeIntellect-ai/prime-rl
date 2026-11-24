import re
from dataclasses import dataclass, field, replace
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist

from prime_rl.configs.trainer import CheckpointConfig, LinearSchedulerConfig, SchedulerConfig
from prime_rl.trainer.ckpt import CheckpointManager, ModelFiles, Progress
from prime_rl.trainer.scheduler import setup_scheduler
from prime_rl.trainer.utils import get_dcp_group
from prime_rl.utils.logger import get_logger
from prime_rl.utils.pathing import get_ckpt_dir, get_step_path

RESUME_STEP = 6


def test_keep_last_ignores_a_step_cut_off_mid_write(tmp_path):
    ckpt_dir = get_ckpt_dir(tmp_path)
    # Steps 1 and 2 finished writing; the run crashed while writing step 3 and resumes from step 2
    for step in (1, 2, 3):
        (get_step_path(ckpt_dir, step) / "trainer").mkdir(parents=True)
    for step in (1, 2):
        (get_step_path(ckpt_dir, step) / "trainer" / ".metadata").touch()

    manager = CheckpointManager(tmp_path, CheckpointConfig(keep_last=1), Mock(spec=ModelFiles))
    manager.maybe_clean()

    assert manager.ckpt_steps == [2]
    assert not get_step_path(ckpt_dir, 1).exists()
    assert get_step_path(ckpt_dir, 2).exists()


@dataclass(frozen=True)
class Run:
    """A run's optimizer and schedule config."""

    lr: float = 1e-2
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    scheduler: SchedulerConfig = field(default_factory=lambda: LinearSchedulerConfig(warmup_steps=4, decay_steps=4))
    max_steps: int | None = 16


BASE = Run()
# The RL trainer runs without max_steps by default, so its schedule has no decay
RL_WARMUP = replace(BASE, scheduler=LinearSchedulerConfig(warmup_steps=8, decay_steps=0), max_steps=None)


@pytest.fixture
def process_group(tmp_path):
    dist.init_process_group("gloo", init_method=f"file://{tmp_path}/process_group", rank=0, world_size=1)
    yield
    get_dcp_group.cache_clear()
    dist.destroy_process_group()


def train(run: Run, checkpoint_step: int | None, manager: CheckpointManager | None = None) -> dict:
    """Train ``run`` from the step after ``checkpoint_step`` to its last step (step 10 without max_steps), set up as
    the trainers do. With a ``manager``, it loads the checkpoint of ``checkpoint_step`` and saves one at RESUME_STEP.
    Returns the param groups after the setup, every group's lr at each step and the warnings of the setup."""
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 2)
    groups = [{"params": [model.weight]}, {"params": [model.bias], "lr": 2 * run.lr}]
    optimizer = torch.optim.AdamW(groups, lr=run.lr, weight_decay=run.weight_decay, betas=run.betas, eps=run.eps)
    scheduler = setup_scheduler(optimizer, run.scheduler, run.max_steps, run.lr, checkpoint_step)
    progress = Progress(step=checkpoint_step or 0)
    warnings = []
    sink = get_logger().add(lambda message: warnings.append(message.record["message"]), level="WARNING")
    if manager is not None and checkpoint_step is not None:
        manager.load(checkpoint_step, model, [optimizer], progress)
    get_logger().remove(sink)
    setup = [{key: value for key, value in group.items() if key != "params"} for group in optimizer.param_groups]
    lrs = []
    for step in range(progress.step + 1, (run.max_steps or 10) + 1):
        lrs.append([group["lr"] for group in optimizer.param_groups])
        model(torch.ones(2, 4)).square().sum().backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        if manager is not None and step == RESUME_STEP:
            manager.save(step, model, [optimizer], Progress(step=step))
    return {"groups": setup, "lrs": lrs, "changed": set(re.findall(r"^group \d+: (\S+) ", "\n".join(warnings), re.M))}


@pytest.mark.parametrize(
    "saved, resumed, skip_optimizer, changed",
    [
        (BASE, replace(BASE, lr=2e-2), False, {"lr", "initial_lr"}),
        (BASE, replace(BASE, lr=2e-2), True, set()),
        (BASE, replace(BASE, weight_decay=0.2, betas=(0.8, 0.9), eps=1e-6), False, {"weight_decay", "betas", "eps"}),
        (BASE, replace(BASE, scheduler=LinearSchedulerConfig(warmup_steps=4, decay_steps=12)), False, {"lr"}),
        (RL_WARMUP, replace(RL_WARMUP, lr=2e-2), False, {"lr", "initial_lr"}),
    ],
    ids=["lr", "lr-skip_optimizer", "weight_decay-betas-eps", "decay_steps", "rl-lr"],
)
def test_resume_trains_with_the_config(tmp_path, process_group, saved, resumed, skip_optimizer, changed):
    """A resume trains with the hyperparameters and the schedule of its config, as a run with that config set up at
    the resume step does, and warns of the changed param group values when it loads the optimizer state."""
    model_files = Mock(spec=ModelFiles, config=None)
    train(saved, None, CheckpointManager(tmp_path, CheckpointConfig(), model_files))
    manager = CheckpointManager(tmp_path, CheckpointConfig(skip_optimizer=skip_optimizer), model_files)
    result = train(resumed, RESUME_STEP, manager)
    expected = train(resumed, RESUME_STEP)
    assert result["groups"] == expected["groups"]
    assert result["lrs"] == expected["lrs"]
    assert result["changed"] == changed
