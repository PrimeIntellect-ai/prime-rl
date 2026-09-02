import uuid
from pathlib import Path

from pydantic import AliasChoices, Field, model_validator

from prime_rl.configs.monitors import EvalMonitorsConfig, MonitorsConfig
from prime_rl.configs.orchestrator import ConcurrencyConfig, EvalSourcesConfig, ScheduledEvalConfig
from prime_rl.configs.shared import ClientConfig, LogConfig, ResumeConfig, RunConfig
from prime_rl.configs.trainer import WeightBroadcastConfig
from prime_rl.utils.config import BaseConfig, default_output_dir


class ServedEvalConfig(EvalSourcesConfig):
    """Eval sources run against a live inference server: the server's client, the
    adaptive concurrency band, and the env-server port range."""

    client: ClientConfig = ClientConfig()
    """Client of the inference server evals run against."""

    concurrency: ConcurrencyConfig = ConcurrencyConfig()
    """Adaptive in-flight episode concurrency, sized by the same controller as
    ``[orchestrator.concurrency]``. Set ``min_inflight = max_inflight`` to pin it."""

    env_server_base_port: int = Field(5000, ge=1, le=65535)
    """First port of the env-server port range: the eval source at position ``i`` is
    served at ``tcp://127.0.0.1:<base + i>``. Sources with an explicit ``serve.address``
    keep it instead, without shifting the other sources' ports."""

    @property
    def env_addresses(self) -> dict[tuple[str, str], str]:
        """Where each eval source's env server lives, keyed by ``("eval", resolved_name)``.
        Same contract as ``OrchestratorConfig.env_addresses``: sources with an explicit
        ``serve.address`` are externally managed; the evals process spawns an env server at
        the derived address for every other source."""
        return {
            ("eval", source.resolved_name): source.serve.address
            or f"tcp://127.0.0.1:{self.env_server_base_port + index}"
            for index, source in enumerate(self.source)
        }


class CheckpointConfig(BaseConfig):
    """Checkpoint the eval progress cursor so an interrupted run can resume."""

    interval: int = Field(1, ge=1)
    """Save after the task cursor advances by N completed groups."""

    keep_last: int | None = Field(1, ge=1)
    """Keep at most this many cursor checkpoints on disk. None keeps all of them."""


class EvalConfig(ServedEvalConfig):
    """``uv run eval``: evaluate the configured sources once against a live inference
    server, then exit. Every source's env server is spawned by the evals process unless
    the source sets ``serve.address``."""

    model: str = Field("Qwen/Qwen3-0.6B", validation_alias=AliasChoices("model", "m"))
    """Name the inference server serves the model under — the ``model`` field of every
    eval request and the startup model check."""

    num_examples: int = Field(-1, validation_alias=AliasChoices("num_examples", "n"))
    """Default eval examples per environment. ``-1`` uses all. Can be overridden per env."""

    group_size: int = Field(1, ge=1, validation_alias=AliasChoices("group_size", "r"))
    """Default rollouts per example. Can be overridden per env."""

    run: RunConfig = Field(default_factory=RunConfig)
    """Run metadata. ``run.name`` names the run directory under ``output_dir``."""

    output_dir: Path = Field(default_factory=default_output_dir, validation_alias=AliasChoices("output_dir", "o"))
    """Directory that groups related runs. Each run writes its artifacts (traces, logs,
    checkpoints) to ``output_dir / run.name``. Defaults to ``$PRL_OUTPUT_DIR`` if set, else ``outputs``."""

    clean: bool = False
    """Delete the run directory (``output_dir / run.name``) before starting. Required to
    overwrite a run directory that contains artifacts from a previous run when not resuming."""

    dry_run: bool = False
    """Resolve and write the config, then exit without evaluating."""

    dashboard: bool = True
    """Start (or reuse) the local dashboard daemon and print its URL."""

    ckpt: CheckpointConfig | None = CheckpointConfig()
    """Checkpoint the task cursor as groups complete. Disable with ``--no-ckpt``."""

    resume: ResumeConfig | None = None
    """Resume from a cursor checkpoint (point at it with the previous run's ``run.name``).
    A bare ``--resume`` loads the latest checkpoint."""

    log: LogConfig = LogConfig()

    monitors: EvalMonitorsConfig = EvalMonitorsConfig()
    """Metric monitors (``monitors.wandb``, ``monitors.file``, ``monitors.prime``)."""

    @property
    def run_dir(self) -> Path:
        assert self.run.dir is not None  # resolved at construction
        return self.output_dir / self.run.dir

    @model_validator(mode="after")
    def auto_setup_run_identity(self):
        """Auto-generate the run name (``<envs>--<model>--<short-id>``) when unset and
        default the run directory, W&B run name and platform evaluation name to it."""
        if self.run.name is None:
            envs = "+".join(dict.fromkeys(source.resolved_name for source in self.source))
            model = self.model.split("/")[-1]
            self.run.name = f"{envs}--{model}--{uuid.uuid4().hex[:8]}".lower()
        if self.run.dir is None:
            self.run.dir = self.run.name
        if self.monitors.wandb is not None and self.monitors.wandb.name is None:
            self.monitors.wandb.name = self.run.name
        if self.monitors.prime is not None and self.monitors.prime.name is None:
            self.monitors.prime.name = self.run.name
        return self


class SFTEvalConfig(ScheduledEvalConfig, ServedEvalConfig):
    """The ``[eval]`` block of the ``sft`` entrypoint: interval-driven eval sources
    against the inference server that receives the trainer's weight broadcasts."""

    cancel_on_new_checkpoint: bool = True
    """Cancel unfinished episodes when a newer trainer checkpoint is ready. Disable to
    finish every triggered eval epoch before loading later weights. The trainer can idle
    while it waits for slow evals."""


class OnlineEvalConfig(SFTEvalConfig):
    """``online-eval``: watch a broadcasts directory for the trainer's weight broadcasts,
    move the inference server onto each of them, and run the due eval sources against the
    updated weights. The ``sft`` launcher writes this config from its ``[eval]`` block;
    with ``weight_broadcast.type = "filesystem"`` it also works standalone against any
    trainer that writes ``broadcasts/step_{n}`` directories with the broadcast markers."""

    model: str = "Qwen/Qwen3-0.6B"
    """Name the inference server serves the model under. The name stays fixed across
    weight updates (weights are swapped in place), so per-step results are told apart by
    ``eval/{env}/policy_version``."""

    broadcasts_dir: Path | None = None
    """Directory to watch for ``step_{n}`` weight broadcasts. Defaults to
    ``<output_dir>/broadcasts``."""

    max_steps: int | None = None
    """Trainer step at which the run ends. The final checkpoint always fires every eval
    env, and the process exits after processing it. If None, runs until terminated."""

    resume_step: int | None = None
    """Trainer step the run resumed from. When set, the startup (base-model) eval is
    skipped; set ``retrigger_on_resume`` to re-fire interval-aligned evals at this step."""

    weight_broadcast: WeightBroadcastConfig | None = None
    """Weight transport. The ``sft`` launcher fills this from its resolved trainer
    transport. None reloads weights from the filesystem broadcasts."""

    output_dir: Path = Field(default_factory=default_output_dir)
    """The run directory, shared with the trainer. Defaults to ``$PRL_OUTPUT_DIR`` if set, else ``outputs``."""

    log: LogConfig = LogConfig()

    monitors: MonitorsConfig = MonitorsConfig()
    """Metric monitors (``monitors.wandb``, ``monitors.file``)."""

    @model_validator(mode="after")
    def auto_setup_broadcasts_dir(self):
        if self.broadcasts_dir is None:
            self.broadcasts_dir = self.output_dir / "broadcasts"
        return self
