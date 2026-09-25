import uuid
from pathlib import Path

from pydantic import AliasChoices, Field, model_validator

from prime_rl.configs.monitors import EvalMonitorsConfig, MonitorsConfig
from prime_rl.configs.orchestrator import (
    ConcurrencyConfig,
    DispatcherConfig,
    EvalSourcesConfig,
    InferenceMetricsConfig,
    ScheduledEvalConfig,
)
from prime_rl.configs.shared import ClientConfig, HeartbeatConfig, LogConfig, RunConfig
from prime_rl.configs.trainer import WeightBroadcastConfig
from prime_rl.utils.config import default_output_dir


class ServedEvalConfig(EvalSourcesConfig):
    """Eval sources run against a live inference server: the server's client and the
    adaptive concurrency band."""

    client: ClientConfig = ClientConfig()
    """Client of the inference server evals run against."""

    concurrency: ConcurrencyConfig = ConcurrencyConfig()
    """Adaptive in-flight episode concurrency, sized by the same controller as
    ``[orchestrator.concurrency]``. Set ``min_inflight = max_inflight`` to pin it."""

    dispatcher: DispatcherConfig = DispatcherConfig()
    """Episode admission: rate limit and burst smoothing (``[dispatcher]``)."""

    inference_metrics: InferenceMetricsConfig = InferenceMetricsConfig()
    """The ``/metrics`` poll of the inference engines (``[inference_metrics]``)."""

    heartbeat: HeartbeatConfig | None = None
    """BetterStack heartbeat for the run: one ping per landed episode — the first
    landed episode is the first beat, so the run's boot never shows up as a
    stale-prone silence. When episodes stop landing, the pings stop and the heartbeat
    goes stale on Better Stack after its grace period. Size that period + grace above
    the longest legitimate gap between episodes."""

    @property
    def env_addresses(self) -> dict[tuple[str, str], str | None]:
        """Where each eval source's env server lives, keyed by ``("eval", resolved_name)``.
        Same contract as ``OrchestratorConfig.env_addresses``: an explicit ``serve.address``
        is an externally managed server; None means the launcher spawns the server and the
        eval learns its address from the file it publishes."""
        return {("eval", source.resolved_name): source.serve.address for source in self.source}


PRIME_INFERENCE_URL = "https://api.pinference.ai/api/v1"


class EvalConfig(ServedEvalConfig):
    """``uv run eval``: evaluate the configured sources once against a live inference
    server, then exit. Every source's env server is spawned by the eval process unless
    the source sets ``serve.address``. Defaults to Prime Inference (``PRIME_API_KEY`` or
    ``prime login``) with the concurrency pinned at 128."""

    model: str = Field("deepseek/deepseek-v4.1-flash", validation_alias=AliasChoices("model", "m"))
    """Model id — the ``model`` field of every eval request and the startup model check."""

    client: ClientConfig = ClientConfig(base_url=PRIME_INFERENCE_URL, api_key_var="PRIME_API_KEY")
    """Client of the inference server. Defaults to Prime Inference."""

    concurrency: ConcurrencyConfig = ConcurrencyConfig(min_inflight=128, max_inflight=128)
    """In-flight episodes, pinned at 128 (``-c N`` repins). External APIs expose no vLLM
    ``/metrics`` to adapt to; against a vLLM server set ``min_inflight < max_inflight`` to
    let the band adapt to KV usage like the orchestrator's."""

    num_examples: int = Field(-1, validation_alias=AliasChoices("num_examples", "n"))
    """Default eval examples per environment. ``-1`` uses all. Can be overridden per env."""

    group_size: int = Field(1, ge=1, validation_alias=AliasChoices("group_size", "r"))
    """Default rollouts per example. Can be overridden per env."""

    run: RunConfig = Field(default_factory=RunConfig)
    """Run metadata. ``run.name`` names the run directory under ``output_dir``."""

    output_dir: Path = Field(default_factory=default_output_dir)
    """Directory that groups related runs. Each run writes its artifacts (traces, logs,
    checkpoints) to ``output_dir / run.name``. Defaults to ``$PRL_OUTPUT_DIR`` if set, else ``outputs``."""

    clean: bool = False
    """Delete the run directory (``output_dir / run.name``) before starting. Required to
    overwrite a run directory that contains artifacts from a previous run when not resuming."""

    dry_run: bool = False
    """Resolve and write the config, then exit without evaluating."""

    dashboard: bool = True
    """Start (or reuse) the local dashboard daemon and print its URL."""

    resume: bool = False
    """Continue the interrupted run named by ``run.name`` from its trace stream: the
    landed episodes rejoin the epoch and only the rollouts still owed run. The model,
    sampling and env config must match the interrupted run; ``num_examples`` and
    ``group_size`` may change."""

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


class SFTOnlineEvalConfig(ScheduledEvalConfig, ServedEvalConfig):
    """The ``[eval]`` block of the ``sft`` entrypoint, and the config of the online-eval
    process the launcher spawns from it: interval-driven eval sources against the inference
    server that receives the trainer's weight broadcasts. The process watches the
    broadcasts directory, moves the inference server onto each broadcast, and runs the
    due sources against the updated weights. The launcher fills the run-level fields
    (``model``, ``weight_broadcast``, ``broadcasts_dir``, ``max_steps``, ``resume_step``,
    ``output_dir``, ``log``, ``monitors``) from the resolved SFT config."""

    cancel_on_new_checkpoint: bool = True
    """Cancel unfinished episodes when a newer trainer checkpoint is ready. Disable to
    finish every triggered eval epoch before loading later weights. The trainer can idle
    while it waits for slow evals."""

    model: str | None = None
    """Name the inference server serves the model under. The name stays fixed across
    weight updates (weights are swapped in place), so per-step results are told apart by
    ``eval/{env}/policy_version``."""

    weight_broadcast: WeightBroadcastConfig | None = None
    """Weight transport. None reloads weights from the filesystem broadcasts."""

    broadcasts_dir: Path | None = None
    """Directory to watch for ``step_{n}`` weight broadcasts. Defaults to
    ``<output_dir>/broadcasts``."""

    max_steps: int | None = None
    """Trainer step at which the run ends. The final checkpoint always fires every eval
    env, and the process exits after processing it. If None, runs until terminated."""

    resume_step: int | None = None
    """Trainer step the run resumed from. When set, the startup (base-model) eval is
    skipped; set ``retrigger_on_resume`` to re-fire interval-aligned evals at this step."""

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
