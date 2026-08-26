from pathlib import Path

from pydantic import Field, model_validator

from prime_rl.configs.monitors import MonitorsConfig
from prime_rl.configs.orchestrator import ConcurrencyConfig, EvalConfig
from prime_rl.configs.shared import ClientConfig, LogConfig
from prime_rl.configs.trainer import WeightBroadcastConfig
from prime_rl.utils.config import BaseConfig, default_output_dir


class EvalsEvalConfig(EvalConfig):
    """Evals against a live inference server. Extends the orchestrator ``EvalConfig``
    (sources, sampling, intervals) with the client of the inference deployment and
    evals-side knobs."""

    client: ClientConfig = ClientConfig()
    """Client of the inference server evals run against. Auto-wired from the
    ``[inference]`` block when the launcher manages the server."""

    env_server_base_port: int = Field(5000, ge=1, le=65535)
    """First port of the env-server port range: the eval source at position ``i`` is
    served at ``tcp://127.0.0.1:<base + i>``. Sources with an explicit ``serve.address``
    keep it instead, without shifting the other sources' ports."""

    concurrency: ConcurrencyConfig = ConcurrencyConfig()
    """Adaptive in-flight episode concurrency (``[eval.concurrency]``), sized by the
    same controller as the orchestrator's ``[orchestrator.concurrency]``."""

    cancel_on_new_checkpoint: bool = True
    """For online evals, cancel unfinished episodes when a newer trainer checkpoint is ready.
    Disable to finish every triggered eval epoch before loading later weights."""

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


class OnlineConfig(BaseConfig):
    """Broadcast-driven online evals: watch a broadcasts directory for new weight
    broadcasts and evaluate each eligible one. Without this block the evals process runs
    every eval source once against the weights the inference server currently serves,
    then exits."""

    broadcasts_dir: Path | None = None
    """Directory to watch for ``step_{n}`` weight broadcasts. Defaults to
    ``<output_dir>/broadcasts``."""

    max_steps: int | None = None
    """Trainer step at which the run ends. The final checkpoint always fires every
    eval env, and the evals process exits after processing it. If None, the evals process
    runs until terminated."""

    resume_step: int | None = None
    """Trainer step the run resumed from. When set, the startup (base-model) eval is
    skipped; set ``eval.retrigger_on_resume`` to re-fire interval-aligned evals at
    this step."""


class EvalsConfig(BaseConfig):
    """``uv run evals``: run the configured evals against a live inference server.
    Standalone (no ``[online]``), one epoch of every eval source runs against the
    served weights and the evals process exits. With ``[online]``, the evals process watches a
    broadcasts directory for new weight broadcasts, updates the inference server through
    the configured transport, and runs the configured evals against the updated weights.
    The ``sft`` launcher writes this config; filesystem mode also works standalone against
    any trainer that writes ``broadcasts/step_{n}`` directories with ``STABLE`` markers."""

    model: str = "Qwen/Qwen3-0.6B"
    """Name the inference server serves the model under — the ``model`` field of every
    eval request and the startup model check. Auto-filled from ``model.name`` by the
    ``sft`` launcher; the name stays fixed across weight updates (weights are
    swapped in place), so per-step results are told apart by ``eval/{env}/policy_version``."""

    eval: EvalsEvalConfig
    """Eval sources, sampling, intervals, concurrency, and the inference client."""

    online: OnlineConfig | None = None
    """Checkpoint watching (``[online]``). None runs the evals once and exits."""

    weight_broadcast: WeightBroadcastConfig | None = None
    """Weight transport for online evals. The ``sft`` launcher fills this from its
    resolved trainer transport. None uses filesystem reloads."""

    output_dir: Path = Field(default_factory=default_output_dir)
    """Directory to write outputs to — rollout traces and logs are written as
    subdirectories. Shared with the trainer for online evals. Defaults to ``$PRL_OUTPUT_DIR`` if set, else ``outputs``."""

    log: LogConfig = LogConfig()

    monitors: MonitorsConfig = MonitorsConfig()
    """Metric monitors (``monitors.wandb``, ``monitors.file``)."""

    @model_validator(mode="after")
    def auto_setup_broadcasts_dir(self):
        if self.online is not None and self.online.broadcasts_dir is None:
            self.online.broadcasts_dir = self.output_dir / "broadcasts"
        return self

    @model_validator(mode="after")
    def validate_skip_first_step_is_online_only(self):
        """``skip_first_step`` gates the base-model eval between checkpoints; a
        standalone run has only that one epoch, so skipping it would exit
        successfully without evaluating anything."""
        if self.online is None and self.eval.skip_first_step:
            raise ValueError(
                "eval.skip_first_step only applies to online evals - a standalone run "
                "would skip its only eval epoch. Remove it or add an [online] block."
            )
        return self
