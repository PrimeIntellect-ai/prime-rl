"""GeneBench round-2 run settings: keep-3 checkpoint retention and the end-of-run export.

The settings are ``GB_*`` string variables. Each component reads them in-process from its own config's
``env_vars`` table (``[trainer.env_vars]`` / ``[orchestrator.env_vars]``). A variable missing there is also
looked up in the process environment, for local runs; that is not a second delivery path on hosted runs (the
backend does not merge the top-level ``[env_vars]`` into the rendered component configs, and round 1's
NCCL_DEBUG from it never reached the trainer), so a dropped component table turns every feature off: check
the "GB settings:" startup line. No typed config fields are added, so the rendered configs keep their schema.

With no ``GB_*`` variable set, every feature is off and stock behavior is unchanged.

This module only uses the standard library, so it can be tested without the GPU stack.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, fields

PREFIX = "GB_"

KEEP3 = "GB_KEEP3"
EVAL_ENV = "GB_EVAL_ENV"
KEEP_METRICS = "GB_KEEP_METRICS"
MAX_EVAL_ERROR = "GB_MAX_EVAL_ERROR"
FINAL_EXPORT = "GB_FINAL_EXPORT"
EXPORT_REPO_PREFIX = "GB_EXPORT_REPO_PREFIX"
FINAL_EVAL_TIMEOUT_S = "GB_FINAL_EVAL_TIMEOUT_S"
WAIT_FOR_EXPORT_S = "GB_WAIT_FOR_EXPORT_S"
EXPECT_FORK_STEP = "GB_EXPECT_FORK_STEP"

# Variable -> GBSettings field. A GB_* key in a config's env_vars that is not listed here is rejected
# (a typo would otherwise silently turn a feature off); one only in the process environment is a warning.
KNOWN_KEYS: dict[str, str] = {
    KEEP3: "keep3",
    EVAL_ENV: "eval_env",
    KEEP_METRICS: "keep_metrics",
    MAX_EVAL_ERROR: "max_eval_error",
    FINAL_EXPORT: "final_export",
    EXPORT_REPO_PREFIX: "export_repo_prefix",
    FINAL_EVAL_TIMEOUT_S: "final_eval_timeout_s",
    WAIT_FOR_EXPORT_S: "wait_for_export_s",
    EXPECT_FORK_STEP: "expect_fork_step",
}

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off", ""}
_METRIC_NAME_RE = re.compile(r"^[A-Za-z0-9_]+$")
_REPO_PREFIX_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")


class GBSettingsError(ValueError):
    """The GB_* variables are invalid, or do not fit the component's config."""


@dataclass(frozen=True)
class GBSettings:
    keep3: bool = False
    """Keep only the last checkpoint and the best one per held-out metric (plus checkpoints whose eval is pending)."""

    eval_env: str = "genebench-heldout"
    """The online eval env whose results rank the checkpoints."""

    keep_metrics: tuple[str, ...] = ("train_reward", "hurdle_frac")
    """Metrics that each hold a best checkpoint: a grader metric name (``eval/<env>/all/agent/metrics/<name>/mean``)
    or a full ``eval/...`` metric key. Higher is better."""

    max_eval_error: float = 0.05
    """A step whose eval error rate is above this cannot be best (its metrics cover only the surviving episodes)."""

    final_export: bool = False
    """At the end of training, export the final kept checkpoints to Hugging Face from the trainer."""

    export_repo_prefix: str = "PrimeIntellect/Nemotron-3.5-Super-genebench-rl"
    """Repo id prefix of the exports; a step's repo is ``<prefix>-step<N>``."""

    final_eval_timeout_s: float = 43200.0
    """How long the trainer waits after its final save for the final and pending evals before exporting."""

    wait_for_export_s: float = 21600.0
    """How long the orchestrator stays up at teardown waiting for the trainer's export to finish."""

    expect_fork_step: int | None = None
    """This run continues another run from this step: refuse to start from scratch or from an earlier step (e.g.
    when the backend dropped ``[resume] dir``, which would train the base model under global step numbers)."""

    sources: dict[str, str] = field(default_factory=dict, compare=False)
    """Where each set variable came from (``config.env_vars`` or ``os.environ``)."""

    @property
    def active(self) -> bool:
        return bool(self.sources)

    @property
    def records_enabled(self) -> bool:
        """The orchestrator writes eval records whenever a feature that ranks checkpoints is on."""
        return self.keep3 or self.final_export

    def metric_key(self, metric: str) -> str:
        if metric.startswith("eval/"):
            return metric
        return f"eval/{self.eval_env}/all/agent/metrics/{metric}/mean"

    @property
    def metric_keys(self) -> dict[str, str]:
        """``{metric name: full metric key}``, in ``keep_metrics`` order."""
        return {metric_name(m): self.metric_key(m) for m in self.keep_metrics}

    def as_dict(self) -> dict:
        out = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "sources"}
        out["keep_metrics"] = list(self.keep_metrics)
        return out

    def describe(self) -> str:
        if not self.active:
            return "GB settings: none set (keep-3 retention and final export off)"
        parts = [f"{k}={v!r}" for k, v in self.as_dict().items()]
        src = ", ".join(f"{k} from {v}" for k, v in sorted(self.sources.items()))
        return f"GB settings: {', '.join(parts)} ({src})"


def metric_name(metric: str) -> str:
    """Short name of a keep metric, used in role names (``best_<name>``)."""
    if not metric.startswith("eval/"):
        return metric
    parts = [p for p in metric.split("/") if p]
    if parts[-1] == "mean" and len(parts) > 1:
        return parts[-2]
    return parts[-1]


def _parse_bool(key: str, value: str) -> bool:
    v = value.strip().lower()
    if v in _TRUE:
        return True
    if v in _FALSE:
        return False
    raise GBSettingsError(f"{key}={value!r} is not a boolean (use 1/0, true/false)")


def _parse_float(key: str, value: str, *, lo: float, hi: float | None = None, lo_open: bool = False) -> float:
    try:
        v = float(value)
    except ValueError:
        raise GBSettingsError(f"{key}={value!r} is not a number") from None
    if v != v or (lo_open and v <= lo) or (not lo_open and v < lo) or (hi is not None and v > hi):
        bound = (
            f"({lo}, {hi if hi is not None else 'inf'})" if lo_open else f"[{lo}, {hi if hi is not None else 'inf'})"
        )
        raise GBSettingsError(f"{key}={value!r} is out of range {bound}")
    return v


def load_gb_settings(
    env_vars: Mapping[str, str] | None, environ: Mapping[str, str] | None = None
) -> tuple[GBSettings, list[str]]:
    """Settings from a component config's ``env_vars`` (which wins) and the process environment.
    Returns the settings and warnings to log. Raises GBSettingsError on an invalid value or an unknown
    GB_* key in ``env_vars``."""
    env_vars = dict(env_vars or {})
    environ = os.environ if environ is None else environ
    warnings: list[str] = []

    unknown_cfg = sorted(k for k in env_vars if k.startswith(PREFIX) and k not in KNOWN_KEYS)
    if unknown_cfg:
        raise GBSettingsError(f"unknown GB_* key(s) in env_vars: {unknown_cfg} (known: {sorted(KNOWN_KEYS)})")
    unknown_env = sorted(k for k in environ if k.startswith(PREFIX) and k not in KNOWN_KEYS)
    if unknown_env:
        warnings.append(f"ignoring unknown GB_* variable(s) in the process environment: {unknown_env}")

    raw: dict[str, str] = {}
    sources: dict[str, str] = {}
    for key in KNOWN_KEYS:
        if key in env_vars:
            raw[key], sources[key] = env_vars[key], "config.env_vars"
        elif key in environ:
            raw[key], sources[key] = environ[key], "os.environ"

    values: dict = {}
    if KEEP3 in raw:
        values["keep3"] = _parse_bool(KEEP3, raw[KEEP3])
    if FINAL_EXPORT in raw:
        values["final_export"] = _parse_bool(FINAL_EXPORT, raw[FINAL_EXPORT])
    if EVAL_ENV in raw:
        env = raw[EVAL_ENV].strip()
        if not env or "/" in env:
            raise GBSettingsError(f"{EVAL_ENV}={raw[EVAL_ENV]!r} must be an eval env name")
        values["eval_env"] = env
    if KEEP_METRICS in raw:
        metrics = tuple(m.strip() for m in raw[KEEP_METRICS].split(",") if m.strip())
        if not metrics:
            raise GBSettingsError(f"{KEEP_METRICS} is empty")
        for m in metrics:
            if not (m.startswith("eval/") or _METRIC_NAME_RE.match(m)):
                raise GBSettingsError(f"{KEEP_METRICS}: {m!r} is neither a metric name nor an eval/... key")
        names = [metric_name(m) for m in metrics]
        if len(set(names)) != len(names):
            raise GBSettingsError(f"{KEEP_METRICS}={raw[KEEP_METRICS]!r} names a metric twice")
        values["keep_metrics"] = metrics
    if MAX_EVAL_ERROR in raw:
        values["max_eval_error"] = _parse_float(MAX_EVAL_ERROR, raw[MAX_EVAL_ERROR], lo=0.0, hi=1.0)
    if EXPORT_REPO_PREFIX in raw:
        prefix = raw[EXPORT_REPO_PREFIX].strip()
        if not _REPO_PREFIX_RE.match(prefix) or re.search(r"-step\d*$", prefix):
            raise GBSettingsError(
                f"{EXPORT_REPO_PREFIX}={raw[EXPORT_REPO_PREFIX]!r} must be '<org>/<name>' without the -step<N> suffix"
            )
        values["export_repo_prefix"] = prefix
    if FINAL_EVAL_TIMEOUT_S in raw:
        values["final_eval_timeout_s"] = _parse_float(
            FINAL_EVAL_TIMEOUT_S, raw[FINAL_EVAL_TIMEOUT_S], lo=0.0, lo_open=True
        )
    if WAIT_FOR_EXPORT_S in raw:
        values["wait_for_export_s"] = _parse_float(WAIT_FOR_EXPORT_S, raw[WAIT_FOR_EXPORT_S], lo=0.0, lo_open=True)
    if EXPECT_FORK_STEP in raw:
        try:
            values["expect_fork_step"] = int(raw[EXPECT_FORK_STEP])
        except ValueError:
            raise GBSettingsError(f"{EXPECT_FORK_STEP}={raw[EXPECT_FORK_STEP]!r} is not a step number") from None
        if values["expect_fork_step"] < 1:
            raise GBSettingsError(f"{EXPECT_FORK_STEP}={raw[EXPECT_FORK_STEP]!r} must be a positive step")

    settings = GBSettings(**values, sources=sources)
    if settings.active and not settings.records_enabled and set(sources) != {EXPECT_FORK_STEP}:
        warnings.append(f"GB_* variables are set but neither {KEEP3} nor {FINAL_EXPORT} is on; they have no effect")
    return settings, warnings


def check_resume_step(settings: GBSettings, resume_step: int | None) -> None:
    """GB_EXPECT_FORK_STEP: fail at startup unless this attempt resumes from the forked step or a later one."""
    expected = settings.expect_fork_step
    if expected is None or (resume_step is not None and resume_step >= expected):
        return
    got = "from scratch" if resume_step is None else f"from step {resume_step}"
    raise GBSettingsError(
        f"{EXPECT_FORK_STEP}={expected}, but this attempt would start {got}: is [resume] dir missing from the rendered "
        "config, or does the fork's checkpoint not resolve?"
    )


def validate_trainer_settings(
    settings: GBSettings,
    *,
    ckpt_enabled: bool,
    keep_last: int | None,
    keep_interval: int | None,
    has_hf_token: bool,
) -> None:
    """Fail at startup on combinations the trainer cannot honor."""
    if settings.final_export and not settings.keep3:
        raise GBSettingsError(f"{FINAL_EXPORT}=1 exports the keep-3 checkpoints and needs {KEEP3}=1")
    if not settings.keep3:
        return
    if not ckpt_enabled:
        raise GBSettingsError(f"{KEEP3}=1 needs checkpointing on ([ckpt])")
    if keep_last is not None or keep_interval is not None:
        raise GBSettingsError(
            f"{KEEP3}=1 replaces the stock retention: unset [ckpt] keep_last ({keep_last}) and "
            f"keep_interval ({keep_interval}), or the stock cleanup deletes what keep-3 keeps"
        )
    if settings.final_export and not has_hf_token:
        raise GBSettingsError(
            f"{FINAL_EXPORT}=1 but HF_TOKEN is not set in the trainer's environment; the export would fail only "
            "after training. Pass it with `prime train ... -e <file with HF_TOKEN>`"
        )


def validate_orchestrator_settings(
    settings: GBSettings,
    *,
    eval_intervals: Mapping[str, int] | None,
    ckpt_interval: int | None,
    retrigger_on_resume: bool,
) -> list[str]:
    """Fail at startup on combinations the orchestrator cannot honor; returns warnings to log."""
    warnings: list[str] = []
    if not settings.records_enabled:
        return warnings
    if not eval_intervals or settings.eval_env not in eval_intervals:
        raise GBSettingsError(
            f"{EVAL_ENV}={settings.eval_env!r} is not an eval env of this run "
            f"(eval envs: {sorted(eval_intervals or {})}); keep-3 ranks checkpoints by its results"
        )
    if not ckpt_interval:
        raise GBSettingsError("keep-3 eval records need orchestrator checkpointing on ([ckpt] interval)")
    eval_interval = eval_intervals[settings.eval_env]
    if ckpt_interval % eval_interval != 0:
        warnings.append(
            f"[ckpt] interval {ckpt_interval} is not a multiple of the {settings.eval_env} eval interval "
            f"{eval_interval}: checkpoints without an eval can only be kept as the last one"
        )
    if not retrigger_on_resume:
        warnings.append(
            "eval.retrigger_on_resume is off: an eval interrupted by a restart is not re-run, so its "
            "checkpoint can only be kept as the last one"
        )
    return warnings
