"""The torch-free part of the GeneBench RL -> Hugging Face export (see hf_export.py).

It holds the interface with the watcher that launches the export run, plus the report, log lines and model card.
It only uses the standard library, so it can be tested without the GPU stack.

The request is four variables: GB_EXPORT_RUN_ID, and GB_EXPORT_STEPS, GB_EXPORT_REPOS and GB_EXPORT_ROLES, which
are comma-separated lists given in the same order. Each step gets its own repo. The last checkpoint is always
exported; when the best step is also the last one, there is a single entry with the role ``best+last``.

`prime train -e` only forwards HF_TOKEN and WANDB_API_KEY to a dedicated (hosted) run and refuses any other key,
so the watcher writes the four values into the top-level [env_vars] table of a per-launch copy of the config:

    [env_vars]
    GB_EXPORT_RUN_ID = "<rl run id>"
    GB_EXPORT_STEPS = "120,200"
    GB_EXPORT_REPOS = "PrimeIntellect/Nemotron-3.5-Super-genebench-rl-step120,PrimeIntellect/Nemotron-3.5-Super-genebench-rl-step200"
    GB_EXPORT_ROLES = "best,last"

and launches it with `prime train <rendered.toml> --volume nemotron-genebench-base -e ~/.config/prime-export/hf.env -y`.
The export reads each variable from the process environment and, when it is not there, from the trainer
config's env_vars (see resolve_export_env), so it does not depend on the hosted launcher exporting [env_vars].
"""

import json
import math
import os
import re
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

HF_ORG = "PrimeIntellect"
REPO_PREFIX = "Nemotron-3.5-Super-genebench-rl-step"
RUNS_ROOT = Path("/volume/runs")
ENV_RUN_ID = "GB_EXPORT_RUN_ID"
ENV_STEPS = "GB_EXPORT_STEPS"
ENV_REPOS = "GB_EXPORT_REPOS"
ENV_ROLES = "GB_EXPORT_ROLES"
ENV_KEYS = (ENV_RUN_ID, ENV_STEPS, ENV_REPOS, ENV_ROLES)
ROLES = ("best", "last", "best+last")
DONE_MARKER = "GB_EXPORT_DONE"
ALL_DONE_MARKER = "GB_EXPORT_ALL_DONE"

# Held-out eval metrics of the RL run, as its orchestrator logs them (monitors/file/metrics.jsonl). The naming
# follows the base eval's dump (eval/genebench-baseline/all/agent/metrics/reward/mean, .../solved_some, ...):
# "pass" is the eval reward_key and becomes rewards/pass; the other reward.json keys become metrics/<key>.
# With group_size 6, solved_some is the share of tasks solved at least once, i.e. pass@6.
HELDOUT_PREFIX = "eval/genebench-heldout/all/agent/"
HELDOUT_KEYS = {
    "train_reward": "metrics/train_reward/mean",
    "hurdle_frac": "metrics/hurdle_frac/mean",
    "pass": "rewards/pass/mean",
    "pass_at_6": "solved_some",
}

_RUN_ID_RE = re.compile(r"^[a-z0-9]{8,64}$")
_STEP_RE = re.compile(r"^[1-9][0-9]*$")


class ExportSpecError(ValueError):
    """The GB_EXPORT_* environment does not describe a valid export."""


@dataclass(frozen=True)
class ExportSpec:
    run_id: str
    step: int
    repo_id: str
    role: str

    @property
    def name(self) -> str:
        return f"step_{self.step}"

    def dcp_dir(self, runs_root: Path | None = None) -> Path:
        return (runs_root or RUNS_ROOT) / self.run_id / "checkpoints" / f"step_{self.step}" / "trainer"


def expected_repo_id(step: int) -> str:
    return f"{HF_ORG}/{REPO_PREFIX}{step}"


def resolve_export_env(
    environ: Mapping[str, str], config_env_vars: Mapping[str, str] | None
) -> tuple[dict[str, str], dict[str, str]]:
    """The GB_EXPORT_* values and where each came from ("environment" or "config env_vars"). The process
    environment wins; a key set in both with different values is an error, since that means two requests."""
    config_env_vars = config_env_vars or {}
    values: dict[str, str] = {}
    sources: dict[str, str] = {}
    for key in ENV_KEYS:
        from_env, from_config = environ.get(key), config_env_vars.get(key)
        if from_env is not None and from_config is not None and from_env.strip() != str(from_config).strip():
            raise ExportSpecError(
                f"{key} differs between the environment ({from_env!r}) and env_vars ({from_config!r})"
            )
        if from_env is not None:
            values[key], sources[key] = from_env, "environment"
        elif from_config is not None:
            values[key], sources[key] = str(from_config), "config env_vars"
    return values, sources


def _split(env: Mapping[str, str], key: str) -> list[str]:
    raw = env.get(key)
    if raw is None or not raw.strip():
        raise ExportSpecError(
            f"{key} is not set (the export run needs {ENV_RUN_ID}, {ENV_STEPS}, {ENV_REPOS}, {ENV_ROLES})"
        )
    items = [item.strip() for item in raw.split(",")]
    if any(not item for item in items):
        raise ExportSpecError(f"{key}={raw!r} has an empty item")
    return items


def parse_export_env(env: Mapping[str, str] | None = None) -> list[ExportSpec]:
    """Read and strictly check the GB_EXPORT_* variables; raises ExportSpecError on anything off."""
    env = os.environ if env is None else env
    run_id = (env.get(ENV_RUN_ID) or "").strip()
    if not run_id:
        raise ExportSpecError(f"{ENV_RUN_ID} is not set")
    if not _RUN_ID_RE.match(run_id):
        raise ExportSpecError(f"{ENV_RUN_ID}={run_id!r} is not a run id (lowercase letters and digits)")
    steps_raw, repos, roles = _split(env, ENV_STEPS), _split(env, ENV_REPOS), _split(env, ENV_ROLES)
    if not len(steps_raw) == len(repos) == len(roles):
        raise ExportSpecError(
            f"{ENV_STEPS}, {ENV_REPOS} and {ENV_ROLES} must list the same number of items: "
            f"{len(steps_raw)} steps, {len(repos)} repos, {len(roles)} roles"
        )
    steps: list[int] = []
    for raw in steps_raw:
        if not _STEP_RE.match(raw):
            raise ExportSpecError(f"{ENV_STEPS}: {raw!r} is not a positive integer step")
        steps.append(int(raw))
    if len(set(steps)) != len(steps):
        raise ExportSpecError(f"{ENV_STEPS}: duplicate steps {steps} (one repo per distinct step)")
    for step, repo in zip(steps, repos):
        if repo != expected_repo_id(step):
            raise ExportSpecError(
                f"{ENV_REPOS}: step {step} must go to {expected_repo_id(step)!r}, got {repo!r} "
                "(repos are listed in the same order as the steps)"
            )
    for role in roles:
        if role not in ROLES:
            raise ExportSpecError(f"{ENV_ROLES}: {role!r} is not one of {list(ROLES)}")
    for tag in ("best", "last"):
        holders = [step for step, role in zip(steps, roles) if tag in role.split("+")]
        if len(holders) > 1:
            raise ExportSpecError(f"{ENV_ROLES}: role {tag!r} is given to several steps {holders}")
    best = [step for step, role in zip(steps, roles) if "best" in role.split("+")]
    last = [step for step, role in zip(steps, roles) if "last" in role.split("+")]
    if not last:
        raise ExportSpecError(
            f"{ENV_ROLES}={','.join(roles)!r}: the last checkpoint must be exported (role 'last' or 'best+last')"
        )
    if best and best[0] > last[0]:
        raise ExportSpecError(f"best step {best[0]} is after the last step {last[0]}")
    return [ExportSpec(run_id, step, repo, role) for step, repo, role in zip(steps, repos, roles)]


def storage_problems(dcp_dir: Path, file_ends: Mapping[str, int]) -> list[str]:
    """Why the DCP checkpoint in ``dcp_dir`` is not fully written, given its .metadata's storage map (relative
    file -> largest offset + length it references): a missing or short data file. The small .metadata lands on
    storage long before the large .distcp files, so .metadata alone does not make a checkpoint complete."""
    if not file_ends:
        return [f"{dcp_dir}/.metadata references no data files"]
    problems = []
    for name, end in sorted(file_ends.items()):
        path = dcp_dir / name
        try:
            size = path.stat().st_size
        except OSError:
            problems.append(f"{path} is missing")
            continue
        if size < end:
            problems.append(f"{path} has {size} bytes, .metadata needs {end}")
    return problems


def has_metadata(dcp_dir: Path) -> bool:
    return (dcp_dir / ".metadata").is_file()


def complete_checkpoint_steps(ckpt_root: Path, is_complete: Callable[[Path], bool] = has_metadata) -> list[int]:
    """Steps under ``ckpt_root`` (runs/<id>/checkpoints) whose trainer checkpoint is complete: it has .metadata
    and passes ``is_complete`` (the export checks every data file .metadata references, see storage_problems)."""
    if not ckpt_root.is_dir():
        return []
    steps = []
    for path in ckpt_root.iterdir():
        match = re.fullmatch(r"step_([0-9]+)", path.name)
        if match and has_metadata(path / "trainer") and is_complete(path / "trainer"):
            steps.append(int(match.group(1)))
    return sorted(steps)


def check_checkpoints(
    specs: list[ExportSpec], runs_root: Path | None = None, is_complete: Callable[[Path], bool] = has_metadata
) -> list[int]:
    """Every requested step has a complete trainer checkpoint, the 'last' one is the newest complete step, and a
    'best' that is the newest step is requested as 'best+last' (one repo per step)."""
    run_id = specs[0].run_id
    ckpt_root = (runs_root or RUNS_ROOT) / run_id / "checkpoints"
    complete = complete_checkpoint_steps(ckpt_root, is_complete)
    missing = [str(spec.dcp_dir(runs_root)) for spec in specs if spec.step not in complete]
    if missing:
        raise ExportSpecError(f"no complete trainer checkpoint at {missing}; complete steps: {complete}")
    for spec in specs:
        if "last" in spec.role.split("+") and spec.step != complete[-1]:
            raise ExportSpecError(
                f"step {spec.step} has role {spec.role!r}, but the newest complete checkpoint is step {complete[-1]}"
            )
        if spec.role == "best" and spec.step == complete[-1]:
            raise ExportSpecError(f"step {spec.step} is the newest complete checkpoint: use the role 'best+last'")
    return complete


def _finite(value) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def heldout_evals(metrics_path: Path) -> dict[int, dict[str, float | None]]:
    """step -> held-out eval metrics, from the RL run's metrics.jsonl. Rows of one step are merged per metric,
    later non-null values winning (a restart re-runs the eval of the step it resumes from, and a step may log its
    eval over several rows). Missing file -> {}."""
    evals: dict[int, dict[str, float | None]] = {}
    if not metrics_path.is_file():
        return evals
    with open(metrics_path) as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            step = _finite(row.get("step")) if isinstance(row, dict) else None
            if step is None or not any(key.startswith(HELDOUT_PREFIX) for key in row):
                continue
            merged = evals.setdefault(int(step), {name: None for name in HELDOUT_KEYS})
            for name, key in HELDOUT_KEYS.items():
                value = _finite(row.get(HELDOUT_PREFIX + key))
                if value is not None:
                    merged[name] = value
    return evals


def best_step(evals: Mapping[int, Mapping[str, float | None]], steps: Iterable[int]) -> int | None:
    """Highest held-out train_reward, then higher pass@6, then the later step (among ``steps`` with an eval)."""
    candidates = [s for s in steps if s in evals and evals[s].get("train_reward") is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda s: (evals[s]["train_reward"], evals[s].get("pass_at_6") or 0.0, s))


def change_group(key: str) -> str:
    if ".mixer.experts." in key:
        return "routed_experts"
    if ".mixer.shared_experts." in key:
        return "shared_experts"
    if ".mixer.gate." in key:
        return "router"
    if "latent_proj" in key:
        return "latent_proj"
    if any(f".mixer.{name}" in key for name in ("q_proj", "k_proj", "v_proj", "o_proj")):
        return "attention"
    if ".mixer." in key:
        return "mamba"
    return "layer_norms" if "layers." in key else key


def quantiles(values: list[float]) -> dict:
    values = sorted(values)
    if not values:
        return {"count": 0}

    def pick(q: float) -> float:
        return values[min(len(values) - 1, int(q * len(values)))]

    return {
        "count": len(values),
        "zero": sum(v == 0 for v in values),
        "min": values[0],
        "p50": pick(0.5),
        "p99": pick(0.99),
        "max": values[-1],
    }


def summarize_changes(changes: Mapping[str, float]) -> dict:
    groups: dict[str, list[float]] = {}
    for key, change in changes.items():
        groups.setdefault(change_group(key), []).append(change)
    return {
        "all": quantiles(list(changes.values())),
        "groups": {group: quantiles(values) for group, values in sorted(groups.items())},
        "largest": sorted(changes.items(), key=lambda item: -item[1])[:20],
    }


def check_changes(
    exported: Mapping[str, float], trained: Mapping[str, float], expected: int, max_change: float
) -> list[str]:
    """Problems that mean a broken export rather than training: wrong count, a huge move (wrong mapping), or most
    tensors equal to the base in the trainer's fp32 weights (checkpoint not loaded)."""
    problems = []
    if len(exported) != expected or len(trained) != expected:
        problems.append(f"{len(exported)}/{expected} swapped tensors ({len(trained)} with fp32 stats)")
    if exported:
        largest = max(exported.items(), key=lambda item: item[1])
        if largest[1] > max_change:
            problems.append(f"largest relative change {largest} > {max_change}")
    if trained:
        unchanged = sum(v == 0 for v in trained.values())
        if unchanged > len(trained) // 2:
            problems.append(f"{unchanged}/{len(trained)} tensors equal the base in fp32 (checkpoint not loaded?)")
    return problems


def step_entry(
    spec: ExportSpec,
    *,
    commit: str | None,
    tensor_count: int,
    swapped: int,
    copied: int,
    sha256: Mapping[str, str],
    exported_changes: Mapping[str, float],
    trained_changes: Mapping[str, float],
    heldout: Mapping[str, float | None] | None,
    runs_root: Path | None = None,
) -> dict:
    exported, trained = summarize_changes(exported_changes), summarize_changes(trained_changes)
    return {
        "step": spec.step,
        "role": spec.role,
        "repo": spec.repo_id,
        "commit": commit,
        "source": {"run_id": spec.run_id, "dcp_dir": str(spec.dcp_dir(runs_root))},
        "tensor_count": tensor_count,
        "swapped_tensors": swapped,
        "copied_tensors": copied,
        "shards": len(sha256),
        "sha256": dict(sorted(sha256.items())),
        # Relative L2 change from the base model: of the exported (base-dtype) tensors, and of the trainer's fp32
        # weights. RL moves weights by far less than one bf16 ulp in places, so the two differ.
        "relative_change": {"all": exported["all"], "groups": exported["groups"]},
        "relative_change_fp32": {"all": trained["all"], "groups": trained["groups"], "largest": trained["largest"]},
        "heldout_eval": dict(heldout) if heldout else None,
    }


def done_line(entry: Mapping) -> str:
    """One log line per exported step: the marker, then compact JSON (no per-shard digests)."""
    summary = {
        "step": entry["step"],
        "role": entry["role"],
        "repo": entry["repo"],
        "commit": entry["commit"],
        "tensor_count": entry["tensor_count"],
        "swapped_tensors": entry["swapped_tensors"],
        "shards": entry["shards"],
        "relative_change_p50": entry["relative_change"]["all"].get("p50"),
        "relative_change_max": entry["relative_change"]["all"].get("max"),
        "heldout_eval": entry["heldout_eval"],
    }
    return f"{DONE_MARKER} {json.dumps(summary, separators=(',', ':'), sort_keys=True)}"


TRANSIENT_HTTP_STATUS = frozenset({408, 423, 425, 429, 500, 502, 503, 504})
RETRY_ATTEMPTS = 5
RETRY_BASE_DELAY = 10.0


# Errors that a repeat cannot fix: bad arguments or local files. Everything else without an HTTP status (network
# errors of requests or httpx, an hf_xet upload error, a timeout) is retried, since every retried call is safe to
# repeat and a needless retry only costs a few minutes before the run fails anyway.
FINAL_ERRORS = (
    FileNotFoundError,
    PermissionError,
    IsADirectoryError,
    NotADirectoryError,
    ValueError,
    TypeError,
    KeyError,
    AttributeError,
    AssertionError,
    NotImplementedError,
)


def is_transient(exc: BaseException) -> bool:
    """Worth retrying a Hugging Face call: an HTTP 408/429/5xx-style status, or an error with no HTTP status that
    is not in FINAL_ERRORS. Other HTTP statuses (401, 403, 404, ...) are final."""
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if isinstance(status, int):
        return status in TRANSIENT_HTTP_STATUS
    return not isinstance(exc, FINAL_ERRORS)


def retry(
    fn: Callable,
    what: str,
    *,
    attempts: int | None = None,
    base_delay: float | None = None,
    log: Callable[[str], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
):
    """Call ``fn()`` until it succeeds, retrying transient errors with exponential backoff (10 s, 20 s, 40 s, ...).
    Only for calls that are safe to repeat: every Hugging Face call of the export is (create_repo with
    exist_ok, uploads of content-addressed files, commits of the same files, reads)."""
    attempts = RETRY_ATTEMPTS if attempts is None else attempts
    delay = RETRY_BASE_DELAY if base_delay is None else base_delay
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:
            if attempt == attempts or not is_transient(exc):
                raise
            if log is not None:
                log(f"{what}: {type(exc).__name__}: {str(exc)[:300]}; retry {attempt}/{attempts - 1} in {delay:.0f}s")
            sleep(delay)
            delay *= 2


def write_json(path: Path, obj) -> None:
    """Write ``obj`` as JSON atomically (temp file + rename), so a reader never sees a partial report."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


def _fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def model_card(spec: ExportSpec, heldout: Mapping[str, float | None] | None, base_repo: str, base_revision: str) -> str:
    role_text = {
        "best": "the checkpoint with the best held-out eval of the run",
        "last": "the last checkpoint of the run",
        "best+last": "the last checkpoint of the run, which also has its best held-out eval",
    }[spec.role]
    if heldout:
        eval_text = (
            f"Held-out eval at this step (20 tasks x 6 samples): mean train_reward {_fmt(heldout.get('train_reward'))}, "
            f"pass@1 {_fmt(heldout.get('pass'))}, pass@6 {_fmt(heldout.get('pass_at_6'))}, "
            f"mean hurdle_frac {_fmt(heldout.get('hurdle_frac'))}"
        )
    else:
        eval_text = "Held-out eval at this step: not found in the run's metrics"
    return f"""---
license: other
license_name: openmdw-1.1
license_link: https://openmdw.ai/license/1-1/
library_name: transformers
pipeline_tag: text-generation
base_model: {base_repo}
---

# {spec.repo_id.split("/", 1)[1]}

The language model of [{base_repo}](https://huggingface.co/{base_repo}) (revision `{base_revision[:8]}`)
after reinforcement learning on GeneBench-Pro genomics data-analysis tasks with prime-rl. It is the text-only
`NemotronHForCausalLM` model that was trained, and the one to evaluate.

- Source: hosted RL run `{spec.run_id}`, step {spec.step}: {role_text}.
- {eval_text}. train_reward = 0.5 * pass + 0.5 * hurdle_frac (share of hurdle tests passed).
- Training: GRPO, 128 rollouts per step (16 per task), AdamW at learning rate 1e-6, RLM agent harness
  (bash, edit, ipython), held-out split of 20 of the 121 tasks.
- Same tensor names, dtypes and shapes, config, tokenizer and chat template as the base text model. The
  multi-token-prediction head (`mtp.*`) was not trained and is the base model's.
- The trainer keeps fp32 weights; each tensor is rounded once to the base file's dtype (BF16, except the F32
  `e_score_correction_bias`). During RL, `A_log` and `D` reached the inference engine in fp32, so here they can
  differ from the evaluated policy by one BF16 rounding step.

Training rendered past assistant turns with their reasoning kept. To match it at inference, pass
`chat_template_kwargs={{"truncate_history_thinking": false}}`.

`export_report.json` records the sha256 of every shard and every trained tensor's relative change from the base.
"""
