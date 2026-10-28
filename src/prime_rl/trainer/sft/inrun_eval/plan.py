"""The plan each node's eval process follows, built by the trainer at the hand-off (``handoff``) from the config and
the export results, and written as JSON next to the node's work dir. Plain data and pure functions."""

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Checkpoint:
    name: str
    model_dir: str
    step: int | None = None
    val_loss: float | None = None
    repo_id: str | None = None
    """Where the export was pushed and verified (the fallback source of a node's copy), else None."""


@dataclass
class Plan:
    run_dir: str
    eval_dir: str
    work_dir: str
    node: int
    num_nodes: int
    master_addr: str
    coord_port: int
    started: float
    deadline: float | None
    """The optional safety cap (``budget_hours``) as a wall-clock time; None: no cap."""
    checkpoints: list[Checkpoint]
    """To evaluate, in order."""
    rounds: list[dict[str, list[int]]]
    """Per round: checkpoint name -> the nodes serving it (every node serves at most one checkpoint per round)."""
    aliases: dict[str, str] = field(default_factory=dict)
    """Checkpoint -> the evaluated checkpoint it is identical to ('best' at the final step is 'last')."""
    skipped: dict[str, str] = field(default_factory=dict)
    """Checkpoints not evaluated, and why."""
    evaluated: dict[str, dict] = field(default_factory=dict)
    """Checkpoints a previous attempt of the run already evaluated completely (a restart): name -> its result."""
    exports: dict[str, dict] = field(default_factory=dict)
    """The export results (``hf_export.ExportResult``), for the cleanup."""
    delete_pushed_exports: bool = False
    config: dict[str, Any] = field(default_factory=dict)
    """``[inrun_eval]``, resolved."""
    gpus_per_node: int = 8
    prime_team_id: str | None = None
    """``PRIME_TEAM_ID`` as the trainer saw it (its environment, else the run config's ``[env_vars]``)."""
    credentials: dict[str, Any] = field(default_factory=dict)
    """What the hand-off found: ``prime_api_key_present`` (never the key), ``prime_team_id_source``."""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> "Plan":
        data = json.loads(text)
        data["checkpoints"] = [Checkpoint(**c) for c in data["checkpoints"]]
        return cls(**data)


def checkpoints_to_eval(
    names: list[str], exports: dict[str, dict], evaluate_aliases: bool = False
) -> tuple[list[Checkpoint], dict[str, str], dict[str, str]]:
    """The exports to evaluate (in ``names`` order), aliases (an export that is a copy of another evaluated one) and
    the skipped ones with the reason. ``evaluate_aliases`` evaluates a copy in its own right (the smoke test)."""
    out: list[Checkpoint] = []
    aliases: dict[str, str] = {}
    skipped: dict[str, str] = {}
    for name in names:
        result = exports.get(name)
        if result is None:
            skipped[name] = "not exported"
            continue
        if not result.get("ok") or not result.get("dir"):
            skipped[name] = "the export failed its checks"
            continue
        same = result.get("same_as")
        if not evaluate_aliases and same is not None and same in names and (exports.get(same) or {}).get("ok"):
            aliases[name] = same
            continue
        out.append(
            Checkpoint(
                name=name,
                model_dir=result["dir"],
                step=result.get("step"),
                val_loss=result.get("val_loss"),
                repo_id=result.get("repo_id") if result.get("verified") else None,
            )
        )
    # An alias's target is evaluated even when it comes later in the list ('best' first, 'last' second).
    return out, aliases, skipped


def previous_result(eval_dir: Path, name: str, step: int | None) -> dict | None:
    """The result of a complete eval of checkpoint ``name`` at ``step`` that ``eval/<name>/summary.json`` already holds
    (an earlier attempt of a restarted run), as ``Driver.run_harness`` returns it; None if there is none."""
    path = Path(eval_dir) / name / "summary.json"
    try:
        summary = json.loads(path.read_text())
    except Exception:
        return None
    if summary.get("checkpoint") != name or summary.get("step") != step or summary.get("complete") is not True:
        return None
    totals = summary.get("summary") or {}
    split = totals.get("split") or {}
    return {
        "summary": str(path),
        "step": step,
        "val_loss": summary.get("val_loss"),
        "overall": totals.get("overall"),
        "val": split.get("val"),
        "train": split.get("train"),
        "complete": True,
        "coverage": summary.get("coverage"),
    }


def make_rounds(names: list[str], num_nodes: int, schedule: str) -> list[dict[str, list[int]]]:
    """``sequential``: one checkpoint per round on every node. ``parallel``: up to ``num_nodes`` checkpoints per
    round, the nodes split into contiguous groups as even as possible (node 0 in the first group)."""
    if not names:
        return []
    if schedule != "parallel" or num_nodes < 2 or len(names) < 2:
        return [{name: list(range(num_nodes))} for name in names]
    rounds = []
    for start in range(0, len(names), num_nodes):
        chunk = names[start : start + num_nodes]
        size, extra = divmod(num_nodes, len(chunk))
        assignment, node = {}, 0
        for i, name in enumerate(chunk):
            count = size + (1 if i < extra else 0)
            assignment[name] = list(range(node, node + count))
            node += count
        rounds.append(assignment)
    return rounds


def round_deadline(now: float, deadline: float | None, reserve: float, rounds_left: int) -> float | None:
    """An equal share of what is left of the cap (minus the reserve) for the next round; None without a cap."""
    if deadline is None:
        return None
    usable = max(0.0, deadline - reserve - now)
    return now + usable / max(1, rounds_left)


def build_plan(
    cfg,
    *,
    run_dir: Path,
    exports: dict[str, dict],
    node: int,
    num_nodes: int,
    gpus_per_node: int,
    master_addr: str,
    delete_pushed_exports: bool = False,
    prime_team_id: str | None = None,
    credentials: dict[str, Any] | None = None,
    now: float | None = None,
) -> Plan:
    """``cfg`` is the resolved ``[inrun_eval]`` config."""
    now = time.time() if now is None else now
    if cfg.model_dir is not None:
        checkpoints = [Checkpoint(name=cfg.model_name, model_dir=str(cfg.model_dir))]
        aliases, skipped = {}, {}
    else:
        checkpoints, aliases, skipped = checkpoints_to_eval(list(cfg.checkpoints), exports, cfg.evaluate_aliases)
    # A restart of a run whose eval already finished a checkpoint at the same step does not evaluate it again.
    evaluated = {}
    if cfg.model_dir is None:
        for ckpt in checkpoints:
            result = previous_result(Path(run_dir) / "eval", ckpt.name, ckpt.step)
            if result is not None:
                evaluated[ckpt.name] = result
        checkpoints = [c for c in checkpoints if c.name not in evaluated]
    work_dir = Path(cfg.work_dir) if cfg.work_dir is not None else Path(run_dir) / ".inrun-eval-work"
    return Plan(
        run_dir=str(run_dir),
        eval_dir=str(Path(run_dir) / "eval"),
        work_dir=str(work_dir / f"node{node}"),
        node=node,
        num_nodes=num_nodes,
        master_addr=master_addr,
        coord_port=cfg.coord_port,
        started=now,
        deadline=None if cfg.budget_hours is None else now + cfg.budget_hours * 3600,
        checkpoints=checkpoints,
        rounds=make_rounds([c.name for c in checkpoints], num_nodes, cfg.schedule),
        aliases=aliases,
        skipped=skipped,
        evaluated=evaluated,
        exports=exports,
        delete_pushed_exports=delete_pushed_exports,
        config=cfg.model_dump(mode="json"),
        gpus_per_node=gpus_per_node,
        prime_team_id=prime_team_id,
        credentials=dict(credentials or {}),
    )
