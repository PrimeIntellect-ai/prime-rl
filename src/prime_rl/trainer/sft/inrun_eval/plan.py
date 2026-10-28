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
    deadline: float
    checkpoints: list[Checkpoint]
    """To evaluate, in order."""
    rounds: list[dict[str, list[int]]]
    """Per round: checkpoint name -> the nodes serving it (every node serves at most one checkpoint per round)."""
    aliases: dict[str, str] = field(default_factory=dict)
    """Checkpoint -> the evaluated checkpoint it is identical to ('best' at the final step is 'last')."""
    skipped: dict[str, str] = field(default_factory=dict)
    """Checkpoints not evaluated, and why."""
    exports: dict[str, dict] = field(default_factory=dict)
    """The export results (``hf_export.ExportResult``), for the cleanup."""
    node_files: list[str] = field(default_factory=list)
    """The export files this node uploads (its share of the world-wide split): what it keeps of its full copies."""
    master_files: list[str] = field(default_factory=list)
    """Non-weight files of an export that only the master keeps."""
    dedupe_exports: bool = False
    delete_pushed_exports: bool = False
    config: dict[str, Any] = field(default_factory=dict)
    """``[inrun_eval]``, resolved."""
    gpus_per_node: int = 8

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> "Plan":
        data = json.loads(text)
        data["checkpoints"] = [Checkpoint(**c) for c in data["checkpoints"]]
        return cls(**data)


def checkpoints_to_eval(
    names: list[str], exports: dict[str, dict]
) -> tuple[list[Checkpoint], dict[str, str], dict[str, str]]:
    """The exports to evaluate (in ``names`` order), aliases (an export that is a copy of another evaluated one) and
    the skipped ones with the reason."""
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
        if same is not None and same in names and (exports.get(same) or {}).get("ok"):
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


def round_deadline(now: float, deadline: float, reserve: float, rounds_left: int) -> float:
    """An equal share of what is left of the budget (minus the reserve) for the next round."""
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
    node_files: list[str] | None = None,
    master_files: list[str] | None = None,
    delete_pushed_exports: bool = False,
    now: float | None = None,
) -> Plan:
    """``cfg`` is the resolved ``[inrun_eval]`` config."""
    now = time.time() if now is None else now
    if cfg.model_dir is not None:
        checkpoints = [Checkpoint(name=cfg.model_name, model_dir=str(cfg.model_dir))]
        aliases, skipped = {}, {}
    else:
        checkpoints, aliases, skipped = checkpoints_to_eval(list(cfg.checkpoints), exports)
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
        deadline=now + cfg.budget_hours * 3600,
        checkpoints=checkpoints,
        rounds=make_rounds([c.name for c in checkpoints], num_nodes, cfg.schedule),
        aliases=aliases,
        skipped=skipped,
        exports=exports,
        node_files=sorted(node_files or []),
        master_files=sorted(master_files or []),
        dedupe_exports=cfg.model_dir is None and node_files is not None,
        delete_pushed_exports=delete_pushed_exports,
        config=cfg.model_dump(mode="json"),
        gpus_per_node=gpus_per_node,
    )
