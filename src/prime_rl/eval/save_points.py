"""Save points of in-flight eval episodes, so a resumed eval continues them.

An episode saves its progress (``vf.EpisodeState.save()``) while it runs. Each save
lands here as ``<run_dir>/save_points/<env>/<dispatch id>.json``, replacing the previous
one, and is removed once the episode lands ok in the trace stream. A resume hands the ones left behind to
the rollouts still owed, which ``EpisodeState.load()`` them."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import orjson

from prime_rl.orchestrator.types import InflightEpisode


@dataclass(frozen=True)
class SavePoint:
    dispatch_id: str
    group_id: str
    state: dict


class SavePoints:
    def __init__(self, run_dir: Path) -> None:
        self.root = run_dir / "save_points"
        self.landing: dict[str, Path] = {}
        """Save points of ok episodes on their way to the trace stream, by episode id."""

    def path(self, env_name: str, dispatch_id: str) -> Path:
        return self.root / env_name / f"{dispatch_id}.json"

    def write(self, meta: InflightEpisode, state: dict) -> None:
        path = self.path(meta.env_name, meta.dispatch_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {"task_key": meta.task.key, "group_id": str(meta.group_id), "state": state}
        tmp = path.with_suffix(".json.tmp")
        tmp.write_bytes(orjson.dumps(record))
        tmp.replace(path)

    def finish(self, meta: InflightEpisode, episode_id: str) -> None:
        """``meta``'s episode finished ok: its save point goes once the episode landed."""
        self.landing[episode_id] = self.path(meta.env_name, meta.dispatch_id)

    def landed(self, episode_id: str) -> None:
        if (path := self.landing.pop(episode_id, None)) is not None:
            path.unlink(missing_ok=True)

    def load(self) -> dict[str, dict[str, list[SavePoint]]]:
        """The save points left behind, by env and task key."""
        found: dict[str, dict[str, list[SavePoint]]] = defaultdict(lambda: defaultdict(list))
        for path in sorted(self.root.glob("*/*.json")):
            record = orjson.loads(path.read_bytes())
            found[path.parent.name][record["task_key"]].append(
                SavePoint(dispatch_id=path.stem, group_id=record["group_id"], state=record["state"])
            )
        return found
