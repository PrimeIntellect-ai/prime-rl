"""Compare two configs for a live update: what changed, and which changes are frozen.

Every field is live by default. Fields marked with pydantic's ``Field(frozen=True)`` cannot
change while a run is going; on a sub-config field, that freezes everything below it."""

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel
from pydantic_core import to_jsonable_python


class ConfigChange(BaseModel):
    path: str
    """Dotted path with list indices: ``train.source[0].sampling.temperature``."""
    old: Any
    new: Any


@dataclass
class ConfigDiff:
    changes: list[ConfigChange] = field(default_factory=list)
    frozen: list[str] = field(default_factory=list)
    """Paths of frozen fields the new config changes."""


def diff_config(old: BaseModel, new: BaseModel) -> ConfigDiff:
    diff = ConfigDiff()
    _diff_value(old, new, "", diff)
    return diff


def _diff_value(old: Any, new: Any, path: str, diff: ConfigDiff) -> None:
    if old == new:
        return
    if isinstance(old, BaseModel) or isinstance(new, BaseModel):
        # Swapping a sub-config for None or for a config of another class is never live.
        if type(old) is not type(new):
            diff.frozen.append(path)
            return
        for name, info in type(old).model_fields.items():
            old_value, new_value = getattr(old, name), getattr(new, name)
            field_path = f"{path}.{name}" if path else name
            if info.frozen and old_value != new_value:
                diff.frozen.append(field_path)
            else:
                _diff_value(old_value, new_value, field_path, diff)
        return
    if isinstance(old, list) and isinstance(new, list) and any(isinstance(item, BaseModel) for item in old + new):
        # Lists of configs (e.g. sources) compare by position; their identifying fields are frozen.
        if len(old) != len(new):
            diff.frozen.append(path)
            return
        for index, (old_item, new_item) in enumerate(zip(old, new)):
            _diff_value(old_item, new_item, f"{path}[{index}]", diff)
        return
    diff.changes.append(ConfigChange(path=path, old=to_jsonable_python(old), new=to_jsonable_python(new)))


def frozen_raw_changes(model: BaseModel, dumped: dict, raw: dict, path: str = "") -> list[str]:
    """Paths of frozen fields that ``raw`` (a submitted config dict) changes from ``dumped``
    (``model`` dumped to JSON). Unlike ``diff_config``, this needs no validated new config,
    so it still reports frozen changes that also make the new config invalid."""
    frozen = []
    for name, info in type(model).model_fields.items():
        if name not in raw or raw[name] == dumped[name]:
            continue
        field_path = f"{path}.{name}" if path else name
        value = getattr(model, name)
        if info.frozen:
            frozen.append(field_path)
        elif isinstance(value, BaseModel) and isinstance(raw[name], dict):
            frozen += frozen_raw_changes(value, dumped[name], raw[name], field_path)
        elif isinstance(value, list) and isinstance(raw[name], list) and len(value) == len(raw[name]):
            for index, item in enumerate(value):
                if isinstance(item, BaseModel) and isinstance(raw[name][index], dict):
                    frozen += frozen_raw_changes(item, dumped[name][index], raw[name][index], f"{field_path}[{index}]")
    return frozen


def merge_resumed_config(base: Any, applied: Any, launch: Any) -> Any:
    """Merge dumped configs on resume: ``base`` is what the checkpointed run was launched
    with, ``applied`` what was live at the checkpoint, ``launch`` what this run is launched
    with. A value the launch config left as it was in ``base`` resumes from ``applied``; any
    value the launch config changed wins."""
    if launch == base:
        return applied
    if isinstance(launch, dict) and isinstance(base, dict) and isinstance(applied, dict):
        return {
            key: merge_resumed_config(base.get(key, _MISSING), applied.get(key, value), value)
            for key, value in launch.items()
        }
    if (
        isinstance(launch, list)
        and isinstance(base, list)
        and isinstance(applied, list)
        and len(launch) == len(base) == len(applied)
    ):
        return [merge_resumed_config(*values) for values in zip(base, applied, launch)]
    return launch


_MISSING = object()
