"""Compare two configs for a live update: what changed, and which changes are frozen.

Every field is live by default. Fields marked with pydantic's ``Field(frozen=True)`` cannot
change while a run is going; on a sub-config field, that freezes everything below it. A frozen
sub-config field marked ``Field(frozen=True, json_schema_extra=LIVE_INSIDE)`` instead lets the
fields below it that are marked ``Field(json_schema_extra=LIVE)`` change, with everything below
them, including swapping their value for None or for another config class."""

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic_core import to_jsonable_python

LIVE = {"live": True}
"""``json_schema_extra`` marker for a field that stays live below a ``LIVE_INSIDE`` field."""

LIVE_INSIDE = {"live_inside": True}
"""``json_schema_extra`` marker for a frozen sub-config field whose ``LIVE`` fields stay live."""


def _marked(info: FieldInfo, key: str) -> bool:
    return isinstance(info.json_schema_extra, dict) and info.json_schema_extra.get(key) is True


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


def _diff_value(old: Any, new: Any, path: str, diff: ConfigDiff, frozen: bool = False) -> None:
    """``frozen``: ``path`` sits below a ``LIVE_INSIDE`` field, so only LIVE fields may change."""
    if old == new:
        return
    if isinstance(old, BaseModel) or isinstance(new, BaseModel):
        # Swapping a sub-config for None or for a config of another class is never live.
        if type(old) is not type(new):
            diff.frozen.append(path)
            return
        for name, info in type(old).model_fields.items():
            old_value, new_value = getattr(old, name), getattr(new, name)
            if old_value == new_value:
                continue
            field_path = f"{path}.{name}" if path else name
            if _marked(info, "live"):
                _diff_live(old_value, new_value, field_path, diff)
            elif info.frozen and not frozen and not _marked(info, "live_inside"):
                diff.frozen.append(field_path)
            elif info.frozen and not frozen:
                # Reported at the frozen field itself, unless only LIVE fields below it changed.
                inner = ConfigDiff()
                _diff_value(old_value, new_value, field_path, inner, frozen=True)
                if inner.frozen:
                    diff.frozen.append(field_path)
                else:
                    diff.changes.extend(inner.changes)
            else:
                _diff_value(old_value, new_value, field_path, diff, frozen=frozen)
        return
    if isinstance(old, list) and isinstance(new, list) and any(isinstance(item, BaseModel) for item in old + new):
        # Lists of configs (e.g. sources) compare by position; their identifying fields are frozen.
        if len(old) != len(new):
            diff.frozen.append(path)
            return
        for index, (old_item, new_item) in enumerate(zip(old, new)):
            _diff_value(old_item, new_item, f"{path}[{index}]", diff, frozen=frozen)
        return
    if frozen:
        diff.frozen.append(path)
        return
    diff.changes.append(ConfigChange(path=path, old=to_jsonable_python(old), new=to_jsonable_python(new)))


def _diff_live(old: Any, new: Any, path: str, diff: ConfigDiff) -> None:
    """A LIVE field changes freely, including to None or to another config class."""
    if isinstance(old, BaseModel) and type(old) is type(new):
        _diff_value(old, new, path, diff)
    else:
        diff.changes.append(ConfigChange(path=path, old=to_jsonable_python(old), new=to_jsonable_python(new)))


def frozen_raw_changes(model: BaseModel, dumped: dict, raw: dict, path: str = "", frozen: bool = False) -> list[str]:
    """Paths of frozen fields that ``raw`` (a submitted config dict) changes from ``dumped``
    (``model`` dumped to JSON). Unlike ``diff_config``, this needs no validated new config,
    so it still reports frozen changes that also make the new config invalid. ``frozen``:
    ``path`` sits below a ``LIVE_INSIDE`` field, so only LIVE fields may change."""
    changes = []
    for name, info in type(model).model_fields.items():
        if name not in raw or raw[name] == dumped[name] or _marked(info, "live"):
            continue
        field_path = f"{path}.{name}" if path else name
        if info.frozen and not frozen and not _marked(info, "live_inside"):
            changes.append(field_path)
            continue
        value = getattr(model, name)
        below_frozen = frozen or bool(info.frozen)
        if isinstance(value, BaseModel) and isinstance(raw[name], dict):
            inner = frozen_raw_changes(value, dumped[name], raw[name], field_path, below_frozen)
        elif isinstance(value, list) and isinstance(raw[name], list) and len(value) == len(raw[name]):
            inner = []
            for index, item in enumerate(value):
                item_path = f"{field_path}[{index}]"
                if isinstance(item, BaseModel) and isinstance(raw[name][index], dict):
                    inner += frozen_raw_changes(item, dumped[name][index], raw[name][index], item_path, below_frozen)
                elif below_frozen and raw[name][index] != dumped[name][index]:
                    inner.append(item_path)
        else:
            inner = [field_path] if below_frozen else []
        if inner and info.frozen and not frozen:
            # Reported at the frozen field itself, as diff_config does.
            changes.append(field_path)
        else:
            changes += inner
    return changes


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
