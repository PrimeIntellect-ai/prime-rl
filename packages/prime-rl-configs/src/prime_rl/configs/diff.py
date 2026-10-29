"""Compare two configs for a live update: what changed, and which changes are frozen.

A field is frozen unless it is ``Annotated`` with ``Live()``. A live field changes freely
with everything below it, including swapping its value for None or for another config
class. An unmarked sub-config is compared field by field, and so is an unmarked list of
sub-configs of the same length, by position: live fields below them stay live. Any other
change to an unmarked field is frozen. A sub-config annotated with ``Frozen()`` is frozen
with everything below it, live fields included."""

from dataclasses import dataclass, field
from typing import Annotated, Any

from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic_core import to_jsonable_python


@dataclass(frozen=True)
class Live:
    """``Annotated`` marker for a field that may change while a run is going."""

    live: bool = True


@dataclass(frozen=True)
class Frozen:
    """``Annotated`` marker for a sub-config whose fields stay frozen, live ones included:
    e.g. a default that only seeds other fields when the config loads."""

    live: bool = False


def _live(info: FieldInfo, live: bool | None) -> bool | None:
    """Whether a field below a field whose state is ``live`` is live: True, False, or
    None while no marker has decided it."""
    if live is not None:
        return live
    for metadata in info.metadata:
        marked = getattr(metadata, "live", None)
        if marked is not None:
            return marked
    return None


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


def _diff_value(old: Any, new: Any, path: str, diff: ConfigDiff, live: bool | None = None) -> None:
    """``live``: True at or below a LIVE field, False at or below a FROZEN one."""
    if old == new:
        return
    if isinstance(old, BaseModel) and type(old) is type(new):
        for name, info in type(old).model_fields.items():
            field_path = f"{path}.{name}" if path else name
            _diff_value(getattr(old, name), getattr(new, name), field_path, diff, _live(info, live))
        return
    if (
        isinstance(old, list)
        and isinstance(new, list)
        and len(old) == len(new)
        and any(isinstance(item, BaseModel) for item in old + new)
    ):
        # Lists of configs (e.g. sources) compare by position; their identifying fields are frozen.
        for index, (old_item, new_item) in enumerate(zip(old, new)):
            _diff_value(old_item, new_item, f"{path}[{index}]", diff, live)
        return
    if live:
        diff.changes.append(ConfigChange(path=path, old=to_jsonable_python(old), new=to_jsonable_python(new)))
    else:
        diff.frozen.append(path)


def frozen_raw_changes(
    model: BaseModel, dumped: dict, raw: dict, path: str = "", live: bool | None = None
) -> list[str]:
    """Paths of frozen fields that ``raw`` (a submitted config dict) changes from ``dumped``
    (``model`` dumped to JSON). Unlike ``diff_config``, this needs no validated new config,
    so it still reports frozen changes that also make the new config invalid."""
    changes = []
    for name, info in type(model).model_fields.items():
        field_live = _live(info, live)
        if name not in raw or raw[name] == dumped[name] or field_live:
            continue
        field_path = f"{path}.{name}" if path else name
        value, raw_value = getattr(model, name), raw[name]
        if isinstance(value, BaseModel) and isinstance(raw_value, dict):
            changes += frozen_raw_changes(value, dumped[name], raw_value, field_path, field_live)
        elif isinstance(value, list) and isinstance(raw_value, list) and len(value) == len(raw_value):
            for index, item in enumerate(value):
                item_path = f"{field_path}[{index}]"
                if isinstance(item, BaseModel) and isinstance(raw_value[index], dict):
                    changes += frozen_raw_changes(item, dumped[name][index], raw_value[index], item_path, field_live)
                elif raw_value[index] != dumped[name][index]:
                    changes.append(item_path)
        else:
            changes.append(field_path)
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
