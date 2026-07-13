from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic_config import BaseConfig as BaseConfig  # noqa: F401
from pydantic_config import cli  # noqa: F401


def to_toml_dict(config: BaseModel, exclude: set[str] | None = None) -> dict:
    """Dump a config to a TOML-serializable dict.

    TOML cannot represent null, so None fields left at their default are dropped
    (they re-resolve identically on re-parse) while explicitly-set None fields
    are encoded as the string ``"None"``, which ``BaseConfig`` converts back to
    ``None``. Dropping those too would silently revert explicit None overrides
    (e.g. ``--trainer.model.compile None``) to their non-None defaults when the
    written config is re-parsed.
    """
    return _encode_model(config, config.model_dump(exclude=exclude, mode="json"))


def _encode_model(model: BaseModel, dumped: dict) -> dict:
    encoded = {}
    for name, value in dumped.items():
        if value is None:
            if name in model.model_fields_set:
                encoded[name] = "None"
            continue
        encoded[name] = _encode_value(getattr(model, name), value)
    return encoded


def _encode_value(attr: Any, value: Any) -> Any:
    if isinstance(attr, BaseModel) and isinstance(value, dict):
        return _encode_model(attr, value)
    if isinstance(value, list) and isinstance(attr, (list, tuple)):
        return [_encode_value(a, v) for a, v in zip(attr, value)]
    if isinstance(value, dict) and isinstance(attr, dict):
        return {k: _encode_value(a, v) for (k, v), a in zip(value.items(), attr.values())}
    return "None" if value is None else value


def find_package_resource(subdir: str) -> Path | None:
    """Find a directory contributed to the `prime_rl` namespace package by any installed wheel.

    Returns None if `subdir` is not present in any wheel — e.g. on a slim
    `prime-rl-configs`-only install where `prime-rl`'s shipped resources
    (templates, etc.) are absent.
    """
    import prime_rl

    for p in prime_rl.__path__:
        candidate = Path(p) / subdir
        if candidate.is_dir():
            return candidate
    return None


def rgetattr(obj: Any, attr_path: str) -> Any:
    """Recursive getattr for dotted paths: rgetattr(cfg, "trainer.model.name")."""
    current = obj
    for attr in attr_path.split("."):
        if not hasattr(current, attr):
            raise AttributeError(f"'{type(current).__name__}' object has no attribute '{attr}'")
        current = getattr(current, attr)
    return current


def rsetattr(obj: Any, attr_path: str, value: Any) -> None:
    """Recursive setattr for dotted paths: rsetattr(cfg, "trainer.model.name", "foo")."""
    if "." not in attr_path:
        return setattr(obj, attr_path, value)
    parent_path, attr = attr_path.rsplit(".", 1)
    setattr(rgetattr(obj, parent_path), attr, value)
