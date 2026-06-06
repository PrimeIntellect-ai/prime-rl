from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic_config import BaseConfig as BaseConfig  # noqa: F401
from pydantic_config import cli  # noqa: F401


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


def get_all_fields(model: BaseModel | type) -> list[str]:
    if isinstance(model, BaseModel):
        model_cls = model.__class__
    else:
        model_cls = model

    fields = []
    for name, field in model_cls.model_fields.items():
        field_type = field.annotation
        fields.append(name)
        if field_type is not None and hasattr(field_type, "model_fields"):
            sub_fields = get_all_fields(field_type)
            fields.extend(f"{name}.{sub}" for sub in sub_fields)
    return fields
