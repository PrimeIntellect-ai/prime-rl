from typing import Any

from pydantic import BaseModel
from pydantic_config import BaseConfig as BaseConfig  # noqa: F401
from pydantic_config import cli  # noqa: F401


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
