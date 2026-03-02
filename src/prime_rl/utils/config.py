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
