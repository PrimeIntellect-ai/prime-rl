from pydantic import BaseModel, field_validator, model_validator
from pydantic_config import BaseConfig as _BaseConfig
from pydantic_config import cli  # noqa: F401


class BaseConfig(_BaseConfig):
    @field_validator("*", mode="before")
    @classmethod
    def empty_str_to_none(cls, v):
        if v == "None":
            return None
        return v

    @model_validator(mode="before")
    @classmethod
    def _default_discriminator_types(cls, data: dict) -> dict:
        """For discriminated-union fields whose default carries a ``type``, inject it when missing.

        Without ``nested_model_default_partial_update`` the ``type`` tag is no
        longer merged in automatically, so we do it here for every field whose
        default instance exposes one.
        """
        if not isinstance(data, dict):
            return data
        for field_name, field_info in cls.model_fields.items():
            val = data.get(field_name)
            if isinstance(val, dict) and "type" not in val:
                default = field_info.default
                if isinstance(default, BaseModel) and hasattr(default, "type"):
                    val["type"] = default.type
        return data


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
