from types import ModuleType

from pydantic import BaseModel

from zeroband.training.orchestrator.genesys.format_utils import extract_last_json


def _load_model_from_code(code_str: str, model_name: str) -> type[BaseModel]:
    """
    Execute `code_str` in a scratch module namespace and return the
    class named `model_name`.

    Raises RuntimeError if the class is missing or not a BaseModel.
    """
    module = ModuleType("dyn_pydantic_cfg")
    try:
        exec(code_str, module.__dict__)
    except Exception as e:
        raise RuntimeError(f"config code failed to execute: {e!r}") from e

    cls = getattr(module, model_name, None)
    if cls is None or not issubclass(cls, BaseModel):
        raise RuntimeError(f"{model_name} not found or not a Pydantic BaseModel")

    # cheap structural self-check (never instantiates)
    cls.model_json_schema()
    return cls


def validate_pydantic_json(
    completion: str, verification_info: dict
) -> tuple[bool, str]:
    payload = extract_last_json(completion)
    if payload is None:
        return 0

    Model = _load_model_from_code(
        verification_info["pydantic_config"],
        verification_info["model_name"],
    )
    try:
        Model.model_validate(payload)
    except Exception:
        return 0

    return 1
