"""Check the public ModelExpress APIs required by the selected transport."""

import importlib
import inspect
from types import ModuleType

MX_INSTALL_HELP = "See docs/scaling.md#modelexpress-refit for the pinned ModelExpress Docker setup."


def import_model_express(module_name: str, *, transport: str) -> ModuleType:
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name not in {module_name, module_name.split(".")[0]}:
            raise
        raise ImportError(f"{transport} requires the optional {module_name} client. {MX_INSTALL_HELP}") from error


def require_mx_refit(*, staging_mode: str | None = None, streaming: bool = False) -> ModuleType:
    """Reject unsupported clients before creating a trainer or generator client."""
    mx = import_model_express("modelexpress_rl", transport="mx_refit")
    required = (
        "FSDPTrainerContext",
        "ModelExpressControlClient",
        "ModelExpressGeneratorClient",
        "ModelExpressGeneratorConfig",
        "ModelExpressTrainerClient",
        "ModelExpressTrainerConfig",
        "TrainerStagingMode",
        "VllmGeneratorContext",
        "WeightPayloadFormat",
        "WeightSource",
        "WeightVersionRef",
        "WeightVersionState",
    )
    missing = [name for name in required if not hasattr(mx, name)]
    if not missing:
        if "wire_dtype_overrides" not in inspect.signature(mx.FSDPTrainerContext).parameters:
            missing.append("FSDPTrainerContext(wire_dtype_overrides=...)")
        if staging_mode is not None and not hasattr(mx.TrainerStagingMode, staging_mode):
            missing.append(f"TrainerStagingMode.{staging_mode}")
        if streaming:
            apply = getattr(mx.ModelExpressGeneratorClient, "apply_weight_streaming", None)
            if (
                not callable(apply)
                or not {"version", "max_staging_bytes"} <= inspect.signature(apply).parameters.keys()
            ):
                missing.append("ModelExpressGeneratorClient.apply_weight_streaming(version=..., max_staging_bytes=...)")
    if missing:
        raise ImportError(
            f"Installed ModelExpress lacks APIs required by mx_refit: {', '.join(missing)}. {MX_INSTALL_HELP}"
        )
    return mx
