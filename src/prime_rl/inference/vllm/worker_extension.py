from importlib import import_module
from types import MappingProxyType
from typing import Literal

WeightTransferType = Literal["nccl", "filesystem", "nixl"]

WORKER_EXTENSION_CLASSES = MappingProxyType(
    {
        "nccl": "prime_rl.inference.vllm.worker.nccl.NCCLWeightUpdateWorker",
        "filesystem": "prime_rl.inference.vllm.worker.filesystem.FileSystemWeightUpdateWorker",
        "nixl": "prime_rl.inference.vllm.worker.nixl.NIXLWeightUpdateWorker",
    }
)


def worker_extension_class_path(
    transfer_type: WeightTransferType,
    *,
    validate_import: bool = False,
) -> str:
    class_path = WORKER_EXTENSION_CLASSES[transfer_type]
    if not validate_import:
        return class_path

    module_name, class_name = class_path.rsplit(".", 1)
    try:
        module = import_module(module_name)
        getattr(module, class_name)
    except (ImportError, AttributeError) as error:
        raise ImportError(f"Could not import the {transfer_type} worker extension {class_path!r}") from error
    return class_path
