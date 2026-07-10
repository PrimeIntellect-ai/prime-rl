"""Canonical Helm controller and PVC values for DynamoGraph deployments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Literal

ControllerMode = Literal["chartManaged", "external"]


def _orchestrator_service() -> dict[str, object]:
    return {
        "enabled": True,
        "type": "ClusterIP",
        "port": 8000,
        "ncclPort": 29501,
    }


def _trainer_service() -> dict[str, object]:
    return {
        "enabled": True,
        "type": "ClusterIP",
        "port": 8000,
        "ncclPort": 29501,
    }


def _trainer_probes() -> dict[str, object]:
    return {
        "enabled": False,
        "startup": {
            "periodSeconds": 10,
            "failureThreshold": 60,
            "timeoutSeconds": 30,
        },
        "liveness": {
            "periodSeconds": 30,
            "failureThreshold": 6,
            "timeoutSeconds": 30,
        },
        "readiness": {
            "periodSeconds": 10,
            "failureThreshold": 3,
            "timeoutSeconds": 30,
        },
    }


def _controller_values(component: Mapping[str, object]) -> dict[str, object]:
    return {key: deepcopy(value) for key, value in component.items() if key not in {"gpu", "placement"}}


def build_chart_runtime_contract(
    *,
    controller_mode: ControllerMode,
    image_reference: str,
    image_pull_secrets: Sequence[str],
    orchestrator_replicas: int,
    trainer_replicas: int,
    orchestrator_command: str | None,
    trainer_command: str | None,
    trainer_gpu_count: int,
    orchestrator_placement: Mapping[str, object],
    trainer_placement: Mapping[str, object],
    shared_pvc: str | None,
    model_cache_pvc: str | None,
    hf_token_secret: str | None,
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    """Return one immutable-by-construction contract and its exact chart values."""
    enabled = controller_mode == "chartManaged"
    image = {
        "reference": image_reference,
        "pullPolicy": "IfNotPresent",
        "pullSecrets": list(image_pull_secrets),
    }
    config = {
        "example": "reverse-text",
        "secrets": {
            "enabled": False,
            "name": "prime-rl-secrets",
        },
    }
    hugging_face = {
        "tokenSecretName": hf_token_secret or "",
        "tokenSecretKey": "HF_TOKEN",
    }
    model_cache = {
        "enabled": model_cache_pvc is not None,
        "existingClaim": model_cache_pvc or "",
        "mountPath": "/model-cache",
    }
    storage = {
        "enabled": enabled or shared_pvc is not None,
        "existingClaim": shared_pvc or "",
        "storageClassName": "nfs",
        "accessModes": ["ReadWriteMany"],
        "size": "1Ti",
        "mountPath": "/data",
    }
    orchestrator = {
        "enabled": enabled,
        "replicas": orchestrator_replicas if enabled else 0,
        "autoStart": enabled,
        "command": orchestrator_command if enabled else "",
        "gpu": {"enabled": False},
        "placement": deepcopy(orchestrator_placement),
        "resources": {"requests": {"memory": "2Gi", "cpu": "1"}},
        "service": _orchestrator_service(),
        "env": [],
    }
    trainer = {
        "enabled": enabled,
        "replicas": trainer_replicas if enabled else 0,
        "autoStart": enabled,
        "command": trainer_command if enabled else "",
        "gpu": {
            "enabled": enabled,
            "count": trainer_gpu_count if enabled else 0,
        },
        "placement": deepcopy(trainer_placement),
        "pytorchCudaAllocConf": "expandable_segments:True",
        "resources": {"requests": {"memory": "4Gi", "cpu": "1"}},
        "service": _trainer_service(),
        "env": [],
        "probes": _trainer_probes(),
    }
    workload = {
        "controllerMode": controller_mode,
        "image": image,
        "config": config,
        "huggingFace": hugging_face,
        "modelCache": model_cache,
        "orchestrator": orchestrator,
        "storage": storage,
        "trainer": trainer,
    }
    values = {
        "image": deepcopy(image),
        "config": deepcopy(config),
        "huggingFace": deepcopy(hugging_face),
        "modelCache": deepcopy(model_cache),
        "orchestrator": _controller_values(orchestrator),
        "storage": deepcopy(storage),
        "trainer": _controller_values(trainer) | {"gpu": deepcopy(trainer["gpu"])},
    }
    return workload, values
