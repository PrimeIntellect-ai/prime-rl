"""Compile a Prime Dynamo inference config into Helm DGD values."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from prime_rl.configs.inference import DisaggregatedInferenceDeploymentConfig, InferenceConfig
from prime_rl.inference.dgd_controller_contract import build_chart_runtime_contract
from prime_rl.inference.dynamo import (
    CHAT_TEMPLATE_ASSET,
    DynamoProcessSpec,
    build_frontend_process,
    build_worker_process,
    resolve_chat_template_content,
    write_role_engine_configs,
)

ENGINE_MOUNT_PATH = "/etc/prime-rl/dynamo"
ENGINE_CONFIG_HASH_ANNOTATION = "prime-rl.nvidia.com/config-sha256"
MANIFEST_HASH_ANNOTATION = "prime-rl.nvidia.com/manifest-sha256"
MANIFEST_HASH_SCOPE_ANNOTATION = "prime-rl.nvidia.com/manifest-sha256-scope"
TOPOLOGY_HASH_ANNOTATION = "prime-rl.nvidia.com/topology-sha256"
WORKLOAD_HASH_ANNOTATION = "prime-rl.nvidia.com/workload-sha256"
MANIFEST_HASH_SCOPE = (
    "resource; json.dumps(sort_keys=true,indent=2)+newline; "
    "exclude=/metadata/annotations/prime-rl.nvidia.com~1manifest-sha256"
)
_DGD_RESERVED_ENV_KEYS = frozenset(
    {
        "CONTAINER_NAME",
        "DYNAMO_PORT",
        "DYN_COMPONENT",
        "DYN_DISCOVERY_BACKEND",
        "DYN_ENABLE_RL",
        "DYN_ENDPOINT",
        "DYN_ENDPOINT_TYPES",
        "DYN_ETCD_ENDPOINTS",
        "DYN_EVENT_PLANE",
        "DYN_FILE_KV",
        "DYN_HEALTH_CHECK_ENABLED",
        "DYN_HTTP_PORT",
        "DYN_KUBE_DISCOVERY_MODE",
        "DYN_NAMESPACE",
        "DYN_NAMESPACE_PREFIX",
        "DYN_NAMESPACE_WORKER_SUFFIX",
        "DYN_PARENT_DGD_K8S_NAME",
        "DYN_PARENT_DGD_K8S_NAMESPACE",
        "DYN_RL_ENDPOINT",
        "DYN_RL_PORT",
        "POD_NAME",
        "POD_NAMESPACE",
        "POD_UID",
        "VLLM_NIXL_SIDE_CHANNEL_HOST",
        "VLLM_NIXL_SIDE_CHANNEL_PORT",
    }
)
_DGD_RESERVED_ENV_PREFIXES = ("DYN_HEALTH_CHECK_", "DYN_SYSTEM_")
_MODEL_CACHE_MOUNT_PATH = "/model-cache"
_CREDENTIAL_ENV_KEY_PATTERNS = (
    re.compile(r"(?:^|_)(?:TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIALS?)(?:_|$)"),
    re.compile(r"(?:^|_)(?:API|ACCESS|PRIVATE|SECRET)_KEY(?:_|$)"),
)
MAX_RELEASE_NAME_LENGTH = 41


@dataclass(frozen=True, slots=True)
class KubernetesToleration:
    key: str
    operator: Literal["Exists", "Equal"] = "Exists"
    effect: Literal["NoSchedule", "PreferNoSchedule", "NoExecute"] | None = "NoSchedule"
    value: str | None = None

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("Kubernetes toleration key must not be empty")
        if self.operator not in ("Exists", "Equal"):
            raise ValueError(f"Unsupported Kubernetes toleration operator: {self.operator!r}")
        if self.effect not in (None, "NoSchedule", "PreferNoSchedule", "NoExecute"):
            raise ValueError(f"Unsupported Kubernetes toleration effect: {self.effect!r}")
        if self.operator == "Exists" and self.value is not None:
            raise ValueError("An Exists toleration cannot define a value")
        if self.operator == "Equal" and self.value is None:
            raise ValueError("An Equal toleration requires a value")

    def as_manifest(self) -> dict[str, str]:
        return {
            "key": self.key,
            "operator": self.operator,
            **({"value": self.value} if self.value is not None else {}),
            **({"effect": self.effect} if self.effect is not None else {}),
        }


def _parse_kubernetes_toleration(value: str) -> KubernetesToleration:
    """Parse one CLI toleration from JSON without evaluating shell-like input."""
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"Kubernetes toleration must be a JSON object: {error.msg}") from error
    if not isinstance(payload, dict):
        raise ValueError("Kubernetes toleration must be a JSON object")
    supported = {"key", "operator", "effect", "value"}
    unknown = sorted(payload.keys() - supported)
    if unknown:
        raise ValueError(f"Kubernetes toleration has unsupported fields: {unknown}")
    if "key" not in payload:
        raise ValueError("Kubernetes toleration requires a key")
    return KubernetesToleration(**payload)


def _unique_tolerations(
    tolerations: tuple[KubernetesToleration, ...],
) -> tuple[KubernetesToleration, ...]:
    unique: list[KubernetesToleration] = []
    seen: set[tuple[tuple[str, str], ...]] = set()
    for toleration in tolerations:
        identity = tuple(sorted(toleration.as_manifest().items()))
        if identity not in seen:
            unique.append(toleration)
            seen.add(identity)
    return tuple(unique)


@dataclass(frozen=True, slots=True)
class GPUSchedulingProfile:
    """Image placement plus the stricter placement required by GPU consumers."""

    runtime_class_name: str | None
    architecture: str
    product: str
    node_pool: str
    node_pool_label: str = "cloud.google.com/gke-nodepool"
    tolerations: tuple[KubernetesToleration, ...] = (KubernetesToleration(key="nvidia.com/gpu"),)
    additional_image_tolerations: tuple[KubernetesToleration, ...] = ()
    additional_gpu_tolerations: tuple[KubernetesToleration, ...] = ()

    def __post_init__(self) -> None:
        required = {
            "architecture": self.architecture,
            "product": self.product,
            "node_pool": self.node_pool,
            "node_pool_label": self.node_pool_label,
        }
        empty = [name for name, value in required.items() if not value]
        if empty:
            raise ValueError(f"GPU scheduling fields must not be empty: {empty}")
        if self.runtime_class_name == "":
            raise ValueError("runtime_class_name must be non-empty or None")
        if not self.tolerations:
            raise ValueError("GPU scheduling requires at least one toleration")
        required_gpu_toleration = KubernetesToleration(key="nvidia.com/gpu")
        if required_gpu_toleration not in self.tolerations:
            raise ValueError("GPU scheduling requires nvidia.com/gpu Exists NoSchedule")

    @property
    def image_node_selector(self) -> dict[str, str]:
        return {
            "kubernetes.io/arch": self.architecture,
            self.node_pool_label: self.node_pool,
        }

    @property
    def node_selector(self) -> dict[str, str]:
        return {
            **self.image_node_selector,
            "nvidia.com/gpu.product": self.product,
        }

    @property
    def image_tolerations(self) -> tuple[KubernetesToleration, ...]:
        required = (
            KubernetesToleration(
                key="kubernetes.io/arch",
                operator="Equal",
                value=self.architecture,
            ),
            KubernetesToleration(key="nvidia.com/gpu"),
            KubernetesToleration(key="prime-rl", operator="Equal", value="true"),
        )
        return _unique_tolerations((*required, *self.additional_image_tolerations))

    @property
    def image_toleration_manifests(self) -> list[dict[str, str]]:
        return [toleration.as_manifest() for toleration in self.image_tolerations]

    @property
    def image_placement(self) -> dict[str, Any]:
        return {
            "nodeSelector": self.image_node_selector,
            "tolerations": self.image_toleration_manifests,
        }

    @property
    def gpu_tolerations(self) -> tuple[KubernetesToleration, ...]:
        return _unique_tolerations((*self.image_tolerations, *self.tolerations, *self.additional_gpu_tolerations))

    @property
    def toleration_manifests(self) -> list[dict[str, str]]:
        return [toleration.as_manifest() for toleration in self.gpu_tolerations]

    @property
    def gpu_placement(self) -> dict[str, Any]:
        placement = {
            "nodeSelector": self.node_selector,
            "tolerations": self.toleration_manifests,
        }
        if self.runtime_class_name is not None:
            placement["runtimeClassName"] = self.runtime_class_name
        return placement


@dataclass(frozen=True)
class DynamoGraphRenderOptions:
    release_name: str
    namespace: str
    image: str
    output_dir: Path
    prime_sha: str
    dynamo_sha: str
    image_digest: str
    run_name: str
    gpu_scheduling: GPUSchedulingProfile
    external_controller: bool = False
    trainer_gpu_count: int = 1
    orchestrator_replicas: int = 1
    trainer_replicas: int = 1
    orchestrator_command: str | None = None
    trainer_command: str | None = None
    model_cache_pvc: str | None = None
    shared_pvc: str | None = None
    image_pull_secrets: tuple[str, ...] = ()
    hf_token_secret: str | None = None

    def __post_init__(self) -> None:
        if re.fullmatch(r"[a-z0-9](?:[-a-z0-9]*[a-z0-9])?", self.release_name) is None:
            raise ValueError("release_name must be a lowercase Kubernetes DNS label")
        if len(self.release_name) > MAX_RELEASE_NAME_LENGTH:
            raise ValueError(
                f"release_name must be at most {MAX_RELEASE_NAME_LENGTH} characters so generated Service names are valid"
            )
        for name, value in (("prime_sha", self.prime_sha), ("dynamo_sha", self.dynamo_sha)):
            if re.fullmatch(r"[0-9a-f]{40}", value) is None:
                raise ValueError(f"{name} must be a full 40-character Git commit SHA")
        if re.fullmatch(r"sha256:[0-9a-f]{64}", self.image_digest) is None:
            raise ValueError("image_digest must be a full sha256 digest")
        if not self.image.endswith(f"@{self.image_digest}"):
            raise ValueError("DGD image must be pinned to image_digest")
        image_name = self.image.rsplit("@", 1)[0]
        image_tag = image_name.rsplit("/", 1)[-1].partition(":")[2]
        if self.prime_sha[:12] not in image_tag or self.dynamo_sha[:12] not in image_tag:
            raise ValueError("DGD image tag must include the Prime and Dynamo commit suffixes")
        if self.trainer_gpu_count < 1:
            raise ValueError("trainer_gpu_count must be at least one")
        if not self.external_controller:
            for component, replicas, command in (
                ("orchestrator", self.orchestrator_replicas, self.orchestrator_command),
                ("trainer", self.trainer_replicas, self.trainer_command),
            ):
                if replicas < 1:
                    raise ValueError(f"{component}_replicas must be at least one in chart-managed mode")
                expected_prefix = f"uv run {component}"
                if command is None or re.match(rf"^uv\s+run\s+{component}(?:\s|$)", command) is None:
                    raise ValueError(f"{component}_command must start with {expected_prefix!r} in chart-managed mode")
        elif self.orchestrator_command is not None or self.trainer_command is not None:
            raise ValueError("external_controller cannot define chart-managed controller commands")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _resource_manifest_canonical(resource: dict[str, Any]) -> bytes:
    annotations = resource["metadata"]["annotations"]
    scoped_annotations = {key: value for key, value in annotations.items() if key != MANIFEST_HASH_ANNOTATION}
    scoped_resource = {
        **resource,
        "metadata": {
            **resource["metadata"],
            "annotations": scoped_annotations,
        },
    }
    return _canonical_json(scoped_resource)


def _worker_topology_binding(services: dict[str, Any]) -> dict[str, Any]:
    return {
        service_name: {
            "role": service["subComponentType"],
            "replicas": service["replicas"],
            "requestsGpu": service["resources"]["requests"]["gpu"],
            "limitsGpu": service["resources"]["limits"]["gpu"],
        }
        for service_name, service in sorted(services.items())
        if service["componentType"] == "worker"
    }


def _release_pod_labels(options: DynamoGraphRenderOptions) -> dict[str, str]:
    return {
        "app.kubernetes.io/name": "prime-rl",
        "app.kubernetes.io/instance": options.release_name,
    }


def _worker_env(process: DynamoProcessSpec) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = [
        {"name": name, "value": value} for name, value in sorted(process.environment().items())
    ]
    values.append(
        {
            "name": "VLLM_NIXL_SIDE_CHANNEL_HOST",
            "valueFrom": {"fieldRef": {"fieldPath": "status.podIP"}},
        }
    )
    return values


def _apply_pod_credentials(
    pod_spec: dict[str, Any],
    container: dict[str, Any],
    options: DynamoGraphRenderOptions,
) -> None:
    if options.image_pull_secrets:
        pod_spec["imagePullSecrets"] = [{"name": name} for name in options.image_pull_secrets]
    if options.model_cache_pvc and not any(item["name"] == "HF_HOME" for item in container.get("env", [])):
        container.setdefault("env", []).append({"name": "HF_HOME", "value": "/model-cache"})
    if options.hf_token_secret and not any(item["name"] == "HF_TOKEN" for item in container.get("env", [])):
        container.setdefault("env", []).append(
            {
                "name": "HF_TOKEN",
                "valueFrom": {
                    "secretKeyRef": {
                        "name": options.hf_token_secret,
                        "key": "HF_TOKEN",
                        "optional": False,
                    }
                },
            }
        )


def _worker_service(
    config: InferenceConfig,
    options: DynamoGraphRenderOptions,
    *,
    role: str,
    replicas: int,
    config_map_name: str,
    engine_file: str,
) -> dict[str, Any]:
    assert config.deployment.type == "disaggregated"
    process = build_worker_process(
        config,
        role,
        Path(ENGINE_MOUNT_PATH) / engine_file,
        nixl_host=None,
        nixl_port=20100,
    )
    container = {
        "image": options.image,
        "imagePullPolicy": "IfNotPresent",
        "command": ["python3", "-m", process.module],
        "args": list(process.arguments),
        "env": _worker_env(process),
        "volumeMounts": [
            {
                "name": "dynamo-engine-config",
                "mountPath": ENGINE_MOUNT_PATH,
                "readOnly": True,
            }
        ],
    }
    pod_spec = {
        **options.gpu_scheduling.gpu_placement,
        "volumes": [
            {
                "name": "dynamo-engine-config",
                "configMap": {"name": config_map_name},
            }
        ],
        "mainContainer": container,
    }
    _apply_pod_credentials(pod_spec, container, options)
    return {
        "componentType": "worker",
        "subComponentType": role,
        "replicas": replicas,
        "extraPodMetadata": {"labels": _release_pod_labels(options)},
        "sharedMemory": {"size": "64Gi"},
        "resources": {
            "requests": {"gpu": str(config.deployment.gpus_per_node)},
            "limits": {"gpu": str(config.deployment.gpus_per_node)},
        },
        "extraPodSpec": pod_spec,
    }


def _add_pvc(resource: dict[str, Any], service: dict[str, Any], name: str | None, mount_point: str) -> None:
    if not name:
        return
    pvcs = resource["spec"].setdefault("pvcs", [])
    if not any(pvc["name"] == name for pvc in pvcs):
        pvcs.append({"name": name, "create": False})
    # Keep PodSpec projection as the single mount source. Older alpha operators
    # do not realize service.volumeMounts, while current alpha-to-beta conversion
    # appends those mounts to extraPodSpec and would otherwise create duplicates.
    pod_spec = service["extraPodSpec"]
    container_mount = {"name": name, "mountPath": mount_point}
    container_mounts = pod_spec["mainContainer"].setdefault("volumeMounts", [])
    if container_mount not in container_mounts:
        if any(mount["mountPath"] == mount_point for mount in container_mounts):
            raise ValueError(f"PVC {name!r} conflicts with an existing container mount at {mount_point!r}")
        container_mounts.append(container_mount)

    pod_volume = {
        "name": name,
        "persistentVolumeClaim": {"claimName": name},
    }
    pod_volumes = pod_spec.setdefault("volumes", [])
    if pod_volume not in pod_volumes:
        if any(volume["name"] == name for volume in pod_volumes):
            raise ValueError(f"PVC {name!r} conflicts with an existing pod volume")
        pod_volumes.append(pod_volume)


def _validate_dgd_environment(config: InferenceConfig) -> None:
    environment_sources = [("global", config.env_vars)]
    if config.deployment.type == "disaggregated":
        environment_sources.extend(
            [
                ("prefill", config.deployment.prefill_env_vars),
                ("decode", config.deployment.decode_env_vars),
            ]
        )
    for source, environment in environment_sources:
        conflicts = sorted(
            key
            for key in environment
            if key in _DGD_RESERVED_ENV_KEYS or any(key.startswith(prefix) for prefix in _DGD_RESERVED_ENV_PREFIXES)
        )
        if conflicts:
            raise ValueError(f"{source} env_vars contains {conflicts}; these DGD keys are operator-owned")


def _validate_typed_credentials(
    config: InferenceConfig,
    options: DynamoGraphRenderOptions,
) -> None:
    environment_sources = [("global", config.env_vars)]
    if config.deployment.type == "disaggregated":
        environment_sources.extend(
            [
                ("prefill", config.deployment.prefill_env_vars),
                ("decode", config.deployment.decode_env_vars),
            ]
        )
    for source, environment in environment_sources:
        credential_conflicts = sorted(
            key for key in environment if any(pattern.search(key.upper()) for pattern in _CREDENTIAL_ENV_KEY_PATTERNS)
        )
        if credential_conflicts:
            raise ValueError(
                f"{source} env_vars contains raw credentials {credential_conflicts}; "
                "use a typed Kubernetes SecretKeyRef instead"
            )
        hf_home = environment.get("HF_HOME")
        if options.model_cache_pvc and hf_home is not None and hf_home != _MODEL_CACHE_MOUNT_PATH:
            raise ValueError(
                f"{source} env_vars sets HF_HOME={hf_home!r}, but the typed model cache mount "
                f"requires {_MODEL_CACHE_MOUNT_PATH!r}"
            )


def build_dgd_values(config: InferenceConfig, options: DynamoGraphRenderOptions) -> dict[str, Any]:
    if config.backend.type != "dynamo" or config.deployment.type != "disaggregated":
        raise ValueError("DGD rendering requires a Dynamo disaggregated inference config")
    deployment: DisaggregatedInferenceDeploymentConfig = config.deployment
    if deployment.num_prefill_nodes != deployment.num_prefill_replicas:
        raise ValueError("DGD rendering currently requires one pod per prefill replica")
    if deployment.num_decode_nodes != deployment.num_decode_replicas:
        raise ValueError("DGD rendering currently requires one pod per decode replica")
    if config.weight_broadcast.type == "filesystem" and not options.shared_pvc:
        raise ValueError("Dynamo filesystem weight broadcast requires a shared existing PVC")
    _validate_dgd_environment(config)
    _validate_typed_credentials(config, options)

    engine_paths = write_role_engine_configs(config, options.output_dir)
    prefill_text = engine_paths["prefill"].read_text()
    decode_text = engine_paths["decode"].read_text()
    engine_data = {
        "prefill-engine.json": prefill_text,
        "decode-engine.json": decode_text,
    }
    chat_template_content = resolve_chat_template_content(config)
    if chat_template_content is not None:
        engine_data[CHAT_TEMPLATE_ASSET] = chat_template_content
    engine_canonical = _canonical_json(engine_data)
    engine_hash = _sha256_bytes(engine_canonical)
    config_map_name = f"{options.release_name}-dynamo-engine-{engine_hash[:12]}"
    runtime_chat_template_path = (
        Path(ENGINE_MOUNT_PATH) / CHAT_TEMPLATE_ASSET if chat_template_content is not None else None
    )
    frontend_process = build_frontend_process(
        config,
        host="0.0.0.0",
        port=8000,
        runtime_chat_template_path=runtime_chat_template_path,
    )
    frontend_container = {
        "image": options.image,
        "imagePullPolicy": "IfNotPresent",
        "command": ["python3", "-m", frontend_process.module],
        "args": list(frontend_process.arguments),
        "env": [{"name": name, "value": value} for name, value in sorted(frontend_process.environment().items())],
        "ports": [
            {"containerPort": 8000, "name": "http"},
            {"containerPort": 8001, "name": "rl"},
        ],
    }
    frontend_pod_spec = {
        **options.gpu_scheduling.image_placement,
        "mainContainer": frontend_container,
    }
    if chat_template_content is not None:
        frontend_container["volumeMounts"] = [
            {
                "name": "dynamo-chat-template",
                "mountPath": ENGINE_MOUNT_PATH,
                "readOnly": True,
            }
        ]
        frontend_pod_spec["volumes"] = [
            {
                "name": "dynamo-chat-template",
                "configMap": {
                    "name": config_map_name,
                    "items": [{"key": CHAT_TEMPLATE_ASSET, "path": CHAT_TEMPLATE_ASSET}],
                },
            }
        ]
    _apply_pod_credentials(frontend_pod_spec, frontend_container, options)
    frontend = {
        "componentType": "frontend",
        "replicas": 1,
        "extraPodMetadata": {"labels": _release_pod_labels(options)},
        "extraPodSpec": frontend_pod_spec,
    }
    prefill = _worker_service(
        config,
        options,
        role="prefill",
        replicas=deployment.num_prefill_replicas,
        config_map_name=config_map_name,
        engine_file="prefill-engine.json",
    )
    decode = _worker_service(
        config,
        options,
        role="decode",
        replicas=deployment.num_decode_replicas,
        config_map_name=config_map_name,
        engine_file="decode-engine.json",
    )
    client_topology = {
        "schema_version": 1,
        "admin_api": "dynamo",
        "base_url": [f"http://{options.release_name}-frontend.{options.namespace}.svc.cluster.local:8000/v1"],
        "rl_base_url": [f"http://{options.release_name}-frontend-rl.{options.namespace}.svc.cluster.local:8001"],
        "dynamo_worker_roles": list(config.dynamo_worker_roles),
        "dynamo_gpus_per_worker": config.dynamo_gpus_per_worker,
    }
    worker_services = {
        "VllmDecodeWorker": decode,
        "VllmPrefillWorker": prefill,
    }
    topology_binding = {
        "clientTopology": client_topology,
        "workerServices": _worker_topology_binding(worker_services),
    }
    topology_canonical = _canonical_json(topology_binding)
    topology_hash = _sha256_bytes(topology_canonical)
    controller_mode = "external" if options.external_controller else "chartManaged"
    workload, chart_values = build_chart_runtime_contract(
        controller_mode=controller_mode,
        image_reference=options.image,
        image_pull_secrets=options.image_pull_secrets,
        orchestrator_replicas=options.orchestrator_replicas,
        trainer_replicas=options.trainer_replicas,
        orchestrator_command=options.orchestrator_command,
        trainer_command=options.trainer_command,
        trainer_gpu_count=options.trainer_gpu_count,
        orchestrator_placement=options.gpu_scheduling.image_placement,
        trainer_placement=options.gpu_scheduling.gpu_placement,
        shared_pvc=options.shared_pvc,
        model_cache_pvc=options.model_cache_pvc,
        hf_token_secret=options.hf_token_secret,
    )
    workload_canonical = _canonical_json(workload)
    workload_hash = _sha256_bytes(workload_canonical)
    annotations = {
        ENGINE_CONFIG_HASH_ANNOTATION: engine_hash,
        "prime-rl.nvidia.com/dynamo-sha": options.dynamo_sha,
        "prime-rl.nvidia.com/image-digest": options.image_digest,
        "prime-rl.nvidia.com/prime-sha": options.prime_sha,
        "prime-rl.nvidia.com/run-name": options.run_name,
        MANIFEST_HASH_SCOPE_ANNOTATION: MANIFEST_HASH_SCOPE,
        TOPOLOGY_HASH_ANNOTATION: topology_hash,
        WORKLOAD_HASH_ANNOTATION: workload_hash,
    }
    resource: dict[str, Any] = {
        "apiVersion": "nvidia.com/v1alpha1",
        "kind": "DynamoGraphDeployment",
        "metadata": {
            "name": options.release_name,
            "namespace": options.namespace,
            "annotations": annotations,
        },
        "spec": {
            "backendFramework": "vllm",
            "services": {
                "Frontend": frontend,
                "VllmDecodeWorker": decode,
                "VllmPrefillWorker": prefill,
            },
        },
    }
    for service in (frontend, prefill, decode):
        _add_pvc(resource, service, options.model_cache_pvc, "/model-cache")
    if config.weight_broadcast.type == "filesystem":
        for service in (prefill, decode):
            _add_pvc(resource, service, options.shared_pvc, "/data")

    manifest_canonical = _resource_manifest_canonical(resource)
    manifest_hash = _sha256_bytes(manifest_canonical)
    annotations = {**annotations, MANIFEST_HASH_ANNOTATION: manifest_hash}
    resource = {
        **resource,
        "metadata": {
            **resource["metadata"],
            "annotations": annotations,
        },
    }
    values: dict[str, Any] = {
        "namespace": options.namespace,
        **chart_values,
        "inference": {
            "enabled": True,
            "mode": "dynamoGraph",
            "dynamoGraph": {
                "controllerMode": controller_mode,
                "clientTopology": client_topology,
                "engineConfig": {
                    "name": config_map_name,
                    "sha256": engine_hash,
                    "canonicalData": engine_canonical.decode(),
                    "annotations": annotations,
                    "data": engine_data,
                },
                "topologyBinding": {
                    "sha256": topology_hash,
                    "canonical": topology_canonical.decode(),
                },
                "workloadBinding": {
                    "sha256": workload_hash,
                    "canonical": workload_canonical.decode(),
                },
                "manifestCanonical": manifest_canonical.decode(),
                "resource": resource,
            },
        },
    }
    return values


def write_dgd_artifacts(config: InferenceConfig, options: DynamoGraphRenderOptions) -> dict[str, Path]:
    options.output_dir.mkdir(parents=True, exist_ok=True)
    values = build_dgd_values(config, options)
    resource = values["inference"]["dynamoGraph"]["resource"]
    paths = {
        "values": options.output_dir / "dynamo-helm-values.json",
        "resource": options.output_dir / "dynamo-graph-deployment.json",
    }
    paths["values"].write_bytes(_canonical_json(values))
    paths["resource"].write_bytes(_canonical_json(resource))
    manifest_entries = []
    for path in sorted(options.output_dir.glob("*.json")):
        manifest_entries.append(f"{_sha256_bytes(path.read_bytes())}  {path.name}")
    manifest = options.output_dir / "artifact-manifest.sha256"
    manifest.write_text("\n".join(manifest_entries) + "\n")
    paths["manifest"] = manifest
    return paths


def _parse_args():
    from prime_rl.inference.dgd_cli import parse_args

    return parse_args()


def main() -> None:
    from prime_rl.inference.dgd_cli import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
