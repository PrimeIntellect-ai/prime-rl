"""Compile a Prime Dynamo inference config into Helm DGD values."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from prime_rl.configs.inference import DisaggregatedInferenceDeploymentConfig, InferenceConfig
from prime_rl.inference.dynamo import (
    CHAT_TEMPLATE_ASSET,
    DynamoProcessSpec,
    build_frontend_process,
    build_worker_process,
    resolve_chat_template_content,
    write_role_engine_configs,
)
from prime_rl.utils.config import cli

ENGINE_MOUNT_PATH = "/etc/prime-rl/dynamo"
MANIFEST_HASH_ANNOTATION = "prime-rl.nvidia.com/manifest-sha256"
MANIFEST_HASH_SCOPE_ANNOTATION = "prime-rl.nvidia.com/manifest-sha256-scope"
MANIFEST_HASH_SCOPE = (
    "resource; json.dumps(sort_keys=true,indent=2)+newline; "
    "exclude=/metadata/annotations/prime-rl.nvidia.com~1manifest-sha256"
)
_DGD_RESERVED_ENV_KEYS = frozenset(
    {
        "DYN_COMPONENT",
        "DYN_DISCOVERY_BACKEND",
        "DYN_ENABLE_RL",
        "DYN_ENDPOINT",
        "DYN_ETCD_ENDPOINTS",
        "DYN_EVENT_PLANE",
        "DYN_FILE_KV",
        "DYN_NAMESPACE",
        "DYN_RL_ENDPOINT",
        "DYN_RL_PORT",
        "DYN_SYSTEM_PORT",
        "VLLM_NIXL_SIDE_CHANNEL_HOST",
        "VLLM_NIXL_SIDE_CHANNEL_PORT",
    }
)


@dataclass(frozen=True, slots=True)
class KubernetesToleration:
    key: str
    operator: Literal["Exists", "Equal"] = "Exists"
    effect: Literal["NoSchedule", "PreferNoSchedule", "NoExecute"] | None = "NoSchedule"
    value: str | None = None

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("Kubernetes toleration key must not be empty")
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


@dataclass(frozen=True, slots=True)
class GPUSchedulingProfile:
    """One image-compatible placement contract for every generated GPU pod."""

    runtime_class_name: str
    architecture: str
    product: str
    node_pool: str
    node_pool_label: str = "cloud.google.com/gke-nodepool"
    tolerations: tuple[KubernetesToleration, ...] = (KubernetesToleration(key="nvidia.com/gpu"),)

    def __post_init__(self) -> None:
        required = {
            "runtime_class_name": self.runtime_class_name,
            "architecture": self.architecture,
            "product": self.product,
            "node_pool": self.node_pool,
            "node_pool_label": self.node_pool_label,
        }
        empty = [name for name, value in required.items() if not value]
        if empty:
            raise ValueError(f"GPU scheduling fields must not be empty: {empty}")
        if not self.tolerations:
            raise ValueError("GPU scheduling requires at least one toleration")

    @property
    def node_selector(self) -> dict[str, str]:
        return {
            "kubernetes.io/arch": self.architecture,
            "nvidia.com/gpu.product": self.product,
            self.node_pool_label: self.node_pool,
        }

    @property
    def toleration_manifests(self) -> list[dict[str, str]]:
        return [toleration.as_manifest() for toleration in self.tolerations]


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
    model_cache_pvc: str | None = None
    shared_pvc: str | None = None
    image_pull_secrets: tuple[str, ...] = ()
    hf_token_secret: str | None = None

    def __post_init__(self) -> None:
        for name, value in (("prime_sha", self.prime_sha), ("dynamo_sha", self.dynamo_sha)):
            if re.fullmatch(r"[0-9a-f]{40}", value) is None:
                raise ValueError(f"{name} must be a full 40-character Git commit SHA")
        if re.fullmatch(r"sha256:[0-9a-f]{64}", self.image_digest) is None:
            raise ValueError("image_digest must be a full sha256 digest")
        if not self.image.endswith(f"@{self.image_digest}"):
            raise ValueError("DGD image must be pinned to image_digest")
        image_tag = self.image.rsplit("@", 1)[0]
        if self.prime_sha[:12] not in image_tag or self.dynamo_sha[:12] not in image_tag:
            raise ValueError("DGD image tag must include the Prime and Dynamo commit suffixes")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _resource_manifest_hash(resource: dict[str, Any]) -> str:
    annotations = resource["metadata"]["annotations"]
    scoped_annotations = {key: value for key, value in annotations.items() if key != MANIFEST_HASH_ANNOTATION}
    scoped_resource = {
        **resource,
        "metadata": {
            **resource["metadata"],
            "annotations": scoped_annotations,
        },
    }
    return _sha256_bytes(_canonical_json(scoped_resource))


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
                        "optional": True,
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
        "runtimeClassName": options.gpu_scheduling.runtime_class_name,
        "nodeSelector": options.gpu_scheduling.node_selector,
        "tolerations": options.gpu_scheduling.toleration_manifests,
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
    service.setdefault("volumeMounts", []).append({"name": name, "mountPoint": mount_point})


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
        conflicts = sorted(_DGD_RESERVED_ENV_KEYS & environment.keys())
        if conflicts:
            raise ValueError(f"{source} env_vars contains {conflicts}; these DGD keys are operator-owned")


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
    engine_hash = _sha256_bytes(_canonical_json(engine_data))
    config_map_name = f"{options.release_name}-dynamo-engine-{engine_hash[:12]}"

    annotations = {
        "prime-rl.nvidia.com/config-sha256": engine_hash,
        "prime-rl.nvidia.com/dynamo-sha": options.dynamo_sha,
        "prime-rl.nvidia.com/image-digest": options.image_digest,
        "prime-rl.nvidia.com/prime-sha": options.prime_sha,
        "prime-rl.nvidia.com/run-name": options.run_name,
        MANIFEST_HASH_SCOPE_ANNOTATION: MANIFEST_HASH_SCOPE,
    }
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
        "ports": [{"containerPort": 8001, "name": "rl"}],
    }
    frontend_pod_spec = {"mainContainer": frontend_container}
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

    manifest_hash = _resource_manifest_hash(resource)
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
        "image": {
            "reference": options.image,
            "pullPolicy": "IfNotPresent",
            "pullSecrets": list(options.image_pull_secrets),
        },
        "inference": {
            "mode": "dynamoGraph",
            "dynamoGraph": {
                "clientTopology": {
                    "schema_version": 1,
                    "admin_api": "dynamo",
                    "base_url": [
                        f"http://{options.release_name}-frontend.{options.namespace}.svc.cluster.local:8000/v1"
                    ],
                    "rl_base_url": [
                        f"http://{options.release_name}-frontend-rl.{options.namespace}.svc.cluster.local:8001"
                    ],
                    "dynamo_worker_roles": list(config.dynamo_worker_roles),
                    "dynamo_gpus_per_worker": config.dynamo_gpus_per_worker,
                },
                "engineConfig": {
                    "name": config_map_name,
                    "sha256": engine_hash,
                    "annotations": annotations,
                    "data": engine_data,
                },
                "resource": resource,
            },
        },
        "trainer": {
            "runtimeClassName": options.gpu_scheduling.runtime_class_name,
            "nodeSelector": options.gpu_scheduling.node_selector,
            "tolerations": options.gpu_scheduling.toleration_manifests,
        },
    }
    if options.model_cache_pvc:
        values["modelCache"] = {
            "enabled": True,
            "existingClaim": options.model_cache_pvc,
            "mountPath": "/model-cache",
        }
    if options.hf_token_secret:
        values["huggingFace"] = {
            "tokenSecretName": options.hf_token_secret,
            "tokenSecretKey": "HF_TOKEN",
        }
    if options.shared_pvc:
        values["storage"] = {
            "enabled": True,
            "existingClaim": options.shared_pvc,
            "mountPath": "/data",
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inference_config", type=Path)
    parser.add_argument("--release-name", required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prime-sha", required=True)
    parser.add_argument("--dynamo-sha", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--run-name")
    parser.add_argument("--gpu-runtime-class", default="nvidia")
    parser.add_argument("--gpu-architecture", required=True)
    parser.add_argument("--gpu-product", required=True)
    parser.add_argument("--gpu-node-pool", required=True)
    parser.add_argument("--gpu-node-pool-label", default="cloud.google.com/gke-nodepool")
    parser.add_argument("--model-cache-pvc")
    parser.add_argument("--shared-pvc")
    parser.add_argument("--image-pull-secret", action="append", default=[])
    parser.add_argument("--hf-token-secret")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = cli(InferenceConfig, args=["@", str(args.inference_config)])
    options = DynamoGraphRenderOptions(
        release_name=args.release_name,
        namespace=args.namespace,
        image=args.image,
        output_dir=args.output_dir,
        prime_sha=args.prime_sha,
        dynamo_sha=args.dynamo_sha,
        image_digest=args.image_digest,
        run_name=args.run_name or args.release_name,
        gpu_scheduling=GPUSchedulingProfile(
            runtime_class_name=args.gpu_runtime_class,
            architecture=args.gpu_architecture,
            product=args.gpu_product,
            node_pool=args.gpu_node_pool,
            node_pool_label=args.gpu_node_pool_label,
        ),
        model_cache_pvc=args.model_cache_pvc,
        shared_pvc=args.shared_pvc,
        image_pull_secrets=tuple(args.image_pull_secret),
        hf_token_secret=args.hf_token_secret,
    )
    for path in write_dgd_artifacts(config, options).values():
        print(path)


if __name__ == "__main__":
    main()
