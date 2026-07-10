import hashlib
import json
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from prime_rl.configs.inference import InferenceConfig
from prime_rl.inference.dgd import (
    DynamoGraphRenderOptions,
    GPUSchedulingProfile,
    KubernetesToleration,
    _parse_args,
    _parse_kubernetes_toleration,
    build_dgd_values,
    write_dgd_artifacts,
)
from prime_rl.inference.dynamo import build_frontend_process, build_worker_process

PRIME_SHA = "1" * 40
DYNAMO_SHA = "2" * 40
IMAGE_DIGEST = f"sha256:{'3' * 64}"
GPU_SCHEDULING = GPUSchedulingProfile(
    runtime_class_name="nvidia",
    architecture="arm64",
    product="NVIDIA-GB200",
    node_pool="customer-gpu-o7v",
)


def inference_config(
    weight_broadcast: str = "nccl",
    *,
    chat_template: str | None = None,
) -> InferenceConfig:
    model = {"name": "Qwen/Qwen3-30B-A3B-Thinking-2507"}
    if chat_template is not None:
        model["chat_template"] = chat_template
    return InferenceConfig.model_validate(
        {
            "backend": {"type": "dynamo"},
            "model": model,
            "parallel": {"tp": 1},
            "weight_broadcast": {"type": weight_broadcast},
            "env_vars": {"HF_HOME": "/model-cache", "HF_HUB_OFFLINE": "1"},
            "deployment": {
                "type": "disaggregated",
                "gpus_per_node": 1,
                "prefill_nodes_per_replica": 1,
                "decode_nodes_per_replica": 1,
                "num_prefill_replicas": 2,
                "num_decode_replicas": 2,
            },
        }
    )


def render_options(tmp_path: Path) -> DynamoGraphRenderOptions:
    return DynamoGraphRenderOptions(
        release_name="p4-math",
        namespace="bis-vllm",
        image=f"nvcr.io/example/prime:prime-{PRIME_SHA[:12]}-dynamo-{DYNAMO_SHA[:12]}@{IMAGE_DIGEST}",
        output_dir=tmp_path,
        prime_sha=PRIME_SHA,
        dynamo_sha=DYNAMO_SHA,
        image_digest=IMAGE_DIGEST,
        run_name="p4-run",
        gpu_scheduling=GPU_SCHEDULING,
        model_cache_pvc="model-cache",
        shared_pvc="p4-shared-data",
        image_pull_secrets=("nvcrimagepullsecret",),
        hf_token_secret="hf-token-secret",
    )


def test_dgd_values_derive_topology_and_role_configs(tmp_path: Path):
    options = render_options(tmp_path)
    values = build_dgd_values(inference_config(), options)
    graph = values["inference"]["dynamoGraph"]
    resource = graph["resource"]
    services = resource["spec"]["services"]

    assert values["image"] == {
        "reference": options.image,
        "pullPolicy": "IfNotPresent",
        "pullSecrets": ["nvcrimagepullsecret"],
    }
    assert resource["apiVersion"] == "nvidia.com/v1alpha1"
    assert resource["metadata"]["namespace"] == "bis-vllm"
    assert services["VllmPrefillWorker"]["replicas"] == 2
    assert services["VllmDecodeWorker"]["replicas"] == 2
    assert services["VllmPrefillWorker"]["resources"]["limits"]["gpu"] == "1"
    assert services["VllmDecodeWorker"]["resources"]["limits"]["gpu"] == "1"
    assert resource["spec"]["pvcs"] == [
        {"name": "model-cache", "create": False},
    ]
    assert values["storage"] == {
        "enabled": True,
        "existingClaim": "p4-shared-data",
        "mountPath": "/data",
    }
    assert values["modelCache"] == {
        "enabled": True,
        "existingClaim": "model-cache",
        "mountPath": "/model-cache",
    }
    assert values["huggingFace"] == {
        "tokenSecretName": "hf-token-secret",
        "tokenSecretKey": "HF_TOKEN",
    }
    assert values["inference"]["dynamoGraph"]["clientTopology"] == {
        "schema_version": 1,
        "admin_api": "dynamo",
        "base_url": ["http://p4-math-frontend.bis-vllm.svc.cluster.local:8000/v1"],
        "rl_base_url": ["http://p4-math-frontend-rl.bis-vllm.svc.cluster.local:8001"],
        "dynamo_worker_roles": ["prefill", "prefill", "decode", "decode"],
        "dynamo_gpus_per_worker": 1,
    }
    for service in services.values():
        pod_spec = service["extraPodSpec"]
        assert pod_spec["mainContainer"]["image"] == options.image
        assert pod_spec["imagePullSecrets"] == [{"name": "nvcrimagepullsecret"}]
        assert any(item["name"] == "HF_TOKEN" for item in pod_spec["mainContainer"]["env"])
        env = {item["name"]: item.get("value") for item in pod_spec["mainContainer"]["env"]}
        assert env["HF_HOME"] == "/model-cache"
        assert env["HF_HUB_OFFLINE"] == "1"

    image_tolerations = [
        {
            "key": "kubernetes.io/arch",
            "operator": "Equal",
            "value": "arm64",
            "effect": "NoSchedule",
        },
        {
            "key": "prime-rl",
            "operator": "Equal",
            "value": "true",
            "effect": "NoSchedule",
        },
    ]
    gpu_tolerations = [
        *image_tolerations,
        {"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"},
    ]
    image_selector = {
        "kubernetes.io/arch": "arm64",
        "cloud.google.com/gke-nodepool": "customer-gpu-o7v",
    }
    gpu_selector = {**image_selector, "nvidia.com/gpu.product": "NVIDIA-GB200"}
    frontend_pod = services["Frontend"]["extraPodSpec"]
    assert frontend_pod["nodeSelector"] == image_selector
    assert frontend_pod["tolerations"] == image_tolerations
    assert "runtimeClassName" not in frontend_pod
    assert frontend_pod["mainContainer"]["ports"] == [
        {"containerPort": 8000, "name": "http"},
        {"containerPort": 8001, "name": "rl"},
    ]
    for role in ("VllmPrefillWorker", "VllmDecodeWorker"):
        worker_pod = services[role]["extraPodSpec"]
        assert worker_pod["runtimeClassName"] == "nvidia"
        assert worker_pod["nodeSelector"] == gpu_selector
        assert worker_pod["tolerations"] == gpu_tolerations
    assert values["orchestrator"] == {
        "nodeSelector": image_selector,
        "tolerations": image_tolerations,
    }
    assert values["trainer"]["runtimeClassName"] == "nvidia"
    assert values["trainer"]["nodeSelector"] == gpu_selector
    assert values["trainer"]["tolerations"] == gpu_tolerations

    prefill_args = services["VllmPrefillWorker"]["extraPodSpec"]["mainContainer"]["args"]
    decode_args = services["VllmDecodeWorker"]["extraPodSpec"]["mainContainer"]["args"]
    prefill_process = build_worker_process(
        inference_config(),
        "prefill",
        Path("/etc/prime-rl/dynamo/prefill-engine.json"),
        nixl_host=None,
        nixl_port=20100,
    )
    decode_process = build_worker_process(
        inference_config(),
        "decode",
        Path("/etc/prime-rl/dynamo/decode-engine.json"),
        nixl_host=None,
        nixl_port=20100,
    )
    frontend_process = build_frontend_process(inference_config(), host="0.0.0.0", port=8000)
    assert prefill_args == list(prefill_process.arguments)
    assert decode_args == list(decode_process.arguments)
    assert services["Frontend"]["extraPodSpec"]["mainContainer"]["args"] == list(frontend_process.arguments)

    prefill_env = {
        item["name"]: item.get("value")
        for item in services["VllmPrefillWorker"]["extraPodSpec"]["mainContainer"]["env"]
    }
    frontend_env = {
        item["name"]: item.get("value") for item in services["Frontend"]["extraPodSpec"]["mainContainer"]["env"]
    }
    assert {key: prefill_env[key] for key in prefill_process.environment()} == prefill_process.environment()
    assert {key: frontend_env[key] for key in frontend_process.environment()} == frontend_process.environment()

    assert {mount["name"] for mount in services["Frontend"]["volumeMounts"]} == {"model-cache"}
    for role in ("VllmPrefillWorker", "VllmDecodeWorker"):
        assert not any(mount["name"] == "p4-shared-data" for mount in services[role]["volumeMounts"])

    prefill = json.loads(graph["engineConfig"]["data"]["prefill-engine.json"])
    decode = json.loads(graph["engineConfig"]["data"]["decode-engine.json"])
    assert prefill["kv_transfer_config"] == decode["kv_transfer_config"]
    assert "kv_events_config" in prefill
    assert "kv_events_config" not in decode
    assert "disaggregation_mode" not in prefill
    assert "enable_rl" not in decode


def test_dgd_artifacts_are_deterministic_and_manifest_verifies(tmp_path: Path):
    paths = write_dgd_artifacts(inference_config(), render_options(tmp_path))
    first_values = paths["values"].read_bytes()
    write_dgd_artifacts(inference_config(), render_options(tmp_path))
    assert paths["values"].read_bytes() == first_values

    for line in paths["manifest"].read_text().splitlines():
        expected, name = line.split("  ", 1)
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == expected

    resource = json.loads(paths["resource"].read_text())
    annotations = resource["metadata"]["annotations"]
    assert annotations["prime-rl.nvidia.com/manifest-sha256-scope"] == (
        "resource; json.dumps(sort_keys=true,indent=2)+newline; "
        "exclude=/metadata/annotations/prime-rl.nvidia.com~1manifest-sha256"
    )
    expected_manifest_hash = annotations["prime-rl.nvidia.com/manifest-sha256"]
    unhashed_resource = deepcopy(resource)
    del unhashed_resource["metadata"]["annotations"]["prime-rl.nvidia.com/manifest-sha256"]
    canonical_resource = (json.dumps(unhashed_resource, indent=2, sort_keys=True) + "\n").encode()
    assert hashlib.sha256(canonical_resource).hexdigest() == expected_manifest_hash


def test_dgd_embeds_and_mounts_content_addressed_chat_template(tmp_path: Path):
    first = build_dgd_values(
        inference_config(chat_template="template-v1: {{ messages }}"),
        render_options(tmp_path / "first"),
    )
    graph = first["inference"]["dynamoGraph"]
    engine_config = graph["engineConfig"]
    frontend = graph["resource"]["spec"]["services"]["Frontend"]["extraPodSpec"]

    assert engine_config["data"]["chat-template.jinja"] == "template-v1: {{ messages }}"
    expected_hash = hashlib.sha256(
        (json.dumps(engine_config["data"], indent=2, sort_keys=True) + "\n").encode()
    ).hexdigest()
    assert engine_config["sha256"] == expected_hash
    assert engine_config["name"].endswith(expected_hash[:12])
    assert frontend["mainContainer"]["args"][-1] == "/etc/prime-rl/dynamo/chat-template.jinja"
    assert frontend["mainContainer"]["volumeMounts"] == [
        {
            "name": "dynamo-chat-template",
            "mountPath": "/etc/prime-rl/dynamo",
            "readOnly": True,
        }
    ]
    assert frontend["volumes"] == [
        {
            "name": "dynamo-chat-template",
            "configMap": {
                "name": engine_config["name"],
                "items": [{"key": "chat-template.jinja", "path": "chat-template.jinja"}],
            },
        }
    ]
    assert not (tmp_path / "first" / "chat-template.jinja").exists()

    second = build_dgd_values(
        inference_config(chat_template="template-v2: {{ messages }}"),
        render_options(tmp_path / "second"),
    )
    second_config = second["inference"]["dynamoGraph"]["engineConfig"]
    assert second_config["name"] != engine_config["name"]
    assert second_config["sha256"] != engine_config["sha256"]


def test_filesystem_broadcast_requires_one_shared_existing_claim(tmp_path: Path):
    options = render_options(tmp_path)
    values = build_dgd_values(inference_config("filesystem"), options)
    resource = values["inference"]["dynamoGraph"]["resource"]
    services = resource["spec"]["services"]

    assert values["storage"] == {
        "enabled": True,
        "existingClaim": "p4-shared-data",
        "mountPath": "/data",
    }
    assert resource["spec"]["pvcs"] == [
        {"name": "model-cache", "create": False},
        {"name": "p4-shared-data", "create": False},
    ]
    assert not any(mount["name"] == "p4-shared-data" for mount in services["Frontend"]["volumeMounts"])
    for role in ("VllmPrefillWorker", "VllmDecodeWorker"):
        assert {"name": "p4-shared-data", "mountPoint": "/data"} in services[role]["volumeMounts"]


def test_filesystem_broadcast_rejects_missing_shared_claim(tmp_path: Path):
    options = replace(render_options(tmp_path), shared_pvc=None)

    with pytest.raises(ValueError, match="shared existing PVC"):
        build_dgd_values(inference_config("filesystem"), options)


@pytest.mark.parametrize(
    ("scope", "key"),
    [
        ("global", "DYN_DISCOVERY_BACKEND"),
        ("global", "DYN_NAMESPACE"),
        ("global", "DYN_NAMESPACE_PREFIX"),
        ("global", "DYN_NAMESPACE_WORKER_SUFFIX"),
        ("global", "DYN_PARENT_DGD_K8S_NAME"),
        ("global", "DYN_PARENT_DGD_K8S_NAMESPACE"),
        ("global", "DYN_KUBE_DISCOVERY_MODE"),
        ("global", "DYN_ENDPOINT_TYPES"),
        ("global", "DYN_SYSTEM_ENABLED"),
        ("global", "DYN_SYSTEM_HOST"),
        ("global", "DYN_SYSTEM_HEALTH_PATH"),
        ("global", "DYN_SYSTEM_LIVE_PATH"),
        ("global", "DYN_SYSTEM_STARTING_HEALTH_STATUS"),
        ("global", "DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS"),
        ("global", "DYN_HEALTH_CHECK_ENABLED"),
        ("global", "POD_NAME"),
        ("global", "POD_NAMESPACE"),
        ("global", "POD_UID"),
        ("global", "CONTAINER_NAME"),
        ("prefill", "DYN_SYSTEM_PORT"),
        ("prefill", "DYN_SYSTEM_PORT1"),
        ("decode", "DYN_ENDPOINT"),
    ],
)
def test_dgd_rejects_operator_owned_environment(scope: str, key: str, tmp_path: Path):
    config_data = inference_config().model_dump(mode="python")
    if scope == "global":
        config_data["env_vars"] = {key: "override"}
    else:
        config_data["deployment"][f"{scope}_env_vars"] = {key: "override"}
    config = InferenceConfig.model_validate(config_data)

    with pytest.raises(ValueError, match=rf"{scope}.*{key}.*operator-owned"):
        build_dgd_values(config, render_options(tmp_path))


@pytest.mark.parametrize("scope", ["global", "prefill", "decode"])
@pytest.mark.parametrize("key", ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"])
def test_dgd_rejects_raw_hugging_face_credentials(scope: str, key: str, tmp_path: Path):
    config_data = inference_config().model_dump(mode="python")
    if scope == "global":
        config_data["env_vars"][key] = "plaintext-secret"
    else:
        config_data["deployment"][f"{scope}_env_vars"] = {key: "plaintext-secret"}
    config = InferenceConfig.model_validate(config_data)

    with pytest.raises(ValueError, match=rf"{scope}.*{key}.*hf_token_secret"):
        build_dgd_values(config, render_options(tmp_path))


@pytest.mark.parametrize("scope", ["global", "prefill", "decode"])
def test_dgd_rejects_hf_home_that_conflicts_with_typed_model_cache(scope: str, tmp_path: Path):
    config_data = inference_config().model_dump(mode="python")
    if scope == "global":
        config_data["env_vars"]["HF_HOME"] = "/wrong-cache"
    else:
        config_data["deployment"][f"{scope}_env_vars"] = {"HF_HOME": "/wrong-cache"}
    config = InferenceConfig.model_validate(config_data)

    with pytest.raises(ValueError, match=rf"{scope}.*HF_HOME.*/model-cache"):
        build_dgd_values(config, render_options(tmp_path))


def test_dgd_allows_hf_home_matching_typed_model_cache(tmp_path: Path):
    values = build_dgd_values(inference_config(), render_options(tmp_path))
    services = values["inference"]["dynamoGraph"]["resource"]["spec"]["services"]

    for service in services.values():
        env = service["extraPodSpec"]["mainContainer"]["env"]
        assert [item for item in env if item["name"] == "HF_HOME"] == [{"name": "HF_HOME", "value": "/model-cache"}]


def test_cli_toleration_parser_is_typed_and_rejects_unknown_fields():
    parsed = _parse_kubernetes_toleration(
        '{"key":"dedicated","operator":"Equal","value":"prime","effect":"NoSchedule"}'
    )
    assert parsed == KubernetesToleration(
        key="dedicated",
        operator="Equal",
        value="prime",
        effect="NoSchedule",
    )

    with pytest.raises(ValueError, match="unsupported fields"):
        _parse_kubernetes_toleration('{"key":"dedicated","command":"touch /tmp/pwned"}')

    with pytest.raises(ValueError, match="operator"):
        _parse_kubernetes_toleration('{"key":"dedicated","operator":"NotARealOperator"}')


def test_cli_accepts_typed_additional_image_and_gpu_tolerations(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dynamo-dgd",
            "inference.toml",
            "--release-name",
            "p4-math",
            "--namespace",
            "bis-vllm",
            "--image",
            "registry/image@sha256:digest",
            "--output-dir",
            "/tmp/artifacts",
            "--prime-sha",
            PRIME_SHA,
            "--dynamo-sha",
            DYNAMO_SHA,
            "--image-digest",
            IMAGE_DIGEST,
            "--gpu-architecture",
            "arm64",
            "--gpu-product",
            "NVIDIA-GB200",
            "--gpu-node-pool",
            "customer-gpu-o7v",
            "--image-toleration",
            '{"key":"image-extra","operator":"Exists"}',
            "--gpu-toleration",
            '{"key":"gpu-extra","operator":"Equal","value":"true"}',
        ],
    )

    args = _parse_args()

    assert args.image_toleration == [KubernetesToleration(key="image-extra")]
    assert args.gpu_toleration == [KubernetesToleration(key="gpu-extra", operator="Equal", value="true")]


def test_gpu_scheduling_changes_manifest_identity(tmp_path: Path):
    first = build_dgd_values(inference_config(), render_options(tmp_path / "first"))
    changed_profile = replace(GPU_SCHEDULING, node_pool="customer-gpu-alternate")
    changed_options = replace(render_options(tmp_path / "second"), gpu_scheduling=changed_profile)
    second = build_dgd_values(inference_config(), changed_options)

    first_resource = first["inference"]["dynamoGraph"]["resource"]
    second_resource = second["inference"]["dynamoGraph"]["resource"]
    assert (
        first_resource["metadata"]["annotations"]["prime-rl.nvidia.com/manifest-sha256"]
        != (second_resource["metadata"]["annotations"]["prime-rl.nvidia.com/manifest-sha256"])
    )


def test_dgd_rejects_native_backend(tmp_path: Path):
    config = InferenceConfig.model_validate({})
    with pytest.raises(ValueError, match="Dynamo disaggregated"):
        build_dgd_values(config, render_options(tmp_path))
