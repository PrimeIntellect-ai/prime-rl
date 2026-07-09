import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from prime_rl.configs.inference import InferenceConfig
from prime_rl.inference.dgd import DynamoGraphRenderOptions, build_dgd_values, write_dgd_artifacts
from prime_rl.inference.dynamo import build_frontend_process, build_worker_process

PRIME_SHA = "1" * 40
DYNAMO_SHA = "2" * 40
IMAGE_DIGEST = f"sha256:{'3' * 64}"


def inference_config(weight_broadcast: str = "nccl") -> InferenceConfig:
    return InferenceConfig.model_validate(
        {
            "backend": {"type": "dynamo"},
            "model": {"name": "Qwen/Qwen3-30B-A3B-Thinking-2507"},
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
    for service in services.values():
        pod_spec = service["extraPodSpec"]
        assert pod_spec["mainContainer"]["image"] == options.image
        assert pod_spec["imagePullSecrets"] == [{"name": "nvcrimagepullsecret"}]
        assert any(item["name"] == "HF_TOKEN" for item in pod_spec["mainContainer"]["env"])
        env = {item["name"]: item.get("value") for item in pod_spec["mainContainer"]["env"]}
        assert env["HF_HOME"] == "/model-cache"
        assert env["HF_HUB_OFFLINE"] == "1"

    gpu_toleration = {"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}
    assert gpu_toleration in services["VllmPrefillWorker"]["extraPodSpec"]["tolerations"]
    assert gpu_toleration in services["VllmDecodeWorker"]["extraPodSpec"]["tolerations"]

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


def test_dgd_rejects_native_backend(tmp_path: Path):
    config = InferenceConfig.model_validate({})
    with pytest.raises(ValueError, match="Dynamo disaggregated"):
        build_dgd_values(config, render_options(tmp_path))
