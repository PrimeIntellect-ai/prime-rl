import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from prime_rl.configs.inference import InferenceConfig
from prime_rl.inference.dgd import DynamoGraphRenderOptions, GPUSchedulingProfile

HELM = shutil.which("helm")
CHART = Path(__file__).parents[3] / "k8s" / "prime-rl"
PRIME_SHA = "1" * 40
DYNAMO_SHA = "2" * 40
IMAGE_DIGEST = f"sha256:{'3' * 64}"
ORCHESTRATOR_COMMAND = "uv run orchestrator @ /app/configs/debug/orch.toml --output-dir /data/outputs"
TRAINER_COMMAND = "uv run trainer @ /app/configs/debug/rl/train.toml --output-dir /data/outputs"
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
    model = {"chat_template": chat_template} if chat_template is not None else {}
    return InferenceConfig.model_validate(
        {
            "backend": {"type": "dynamo"},
            "model": model,
            "weight_broadcast": {"type": weight_broadcast},
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


def render_options(
    tmp_path: Path,
    *,
    external_controller: bool = False,
    release_name: str = "p4-math",
    shared_pvc: str | None = None,
    trainer_gpu_count: int = 1,
) -> DynamoGraphRenderOptions:
    return DynamoGraphRenderOptions(
        release_name=release_name,
        namespace="bis-vllm",
        image=f"nvcr.io/example/prime:prime-{PRIME_SHA[:12]}-dynamo-{DYNAMO_SHA[:12]}@{IMAGE_DIGEST}",
        output_dir=tmp_path,
        prime_sha=PRIME_SHA,
        dynamo_sha=DYNAMO_SHA,
        image_digest=IMAGE_DIGEST,
        run_name="p4-run",
        gpu_scheduling=GPU_SCHEDULING,
        external_controller=external_controller,
        trainer_gpu_count=trainer_gpu_count,
        orchestrator_command=None if external_controller else ORCHESTRATOR_COMMAND,
        trainer_command=None if external_controller else TRAINER_COMMAND,
        model_cache_pvc="model-cache",
        hf_token_secret="hf-token-secret",
        shared_pvc=shared_pvc,
        image_pull_secrets=("nvcrimagepullsecret",),
    )


def helm_template(
    *args: str,
    release_name: str = "p4-math",
    release_namespace: str = "bis-vllm",
) -> str:
    if HELM is None:
        pytest.skip("helm is not installed")
    return subprocess.run(
        [HELM, "template", release_name, str(CHART), "--namespace", release_namespace, *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def rendered_documents(rendered: str) -> list[dict]:
    return [document for document in yaml.safe_load_all(rendered) if document]


def rendered_resource(rendered: str, kind: str, name: str) -> dict:
    return next(
        document
        for document in rendered_documents(rendered)
        if document.get("kind") == kind and document.get("metadata", {}).get("name") == name
    )


def labels_match(selector: dict[str, str], labels: dict[str, str]) -> bool:
    return selector.items() <= labels.items()


def toleration_identity(toleration: dict[str, str]) -> tuple[str, str, str | None, str | None]:
    return (
        toleration["key"],
        toleration["operator"],
        toleration.get("value"),
        toleration.get("effect"),
    )


def write_values_mutation(
    source: Path,
    target: Path,
    path: tuple[str, ...],
    replacement: object,
) -> None:
    values = json.loads(source.read_text())
    parent = values
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = replacement
    target.write_text(json.dumps(values))


def canonical_json(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def rewrite_valid_integrity(values: dict, workload: dict) -> None:
    graph = values["inference"]["dynamoGraph"]
    workload_canonical = canonical_json(workload)
    workload_hash = hashlib.sha256(workload_canonical.encode()).hexdigest()
    graph["workloadBinding"] = {
        "canonical": workload_canonical,
        "sha256": workload_hash,
    }

    resource = graph["resource"]
    resource_annotations = resource["metadata"]["annotations"]
    resource_annotations["prime-rl.nvidia.com/workload-sha256"] = workload_hash
    graph["engineConfig"]["annotations"]["prime-rl.nvidia.com/workload-sha256"] = workload_hash

    scoped_resource = json.loads(json.dumps(resource))
    scoped_resource["metadata"]["annotations"].pop("prime-rl.nvidia.com/manifest-sha256")
    manifest_canonical = canonical_json(scoped_resource)
    manifest_hash = hashlib.sha256(manifest_canonical.encode()).hexdigest()
    graph["manifestCanonical"] = manifest_canonical
    resource_annotations["prime-rl.nvidia.com/manifest-sha256"] = manifest_hash
    graph["engineConfig"]["annotations"]["prime-rl.nvidia.com/manifest-sha256"] = manifest_hash
