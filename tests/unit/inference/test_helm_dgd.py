import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from prime_rl.configs.inference import InferenceConfig
from prime_rl.inference.dgd import DynamoGraphRenderOptions, GPUSchedulingProfile, write_dgd_artifacts

HELM = shutil.which("helm")
CHART = Path(__file__).parents[3] / "k8s" / "prime-rl"
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


def render_options(tmp_path: Path, *, shared_pvc: str | None = None) -> DynamoGraphRenderOptions:
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
        hf_token_secret="hf-token-secret",
        shared_pvc=shared_pvc,
        image_pull_secrets=("nvcrimagepullsecret",),
    )


def helm_template(*args: str, release_name: str = "p4-math") -> str:
    if HELM is None:
        pytest.skip("helm is not installed")
    return subprocess.run(
        [HELM, "template", release_name, str(CHART), *args],
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


def test_native_chart_still_renders_inference_statefulset():
    rendered = helm_template()
    assert "name: p4-math-inference\n" in rendered
    assert "kind: DynamoGraphDeployment" not in rendered
    assert "p4-math-inference-0.p4-math-inference-headless" in rendered


def test_chart_rejects_unknown_inference_mode():
    with pytest.raises(subprocess.CalledProcessError):
        helm_template("--set", "inference.mode=typo")


def test_dgd_chart_renders_generated_graph_without_inference_statefulset(tmp_path: Path):
    options = render_options(tmp_path)
    paths = write_dgd_artifacts(inference_config(), options)
    rendered = helm_template("-f", str(paths["values"]))
    graph = json.loads(paths["resource"].read_text())
    rendered_graph = rendered_resource(rendered, "DynamoGraphDeployment", "p4-math")

    assert rendered.count("kind: DynamoGraphDeployment") == 1
    assert rendered.count("kind: ConfigMap") == 1
    assert "name: p4-math-inference\n" not in rendered
    assert "name: p4-math-frontend-rl" in rendered
    assert "http://p4-math-frontend.bis-vllm.svc.cluster.local:8000/v1" in rendered
    assert "http://p4-math-frontend-rl.bis-vllm.svc.cluster.local:8001" in rendered
    assert graph["spec"]["services"]["VllmPrefillWorker"]["replicas"] == 2
    assert graph["spec"]["services"]["VllmDecodeWorker"]["replicas"] == 2
    assert rendered_graph["spec"]["services"]["Frontend"]["extraPodSpec"]["mainContainer"]["ports"] == [
        {"containerPort": 8000, "name": "http"},
        {"containerPort": 8001, "name": "rl"},
    ]
    assert rendered.count(f'image: "{options.image}"') == 2
    assert rendered.count(f"image: {options.image}") == 3
    assert rendered.count("nvcrimagepullsecret") == 5
    assert rendered.count("name: DYN_RL_TOPOLOGY") == 2
    assert rendered.count("claimName: model-cache") == 2
    assert rendered.count("name: HF_TOKEN") == 5
    assert rendered.count("name: HF_HOME") == 5
    assert rendered.count("cloud.google.com/gke-nodepool: customer-gpu-o7v") == 5
    assert rendered.count("kubernetes.io/arch: arm64") == 5
    assert rendered.count("nvidia.com/gpu.product: NVIDIA-GB200") == 3
    assert rendered.count("runtimeClassName: nvidia") == 3
    assert rendered.count("key: kubernetes.io/arch") == 5
    assert rendered.count("key: nvidia.com/gpu") == 3
    assert rendered.count("key: prime-rl") == 5
    assert not any(kind in rendered for kind in ("kind: ClusterRole", "kind: CustomResourceDefinition"))


def test_dgd_chart_renders_chat_template_configmap_and_frontend_mount(tmp_path: Path):
    paths = write_dgd_artifacts(
        inference_config(chat_template="template-marker: {{ messages }}"),
        render_options(tmp_path),
    )
    rendered = helm_template("-f", str(paths["values"]))
    values = json.loads(paths["values"].read_text())
    config_map = rendered_resource(
        rendered,
        "ConfigMap",
        values["inference"]["dynamoGraph"]["engineConfig"]["name"],
    )

    assert config_map["data"]["chat-template.jinja"] == "template-marker: {{ messages }}"
    assert "/etc/prime-rl/dynamo/chat-template.jinja" in rendered
    assert "name: dynamo-chat-template" in rendered
    assert "key: chat-template.jinja" in rendered


def test_dgd_chart_preserves_exact_content_addressed_configmap_bytes(tmp_path: Path):
    paths = write_dgd_artifacts(
        inference_config(chat_template="EXACT: {{ messages }}"),
        render_options(tmp_path),
    )
    values = json.loads(paths["values"].read_text())
    rendered = helm_template("-f", str(paths["values"]))
    config_map_name = values["inference"]["dynamoGraph"]["engineConfig"]["name"]
    config_map = rendered_resource(rendered, "ConfigMap", config_map_name)
    expected_data = values["inference"]["dynamoGraph"]["engineConfig"]["data"]

    assert config_map["data"] == expected_data
    expected_hash = values["inference"]["dynamoGraph"]["engineConfig"]["sha256"]
    canonical_data = (json.dumps(config_map["data"], indent=2, sort_keys=True) + "\n").encode()
    assert hashlib.sha256(canonical_data).hexdigest() == expected_hash


def test_dgd_chart_rejects_release_name_mismatch(tmp_path: Path):
    paths = write_dgd_artifacts(inference_config(), render_options(tmp_path))

    with pytest.raises(subprocess.CalledProcessError) as error:
        helm_template("-f", str(paths["values"]), release_name="other-release")

    assert "must match embedded DynamoGraphDeployment metadata.name" in error.value.stderr


def test_dgd_chart_rejects_namespace_mismatch(tmp_path: Path):
    paths = write_dgd_artifacts(inference_config(), render_options(tmp_path))

    with pytest.raises(subprocess.CalledProcessError) as error:
        helm_template("-f", str(paths["values"]), "--set", "namespace=other-namespace")

    assert "must match embedded DynamoGraphDeployment metadata.namespace" in error.value.stderr


@pytest.mark.parametrize(
    ("component", "name"),
    [
        ("orchestrator", "DYN_RL_TOPOLOGY"),
        ("orchestrator", "HF_TOKEN"),
        ("trainer", "HF_HOME"),
    ],
)
def test_dgd_chart_rejects_raw_env_that_overrides_typed_contract(
    component: str,
    name: str,
    tmp_path: Path,
):
    paths = write_dgd_artifacts(inference_config(), render_options(tmp_path))
    overlay = tmp_path / f"{component}-{name}.json"
    overlay.write_text(json.dumps({component: {"env": [{"name": name, "value": "override"}]}}))

    with pytest.raises(subprocess.CalledProcessError) as error:
        helm_template("-f", str(paths["values"]), "-f", str(overlay))

    assert f"{component}.env cannot override generated {name}" in error.value.stderr


def test_filesystem_broadcast_reuses_existing_claim_without_rendering_pvc(tmp_path: Path):
    paths = write_dgd_artifacts(
        inference_config("filesystem"),
        render_options(tmp_path, shared_pvc="p4-shared-data"),
    )
    rendered = helm_template("-f", str(paths["values"]))

    assert "kind: PersistentVolumeClaim" not in rendered
    assert rendered.count("claimName: p4-shared-data") == 2


def test_dgd_chart_rejects_mutable_prime_runtime_image(tmp_path: Path):
    paths = write_dgd_artifacts(inference_config(), render_options(tmp_path))

    with pytest.raises(subprocess.CalledProcessError):
        helm_template(
            "-f",
            str(paths["values"]),
            "--set",
            "image.reference=nvcr.io/example/prime:latest",
        )


def test_dgd_chart_rejects_runtime_image_that_differs_from_workers(tmp_path: Path):
    paths = write_dgd_artifacts(inference_config(), render_options(tmp_path))
    other_digest = f"sha256:{'4' * 64}"

    with pytest.raises(subprocess.CalledProcessError) as error:
        helm_template(
            "-f",
            str(paths["values"]),
            "--set",
            f"image.reference=nvcr.io/example/prime:reviewed@{other_digest}",
        )

    assert "must use the same image.reference" in error.value.stderr


def test_dgd_rejects_image_without_matching_digest(tmp_path: Path):
    with pytest.raises(ValueError, match="must be pinned"):
        DynamoGraphRenderOptions(
            release_name="p4-math",
            namespace="bis-vllm",
            image="nvcr.io/example/prime:p4",
            output_dir=tmp_path,
            prime_sha=PRIME_SHA,
            dynamo_sha=DYNAMO_SHA,
            image_digest=IMAGE_DIGEST,
            run_name="p4-run",
            gpu_scheduling=GPU_SCHEDULING,
        )


def test_dgd_rejects_image_without_commit_suffixes(tmp_path: Path):
    with pytest.raises(ValueError, match="commit suffixes"):
        DynamoGraphRenderOptions(
            release_name="p4-math",
            namespace="bis-vllm",
            image=f"nvcr.io/example/prime:p4@{IMAGE_DIGEST}",
            output_dir=tmp_path,
            prime_sha=PRIME_SHA,
            dynamo_sha=DYNAMO_SHA,
            image_digest=IMAGE_DIGEST,
            run_name="p4-run",
            gpu_scheduling=GPU_SCHEDULING,
        )
