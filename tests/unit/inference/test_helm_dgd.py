import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from prime_rl.inference.dgd import DynamoGraphRenderOptions, write_dgd_artifacts
from tests.unit.inference.helm_dgd_test_utils import (
    DYNAMO_SHA,
    GPU_SCHEDULING,
    IMAGE_DIGEST,
    ORCHESTRATOR_COMMAND,
    PRIME_SHA,
    TRAINER_COMMAND,
    helm_template,
    inference_config,
    labels_match,
    render_options,
    rendered_documents,
    rendered_resource,
    rewrite_valid_integrity,
    toleration_identity,
)


def test_native_chart_still_renders_inference_statefulset():
    rendered = helm_template()
    assert "name: p4-math-inference\n" in rendered
    assert "kind: DynamoGraphDeployment" not in rendered
    assert "p4-math-inference-0.p4-math-inference-headless" in rendered


def test_chart_rejects_unknown_inference_mode():
    with pytest.raises(subprocess.CalledProcessError):
        helm_template("--set", "inference.mode=typo")


def test_native_chart_rejects_invalid_image_pull_policy():
    with pytest.raises(subprocess.CalledProcessError) as error:
        helm_template("--set", "image.pullPolicy=Sometimes")

    assert "image/pullPolicy" in error.value.stderr
    assert "'Always', 'IfNotPresent', 'Never'" in error.value.stderr


def test_chart_release_name_boundary_keeps_every_resource_name_valid():
    release_name = "a" * 41
    rendered = helm_template(release_name=release_name, release_namespace="default")

    assert all(len(document["metadata"]["name"]) <= 63 for document in rendered_documents(rendered))


def test_chart_rejects_release_name_that_would_overflow_service_names():
    with pytest.raises(subprocess.CalledProcessError) as error:
        helm_template(release_name="a" * 42, release_namespace="default")

    assert "at most 41 characters" in error.value.stderr


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
    assert rendered.count("claimName: model-cache") == 5
    assert rendered.count("name: HF_TOKEN") == 5
    assert rendered.count("name: HF_HOME") == 5
    chart_pods = {
        component: rendered_resource(rendered, "StatefulSet", f"p4-math-{component}")["spec"]["template"]["spec"]
        for component in ("orchestrator", "trainer")
    }
    dgd_pods = {
        component: rendered_graph["spec"]["services"][service]["extraPodSpec"]
        for component, service in {
            "frontend": "Frontend",
            "prefill": "VllmPrefillWorker",
            "decode": "VllmDecodeWorker",
        }.items()
    }
    pods = {**chart_pods, **dgd_pods}
    image_selector = {
        "cloud.google.com/gke-nodepool": "customer-gpu-o7v",
        "kubernetes.io/arch": "arm64",
    }
    gpu_selector = {**image_selector, "nvidia.com/gpu.product": "NVIDIA-GB200"}
    required_tolerations = {
        ("kubernetes.io/arch", "Equal", "arm64", "NoSchedule"),
        ("nvidia.com/gpu", "Exists", None, "NoSchedule"),
        ("prime-rl", "Equal", "true", "NoSchedule"),
    }

    assert set(pods) == {"orchestrator", "trainer", "frontend", "prefill", "decode"}
    for component, pod in pods.items():
        assert {toleration_identity(item) for item in pod["tolerations"]} == required_tolerations
        container = pod["containers"][0] if component in chart_pods else pod["mainContainer"]
        hf_token = next(item for item in container["env"] if item["name"] == "HF_TOKEN")
        assert hf_token["valueFrom"]["secretKeyRef"]["optional"] is False
        if component in {"trainer", "prefill", "decode"}:
            assert pod["nodeSelector"] == gpu_selector
            assert pod["runtimeClassName"] == "nvidia"
        else:
            assert pod["nodeSelector"] == image_selector
            assert "runtimeClassName" not in pod

    assert "nvidia.com/gpu" not in chart_pods["orchestrator"]["containers"][0]["resources"].get("requests", {})
    assert chart_pods["trainer"]["containers"][0]["resources"]["requests"]["nvidia.com/gpu"] == 1
    assert chart_pods["orchestrator"]["containers"][0]["args"] == [ORCHESTRATOR_COMMAND]
    assert chart_pods["trainer"]["containers"][0]["args"] == [TRAINER_COMMAND]
    assert rendered_resource(rendered, "StatefulSet", "p4-math-orchestrator")["spec"]["replicas"] == 1
    assert rendered_resource(rendered, "StatefulSet", "p4-math-trainer")["spec"]["replicas"] == 1
    assert not any(kind in rendered for kind in ("kind: ClusterRole", "kind: CustomResourceDefinition"))


def test_external_controller_mode_renders_only_five_dgd_inference_pods(tmp_path: Path):
    paths = write_dgd_artifacts(
        inference_config(),
        render_options(tmp_path, external_controller=True),
    )
    rendered = helm_template("-f", str(paths["values"]))
    graph = rendered_resource(rendered, "DynamoGraphDeployment", "p4-math")

    assert "kind: StatefulSet" not in rendered
    assert "kind: PersistentVolumeClaim" not in rendered
    assert "p4-math-shared-data" not in rendered
    assert sum(service["replicas"] for service in graph["spec"]["services"].values()) == 5
    assert rendered.count("kind: DynamoGraphDeployment") == 1
    assert rendered.count("name: p4-math-frontend-rl") == 1


@pytest.mark.parametrize("external_controller", [False, True])
def test_dgd_chart_renders_without_gpu_runtime_class(
    tmp_path: Path,
    external_controller: bool,
):
    options = render_options(tmp_path, external_controller=external_controller)
    options = replace(
        options,
        gpu_scheduling=replace(options.gpu_scheduling, runtime_class_name=None),
    )
    paths = write_dgd_artifacts(inference_config(), options)

    rendered = helm_template("-f", str(paths["values"]))
    graph = rendered_resource(rendered, "DynamoGraphDeployment", "p4-math")
    workload = json.loads(
        json.loads(paths["values"].read_text())["inference"]["dynamoGraph"]["workloadBinding"]["canonical"]
    )

    assert "runtimeClassName" not in workload["trainer"]["placement"]
    for role in ("VllmPrefillWorker", "VllmDecodeWorker"):
        assert "runtimeClassName" not in graph["spec"]["services"][role]["extraPodSpec"]
    if external_controller:
        assert "kind: StatefulSet" not in rendered
    else:
        trainer = rendered_resource(rendered, "StatefulSet", "p4-math-trainer")
        assert "runtimeClassName" not in trainer["spec"]["template"]["spec"]


def test_dgd_chart_projects_model_cache_once_into_operator_pod_specs(tmp_path: Path):
    paths = write_dgd_artifacts(inference_config(), render_options(tmp_path))

    rendered = helm_template("-f", str(paths["values"]))
    graph = rendered_resource(rendered, "DynamoGraphDeployment", "p4-math")
    expected_mount = {"name": "model-cache", "mountPath": "/model-cache"}
    expected_volume = {
        "name": "model-cache",
        "persistentVolumeClaim": {"claimName": "model-cache"},
    }

    for service in graph["spec"]["services"].values():
        assert "volumeMounts" not in service
        pod_spec = service["extraPodSpec"]
        assert pod_spec["mainContainer"].get("volumeMounts", []).count(expected_mount) == 1
        assert pod_spec.get("volumes", []).count(expected_volume) == 1


def test_chart_managed_trainer_uses_exact_bound_gpu_resources(tmp_path: Path):
    paths = write_dgd_artifacts(
        inference_config(),
        render_options(tmp_path, trainer_gpu_count=4),
    )
    values = json.loads(paths["values"].read_text())
    rendered = helm_template("-f", str(paths["values"]))
    trainer = rendered_resource(rendered, "StatefulSet", "p4-math-trainer")
    resources = trainer["spec"]["template"]["spec"]["containers"][0]["resources"]
    workload = json.loads(values["inference"]["dynamoGraph"]["workloadBinding"]["canonical"])

    assert workload["trainer"]["gpu"] == {"enabled": True, "count": 4}
    assert resources["requests"]["nvidia.com/gpu"] == 4
    assert resources["limits"]["nvidia.com/gpu"] == 4


def test_legacy_statefulset_selectors_remain_upgrade_compatible():
    documents = rendered_documents(helm_template())
    controllers = [document for document in documents if document["kind"] == "StatefulSet"]

    assert len(controllers) == 3
    for controller in controllers:
        role = controller["metadata"]["labels"]["role"]
        assert controller["spec"]["selector"]["matchLabels"] == {
            "app": "prime-rl",
            "example": "reverse-text",
            "role": role,
        }
        assert controller["spec"]["template"]["metadata"]["labels"]["app.kubernetes.io/instance"] == "p4-math"


def test_chart_service_selectors_are_release_disjoint():
    releases = {release: rendered_documents(helm_template(release_name=release)) for release in ("alpha", "beta")}
    pod_labels = {
        release: [
            document["spec"]["template"]["metadata"]["labels"]
            for document in documents
            if document["kind"] == "StatefulSet"
        ]
        for release, documents in releases.items()
    }

    for release, documents in releases.items():
        other_release = "beta" if release == "alpha" else "alpha"
        services = [document for document in documents if document["kind"] == "Service"]
        assert len(services) == 6

        for service in services:
            selector = service["spec"]["selector"]
            assert selector["app.kubernetes.io/instance"] == release
            assert any(labels_match(selector, labels) for labels in pod_labels[release])
            assert not any(labels_match(selector, labels) for labels in pod_labels[other_release])


def test_dgd_rl_service_selector_is_release_disjoint(tmp_path: Path):
    release_pods: dict[str, dict[str, str]] = {}
    release_services: dict[str, dict[str, str]] = {}
    for release in ("alpha", "beta"):
        output_dir = tmp_path / release
        paths = write_dgd_artifacts(inference_config(), render_options(output_dir, release_name=release))
        rendered = helm_template("-f", str(paths["values"]), release_name=release)
        release_pods[release] = {
            # Grove owns and rewrites the conventional app labels on realized
            # pods, but Dynamo's identity labels remain stable.
            "app.kubernetes.io/name": f"{release}-0-frontend",
            "nvidia.com/dynamo-graph-deployment-name": release,
            "nvidia.com/dynamo-component": "Frontend",
            "nvidia.com/dynamo-component-type": "frontend",
        }
        release_services[release] = rendered_resource(rendered, "Service", f"{release}-frontend-rl")["spec"]["selector"]

    for release in ("alpha", "beta"):
        other_release = "beta" if release == "alpha" else "alpha"
        selector = release_services[release]
        assert selector == {
            "nvidia.com/dynamo-graph-deployment-name": release,
            "nvidia.com/dynamo-component": "Frontend",
            "nvidia.com/dynamo-component-type": "frontend",
        }
        assert labels_match(selector, release_pods[release])
        assert not labels_match(selector, release_pods[other_release])


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


def test_dgd_chart_uses_canonical_workload_contract_as_sole_authority(tmp_path: Path):
    paths = write_dgd_artifacts(inference_config(), render_options(tmp_path))
    values = json.loads(paths["values"].read_text())
    workload_binding = values["inference"]["dynamoGraph"]["workloadBinding"]
    workload = json.loads(workload_binding["canonical"])

    assert hashlib.sha256(workload_binding["canonical"].encode()).hexdigest() == workload_binding["sha256"]
    assert workload["controllerMode"] == "chartManaged"
    for key in ("config", "huggingFace", "image", "modelCache", "storage"):
        assert workload[key] == values[key]
    assert values["orchestrator"] == {
        key: value for key, value in workload["orchestrator"].items() if key not in {"gpu", "placement"}
    }
    assert values["trainer"] == {key: value for key, value in workload["trainer"].items() if key != "placement"}
    assert workload["orchestrator"]["placement"]["nodeSelector"] == {
        "cloud.google.com/gke-nodepool": "customer-gpu-o7v",
        "kubernetes.io/arch": "arm64",
    }
    assert workload["trainer"]["placement"]["nodeSelector"] == {
        "cloud.google.com/gke-nodepool": "customer-gpu-o7v",
        "kubernetes.io/arch": "arm64",
        "nvidia.com/gpu.product": "NVIDIA-GB200",
    }


@pytest.mark.parametrize(
    ("component", "selector_key"),
    [
        ("orchestrator", "kubernetes.io/arch"),
        ("orchestrator", "cloud.google.com/gke-nodepool"),
        ("trainer", "kubernetes.io/arch"),
        ("trainer", "cloud.google.com/gke-nodepool"),
        ("trainer", "nvidia.com/gpu.product"),
    ],
)
def test_dgd_chart_rejects_rehashed_chart_selector_mutations(
    component: str,
    selector_key: str,
    tmp_path: Path,
):
    paths = write_dgd_artifacts(inference_config(), render_options(tmp_path))
    values = json.loads(paths["values"].read_text())
    workload = json.loads(values["inference"]["dynamoGraph"]["workloadBinding"]["canonical"])
    selector = workload[component]["placement"]["nodeSelector"]
    selector[f"tampered.example/{selector_key.rsplit('/', 1)[-1]}"] = selector.pop(selector_key)
    rewrite_valid_integrity(values, workload)
    mutation = tmp_path / f"{component}-selector.json"
    mutation.write_text(json.dumps(values))

    with pytest.raises(subprocess.CalledProcessError) as error:
        helm_template("-f", str(mutation))

    assert f"{component} placement must match" in error.value.stderr


def test_dgd_chart_rejects_rehashed_runtime_class_mutation(tmp_path: Path):
    paths = write_dgd_artifacts(inference_config(), render_options(tmp_path))
    values = json.loads(paths["values"].read_text())
    workload = json.loads(values["inference"]["dynamoGraph"]["workloadBinding"]["canonical"])
    workload["trainer"]["placement"]["runtimeClassName"] = "tampered"
    rewrite_valid_integrity(values, workload)
    mutation = tmp_path / "runtime-class.json"
    mutation.write_text(json.dumps(values))

    with pytest.raises(subprocess.CalledProcessError) as error:
        helm_template("-f", str(mutation))

    assert "trainer placement must match" in error.value.stderr


@pytest.mark.parametrize("component", ["orchestrator", "trainer"])
@pytest.mark.parametrize("toleration_key", ["kubernetes.io/arch", "nvidia.com/gpu", "prime-rl"])
def test_dgd_chart_rejects_every_rehashed_required_toleration_mutation(
    component: str,
    toleration_key: str,
    tmp_path: Path,
):
    paths = write_dgd_artifacts(inference_config(), render_options(tmp_path))
    values = json.loads(paths["values"].read_text())
    workload = json.loads(values["inference"]["dynamoGraph"]["workloadBinding"]["canonical"])
    toleration = next(item for item in workload[component]["placement"]["tolerations"] if item["key"] == toleration_key)
    toleration["effect"] = "NoExecute"
    rewrite_valid_integrity(values, workload)
    mutation = tmp_path / f"{component}-{toleration_key.replace('/', '-')}.json"
    mutation.write_text(json.dumps(values))

    with pytest.raises(subprocess.CalledProcessError) as error:
        helm_template("-f", str(mutation))

    assert f"{component} placement must match" in error.value.stderr


def test_dgd_chart_ignores_legacy_component_placement_overlays(tmp_path: Path):
    paths = write_dgd_artifacts(inference_config(), render_options(tmp_path))
    overlay = tmp_path / "legacy-placement.json"
    overlay.write_text(
        json.dumps(
            {
                "orchestrator": {
                    "nodeSelector": {"tampered": "true"},
                    "runtimeClassName": "tampered",
                    "tolerations": [{"key": "tampered", "operator": "Exists"}],
                },
                "trainer": {
                    "nodeSelector": {"tampered": "true"},
                    "runtimeClassName": "tampered",
                    "tolerations": [{"key": "tampered", "operator": "Exists"}],
                },
            }
        )
    )
    rendered = helm_template("-f", str(paths["values"]), "-f", str(overlay))

    orchestrator = rendered_resource(rendered, "StatefulSet", "p4-math-orchestrator")["spec"]["template"]["spec"]
    trainer = rendered_resource(rendered, "StatefulSet", "p4-math-trainer")["spec"]["template"]["spec"]
    assert orchestrator["nodeSelector"] == {
        "cloud.google.com/gke-nodepool": "customer-gpu-o7v",
        "kubernetes.io/arch": "arm64",
    }
    assert "runtimeClassName" not in orchestrator
    assert trainer["nodeSelector"] == {
        "cloud.google.com/gke-nodepool": "customer-gpu-o7v",
        "kubernetes.io/arch": "arm64",
        "nvidia.com/gpu.product": "NVIDIA-GB200",
    }
    assert trainer["runtimeClassName"] == "nvidia"


def test_filesystem_broadcast_reuses_existing_claim_without_rendering_pvc(tmp_path: Path):
    paths = write_dgd_artifacts(
        inference_config("filesystem"),
        render_options(tmp_path, shared_pvc="p4-shared-data"),
    )
    rendered = helm_template("-f", str(paths["values"]))

    assert "kind: PersistentVolumeClaim" not in rendered
    assert rendered.count("claimName: p4-shared-data") == 4


def test_external_filesystem_broadcast_binds_existing_claim_without_rendering_pvc(tmp_path: Path):
    paths = write_dgd_artifacts(
        inference_config("filesystem"),
        render_options(
            tmp_path,
            external_controller=True,
            shared_pvc="p4-shared-data",
        ),
    )
    rendered = helm_template("-f", str(paths["values"]))
    values = json.loads(paths["values"].read_text())
    graph = rendered_resource(rendered, "DynamoGraphDeployment", "p4-math")
    services = graph["spec"]["services"]
    workload = json.loads(values["inference"]["dynamoGraph"]["workloadBinding"]["canonical"])

    assert "kind: PersistentVolumeClaim" not in rendered
    assert workload["storage"] == {
        "enabled": True,
        "existingClaim": "p4-shared-data",
        "storageClassName": "nfs",
        "accessModes": ["ReadWriteMany"],
        "size": "1Ti",
        "mountPath": "/data",
    }
    assert graph["spec"]["pvcs"] == [
        {"create": False, "name": "model-cache"},
        {"create": False, "name": "p4-shared-data"},
    ]
    assert all("volumeMounts" not in service for service in services.values())
    for role in ("VllmPrefillWorker", "VllmDecodeWorker"):
        pod_spec = services[role]["extraPodSpec"]
        assert pod_spec["mainContainer"]["volumeMounts"].count({"name": "p4-shared-data", "mountPath": "/data"}) == 1
        assert (
            pod_spec["volumes"].count(
                {
                    "name": "p4-shared-data",
                    "persistentVolumeClaim": {"claimName": "p4-shared-data"},
                }
            )
            == 1
        )


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

    assert "image configuration must match the workload binding" in error.value.stderr


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
