import json
import subprocess
from pathlib import Path

import pytest

from prime_rl.inference.dgd import write_dgd_artifacts
from tests.unit.inference.helm_dgd_test_utils import (
    helm_template,
    inference_config,
    render_options,
    rewrite_valid_integrity,
    write_values_mutation,
)


@pytest.mark.parametrize(
    ("case", "path", "replacement", "error_fragment"),
    [
        (
            "engine-data",
            ("inference", "dynamoGraph", "engineConfig", "data", "prefill-engine.json"),
            "tampered",
            "engineConfig.data must match its canonical payload",
        ),
        (
            "engine-sha",
            ("inference", "dynamoGraph", "engineConfig", "sha256"),
            "0" * 64,
            "engineConfig.sha256 must match its canonical payload",
        ),
        (
            "engine-name",
            ("inference", "dynamoGraph", "engineConfig", "name"),
            "p4-math-dynamo-engine-000000000000",
            "engineConfig.name must be content-addressed",
        ),
        (
            "dgd-config-sha",
            (
                "inference",
                "dynamoGraph",
                "resource",
                "metadata",
                "annotations",
                "prime-rl.nvidia.com/config-sha256",
            ),
            "0" * 64,
            "DynamoGraphDeployment config-sha256 must match engineConfig.sha256",
        ),
        (
            "manifest-sha",
            (
                "inference",
                "dynamoGraph",
                "resource",
                "metadata",
                "annotations",
                "prime-rl.nvidia.com/manifest-sha256",
            ),
            "0" * 64,
            "manifest-sha256 must match its canonical payload",
        ),
        (
            "workload-sha",
            ("inference", "dynamoGraph", "workloadBinding", "sha256"),
            "0" * 64,
            "workload binding sha256 must match its canonical payload",
        ),
        (
            "dgd-workload-sha",
            (
                "inference",
                "dynamoGraph",
                "resource",
                "metadata",
                "annotations",
                "prime-rl.nvidia.com/workload-sha256",
            ),
            "0" * 64,
            "workload binding annotation must match workloadBinding.sha256",
        ),
    ],
)
def test_dgd_chart_rejects_content_identity_mutations(
    case: str,
    path: tuple[str, ...],
    replacement: object,
    error_fragment: str,
    tmp_path: Path,
):
    paths = write_dgd_artifacts(inference_config(), render_options(tmp_path))
    mutation = tmp_path / f"{case}.json"
    write_values_mutation(paths["values"], mutation, path, replacement)

    with pytest.raises(subprocess.CalledProcessError) as error:
        helm_template("-f", str(mutation))

    assert error_fragment in error.value.stderr


@pytest.mark.parametrize(
    ("case", "path", "replacement"),
    [
        (
            "client-roles",
            ("inference", "dynamoGraph", "clientTopology", "dynamo_worker_roles"),
            ["prefill", "decode", "decode", "decode"],
        ),
        (
            "client-gpus",
            ("inference", "dynamoGraph", "clientTopology", "dynamo_gpus_per_worker"),
            2,
        ),
        (
            "worker-replicas",
            (
                "inference",
                "dynamoGraph",
                "resource",
                "spec",
                "services",
                "VllmDecodeWorker",
                "replicas",
            ),
            3,
        ),
        (
            "worker-gpus",
            (
                "inference",
                "dynamoGraph",
                "resource",
                "spec",
                "services",
                "VllmPrefillWorker",
                "resources",
                "limits",
                "gpu",
            ),
            "2",
        ),
    ],
)
def test_dgd_chart_rejects_topology_mutations(
    case: str,
    path: tuple[str, ...],
    replacement: object,
    tmp_path: Path,
):
    paths = write_dgd_artifacts(inference_config(), render_options(tmp_path))
    mutation = tmp_path / f"{case}.json"
    write_values_mutation(paths["values"], mutation, path, replacement)

    with pytest.raises(subprocess.CalledProcessError) as error:
        helm_template("-f", str(mutation))

    assert "topology binding" in error.value.stderr


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
    ("override_flag", "enabled_value"),
    [
        ("--set", "false"),
        ("--set-string", "false"),
        ("--set-string", "true"),
    ],
)
def test_dgd_chart_requires_boolean_true_inference_with_schema_skipped(
    override_flag: str,
    enabled_value: str,
    tmp_path: Path,
):
    paths = write_dgd_artifacts(inference_config(), render_options(tmp_path))

    with pytest.raises(subprocess.CalledProcessError) as error:
        helm_template(
            "--skip-schema-validation",
            "-f",
            str(paths["values"]),
            override_flag,
            f"inference.enabled={enabled_value}",
        )

    assert "inference.enabled must be boolean true in dynamoGraph mode" in error.value.stderr


def test_dgd_chart_cannot_switch_mode_and_skip_integrity_validation(tmp_path: Path):
    paths = write_dgd_artifacts(inference_config(), render_options(tmp_path))

    with pytest.raises(subprocess.CalledProcessError) as error:
        helm_template(
            "--skip-schema-validation",
            "-f",
            str(paths["values"]),
            "--set",
            "inference.mode=statefulset",
        )

    assert "generated DynamoGraph contract requires dynamoGraph mode" in error.value.stderr


@pytest.mark.parametrize(
    ("external_controller", "overrides", "error_fragment"),
    [
        (False, ("--set", "orchestrator.enabled=false"), "orchestrator.enabled must match"),
        (False, ("--set", "trainer.enabled=false"), "trainer.enabled must match"),
        (False, ("--set", "trainer.gpu.enabled=false"), "trainer GPU configuration must match"),
        (False, ("--set", "trainer.gpu.count=2"), "trainer GPU configuration must match"),
        (
            False,
            ("--set", r"orchestrator.resources.requests.nvidia\.com/gpu=1"),
            "orchestrator resources cannot set NVIDIA extended resource",
        ),
        (
            False,
            ("--set", r"orchestrator.resources.limits.nvidia\.com/mig-1g\.23gb=1"),
            "orchestrator resources cannot set NVIDIA extended resource",
        ),
        (
            False,
            ("--set", r"orchestrator.resources.requests.nvidia\.com/gpu\.shared=1"),
            "orchestrator resources cannot set NVIDIA extended resource",
        ),
        (
            False,
            ("--set", r"trainer.resources.requests.nvidia\.com/gpu=2"),
            "trainer resources cannot set NVIDIA extended resource",
        ),
        (
            False,
            ("--set", r"trainer.resources.limits.nvidia\.com/mig-1g\.23gb=1"),
            "trainer resources cannot set NVIDIA extended resource",
        ),
        (True, ("--set", "orchestrator.enabled=true"), "orchestrator.enabled must match"),
        (True, ("--set", "trainer.enabled=true"), "trainer.enabled must match"),
        (True, ("--set", "trainer.gpu.enabled=true"), "trainer GPU configuration must match"),
        (True, ("--set", "storage.enabled=true"), "storage configuration must match"),
        (
            True,
            ("--set", "inference.dynamoGraph.controllerMode=chartManaged"),
            "controllerMode must match",
        ),
    ],
)
def test_dgd_chart_rejects_workload_contract_overlays_with_schema_skipped(
    external_controller: bool,
    overrides: tuple[str, ...],
    error_fragment: str,
    tmp_path: Path,
):
    paths = write_dgd_artifacts(
        inference_config(),
        render_options(tmp_path, external_controller=external_controller),
    )

    with pytest.raises(subprocess.CalledProcessError) as error:
        helm_template(
            "--skip-schema-validation",
            "-f",
            str(paths["values"]),
            *overrides,
        )

    assert error_fragment in error.value.stderr


@pytest.mark.parametrize("external_controller", [False, True])
def test_dgd_chart_rejects_rehashed_workload_mode_contradictions(
    external_controller: bool,
    tmp_path: Path,
):
    paths = write_dgd_artifacts(
        inference_config(),
        render_options(tmp_path, external_controller=external_controller),
    )
    values = json.loads(paths["values"].read_text())
    graph = values["inference"]["dynamoGraph"]
    workload = json.loads(graph["workloadBinding"]["canonical"])

    if external_controller:
        workload["trainer"]["enabled"] = True
        workload["trainer"]["gpu"] = {"enabled": True, "count": 1}
        values["trainer"] = {"enabled": True, "gpu": {"enabled": True, "count": 1}}
        error_fragment = "external mode forbids chart-managed controller workloads"
    else:
        workload["orchestrator"]["enabled"] = False
        values["orchestrator"]["enabled"] = False
        error_fragment = "chartManaged mode requires orchestrator, trainer"

    rewrite_valid_integrity(values, workload)
    mutation = tmp_path / "contradictory-workload.json"
    mutation.write_text(json.dumps(values))

    with pytest.raises(subprocess.CalledProcessError) as error:
        helm_template("--skip-schema-validation", "-f", str(mutation))

    assert error_fragment in error.value.stderr


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
