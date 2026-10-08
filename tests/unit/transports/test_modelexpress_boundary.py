"""Keep the framework adapter on ModelExpress's public client surface."""

import ast
import pickle
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    "relative_path",
    [
        "src/prime_rl/transports/weights/modelexpress.py",
        "src/prime_rl/inference/vllm/worker/modelexpress.py",
    ],
)
def test_modelexpress_adapter_uses_only_public_clients(relative_path):
    source = ast.parse((ROOT / relative_path).read_text())
    client_methods = {
        "_trainer": {"bind_tensors", "publish_version", "release_version", "close"},
        "_control": {
            "create_trainer_mesh",
            "create_weight_version",
            "get_weight_version",
            "delete_weight_version",
            "delete_trainer_mesh",
            "close",
        },
        "_generator": {"stage_weight", "apply_weight", "close"},
    }
    for node in ast.walk(source):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.startswith("modelexpress"):
                assert module == "modelexpress_rl"
                assert all(not name.name.startswith("_") for name in node.names)
            assert not module.startswith(("grpc", "nixl"))
        if isinstance(node, ast.Import):
            assert all(not name.name.startswith(("modelexpress", "grpc", "nixl")) for name in node.names)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute):
            owner = node.value
            if isinstance(owner.value, ast.Name) and owner.value.id == "self" and owner.attr in client_methods:
                assert not node.attr.startswith("_")
                if isinstance(node.ctx, ast.Load) and node.attr in client_methods[owner.attr]:
                    continue
                assert (owner.attr, node.attr) == ("_trainer", "worker_id")


@pytest.mark.parametrize("fail_install", [False, True])
def test_modelexpress_worker_uses_stage_apply_release(fail_install):
    from prime_rl.inference.vllm.worker.modelexpress import ModelExpressWeightUpdateWorker

    calls = []
    staged = SimpleNamespace(release=lambda: calls.append("release"))

    def stage_weight(*, version):
        assert version.version_id == "version-a"
        calls.append("stage")
        return staged

    def apply_weight(handle):
        assert handle is staged
        calls.append("apply")
        if fail_install:
            raise RuntimeError("installation failed")

    worker = ModelExpressWeightUpdateWorker()
    worker._worker_id = "worker-0"
    worker._generator = SimpleNamespace(stage_weight=stage_weight, apply_weight=apply_weight)
    if fail_install:
        with pytest.raises(RuntimeError, match="installation failed"):
            worker.update_weights_from_path(version_uid="version-a")
    else:
        result = worker.update_weights_from_path(version_uid="version-a")
        assert result == {"worker_id": "worker-0", "version_uid": "version-a"}
    assert calls == ["stage", "apply", "release"]


def test_modelexpress_worker_forwards_generator_buffer_config():
    source = ast.parse((ROOT / "src/prime_rl/inference/vllm/worker/modelexpress.py").read_text())
    config = next(
        node
        for node in ast.walk(source)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ModelExpressGeneratorConfig"
    )
    keywords = {item.arg: item.value for item in config.keywords}
    for name in ("staging_buffer_bytes", "staging_buffers_count"):
        assert isinstance(keywords[name], ast.Name)
        assert keywords[name].id == name


@pytest.mark.parametrize("outcome", ["success", "timeout", "wrong_uid", "read_error"])
def test_modelexpress_installation_outcome_reaches_non_master_without_shared_storage(tmp_path, outcome):
    source = ast.parse((ROOT / "src/prime_rl/transports/weights/modelexpress.py").read_text())
    sender = next(
        node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "ModelExpressWeightSender"
    )
    broadcast = next(node for node in sender.body if isinstance(node, ast.FunctionDef) and node.name == "_broadcast")
    publication = next(
        index
        for index, node in enumerate(broadcast.body)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "publish_version"
    )
    # Execute the real acknowledgment block without importing the GPU runtime.
    module = ast.parse("def acknowledge(self, step_dir, version): pass")
    module.body[0].body = broadcast.body[publication + 1 :]
    messages = []
    calls = []
    rank = 0

    def broadcast_object_list(values, src):
        assert src == 0
        assert len(values) == 1 and isinstance(values[0], bool)
        if rank == 0:
            messages.append(pickle.dumps(values))
        else:
            values[:] = pickle.loads(messages[0])

    namespace = {
        "time": time,
        "dist": SimpleNamespace(
            broadcast_object_list=broadcast_object_list,
            barrier=lambda: calls.append((rank, "barrier")),
        ),
        "INSTALLED_MARKER": ".installed",
    }
    exec(compile(ast.fix_missing_locations(module), "<installation acknowledgment>", "exec"), namespace)
    installed = tmp_path / ".installed"
    if outcome == "read_error":
        installed.mkdir()
    elif outcome != "timeout":
        installed.write_text("version-a" if outcome == "success" else "version-b")
    error_type = {"timeout": TimeoutError, "wrong_uid": RuntimeError, "read_error": IsADirectoryError}
    errors = []
    for rank in (0, 1):
        sender = SimpleNamespace(
            world=SimpleNamespace(is_master=rank == 0),
            timeout=0,
            _trainer=SimpleNamespace(release_version=lambda **kwargs: calls.append((rank, "release"))),
        )
        step_dir = tmp_path if rank == 0 else tmp_path / "non_master_local_fs"
        if outcome == "success":
            namespace["acknowledge"](sender, step_dir, SimpleNamespace(version_id="version-a"))
        else:
            expected_error = error_type[outcome] if rank == 0 else RuntimeError
            with pytest.raises(expected_error) as exc:
                namespace["acknowledge"](sender, step_dir, SimpleNamespace(version_id="version-a"))
            errors.append(str(exc.value))
    assert len(messages) == 1
    assert calls == ([(0, "release"), (0, "barrier"), (1, "release"), (1, "barrier")] if outcome == "success" else [])
    if errors:
        assert errors[1] == "Inference weight installation failed on trainer rank zero"
