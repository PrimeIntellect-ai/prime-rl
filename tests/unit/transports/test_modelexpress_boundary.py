"""Keep the framework adapter on ModelExpress's public client surface."""

import ast
import pickle
import time
import uuid
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

    if relative_path == "src/prime_rl/transports/weights/modelexpress.py":
        mesh = next(
            node
            for node in ast.walk(source)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "create_trainer_mesh"
        )
        key = next(item.value for item in mesh.keywords if item.arg == "idempotency_key")
        expression = compile(ast.Expression(key), "<mesh idempotency key>", "eval")
        namespace = {
            "uuid": uuid,
            "self": SimpleNamespace(model_name="model-a"),
            "workers": {"rank-0": None, "rank-1": None},
        }
        expected = eval(expression, namespace)
        namespace["workers"] = {"rank-1": None, "rank-0": None}
        assert eval(expression, namespace) == expected
        namespace["workers"] = {"restarted-rank-0": None, "rank-1": None}
        assert eval(expression, namespace) != expected
        namespace["workers"] = {"rank-0": None, "rank-1": None}
        namespace["self"].model_name = "model-b"
        assert eval(expression, namespace) != expected


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
    worker._generator = SimpleNamespace(stage_weight=stage_weight, apply_weight=apply_weight)
    worker.model_runner = SimpleNamespace(get_model=lambda: SimpleNamespace())
    if fail_install:
        with pytest.raises(RuntimeError, match="installation failed"):
            worker.update_weights_from_modelexpress(version_uid="version-a")
    else:
        result = worker.update_weights_from_modelexpress(version_uid="version-a")
        assert result is None
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


def test_refresh_copies_current_weights_into_bound_copies():
    import torch
    from torch import nn

    from prime_rl.transports.weights.modelexpress import ModelExpressWeightSender

    class Packed(nn.Module):
        """Its state dict splits ``packed`` into copies, like a packed parameter sharded along its packing dimension."""

        def __init__(self):
            super().__init__()
            self.packed = nn.Parameter(torch.zeros(4))
            self.plain = nn.Parameter(torch.zeros(2))

            def split(module, state_dict, prefix, local_metadata):
                packed = state_dict.pop(f"{prefix}packed")
                state_dict[f"{prefix}q"], state_dict[f"{prefix}k"] = packed[:2].clone(), packed[2:].clone()

            self.register_state_dict_post_hook(split)

    model = Packed()
    sender = SimpleNamespace(_bound=model.state_dict())
    with torch.no_grad():
        model.packed.copy_(torch.arange(4.0))
        model.plain.add_(1)
    ModelExpressWeightSender._refresh_bound_tensors(sender, model)
    assert sender._bound["q"].tolist() == [0.0, 1.0] and sender._bound["k"].tolist() == [2.0, 3.0]
    assert sender._bound["plain"].data_ptr() == model.plain.data_ptr()
