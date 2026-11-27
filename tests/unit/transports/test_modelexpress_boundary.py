"""Keep the framework adapter on ModelExpress's public client surface."""

import ast
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
            "list_weight_versions",
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
    if fail_install:
        with pytest.raises(RuntimeError, match="installation failed"):
            worker.update_weights_from_modelexpress(version_uid="version-a")
    else:
        result = worker.update_weights_from_modelexpress(version_uid="version-a")
        assert result is None
    assert calls == ["stage", "apply", "release"]


def test_modelexpress_worker_initializes_once_per_worker_id(monkeypatch):
    import torch

    import prime_rl.inference.vllm.worker.modelexpress as worker_module

    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(uuid="gpu-a"))
    monkeypatch.setattr(worker_module, "os", SimpleNamespace(getpid=lambda: 100))
    closed = []
    running = SimpleNamespace(worker_id="session-7:gpu-a:100", close=lambda: closed.append(True))
    worker = worker_module.ModelExpressWeightUpdateWorker()
    worker.device = torch.device("cuda", 0)
    worker._generator = running

    # The same worker id keeps the running client without touching the engine.
    worker.init_broadcaster("trainer", 8001, 60, "session-7")
    assert worker._generator is running and not closed

    # A new worker id closes the running client before initializing another one from the engine.
    worker.model_runner = SimpleNamespace()
    with pytest.raises(AttributeError, match="model_config"):
        worker.init_broadcaster("trainer", 8001, 60, "session-8")
    assert closed == [True]


@pytest.mark.parametrize(
    "workers",
    [
        # One server running four data-parallel engines: every engine's worker has rank 0.
        [("gpu-0", 10, 0, 0), ("gpu-1", 11, 0, 1), ("gpu-2", 12, 0, 2), ("gpu-3", 13, 0, 3)],
        # Two external-LB servers of TP=2, each seeing only its own two GPUs as devices 0 and 1.
        [("gpu-0", 10, 0, 0), ("gpu-1", 11, 1, 1), ("gpu-2", 20, 0, 0), ("gpu-3", 21, 1, 1)],
        # Prefill and decode deployments on two nodes: each starts at rank 0 on device 0, in its own container.
        [("gpu-0", 10, 0, 0), ("gpu-8", 10, 0, 0)],
        # Two engines sharing one GPU.
        [("gpu-0", 10, 0, 0), ("gpu-0", 20, 0, 0)],
    ],
    ids=["dp4", "external-lb-tp2", "pd", "shared-gpu"],
)
def test_modelexpress_worker_ids_are_unique_without_layout_knowledge(monkeypatch, workers):
    """Each worker registers with MX under its own id, which also names its version lease and NIXL agent. Every
    engine receives the same initialization, so the id comes from the worker's own GPU and process."""
    import torch
    import transformers
    from torch import nn

    import prime_rl.inference.vllm.worker.modelexpress as worker_module
    import prime_rl.trainer.models as models

    registered = []

    class FakeGeneratorClient:
        @staticmethod
        def initialize(config):
            registered.append(config.worker_id)
            return SimpleNamespace(close=lambda: None)

    monkeypatch.setattr(worker_module, "ModelExpressGeneratorClient", FakeGeneratorClient)
    monkeypatch.setattr(worker_module, "VllmGeneratorContext", lambda **kwargs: kwargs)
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        models, "get_custom_causal_lm_cls", lambda config: SimpleNamespace(conversion_chain=lambda config: None)
    )
    monkeypatch.setattr(worker_module.atexit, "register", lambda *args: None)

    for gpu_uuid, pid, rank, device_index in workers:
        monkeypatch.setattr(
            torch.cuda, "get_device_properties", lambda device, uuid=gpu_uuid: SimpleNamespace(uuid=uuid)
        )
        monkeypatch.setattr(worker_module, "os", SimpleNamespace(getpid=lambda pid=pid: pid))
        worker = worker_module.ModelExpressWeightUpdateWorker()
        worker.rank = rank
        worker.device = torch.device("cuda", device_index)
        worker.vllm_config = None
        worker.model_runner = SimpleNamespace(
            model_config=SimpleNamespace(model="model-a", trust_remote_code=False), get_model=nn.Module
        )
        worker.init_broadcaster("host", 8001, 60, "session")

    assert registered == [f"session:{gpu_uuid}:{pid}" for gpu_uuid, pid, _, _ in workers]
    assert len(set(registered)) == len(workers)


def test_modelexpress_sender_publishes_current_values_of_state_dict_copies(tmp_path, monkeypatch):
    """MX stages each version from the dict bound at the first update. A state-dict entry that is a copy of its
    parameter, as a fused qkv split along its FSDP-sharded dimension is, must still carry each update's values, and
    the copies must not stay allocated between updates."""
    import gc
    import weakref

    import torch
    from torch import nn

    import prime_rl.transports.weights.modelexpress as sender_module
    from prime_rl.configs.shared import ModelExpressWeightBroadcastConfig

    class PackedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.packed = nn.Parameter(torch.zeros(4))
            self.plain = nn.Parameter(torch.zeros(2))
            self.register_state_dict_post_hook(PackedModel.split_into_copies)

        @staticmethod
        def split_into_copies(module, state_dict, prefix, local_metadata):
            packed = state_dict.pop(f"{prefix}packed")
            state_dict[f"{prefix}q"], state_dict[f"{prefix}k"] = packed.clone().split(2)

    published = []

    class FakeTrainer:
        worker_id = "trainer-0"

        def bind_tensors(self, tensors):
            self.bound = tensors
            return SimpleNamespace(logical_shard_id="shard-0", metadata_endpoint="host:5655")

        def publish_version(self, *, version):
            published.append({name: tensor.clone() for name, tensor in self.bound.items()})
            self.copies = [weakref.ref(self.bound[name]) for name in ("q", "k")]
            (tmp_path / f"step_{len(published)}" / ".installed").write_text(version.version_id)

        def release_version(self, *, version):
            pass

        def close(self):
            pass

    versions = iter(range(100))
    control = SimpleNamespace(
        create_trainer_mesh=lambda **kwargs: SimpleNamespace(mesh_id="mesh-0"),
        create_weight_version=lambda **kwargs: SimpleNamespace(version_id=f"version-{next(versions)}"),
        delete_trainer_mesh=lambda mesh_id: None,
        close=lambda: None,
    )
    monkeypatch.setattr(sender_module.ModelExpressTrainerClient, "initialize", lambda config: FakeTrainer())
    monkeypatch.setattr(sender_module.ModelExpressControlClient, "connect", lambda **kwargs: control)
    monkeypatch.setattr(sender_module, "ModelExpressTrainerConfig", lambda **kwargs: kwargs)
    monkeypatch.setattr(sender_module, "FSDPTrainerContext", lambda **kwargs: kwargs)
    monkeypatch.setattr(sender_module, "all_gather_bytes", lambda payload: [payload])
    monkeypatch.setattr(sender_module, "broadcast_bytes", lambda payload: payload)
    monkeypatch.setattr(sender_module.dist, "barrier", lambda: None)
    monkeypatch.setattr(sender_module.atexit, "register", lambda *args: None)
    monkeypatch.delenv("MX_METADATA_PORT", raising=False)

    sender = sender_module.ModelExpressWeightSender(
        tmp_path, ModelExpressWeightBroadcastConfig(timeout=5), model_name="model-a"
    )
    sender.world = SimpleNamespace(is_master=True, local_rank=0)

    model = PackedModel()
    for step in (1, 2, 3):
        with torch.no_grad():
            model.packed.copy_(torch.arange(4.0) + step)
            model.plain.fill_(step)
        (tmp_path / f"step_{step}").mkdir()
        sender._broadcast(model, step, tmp_path / f"step_{step}")
        gc.collect()
        assert all(copy() is None for copy in sender._trainer.copies)

    for step, tensors in enumerate(published, start=1):
        assert torch.equal(tensors["q"], torch.tensor([0.0, 1.0]) + step)
        assert torch.equal(tensors["k"], torch.tensor([2.0, 3.0]) + step)
        assert torch.equal(tensors["plain"], torch.full((2,), float(step)))


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
    from prime_rl.transports.weights.base import marker_exists

    source = ast.parse((ROOT / "src/prime_rl/transports/weights/modelexpress.py").read_text())
    sender = next(
        node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "ModelExpressWeightSender"
    )
    broadcast = next(node for node in sender.body if isinstance(node, ast.FunctionDef) and node.name == "_broadcast")
    publication = next(
        index
        for index, node in enumerate(broadcast.body)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "_publish"
    )
    # Execute the real acknowledgment block without importing the GPU runtime.
    module = ast.parse("def acknowledge(self, step_dir, version): pass")
    module.body[0].body = broadcast.body[publication + 1 :]
    messages = []
    calls = []
    rank = 0

    def broadcast_bytes(value):
        if rank == 0:
            messages.append(value)
            return value
        return messages[0]

    namespace = {
        "time": time,
        "broadcast_bytes": broadcast_bytes,
        "dist": SimpleNamespace(
            barrier=lambda: calls.append((rank, "barrier")),
        ),
        "INSTALLED_MARKER": ".installed",
        "marker_exists": marker_exists,
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
            _tensors={},
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


def test_integer_parameters_keep_their_values_through_a_layerwise_reload():
    """DeepSeek-V4's hash gate: an integer routing table beside a float weight, which the routed experts alias.

    ModelExpress refits through vLLM's layerwise reload and writes only floating-point tensors. After the conversion,
    the table is outside the parameters a refit must cover, the reload installs the new float weight, and the table
    (and the experts' alias of it) keeps the values loaded at startup.
    """
    import torch
    from torch import nn
    from vllm.model_executor.model_loader.reload.layerwise import (
        finalize_layerwise_reload,
        initialize_layerwise_reload,
        record_metadata_for_reloading,
    )

    from prime_rl.inference.vllm.worker.modelexpress import keep_integer_parameters_as_buffers

    gate = nn.Module()
    gate.weight = nn.Parameter(torch.zeros(4, 8), requires_grad=False)
    gate.tid2eid = nn.Parameter(torch.arange(12).reshape(6, 2), requires_grad=False)
    experts = nn.Module()
    object.__setattr__(experts, "hash_indices_table", gate.tid2eid)
    model = nn.Module()
    model.gate, model.experts = gate, experts
    table = gate.tid2eid.detach().clone()
    record_metadata_for_reloading(model)

    assert keep_integer_parameters_as_buffers(model) == ["gate.tid2eid"]
    assert [name for name, _ in model.named_parameters()] == ["gate.weight"]
    assert "tid2eid" in gate._non_persistent_buffers_set
    assert experts.hash_indices_table is gate.tid2eid and not isinstance(experts.hash_indices_table, nn.Parameter)

    initialize_layerwise_reload(model)
    gate.weight.weight_loader(gate.weight, torch.ones(4, 8))
    finalize_layerwise_reload(model, model_config=None)

    assert torch.equal(gate.weight, torch.ones(4, 8))
    assert torch.equal(gate.tid2eid, table)
    assert experts.hash_indices_table is gate.tid2eid


def test_modelexpress_pull_sender_retires_each_version_before_refilling_its_host_copy(tmp_path, monkeypatch):
    """Under pull the trainer holds one version: before publishing the next, it stops new reads of the previous one
    and waits for those in flight. It never waits for an engine to install, except for the last version at close."""
    import torch
    from torch import nn

    import prime_rl.transports.weights.modelexpress as sender_module
    from prime_rl.configs.shared import ModelExpressWeightBroadcastConfig

    calls = []
    lease_held = [1]

    class FakeTrainer:
        worker_id = "trainer-0"

        def bind_tensors(self, tensors):
            return SimpleNamespace(logical_shard_id="shard-0", metadata_endpoint="host:5655")

        def publish_version(self, *, version):
            calls.append(("publish", version.version_id))

        def release_version(self, *, version):
            if lease_held[0]:
                lease_held[0] -= 1
                calls.append(("release refused", version.version_id))
                raise RuntimeError("weight version has an active lease")
            calls.append(("release", version.version_id))

        def close(self):
            pass

    def list_weight_versions(model_name):
        return [
            SimpleNamespace(
                version_id="dead", version_number=7, state=sender_module.WeightVersionState.READY, trainer_mesh_id="old"
            ),
            SimpleNamespace(
                version_id="gone",
                version_number=6,
                state=sender_module.WeightVersionState.RELEASING,
                trainer_mesh_id="old",
            ),
        ]

    control = SimpleNamespace(
        create_trainer_mesh=lambda **kwargs: SimpleNamespace(mesh_id="mesh-0"),
        create_weight_version=lambda **kwargs: SimpleNamespace(version_id=f"version-{kwargs['version_number']}"),
        delete_weight_version=lambda version_id: calls.append(("retire", version_id)),
        list_weight_versions=list_weight_versions,
        delete_trainer_mesh=lambda mesh_id: None,
        close=lambda: None,
    )
    monkeypatch.setattr(sender_module.ModelExpressTrainerClient, "initialize", lambda config: FakeTrainer())
    monkeypatch.setattr(sender_module.ModelExpressControlClient, "connect", lambda **kwargs: control)
    monkeypatch.setattr(sender_module, "ModelExpressTrainerConfig", lambda **kwargs: kwargs)
    monkeypatch.setattr(sender_module, "FSDPTrainerContext", lambda **kwargs: kwargs)
    monkeypatch.setattr(sender_module, "all_gather_bytes", lambda payload: [payload])
    monkeypatch.setattr(sender_module, "broadcast_bytes", lambda payload: payload)
    monkeypatch.setattr(sender_module.dist, "barrier", lambda: None)
    monkeypatch.setattr(sender_module.atexit, "register", lambda *args: None)
    monkeypatch.delenv("MX_METADATA_PORT", raising=False)

    sender = sender_module.ModelExpressWeightSender(
        tmp_path, ModelExpressWeightBroadcastConfig(timeout=5, pull=True), model_name="model-a"
    )
    assert not sender.acknowledged
    sender.world = SimpleNamespace(is_master=True, local_rank=0)
    model = nn.Linear(2, 2)
    for step in (0, 1, 2):
        sender.step_dir(step).mkdir(parents=True)
        sender._broadcast(model, step, sender.step_dir(step))
        assert sender._tensors == {}
    (sender.step_dir(2) / ".installed").write_text("version-2")
    sender.close()
    assert calls == [
        ("retire", "dead"),
        ("publish", "version-0"),
        ("retire", "version-0"),
        ("release refused", "version-0"),
        ("release", "version-0"),
        ("publish", "version-1"),
        ("retire", "version-1"),
        ("release", "version-1"),
        ("publish", "version-2"),
        ("retire", "version-2"),
        ("release", "version-2"),
    ]
    assert torch.is_tensor(model.weight)


def test_modelexpress_pull_receiver_waits_until_every_engine_serves_a_version_of_this_trainer(tmp_path):
    import asyncio

    from prime_rl.configs.shared import ModelExpressWeightBroadcastConfig
    from prime_rl.transports.weights import setup_weight_receiver
    from prime_rl.transports.weights.base import FINISHED_MARKER

    served = [[None, 3], [3, 5], [3, 4]]

    class AdminPlane:
        async def served_versions(self):
            return served.pop(0)

    receiver = setup_weight_receiver(
        tmp_path, ModelExpressWeightBroadcastConfig(timeout=5, pull=True), AdminPlane(), "model-a"
    )
    receiver.poll_interval = 0
    assert receiver.engines_pull
    for step in (3, 4):
        receiver.step_dir(step).mkdir(parents=True)
        (receiver.step_dir(step) / FINISHED_MARKER).touch()
    (receiver.step_dir(3) / ".mx_version").write_text("version-3")
    # v5 has no finished broadcast (it predates a trainer restart), so the second answer does not count
    asyncio.run(receiver.receive(3))
    assert served == []
    assert (receiver.step_dir(3) / ".installed").read_text() == "version-3"
