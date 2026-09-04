"""Rail defaults for mx_refit and their production call sites."""

import ast
from pathlib import Path
from runpy import run_path

import pytest

_SOURCE_ROOT = Path(__file__).parents[3] / "src" / "prime_rl"
apply_rdma_defaults = run_path(_SOURCE_ROOT / "transports" / "weights" / "mx_rdma.py")["apply_rdma_defaults"]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in ("MX_RDMA_NIC_PIN", "MX_RESHARD_MIN_GBPS"):
        monkeypatch.delenv(key, raising=False)


def test_defaults_opt_into_spreading_and_reporting(monkeypatch):
    """Unset means "spread the rails", which is the whole point of the change.

    Left to UCX, four inference readers in one pod converged on a single NIC at
    1.5 GB/s each. Spread one per rail they ran at 6.75 each, and aggregate went
    from 6.1 to 27.0 GB/s -- lower with four readers than with one, so the
    default matters more than a tuning knob usually would.
    """
    apply_rdma_defaults()

    import os

    assert os.environ["MX_RDMA_NIC_PIN"] == "auto"

    # Asserted as a band rather than a literal, because a literal is what let the
    # wrong number through in the first place. The floor is per rank, so it has to
    # clear the collapsed rate (~12 Gbps) and stay clearly under the rate a healthy
    # reader sees while its three siblings are also reading (~54 Gbps, not the
    # ~208 a lone reader gets). An earlier default of 50 satisfied a hardcoded
    # equality check perfectly while sitting 8% below healthy, i.e. firing on the
    # good runs. The margins here are what make the guard usable, so they are the
    # thing worth testing.
    floor = float(os.environ["MX_RESHARD_MIN_GBPS"])
    collapsed_gbps, healthy_per_reader_gbps = 12.0, 54.0
    assert floor >= 2 * collapsed_gbps, (
        f"floor {floor} is too close to the {collapsed_gbps} Gbps collapse to distinguish a fault from noise"
    )
    assert floor <= healthy_per_reader_gbps / 2, (
        f"floor {floor} leaves too little headroom under the healthy per-reader "
        f"{healthy_per_reader_gbps} Gbps and will fire on good runs"
    )


def test_an_operator_override_is_never_clobbered(monkeypatch):
    """Anyone who set these meant it.

    Someone pinning ``UCX_NET_DEVICES`` by hand, or running a fabric where 5
    Gbps is a normal rate rather than a fault, must not have that decision
    silently replaced at client construction. ``setdefault`` rather than
    assignment is the entire contract of this function.
    """
    monkeypatch.setenv("MX_RDMA_NIC_PIN", "mlx5_4:1")
    monkeypatch.setenv("MX_RESHARD_MIN_GBPS", "5")

    apply_rdma_defaults()

    import os

    assert os.environ["MX_RDMA_NIC_PIN"] == "mlx5_4:1"
    assert os.environ["MX_RESHARD_MIN_GBPS"] == "5"


def test_disabling_the_pin_explicitly_is_respected(monkeypatch):
    """ "off" is a real answer and must survive.

    Distinct from the override case: an empty or explicitly-disabled value is
    the one most likely to be mistaken for "unset" by a careless implementation,
    and that would re-enable a probe the operator switched off.
    """
    monkeypatch.setenv("MX_RDMA_NIC_PIN", "")

    apply_rdma_defaults()

    import os

    assert os.environ["MX_RDMA_NIC_PIN"] == ""


def test_calling_twice_is_stable(monkeypatch):
    """Both transports call this, and a process can be both in a colocated run."""
    apply_rdma_defaults()
    apply_rdma_defaults()

    import os

    assert os.environ["MX_RDMA_NIC_PIN"] == "auto"


def _call_names_in_method(path: Path, class_name: str, method_name: str) -> list[str]:
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(
        node
        for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name
    )

    def call_name(call: ast.Call) -> str:
        parts: list[str] = []
        current = call.func
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        return ".".join(reversed(parts))

    return [call_name(node) for node in ast.walk(method) if isinstance(node, ast.Call)]


@pytest.mark.parametrize(
    ("path", "class_name", "method_name", "client_initializer"),
    [
        (
            _SOURCE_ROOT / "inference" / "vllm" / "worker" / "mx_refit.py",
            "MXRefitUpdateWorker",
            "init_broadcaster",
            "ModelExpressGeneratorClient.initialize",
        ),
    ],
)
def test_production_paths_apply_defaults_before_client_initialization(
    path, class_name, method_name, client_initializer
):
    """The pin must be active before either side constructs its NIXL agent."""
    calls = _call_names_in_method(path, class_name, method_name)

    assert calls.count("apply_rdma_defaults") == 1
    assert calls.index("apply_rdma_defaults") < calls.index(client_initializer)


def test_fsdp_publishers_remain_unpinned():
    """Receivers need every FSDP source, including across isolated rail subnets."""
    path = _SOURCE_ROOT / "transports" / "weights" / "mx_refit.py"
    calls = _call_names_in_method(path, "MXRefitWeightSender", "_initialize")

    assert "apply_rdma_defaults" not in calls
    assert "ModelExpressTrainerClient.initialize" in calls


def test_fsdp_refit_supplies_explicit_trainer_engine_context():
    """FULL_TENSOR initialization must select MX's FSDP geometry adapter."""
    path = _SOURCE_ROOT / "transports" / "weights" / "mx_refit.py"
    tree = ast.parse(path.read_text())
    config_call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ModelExpressTrainerConfig"
    )
    engine_context = next(keyword.value for keyword in config_call.keywords if keyword.arg == "engine_context")

    assert isinstance(engine_context, ast.Call)
    assert isinstance(engine_context.func, ast.Name)
    assert engine_context.func.id == "FSDPTrainerContext"


def test_fsdp_refit_uses_supported_create_version_arguments():
    """PrimeRL must not pass fields absent from MX's public control API."""
    path = _SOURCE_ROOT / "transports" / "weights" / "mx_refit.py"
    tree = ast.parse(path.read_text())
    create_call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "create_weight_version"
    )

    assert {keyword.arg for keyword in create_call.keywords} == {
        "model_name",
        "idempotency_key",
        "payload_format",
        "expected_source_slots",
        # Caller-selected identity. MX would generate one if omitted, but then only
        # the trainer would know it, which is what forced the shared-PVC marker.
        "uid",
    }
