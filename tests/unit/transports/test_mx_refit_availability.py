"""Optional ModelExpress client tests."""

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest

import prime_rl.transports.weights as weights
from prime_rl.utils.mx_compat import import_model_express, require_mx_refit


@pytest.fixture
def without_client(monkeypatch):
    for module in (
        "modelexpress",
        "modelexpress.client",
        "modelexpress_rl",
        "prime_rl.transports.weights.nixl",
        "prime_rl.transports.weights.mx_refit",
    ):
        monkeypatch.setitem(sys.modules, module, None)


def test_factory_imports_without_client(without_client):
    reloaded = importlib.reload(weights)
    assert reloaded.setup_weight_sender is not None


@pytest.mark.parametrize("transport", ["nixl", "mx_refit"])
def test_selecting_mx_refit_without_the_client_says_so(without_client, transport):
    with pytest.raises(ImportError, match=rf"{transport} requires.*docs/scaling.md"):
        getattr(weights, f"_{transport}_classes")()


def test_missing_client_dependency_is_not_reported_as_missing_client(monkeypatch):
    failure = ModuleNotFoundError("No module named 'grpc'", name="grpc")

    def missing_dependency(name):
        raise failure

    monkeypatch.setattr(importlib, "import_module", missing_dependency)
    with pytest.raises(ModuleNotFoundError) as raised:
        import_model_express("modelexpress_rl", transport="mx_refit")
    assert raised.value is failure


@pytest.fixture
def client_api(monkeypatch):
    module = ModuleType("modelexpress_rl")
    for name in (
        "ModelExpressControlClient",
        "ModelExpressGeneratorConfig",
        "ModelExpressTrainerClient",
        "ModelExpressTrainerConfig",
        "VllmGeneratorContext",
        "WeightPayloadFormat",
        "WeightSource",
        "WeightVersionRef",
        "WeightVersionState",
    ):
        setattr(module, name, object())
    module.FSDPTrainerContext = lambda wire_dtype_overrides=None: None
    module.TrainerStagingMode = SimpleNamespace(COPY_TO_HOST="COPY_TO_HOST")
    module.ModelExpressGeneratorClient = SimpleNamespace(
        apply_weight_streaming=lambda *, version, max_staging_bytes: None
    )
    monkeypatch.setitem(sys.modules, "modelexpress_rl", module)
    return module


@pytest.mark.parametrize("missing", [None, "wire_dtype_overrides", "COPY_TO_HOST", "apply_weight_streaming"])
def test_client_capabilities_fail_before_initializing_a_model(client_api, missing):
    if missing == "wire_dtype_overrides":
        client_api.FSDPTrainerContext = lambda: None
    elif missing == "COPY_TO_HOST":
        client_api.TrainerStagingMode = SimpleNamespace()
    elif missing == "apply_weight_streaming":
        client_api.ModelExpressGeneratorClient = SimpleNamespace(apply_weight_streaming=lambda *, version: None)

    if missing is None:
        assert require_mx_refit(staging_mode="COPY_TO_HOST", streaming=True) is client_api
    else:
        with pytest.raises(ImportError, match=missing):
            require_mx_refit(staging_mode="COPY_TO_HOST", streaming=True)
        if missing == "apply_weight_streaming":
            assert require_mx_refit(staging_mode="COPY_TO_HOST") is client_api
