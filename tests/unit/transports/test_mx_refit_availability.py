"""Optional ModelExpress client tests."""

import importlib
import sys

import pytest

import prime_rl.transports.weights as weights


@pytest.fixture
def without_client(monkeypatch):
    monkeypatch.setitem(sys.modules, "modelexpress_rl", None)
    monkeypatch.delitem(sys.modules, "prime_rl.transports.weights.mx_refit", raising=False)


def test_factory_imports_without_client(without_client):
    reloaded = importlib.reload(weights)
    assert reloaded.setup_weight_sender is not None


def test_selecting_mx_refit_without_the_client_says_so(without_client):
    with pytest.raises(ImportError, match="modelexpress_rl"):
        weights._mx_refit_classes()
