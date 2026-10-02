import inspect

import pytest

modelexpress_rl = pytest.importorskip("modelexpress_rl")

from prime_rl.inference.vllm.worker.mx_refit import MXRefitUpdateWorker  # noqa: E402

VERSION = "run:3"


class _Generator:
    def __init__(self):
        self.calls = []

    def apply_weight_streaming(self, **kwargs):
        self.calls.append(("apply_weight_streaming", kwargs))
        return {"batches": 39}

    def stage_weight(self, **kwargs):
        self.calls.append(("stage_weight", kwargs))
        raise AssertionError("a staging budget must select bounded streaming")


def _worker(generator):
    worker = object.__new__(MXRefitUpdateWorker)
    worker._generator = generator
    worker.rank = 0
    return worker


def test_staging_budget_routes_through_bounded_streaming(monkeypatch):
    monkeypatch.setenv("MX_REFIT_STAGING_BYTES", str(2 * 1024**3))
    generator = _Generator()

    _worker(generator).update_weights_from_path(version_uid=VERSION)

    [(method, kwargs)] = generator.calls
    assert method == "apply_weight_streaming"
    assert kwargs["version"] == modelexpress_rl.WeightVersionRef(VERSION)
    assert kwargs["max_staging_bytes"] == 2 * 1024**3


def test_bounded_streaming_call_matches_the_installed_client():
    """A signature change in ModelExpress should fail here, not on the cluster."""
    client = modelexpress_rl.ModelExpressGeneratorClient
    assert hasattr(client, "apply_weight_streaming"), (
        "installed modelexpress_rl has no bounded streaming; GLM needs ai-dynamo/modelexpress#749"
    )
    inspect.signature(client.apply_weight_streaming).bind(
        None, version=modelexpress_rl.WeightVersionRef(VERSION), max_staging_bytes=1
    )
