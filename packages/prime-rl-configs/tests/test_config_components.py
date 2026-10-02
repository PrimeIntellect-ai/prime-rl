from types import SimpleNamespace

import pytest
from pydantic import create_model

from prime_rl.utils.config import dump_rl_components


@pytest.mark.parametrize("multi_node", [False, True])
def test_component_serialization_keeps_new_fields_and_nulls(multi_node):
    Component = create_model("Component", new_option=(int | None, None))
    Inference = create_model(
        "Inference",
        new_option=(int | None, None),
        deployment=(dict, {}),
        router=(dict | None, {"port": 8000}),
        output_dir=(str, "outputs"),
    )
    config = SimpleNamespace(
        trainer=Component(),
        orchestrator=Component(new_option=7),
        inference=Inference(),
        deployment=SimpleNamespace(type="multi_node" if multi_node else "single_node"),
    )
    components = dump_rl_components(config)
    assert components["trainer"]["new_option"] is None
    assert components["orchestrator"]["new_option"] == 7
    assert "deployment" not in components["inference"]
    assert "output_dir" not in components["inference"]
    assert components["inference"]["router"] == (None if multi_node else {"port": 8000})
    assert config.inference.router == {"port": 8000}
    config.inference = None
    assert set(dump_rl_components(config)) == {"trainer", "orchestrator"}
