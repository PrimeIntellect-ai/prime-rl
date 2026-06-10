"""Import-census contract test for the pinned verifiers fork.

verifiers is consumed as an editable git-submodule path dep (deps/verifiers),
so uv.lock cannot detect API drift on a repin. This census enumerates every
symbol prime-rl imports from verifiers (derived by grepping src/prime_rl/ and
packages/ for ``from verifiers ... import`` and ``vf.<attr>`` usages). A repin
that drops or moves a symbol fails here, pointedly, instead of at runtime.
"""

import importlib

import pytest

# (module, symbol) — keep flat and boring; regenerate by re-running the grep.
VERIFIERS_CONTRACT: list[tuple[str, str]] = [
    ("verifiers", "ClientConfig"),
    ("verifiers", "ClientType"),
    ("verifiers", "Environment"),
    ("verifiers", "GenerationPlan"),
    ("verifiers", "GenerationTarget"),
    ("verifiers", "MemberGenerationPlan"),
    ("verifiers", "MultiAgentEnv"),
    ("verifiers", "RolloutInput"),
    ("verifiers", "RolloutOutput"),
    ("verifiers", "RolloutTimeoutError"),
    ("verifiers", "TrajectoryStep"),
    ("verifiers", "load_environment"),
    ("verifiers", "rollout_to_member_rollouts"),
    ("verifiers.clients", "resolve_client"),
    ("verifiers.serve", "ZMQEnvClient"),
    ("verifiers.serve", "ZMQEnvServer"),
    ("verifiers.types", "MemberRollout"),
    ("verifiers.utils.async_utils", "EventLoopLagMonitor"),
    ("verifiers.utils.async_utils", "EventLoopLagStats"),
    ("verifiers.utils.async_utils", "maybe_retry"),
    ("verifiers.utils.client_utils", "setup_openai_client"),
    ("verifiers.utils.save_utils", "make_serializable"),
    ("verifiers.utils.save_utils", "state_to_output"),
    ("verifiers.utils.serve_utils", "decode_tensor_payload"),
    ("verifiers.utils.serve_utils", "get_free_port"),
]


@pytest.mark.parametrize("module, symbol", VERIFIERS_CONTRACT, ids=[f"{m}.{s}" for m, s in VERIFIERS_CONTRACT])
def test_verifiers_symbol_exists(module: str, symbol: str):
    mod = importlib.import_module(module)
    assert hasattr(mod, symbol), (
        f"{module}.{symbol} is missing from the pinned verifiers — prime-rl imports it "
        f"(grep src/prime_rl/ and packages/). The repin broke this contract: either "
        f"restore the symbol upstream or port every call site in the same change."
    )


def test_member_rollout_keeps_training_identity_fields():
    """prime-rl keys RAE baselines on (task, example_id, member_id) and trains
    on reward — losing any of these silently corrupts credit assignment."""
    from verifiers.types import MemberRollout

    required = {"task", "example_id", "member_id", "reward"}
    missing = required - set(MemberRollout.__annotations__)
    assert not missing, f"MemberRollout lost training-identity fields: {sorted(missing)}"
