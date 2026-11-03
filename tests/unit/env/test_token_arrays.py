import numpy as np
from verifiers.v1.graph import MessageNode
from verifiers.v1.types import AssistantMessage

from prime_rl.env.delta import dump, pack, unpack


def assistant(**fields) -> MessageNode:
    return MessageNode(message=AssistantMessage(content="a"), sampled=True, **fields)


def test_token_arrays_wire_round_trip():
    node = assistant(
        token_ids=[1, 2, 130_000],
        mask=[False, True, True],
        logprobs=[-0.125, -0.25],
        loss_weights={"rl": [0.0, 1.0, 1.0]},
    )
    wire = unpack(pack(dump(node, context={"token_arrays": True})))
    assert wire["token_ids"]["__nd__"] and wire["loss_weights"]["rl"]["__nd__"]
    back = MessageNode.model_validate(wire)
    assert back.token_ids.tolist() == [1, 2, 130_000]
    assert back.mask.tolist() == [False, True, True]
    assert back.logprobs.tolist() == [-0.125, -0.25]
    assert back.loss_weights["rl"].tolist() == [0.0, 1.0, 1.0]
    # Without the context the wire keeps lists, which validate the same way.
    plain = unpack(pack(dump(node)))
    assert plain["token_ids"] == [1, 2, 130_000]
    assert MessageNode.model_validate(plain).token_ids.dtype == np.int32
