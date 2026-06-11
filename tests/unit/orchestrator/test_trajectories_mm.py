"""Multimodal egress through the v1 message graph.

A VLM rollout's images ride transiently on the graph (`MessageNode.multi_modal_data`): each
image is attributed to the node whose message introduced it (by content part), `Branch.multi_modal_data`
concatenates them in path order, and `trace_to_samples` rebuilds the flat `mm_kwargs` /
`mm_token_type_ids` the trainer forwards. These tests pin that chain end-to-end on synthetic
traces (no model), including the reused-prefix skip across turns and the exclude-from-wire invariant.
"""

import json

import numpy as np
import torch
import verifiers.v1 as vf
from renderers.base import MultiModalData
from verifiers.v1 import graph
from verifiers.v1.task import Task
from verifiers.v1.types import AssistantMessage, Response, TurnTokens

IMG = 999  # stand-in for an image-pad token id


def _image_mmd(lengths: list[int], fill: float) -> MultiModalData:
    """A MultiModalData for `len(lengths)` images, in prompt order; each `pixel_values` is a
    length×4 block (distinct per image via `fill`). Items are attributed by content part, so no
    placeholders are needed."""
    items = [
        {
            "pixel_values": np.full((ln, 4), fill, dtype=np.float32),
            "image_grid_thw": np.array([[1, ln, 1]], dtype=np.int64),
        }
        for ln in lengths
    ]
    return MultiModalData(
        mm_hashes={"image": [f"h{i}" for i in range(len(items))]},
        mm_items={"image": items},
    )


def _response(prompt_ids, comp_ids, spans, mmd, content="x") -> Response:
    return Response(
        id="r",
        created=0,
        model="m",
        message=AssistantMessage(content=content),
        finish_reason="stop",
        tokens=TurnTokens(
            prompt_ids=prompt_ids,
            completion_ids=comp_ids,
            completion_logprobs=[-0.1] * len(comp_ids),
            message_spans=spans,
            multi_modal_data=mmd,
        ),
    )


def _img_part(url: str) -> vf.ImageUrlContentPart:
    return vf.ImageUrlContentPart(image_url=vf.ImageUrlSource(url=url))


def test_single_image_branch_roundtrip():
    user = vf.UserMessage(content=[vf.TextContentPart(text="what color?"), _img_part("data:x")])
    prompt_ids = [10, 11, IMG, IMG, 12]  # image-pad at indices 2,3
    comp_ids = [20, 21]
    mmd = _image_mmd([2], fill=1.0)
    trace = vf.Trace[Task](task=Task(idx=0, instruction=[user]))
    graph.add_turn(trace, [user], _response(prompt_ids, comp_ids, [(0, 5)], mmd))

    # Attribution: the image's item lands on the user node that introduced it.
    node = trace.nodes[0]
    assert node.multi_modal_data is not None
    assert len(node.multi_modal_data.mm_items["image"]) == 1

    branch = trace.branches[0]
    assert branch.token_ids == prompt_ids + comp_ids  # concat invariant
    assert len(branch.multi_modal_data.mm_items["image"]) == 1

    sample = vf_trace_to_samples(trace, {IMG: 1})[0]
    # mm_token_type_ids come from the token ids (the image-pad mapping), not placeholders.
    assert sample.mm_token_type_ids == [0, 0, 1, 1, 0, 0, 0]
    ones = [i for i, v in enumerate(sample.mm_token_type_ids) if v == 1]
    assert ones == [2, 3]  # exactly the image-pad token positions

    pv = sample.mm_kwargs["pixel_values"]
    assert pv.shape == [2, 4] and pv.dtype == "float32"
    decoded = torch.frombuffer(bytearray(pv.data), dtype=getattr(torch, pv.dtype)).reshape(pv.shape)
    assert torch.allclose(decoded, torch.full((2, 4), 1.0))

    # exclude-from-wire: no pixel tensors in the serialized trace
    wire = json.dumps(trace.to_wire())
    assert "pixel_values" not in wire and "image_grid_thw" not in wire


def test_multiturn_multi_image():
    """Two turns, one image each. The prefix (turn-0 user + assistant) is reused; the turn-1
    image attaches to the new user node, and the two images concatenate in path (token) order —
    the reused prefix keeps its own image (not overwritten by the re-render)."""
    user0 = vf.UserMessage(content=[vf.TextContentPart(text="t0"), _img_part("data:a")])
    # turn 0: [10, IMGA, 11] -> completion [20]; image A
    r0 = _response([10, IMG, 11], [20], [(0, 3)], _image_mmd([1], fill=10.0), content="x")
    trace = vf.Trace[Task](task=Task(idx=0, instruction=[user0]))
    graph.add_turn(trace, [user0], r0)
    assistant0 = trace.nodes[1].message  # reuse the exact assistant message so the prefix dedups

    user1 = vf.UserMessage(content=[vf.TextContentPart(text="t1"), _img_part("data:b")])
    # turn 1 re-renders the whole convo: [10, IMGA, 11, 20, 30, IMGB, 31] -> completion [40].
    # mmd carries both images (A reused -> skipped, B new -> attached to the user1 node).
    r1 = _response([10, IMG, 11, 20, 30, IMG, 31], [40], [(0, 3), (3, 4), (4, 7)], _image_mmd([1, 1], fill=20.0))
    graph.add_turn(trace, [user0, assistant0, user1], r1)

    branch = trace.branches[0]
    assert branch.token_ids == [10, IMG, 11, 20, 30, IMG, 31, 40]
    assert len(branch.multi_modal_data.mm_items["image"]) == 2  # both images, in token order

    sample = vf_trace_to_samples(trace, {IMG: 1})[0]
    assert sample.mm_token_type_ids == [0, 1, 0, 0, 0, 1, 0, 0]
    pv = sample.mm_kwargs["pixel_values"]
    assert pv.shape == [2, 4]  # two 1-row images concatenated on dim 0
    decoded = torch.frombuffer(bytearray(pv.data), dtype=getattr(torch, pv.dtype)).reshape(pv.shape)
    # image order is token order: A (fill 10, kept from turn 0) then B (fill 20, turn 1)
    assert decoded[0, 0].item() == 10.0 and decoded[1, 0].item() == 20.0


def vf_trace_to_samples(trace, mapping):
    from prime_rl.orchestrator.trajectories import trace_to_samples

    return trace_to_samples(trace, env_name="t", mm_token_type_ids_mapping=mapping)
