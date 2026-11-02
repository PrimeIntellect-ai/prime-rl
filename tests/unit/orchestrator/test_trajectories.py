import pytest
import verifiers.v1 as vf

from prime_rl.orchestrator.trajectories import trace_to_samples

PAD = 9
IMAGE = vf.ImageUrlContentPart(image_url=vf.ImageUrlSource(url="data:image/png;base64,"))


def _node(message, token_ids, *, parent=None, sampled=False, placeholders=None):
    return vf.MessageNode(
        message=message,
        token_ids=token_ids,
        mask=[sampled] * len(token_ids),
        logprobs=[-0.1] * len(token_ids) if sampled else [],
        sampled=sampled,
        parent=parent,
        mm_placeholders=placeholders or {},
    )


def _trace(nodes):
    return vf.Trace[vf.TaskData](
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt=None)),
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        nodes=nodes,
        ok=True,
    )


def test_placeholder_token_ids_outside_images_train_as_text():
    """A sampled pad and a pad in tool text (before the tool's image) are text tokens."""
    nodes = [
        _node(
            vf.UserMessage(content=[IMAGE, vf.TextContentPart(text="q")]),
            [1, PAD, PAD, 2],
            placeholders={"image": [(1, 2)]},
        ),
        _node(vf.AssistantMessage(content="a"), [3, PAD, 4], parent=0, sampled=True),
        _node(
            vf.ToolMessage(content=[vf.TextContentPart(text="pad"), IMAGE], tool_call_id="c"),
            [5, PAD, 6, PAD, PAD, PAD],
            parent=1,
            placeholders={"image": [(3, 3)]},
        ),
        _node(vf.AssistantMessage(content="b"), [7, 8], parent=2, sampled=True),
    ]
    [sample] = trace_to_samples(_trace(nodes))
    assert [(image.offset, image.length) for image in sample.mm_refs.images] == [(1, 2), (10, 3)]
    assert sample.mm_token_type_ids.tolist() == [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0]


def test_image_without_placeholder_is_an_error():
    nodes = [
        _node(vf.UserMessage(content=[IMAGE, vf.TextContentPart(text="q")]), [1, PAD, 2]),
        _node(vf.AssistantMessage(content="a"), [3], parent=0, sampled=True),
    ]
    with pytest.raises(ValueError, match="1 inline images but 0 image placeholder ranges"):
        trace_to_samples(_trace(nodes))
