import verifiers.v1 as vf

from prime_rl.orchestrator.trajectories import trace_to_samples

PAD = 9


def _trace(completion_ids: list[int]) -> vf.Trace:
    image = vf.ImageUrlContentPart(image_url=vf.ImageUrlSource(url="data:image/png;base64,"))
    prompt_ids = [1, PAD, PAD, 2]
    nodes = [
        vf.MessageNode(
            message=vf.UserMessage(content=[image, vf.TextContentPart(text="q")]),
            token_ids=prompt_ids,
            mask=[False] * len(prompt_ids),
            logprobs=[0.0] * len(prompt_ids),
        ),
        vf.MessageNode(
            message=vf.AssistantMessage(content="a"),
            token_ids=completion_ids,
            mask=[True] * len(completion_ids),
            logprobs=[-0.1] * len(completion_ids),
            sampled=True,
            parent=0,
        ),
    ]
    return vf.Trace[vf.TaskData](
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt=None)),
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        nodes=nodes,
        mm_token_type_id_map={PAD: 1},
        ok=True,
    )


def test_sampled_image_placeholder_skips_branch():
    [sample] = trace_to_samples(_trace([3, 4]))
    assert [(image.offset, image.length) for image in sample.mm_refs.images] == [(1, 2)]
    assert trace_to_samples(_trace([3, PAD, 4])) == []
