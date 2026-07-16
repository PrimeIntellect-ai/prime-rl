import torch

from prime_rl.weight_transfer.kernel_graph import KernelGraphRecorder


def test_records_and_replays_qwen_style_kernel_layout_and_dtype() -> None:
    w13 = torch.arange(2 * 8 * 4, dtype=torch.bfloat16).reshape(2, 8, 4)
    router = torch.arange(8, dtype=torch.bfloat16)

    recorder = KernelGraphRecorder()
    recorder.add_input("w13_weight", w13)
    recorder.add_input("e_score_correction_bias", router)
    with recorder:
        gate, up = w13.chunk(2, dim=1)
        kernel_w13 = torch.cat((up, gate), dim=1).transpose(-1, -2).contiguous()
        kernel_router = router.to(torch.float32)
    graph = recorder.finish(
        {
            "w13_weight": kernel_w13,
            "e_score_correction_bias": kernel_router,
        }
    )

    next_w13 = (w13 + 5).clone()
    next_router = (router + 3).clone()
    graph = type(graph).decode(graph.encode())
    outputs = graph.replay(
        {
            "w13_weight": next_w13,
            "e_score_correction_bias": next_router,
        }
    )

    gate, up = next_w13.chunk(2, dim=1)
    assert torch.equal(outputs["w13_weight"], torch.cat((up, gate), dim=1).transpose(-1, -2).contiguous())
    assert outputs["e_score_correction_bias"].dtype is torch.float32
    assert torch.equal(outputs["e_score_correction_bias"], next_router.float())


def test_captures_input_independent_tensor_constants() -> None:
    weight = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    permutation = torch.tensor([2, 0, 1])

    recorder = KernelGraphRecorder()
    recorder.add_input("weight", weight)
    with recorder:
        output = weight.index_select(0, permutation)
    graph = recorder.finish({"weight": output})
    graph = type(graph).decode(graph.encode())

    updated = weight + 100
    assert torch.equal(graph.replay({"weight": updated})["weight"], updated[permutation])
