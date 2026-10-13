"""Compare #4074's PyTorch combine with the experimental fused CUDA path."""

import argparse
import json

import torch

from prime_rl.trainer.distributed.token_dispatcher import _combine_routed_output


def measure(fn, repeats, stream):
    for _ in range(6):
        fn()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    allocated = torch.cuda.memory_allocated()
    result = fn()
    torch.cuda.synchronize()
    extra_bytes = torch.cuda.max_memory_allocated() - allocated
    del result

    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    end.synchronize()
    launch_ms = start.elapsed_time(end) / repeats

    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(6):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured_result = fn()
    for _ in range(6):
        graph.replay()
    start.record()
    for _ in range(repeats):
        graph.replay()
    end.record()
    end.synchronize()
    graph_ms = start.elapsed_time(end) / repeats
    del captured_result
    return dict(launch_ms=launch_ms, cuda_graph_ms=graph_ms, peak_extra_bytes=extra_bytes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tokens", type=int)
    parser.add_argument("hidden", type=int)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    values = torch.randn(args.tokens * args.top_k, args.hidden, device="cuda", dtype=torch.bfloat16)
    order = torch.randperm(len(values), device="cuda")
    scores = torch.rand(len(values), device="cuda")
    grad = torch.randn(args.tokens, args.hidden, device="cuda", dtype=torch.bfloat16)
    report = dict(torch=torch.__version__, device=torch.cuda.get_device_name(), shape=list(values.shape), modes=[])
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for weighted in (True, False):
            weights = scores if weighted else None
            for fused in (False, True):

                def combine(x, indices, s):
                    return _combine_routed_output(
                        x,
                        num_tokens=args.tokens,
                        route_indices_experts_sorted=indices,
                        scores_after_experts=s,
                        use_fused=fused,
                    )

                for compiled in (False, True):
                    fn = torch.compile(combine, fullgraph=True, dynamic=True) if compiled else combine
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        x = values.clone().requires_grad_()
                        s = weights.clone().requires_grad_() if weighted else None
                        inputs = (x, s) if weighted else (x,)
                        output = fn(x, order, s)
                    torch.cuda.current_stream().wait_stream(stream)

                    def forward():
                        with torch.no_grad():
                            return fn(values, order, weights)

                    def backward():
                        return torch.autograd.grad(output, inputs, grad, retain_graph=True)

                    def forward_backward():
                        return torch.autograd.grad(fn(x, order, s), inputs, grad)

                    report["modes"].append(
                        dict(
                            weighted=weighted,
                            fused=fused,
                            compiled=compiled,
                            forward=measure(forward, args.repeats, stream),
                            backward=measure(backward, args.repeats, stream),
                            forward_backward=measure(forward_backward, args.repeats, stream),
                        )
                    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
