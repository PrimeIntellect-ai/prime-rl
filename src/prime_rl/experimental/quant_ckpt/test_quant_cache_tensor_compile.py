import torch
from torch._dynamo.backends.common import aot_autograd
from torch._dynamo.testing import CompileCounterWithBackend
from torch._functorch.partitioners import min_cut_rematerialization_partition

from prime_rl.experimental.quant_ckpt.fake_kernels import QuantLinear, _calls
from prime_rl.experimental.quant_ckpt.quant_cache_tensor import QuantCacheTensor


def test_cse_dedupes_sibling_calls_under_compile():
    def fn(x):
        a = torch.ops.proto.fake_quantize(x, 2.0)
        b = torch.ops.proto.fake_quantize(x, 2.0)
        return (a + b).sum()

    # Behavioral check: compile with the real "inductor" backend and confirm only one
    # real call happens. NB: CompileCounterWithBackend.graphs captures the dynamo-level
    # graph *before* AOTAutograd's joint-graph partitioning runs, so it still shows two
    # separate nodes even though only one real call executes (see README "Blockers and
    # surprises" — cnt.graphs is the wrong place to look for CSE's effect).
    cnt = CompileCounterWithBackend("inductor")
    compiled_fn = torch.compile(fn, backend=cnt, fullgraph=True)

    x = torch.randn(4, 4, requires_grad=True)
    out = compiled_fn(x)
    out.backward()

    assert _calls["fake_quantize"] == 1

    # Structural check: to see the actual post-CSE forward graph, hook AOTAutograd's own
    # fw_compiler with the same partitioner "inductor" uses internally
    # (min_cut_rematerialization_partition — the one that runs fx_graph_cse). This is the
    # only way to observe CSE's node-merging directly, since a generic aot_autograd()
    # backend's default partitioner does not call fx_graph_cse at all.
    captured = {}

    def fw_compiler(gm, example_inputs):
        captured["fw"] = gm
        return gm.forward

    def bw_compiler(gm, example_inputs):
        return gm.forward

    graph_backend = aot_autograd(
        fw_compiler=fw_compiler, bw_compiler=bw_compiler, partition_fn=min_cut_rematerialization_partition
    )
    graph_compiled_fn = torch.compile(fn, backend=graph_backend, fullgraph=True)
    x2 = torch.randn(4, 4, requires_grad=True)
    graph_compiled_fn(x2).backward()

    fake_quantize_nodes = [
        node
        for node in captured["fw"].graph.nodes
        if node.op == "call_function" and "fake_quantize" in str(node.target)
    ]
    assert len(fake_quantize_nodes) == 1


def test_cse_dedupes_fwd_and_wgrad_across_sibling_calls_under_compile():
    # Compile equivalent of test_fwd_and_wgrad_quantize_dedup_independently_across_sibling_calls:
    # same w1/w3 sharing one h, but here there's no QuantCacheTensor at all — h is a plain
    # tensor, and QuantLinear (defined in fake_kernels) is compiled directly. Checks that
    # compile gets both dedups (fwd-layout cast, wgrad-layout cast) for free, same as
    # claim 2 predicts.
    def fn(h, w1, w3):
        y1 = QuantLinear.apply(h, w1)
        y3 = QuantLinear.apply(h, w3)
        return y1.sum() + y3.sum()

    # Behavioral check: real inductor backend, confirm each op collapses to one real call.
    cnt = CompileCounterWithBackend("inductor")
    compiled_fn = torch.compile(fn, backend=cnt, fullgraph=True)

    torch.manual_seed(0)
    h = torch.randn(4, 3, requires_grad=True)
    w1 = torch.randn(3, 5, requires_grad=True)
    w3 = torch.randn(3, 5, requires_grad=True)
    out = compiled_fn(h, w1, w3)
    out.backward()

    assert _calls["fake_quantize_fwd"] == 1
    assert _calls["fake_quantize_wgrad"] == 1

    # Structural check, same technique as test_cse_dedupes_sibling_calls_under_compile: hook
    # AOTAutograd's own fw_compiler/bw_compiler with the real partitioner to see where the
    # deduped nodes actually land. Surprise: fake_quantize_wgrad(h, 2.0) only depends on h,
    # not on grad_output, so min_cut_rematerialization_partition relocates it into the
    # *forward* graph as a saved intermediate — the backward graph doesn't call it at all,
    # it just consumes the already-computed value. CSE then merges w1's and w3's identical
    # calls the same way regardless of which graph they end up in, so the count is checked
    # across both graphs combined rather than asserting a specific one.
    captured = {"fw": [], "bw": []}

    def fw_compiler(gm, example_inputs):
        captured["fw"].append(gm)
        return gm.forward

    def bw_compiler(gm, example_inputs):
        captured["bw"].append(gm)
        return gm.forward

    graph_backend = aot_autograd(
        fw_compiler=fw_compiler, bw_compiler=bw_compiler, partition_fn=min_cut_rematerialization_partition
    )
    graph_compiled_fn = torch.compile(fn, backend=graph_backend, fullgraph=True)
    h2 = torch.randn(4, 3, requires_grad=True)
    w1_2 = torch.randn(3, 5, requires_grad=True)
    w3_2 = torch.randn(3, 5, requires_grad=True)
    graph_compiled_fn(h2, w1_2, w3_2).backward()

    def count_nodes(gms, needle):
        return sum(
            1
            for gm in gms
            for node in gm.graph.nodes
            if node.op == "call_function" and needle in str(node.target)
        )

    assert count_nodes(captured["fw"] + captured["bw"], "fake_quantize_fwd") == 1
    assert count_nodes(captured["fw"] + captured["bw"], "fake_quantize_wgrad") == 1
    # Confirmed empirically, not assumed: both land in the forward graph as call_function
    # nodes. (The backward graph's *code* does mention "fake_quantize_wgrad" — but only as
    # a placeholder parameter name for the already-computed saved value, not a call node;
    # count_nodes filters on node.op == "call_function" specifically to avoid that trap.)
    assert count_nodes(captured["fw"], "fake_quantize_wgrad") == 1
    assert count_nodes(captured["bw"], "fake_quantize_wgrad") == 0


def test_eager_does_not_dedupe_for_contrast():
    def fn(x):
        a = torch.ops.proto.fake_quantize(x, 2.0)
        b = torch.ops.proto.fake_quantize(x, 2.0)
        return (a + b).sum()

    x = torch.randn(4, 4, requires_grad=True)
    out = fn(x)
    out.backward()

    assert _calls["fake_quantize"] == 2


def test_quant_cache_tensor_under_compile():
    # Exploratory: pre-warm the cache *eagerly* on a wrapped tensor before it ever enters
    # a compiled region, then mutate the underlying data in place (same identity, same
    # shape/dtype, so no recompile) and call the compiled function again. This is the
    # scenario that could trigger claim 1's "frozen stale result" failure mode, since it
    # gives dynamo a genuinely warm cache to observe at first trace time.
    def fn(w):
        return torch.ops.proto.fake_quantize(w, 2.0)

    x1 = torch.ones(4, 4)
    wrapped = QuantCacheTensor.from_tensor(x1)
    eager_result = torch.ops.proto.fake_quantize(wrapped, 2.0)  # warms wrapped._cache
    assert _calls["fake_quantize"] == 1

    cnt = CompileCounterWithBackend("inductor")
    compiled_fn = torch.compile(fn, backend=cnt)

    graph_break = False
    try:
        r1 = compiled_fn(wrapped)
        wrapped._data.copy_(torch.full((4, 4), 2.0))  # mutate in place, no shape/dtype change
        r2 = compiled_fn(wrapped)
    except Exception as e:
        graph_break = True
        print(f"Compiled call raised: {e!r}")

    print(f"frame_count={cnt.frame_count}")
    if cnt.graphs:
        cnt.graphs[-1].print_readable()

    if not graph_break:
        expected_r2 = torch.full((4, 4), 4.0, dtype=torch.float16)
        print(f"r1={r1}, r2={r2}, expected_r2={expected_r2}, _calls={_calls}")
        # Observed: no graph break, no stale/frozen value. r2 correctly reflects the
        # mutated data, and _calls increments on every compiled invocation (2 -> 3),
        # proving the real op reruns each time rather than returning a cached result.
        # See README "Blockers and surprises" for why: __tensor_flatten__ excludes
        # _cache, so __tensor_unflatten__ always reconstructs the subclass with a fresh,
        # empty cache during trace-time fakification — dynamo's traced view of the
        # object never sees a warm cache, so it can only ever bake in "always call the
        # real op." This is what makes claim 1's correctness-bug half unreachable for
        # this specific implementation, at the cost of the cache being fully inert
        # (silently never helps) under compile — confirming the other half of claim 1.
        assert torch.equal(r2, expected_r2)
        assert not torch.equal(r2, eager_result)
        assert _calls["fake_quantize"] == 3
        assert cnt.frame_count == 1
    else:
        # dynamo declined to trace through the subclass at all.
        assert cnt.frame_count == 0
