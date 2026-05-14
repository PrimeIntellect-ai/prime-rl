# Tree Training on Hybrid Models

This note evaluates how Tree Training interacts with hybrid models that use
linear attention or SSM layers, such as Qwen3.5-style GatedDeltaNet or
Nemotron-H Mamba layers.

## Current Prod Hybrid Path

The current production hybrid path is optimized for packed sequences with a
linear history. It preserves the main training stack:

- Flash-linear-attention `chunk_gated_delta_rule` for Qwen3.5 linear attention.
- Causal Conv1d sequence-boundary resets through `seq_idx`.
- `cu_seqlens` derived from packed `position_ids`.
- Flash attention for full-attention layers.
- FSDP, grouped MoE, FP8 linear replacement, fused loss or fused LM head when
  configured.
- Context parallelism for hybrid layers only under `cp_style = "ulysses"`.

The key assumption is that every packed segment is an ordinary sequence. A
DFS-packed tree violates this: sibling branches must inherit the same parent
state, not the final state of the previous sibling in DFS order.

The current Tree Training fast path is therefore dense/full-attention only. Its
FlexAttention block mask can constrain softmax attention, but it cannot make a
linear-attention or SSM scan branch-aware.

## Why Hybrid Tree Packing Is Hard

Linear attention and SSM layers are recurrent scans. For a tree, each node needs
to run from its parent node's final state:

```text
state[root] = zero
for node in topological_tree_order:
    outputs[node], state[node] = mixer(
        tokens=node.tokens,
        initial_state=state[parent(node)],
    )
```

For Qwen3.5-style GatedDeltaNet, the scan state is not the only state that must
branch correctly. The causal Conv1d before the scan also has history, so each
node must inherit the parent path's last `kernel_size - 1` projected tokens.

For Mamba-style layers, the same issue exists with the SSM state and the
pre-scan convolution state, but the exposed kernel API and state layout differ.

## Option 1: Branchwise Linear Mixer

Expand each root-to-leaf branch only for hybrid token-mixer layers, run the
existing linear-attention or Mamba kernel on ordinary branch sequences, then
scatter the outputs back into the packed tree layout.

This is the safest short-term option and is the easiest to prove correct.

Preserves:

- Existing FLA or Mamba kernels.
- Existing Conv1d reset behavior through `cu_seqlens` or `seq_idx`.
- Most model, optimizer, FSDP, MoE, dtype, and loss infrastructure.
- Full-attention tree masking for dense attention layers if implemented as a
  mixed path.

Loses:

- Deduplication inside the linear-attention or SSM layers.
- Much of the expected Tree Training speedup on heavily hybrid models, because
  most layers may be linear-attention or SSM layers.
- Clean context-parallel behavior without additional work.
- Some memory and launch efficiency due to branch gather/scatter.

Assessment: good as an exact reference or bridge, but not a strong production
performance story. On heavily branched caterpillars it can collapse toward
per-branch cost for the dominant hybrid layers.

## Option 2: Tree-State Scan

Run each tree node once. For every hybrid layer, initialize the node scan from
its parent node's final recurrent state and parent convolution cache, then save
the node's final state for its children.

For Qwen3.5 GatedDeltaNet, this is plausible because the FLA kernel already
supports variable-length inputs with per-segment initial states:

```text
initial_state: [N, H, K, V]
cu_seqlens:    [N + 1]
final_state:   [N, H, K, V]
```

Prime-RL's current wrapper uses the production linear-sequence mode and passes
`initial_state = None`, so Tree Training would need a new tree-aware path around
that existing FLA kernel.

The main implementation risk is not mathematical correctness, but avoiding a
separate kernel launch per small tree node. The path should be a segmented
wavefront scan:

1. Collapse single-child chains and split only at branch points. For
   caterpillar-shaped data, this makes launches proportional to branch depth,
   not token count or node count.
2. Batch all segments whose parent state is already available. Gather their
   parent final states into `initial_state`, flatten the child segments, build
   `cu_seqlens`, and call `chunk_gated_delta_rule` once for the whole wavefront.
3. Run pointwise work once over the packed tree tensor. Input projections,
   gates, norms, and output projection should stay dense over the packed tree;
   only the recurrent scan needs tree semantics.
4. Handle the pre-scan causal Conv1d with cached-prefix padding first: prepend
   each segment with the parent path's last `kernel_size - 1` projected tokens,
   run batched causal Conv1d over flattened segments, then drop prefix outputs.
5. Add bucketing or CUDA graphs only after the wavefront path works. They reduce
   CPU overhead but do not replace batching segments into fewer GPU launches.

This should avoid death by tiny kernels on caterpillar-like examples. It will
not be as efficient as one contiguous linear-sequence FLA call, but it preserves
tree deduplication and reuses the mature production scan kernel.

Preserves:

- Tree deduplication for linear-attention layers.
- Existing FLA math kernels, at least per node.
- Full-attention Tree Training via FlexAttention for full-attention layers.
- Most optimizer, FSDP, MoE, dtype, and loss machinery.

Loses:

- The current single large contiguous FLA call.
- Some launch efficiency from wavefront-level segmented calls.
- Current context-parallel assumptions: the Qwen3.5 CP context currently assumes
  one global linear sequence, not a branching state graph.
- Simplicity; activation and state bookkeeping become model-specific.

Assessment: the best practical production direction. It should start with
Qwen3.5 GatedDeltaNet, because its state API is closest to what Tree Training
needs. A prototype is likely days to a week; a robust production path is more
like one to three weeks, mostly because of state bookkeeping, conv-cache
inheritance, equivalence tests, and performance tuning.

## Option 3: Custom Tree Scan Kernel

Build a native tree-aware scan kernel for each hybrid token mixer. The kernel
would handle state fanout from parent nodes to children, convolution-cache
inheritance, and recurrent state updates without materializing per-branch
duplicates.

For Qwen3.5 GatedDeltaNet, this means replacing or extending the FLA segmented
scan with a kernel that understands tree metadata directly. The forward pass
needs to propagate parent states to children and produce per-node final states.
The backward pass is harder: gradients flowing from multiple children must
accumulate into the parent recurrent state and convolution cache in a way that
matches the per-branch reference.

Preserves:

- Tree deduplication.
- Large fused-kernel execution.
- The best chance of matching production hybrid throughput.
- A path toward context-parallel tree execution.

Loses:

- Reuse of mature production kernels.
- Simplicity and short-term maintainability.
- Model generality: Qwen3.5 GatedDeltaNet, Mamba, and other hybrid mixers have
  different state layouts and recurrence semantics.

Assessment: this is the eventual high-performance path, but it is a kernel
project, not a small extension to the current Tree Training PR. A forward-only
prototype may be manageable, but a production autograd kernel with numerics,
ragged metadata, conv-cache handling, tests, and CP support is likely a
multi-week effort for Qwen3.5 alone. A generic hybrid kernel covering
GatedDeltaNet and Mamba/Nemotron-H should be treated as a separate project.

## Recommendation

For the current Tree Training PR, hybrid linear-attention and SSM models should
be explicitly guarded as unsupported unless an experimental hybrid mode is
enabled.

Recommended follow-up sequence:

1. Add a runtime/config guard for Tree Training on models with
   `layer_type == "linear_attention"` or Mamba layers.
2. Implement `hybrid_tree_mode = "branchwise_mixer"` as an exact reference.
3. Implement a Qwen3.5-specific tree-state scan path using FLA
   `initial_state` / `output_final_state` plus parent convolution-cache
   inheritance, with wavefront batching to avoid per-node kernel launches.
4. Treat Mamba/Nemotron-H as a separate follow-up because the state API differs.
5. Consider custom kernels only after the tree-state scan demonstrates a useful
   end-to-end gain.
