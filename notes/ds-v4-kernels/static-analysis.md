# Static analysis: by-eye bottlenecks

Merged from four independent read-only reviews (attention core; compressor and indexer; mHC and
norms; packing, rotary and plumbing), deduplicated and ranked. Every finding is annotated with
what the measurements in `measured.md` say about it, because several by-eye estimates did not
survive contact with the profiler.

Format: `severity | file:line | what is allocated or launched | why it is wasteful | what would
replace it | saving`. Bytes are `b = 1`, bf16, per layer unless stated. `e = t/4` for CSA,
`t/128` for HCA.

## Confirmed by measurement

### CRITICAL

**`attention.py:155-162` - the dense score tensor, three copies live at once.**
`eager_attention_with_sinks` allocates six sequential full-size `(1, h, t, t+e+1)` bf16 buffers:
the `bmm` output, `* scaling`, `+ attention_mask`, the `cat` with sink logits, the row-max
subtraction, and the softmax. Three are live simultaneously at the peak and one is retained for
backward.
*Measured:* exactly. The fitted model is `fwd_peak = 3.005 * S + 405 KB/token` with 1.1% worst
error, and the attribution log names five distinct 10.00 GB allocations at `(1, 64, 8192, 10241)`
for `attn-csa` at `t = 8192`: `aten.bmm`, `aten.mul`, `aten.sub`, `aten.cat`, and
`aten._softmax_backward_data`. This is 97% of a whole decoder layer's forward transient (33.35 GB
of 34.50 GB, MoE included).
*Replace with:* a flash/streaming kernel with online softmax, the per-head sink folded into the
running denominator, tiled over the 128-wide band plus the gathered top-k entry columns.
*Saving:* everything quadratic. `641 / 81921 = 0.8%` of CSA score columns at `t = 65536` are ever
unmasked. See `memory-model.md`.

**`attention.py:155` and `:165` - dense QK^T and PV over the full `t x (t+e)` grid.**
Only a 128-wide causal band of the token columns is ever unmasked, and only `index_topk = 512` of
the `e` entry columns survive the indexer's selection. Everything else is computed, written,
masked to `finfo.min`, exponentiated to zero, and multiplied by V.
*Measured:* `attn-sliding` forward+backward is 810 ms at `t = 16384`; the useful fraction of the
band work is `128 / 16384 = 0.8%`.
*Saving:* ~64x on the band at `t = 8192`, ~256x at `t = 32768`.

**`attention.py:183-189` - `build_sliding_window_mask` materializes `13 t^2` bytes of transients.**
An int64 `(t,t)` `distance` (8 bytes per element, four times the size of the bf16 mask it
produces), five `(t,t)` bool tensors, a `(t,t)` zeros, and a `masked_fill_`, all to express a
128-wide causal band clipped at document boundaries.
*Measured:* `packed-context` peaks at 14.00 GB at `t = 32768` against a predicted `13 t^2` =
13.6 GB, and retains 2.01 GB, which then lives for the whole 43-layer forward.
*Replace with:* `(cu_seqlens, sliding_window)` passed to the kernel; build nothing.

### HIGH

**`attention.py:463-466` - the Lightning Indexer's three fp32 `(1, t, 64, e)` copies.**
The `matmul` output, the `relu` output, and the `scores * weights` product, each
`256 * t * e = 64 t^2` bytes in fp32 for CSA.
*Measured:* the attribution log shows all three at exactly 4.00 GB for `t = 8192`, and the
`indexer` module peaks at 12.46 GB. But see the correction below: this peak is **not additive**
with the attention peak, so its contribution to the binding constraint is zero.
*Replace with:* folding `softmax_scale` into `weights` and using
`torch.einsum("bthe,bth->bte", scores, weights)` removes two of the three copies in two lines. A
fused kernel `out[t,e] = scale * sum_h w[t,h] * relu(sum_d q[t,h,d] k[e,d])` removes the last.
*Saving:* 8 GB of transient per CSA layer at `t = 8192`, 128 GB at `t = 32768`. Worth doing, but
it does not move the OOM ceiling.

**`attention.py:672-674` - the mask `cat` onto the entry axis, rebuilt in all 41 compressed layers.**
`torch.cat([attention_mask.expand(...), block_bias], dim=-1)` copies the shared `(t,t)` half every
time just to append `block_bias`.
*Saving:* 168 MB per CSA layer at `t = 8192`, 2.68 GB at `t = 32768`; 6.2 GB and 99.5 GB summed
over 41 layers.

**`attention.py:547-549` - CSA `block_bias` is a dense `(1,1,t,e+1)` `-inf` fill for 512 zeros/row.**
Density of non-`-inf` cells is 21.9% at `t = 8192` and 6.05% at `t = 32768`.
*Replace with:* keep `top_k_indices` as the representation and let the attention kernel gather.
*Saving:* 33.6 MB per layer at `t = 8192`, 537 MB at `t = 32768`, plus it is the enabling change
for the sparse kernel.

**`hyperconnections.py:51` - three full-width fp32 `(1, t, 16384)` tensors per hyper-connection.**
The `.float()` cast, the `x.float().square()` intermediate inside
`DeepseekV4UnweightedRMSNorm.forward`, and the norm output. 86 instances.
*Measured:* `hyperconnection` peaks at 6.64 GB and retains 4.29 GB at `t = 32768`, i.e. 217 KB per
token per instance. That is the largest single linear term in the model.
*Replace with:* `quack.rmsnorm(x, None, eps=...)` is a one-line drop-in that accepts `weight=None`
and fp32 input and covers `N = 16384`. Megatron's `fused_proj_rms_compute_h` fuses norm and
projection together, but it is cuTile-only and therefore unavailable on H200 in practice.

**`hyperconnections.py:59-63` - the Sinkhorn loop is 39 normalization steps, 119 launches.**
`hc_sinkhorn_iters = 20` gives one column normalization plus `19 * 2` more, each a
`sum` / `+ eps` / `div` triple on a `(1, t, 4, 4)` fp32 tensor.
*Measured:* 10,234 launches per 43-layer forward. `hyperconnection` forward is 8.98 ms at
`t = 32768` against a `rmsnorm` forward of 0.14 ms, so the loop is launch-bound, not
bandwidth-bound.
*Replace with:* Megatron's `fused_sinkhorn` (Triton, sm90-clean, semantics verified byte-identical
to prime-rl's, see `megatron-survey.md`).
*Saving:* 10,234 launches to 86; and the saved-for-backward set drops from the whole 40-tensor
trajectory to a single `(1,t,4,4)` initial matrix.

**`hyperconnections.py:65` and `modeling_deepseek_v4.py:65-67, 73-75` - full-size products
materialized before a reduction.** `(pre.unsqueeze(-1) * hidden_streams).sum(dim=2)` builds a
`(1,t,4,4096)` product in fp32 (silent promotion, since `pre` is fp32) purely to reduce it. The
two decoder-layer expressions each allocate three full `(1,t,4,4096)` bf16 tensors.
*Replace with:* `torch.einsum("btn,btnh->bth", pre.to(dtype), hidden_streams)`; Megatron's
`fused_h_aggregate` and `fused_h_post_bda` are the fused forms, both Triton and both sm90-clean.

**`rotary.py:44-49` - the closing `torch.cat` copies 448 of 512 channels unchanged.**
`apply_rotary_pos_emb_interleaved` rotates the trailing 64 of 512 channels but reads and writes
the whole tensor, plus five fp32 temporaries and two `repeat_interleave` clones. Called 212 times
per model forward (3 per sliding layer, 6 per CSA, 4 per HCA).
*Replace with:* an in-place fused rope on the trailing 64 channels. Megatron's
`fused_mla_rope_inplace` is the closest existing kernel, but prime-rl uses adjacent-pair
interleaving, which that kernel explicitly refuses, and half-width `cos`/`sin` tables. Adapt, not
copy.

### MEDIUM

- `attention.py:416-423` - the compressor's advanced-indexing gathers, the out-of-place
  `+ self.position_bias`, and a full fp32 softmax over the pooling window before the cast back.
  *Measured:* `compressor-hca` is genuinely cheap (0.19 GB, 0.95 ms at `t = 32768`);
  `compressor-csa` is not, but its cost is the indexer inside it, not the pooling.
- `attention.py:399-405` - `_overlap_with_previous_window` allocates six tensors and ~10 launches
  to express "shift the `Ca` series one entry later". The two `masked_fill`s are out-of-place on
  freshly allocated tensors and could be in-place.
- `attention.py:349-350` - `token_entry_causal_mask` builds three `(1,t,e)` bool tensors to encode
  a per-row contiguous prefix, which is two integer bounds per query row.
- `attention.py:386` - every compressor constructs its own `DeepseekV4RotaryEmbedding`; 62
  instances compute bit-identical entry-position tables per forward.
- `attention.py:286` and `utils/sequence.py:42,51` - eight device-to-host syncs per model forward
  (`int(seq_lens.sum())`, `get_cu_seqlens_from_seq_lens`'s two `.item()` calls, two
  `repeat_interleave` without `output_size`, two `int(counts.sum())`, and `check_position_ids`'s
  `.any()`). The comment at `attention.py:285` claiming no device sync is wrong for the real
  trainer path, where `seq_lens` is a CUDA tensor.
- `hyperconnections.py:52` - `self.fn.float()` recasts a `(24, 16384)` parameter on every forward
  of every one of the 86 instances. FSDP's `MixedPrecisionPolicy` makes the parameter genuinely
  bf16, so this is a real cast, not a no-op.
- `modeling_deepseek_v4.py:263` - `.expand(-1,-1,4,-1).contiguous()` materializes four
  byte-identical copies of the embedding to seed the mHC streams.

### LOW

- `attention.py:512` - `topk` returns fp32 values that are discarded, and int64 indices for an
  index space bounded by `e <= 8192`.
- `attention.py:515, 546` - two `torch.full_like` allocations of `(1,t,512)` int64 exist only to
  carry a constant into `torch.where`, which accepts scalars.
- `attention.py:576-578` - HCA's `block_bias` is a dense tensor expressing a pure causal
  threshold, which is one integer per query row.
- `attention.py:692` - the grouped-linear `bmm` output is cloned by `.flatten(2)` because cuBLAS
  writes group-major and the consumer wants token-major.

## Claims that did not survive measurement

**Refuted: "the indexer's autograd graph is retained through the forward."** One review rated this
CRITICAL, estimating 176 GB of retained memory model-wide at `t = 8192`. It is wrong. The indexer
returns `topk(...).indices`, an int64 tensor with no `grad_fn`, so the subgraph is unreachable and
freed the moment `DeepseekV4Indexer.forward` returns. *Measured:* the `indexer` module has a
12.46 GB forward peak and retains **0.03 GB** at `t = 8192`, and `attn-csa` retains only 2.07 GB
more than `attn-sliding`, not the ~8 GB the claim requires. The indexer's peak is also not
additive with the attention peak (see `memory-model.md`).

The *underlying observation* behind that claim is nonetheless correct and important: no gradient
reaches the indexer. That is a correctness finding, not a memory finding, and it is recorded in
`README.md`.

**Refuted: "`aten::bmm` is a selective-checkpointing save target, so the score tensor survives
AC."** True as stated about `DEFAULT_SELECTIVE_TARGETS`, but the default mode is `"full"`
(`configs/trainer.py:32`), under which `_mandatory_checkpoint_policy` saves nothing inside a
block. *Measured:* at `t = 8192`, `decoder-csa` with `ac="full"` retains 0.33 GB against 18.23 GB
with `ac="none"`. The finding only applies if someone switches to `selective`, and it is a good
argument for not doing so.

**Unresolved: whether the explicit row-max subtraction at `attention.py:161` is redundant.** One
review rated it HIGH on the grounds that ATen's softmax already subtracts the row max internally
and that rounding `x - max` back to bf16 loses precision. The code comment asserts the opposite
("without it the exponentials overflow in bf16"). Both are plausible and this is a numerics
question, not a memory one, so it was left alone: this phase changes no model code. It is worth
one focused experiment before any kernel work assumes the subtraction can be dropped, because
removing it would free a full `S(t)` copy, roughly a third of the forward peak.
