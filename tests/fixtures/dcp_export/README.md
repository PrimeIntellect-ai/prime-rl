# dcp_export golden fixtures — provenance

The two `*_legacy_export_golden.json` fixtures are the immutable goldens for
`tests/unit/utils/test_dcp_export_compatibility.py`. Each records a real tiny PrimeRL
model's prime-format source state and the HF-format state MAIN's legacy export produces
for it: every key with shape, dtype, and raw little-endian tensor bytes (hex).

Generated ONCE by replaying MAIN's legacy export conversion semantics on CPU — no GPU,
no DCP gather (single full CPU state), no pretrained weights (tiny seeded
`torch.manual_seed(20260101)` init), and no use of the CPU exporter under test:

1. Build the real PrimeRL model class on CPU from the tiny config recorded in the fixture
   (`LlamaForCausalLM._from_config` / `Qwen3MoeForCausalLM._from_config`).
   1b. Apply the production init API to the prime-rl custom modules HF `post_init` does
   not initialize: `MoE` (covers `GroupedExperts` + router + shared expert) and bare
   `GroupedExperts`/`TokenChoiceTopKRouter`/`FeedForward` modules get
   `.init_weights(config.initializer_range)` (plus the CPU `buffer_device` for `MoE`)
   before the state is dumped. `GroupedExperts`
   parameters are `torch.empty` until this runs — the trainer normally loads real weights
   over them, so the empty state is expected in production and this init step is a
   fixture-only concern. The generator asserts every floating source and golden tensor is
   finite before writing the fixtures.
2. Emulate the legacy `gather_weights_parallel(model, dtype=torch.bfloat16)` single-rank
   gather (fp32 -> bf16; no keep_in_fp32 tensors in these families).
3. Emulate the legacy tied-alias drop (both fixtures are untied; `dropped_tied_aliases` is
   recorded in the fixture).
4. Apply MAIN's `prime_rl.utils.weights.convert_state_dict_to_hf` — the exact function
   MAIN's legacy converter `tools/convert_dcp_to_bf16.py` calls after its GPU gather.

Reference codebase: MAIN at immutable commit
`9546001f7361f580836445f254ee958f8c41e4fc` (the fixture's `provenance` block records the
commit, seed, torch version, and the expert init). The generator ran in the branch
project's full CPU env (torch 2.13.0+cu130, transformers 5.19.0, python 3.12.15, no GPU)
and asserted the critical modules (`prime_rl.utils.weights`, both model packages)
resolved from MAIN.

Configs: llama_dense = hidden 8, heads 2, kv 1, intermediate 16, 1 layer, vocab 16, untied,
`mlp_bias=True` (real gate/up/down bias tensors, identical in PrimeRL and native HF);
qwen3_moe = same plus moe_intermediate 16, 4 experts, top-2, sparse step 1, untied,
`attention_bias=False` (as in real Qwen3-MoE checkpoints). PrimeRL ties o_proj bias to
`output_bias` while native HF ties it to `attention_bias` (a separate existing model mismatch), so
attention_bias=True would leave `o_proj.bias` missing on the HF load; both fixtures keep
diagnostics empty by design.

The GPU CLI and CUDA gather were not executed when generating these fixtures.
