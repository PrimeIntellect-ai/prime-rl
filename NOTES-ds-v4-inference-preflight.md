# DeepSeek V4 Flash inference pre-flight — working log

Working log for the standalone vLLM serving check of `deepseek-ai/DeepSeek-V4-Flash-0731` on this
8xH200 node. Kept for handoff/debugging context behind the commits it produced.

## Environment

- Node: 8x H200, all free at start.
- Commit: `2b592ba5c` (feat/ds-v4).
- `HF_HOME=/beegfs/garrett/huggingface` (772T free).
- `nvcc` 12.9.

## Timeline

- `uv sync --all-extras --all-packages` — OK, `vllm==0.26.0+cu129` installed.
- Pre-staged checkpoint via `uv run hf download deepseek-ai/DeepSeek-V4-Flash-0731` (the `hf` CLI,
  not `huggingface-cli`, which isn't installed). Completed cleanly: 48/48 safetensors shards,
  `model.safetensors.index.json` total_size 166,878,536,440 bytes matches actual dir size (~156G on
  disk via `du`, discrepancy is safetensors header overhead vs raw tensor bytes — not a concern), no
  `.incomplete` files, no broken symlinks.
- Starting vLLM server via `uv run inference @ examples/advanced/deepseek-v4-flash/inference.toml`
  in a detached tmux session (`ds-v4-inference`), no flags added beyond the config.

## Bug found and fixed: legacy `compress_ratios`/`num_hash_layers` schema

First boot crashed immediately (before touching any GPU) with:

```
AttributeError: property of 'DeepseekV4Config' object has no setter
```

Root cause: the real checkpoint's `config.json` ships the V3-flavoured legacy schema
(`compress_ratios`: flat per-layer list of 0/4/128; `num_hash_layers`: 3) rather than the
modern `layer_types`/`compress_rates`/`mlp_layer_types` fields. vLLM's own shadow
`DeepseekV4Config` (`vllm/transformers_utils/configs/deepseek_v4.py`) never translates
these legacy fields (unlike the real transformers-native `DeepseekV4Config`), so
`PretrainedConfig.__init__`'s generic kwargs loop tries to `setattr` the raw
`compress_ratios` list directly — which collides with this repo's own
`monkey_patch_deepseek_v4_compress_ratios` (a read-only property of the same name,
already in `src/prime_rl/inference/patches.py`, which assumed the raw key would never
be present). That existing patch was written/tested against a checkpoint using the
modern schema, so it never hit this collision before.

**Revision 1 (reverted):** first attempt added a separate `monkey_patch_deepseek_v4_
legacy_layer_types` that popped `compress_ratios`/`num_hash_layers` and translated them
into `layer_types`/`compress_rates`/`mlp_layer_types`, mirroring the real transformers
class. This crashed the actual GPU boot with a *new* error:
`AttributeError: 'DeepseekV4Config' object has no attribute 'num_hash_layers'` — every
DeepSeek V4 backend (`vllm/models/deepseek_v4/{nvidia,amd,xpu}/model.py`) reads
`config.num_hash_layers` directly, never `mlp_layer_types`, so popping it broke a real
read. Also traced why the *original* read-only `compress_ratios` property patch existed
at all: `vllm/models/deepseek_v4/attention.py` only ever indexes
`config.compress_ratios[layer_id]` directly — it doesn't use `layer_types` at all, so
translating into `layer_types` was solving a problem vLLM's model code doesn't have.

**Revision 2 (kept):** simplified to the actual minimal fix — gave the existing
`monkey_patch_deepseek_v4_compress_ratios` property a **setter**, so
`PretrainedConfig.__init__`'s generic `setattr(self, "compress_ratios", <raw list>)`
stores the checkpoint's real list instead of crashing, and the getter returns it verbatim
when present (falling back to the `compress_rates`/`layer_types`-derived value only for
checkpoints that ship the modern schema and have no raw `compress_ratios` at all — the
property's original purpose). No separate patch needed; `num_hash_layers`/`layer_types`
are left completely untouched. Verified in isolation (CPU-only, no GPU): `compress_ratios`
now returns the checkpoint's real per-layer values unmodified (`[0, 0, 4, 128, ...]`,
length 46, `num_hidden_layers`=43 so only the first 43 are ever read), and
`num_hash_layers` stays `3` as a plain attribute.

Confirmed not a vLLM-version issue (checked via subagent against vLLM `main`):
`vllm/transformers_utils/configs/deepseek_v4.py` on `main` is still the same unmodified
23-line shim, no legacy-kwarg translation, no `compress_ratios` property. Upstream issue
`vllm-project/vllm#42741` and its two open/unmerged PRs (#43443, #44031) are about the
*opposite* direction (transformers >= 4.57 removing `compress_ratios` from its own
config, breaking vLLM's attention code that reads it) and don't touch the config-parsing
shim at all — they wouldn't fix this crash even once merged, since this crash happens
earlier, inside `PretrainedConfig.__init__`'s kwargs loop, before attention code ever
runs. The local monkeypatch is addressing a real, currently-unfixed upstream gap.

Noted: this is a fragile fix (monkeypatching a vendored library's config class), and
should get a real upstream fix eventually — but it follows the exact established pattern
already used throughout `src/prime_rl/inference/patches.py` for a half-dozen other
DeepSeek V4 vLLM gaps, and unblocks this pre-flight check now.

## Bug found and fixed: DeepGEMM JIT cache dir poisoned to another user's `/tmp` path

Config parsing now succeeds and the engine gets as far as loading weights (20 GiB/rank,
~68s) before crashing during memory profiling / kernel warmup:

```
RuntimeError: Assertion error (.../utils/system.hpp:76): Failed to make directory:
/tmp/.research_deep_gemm/tmp/<pid>-..., created: false, value: 13
```

errno 13 = permission denied. `/tmp/.research_deep_gemm` on this shared node is owned by
a different user (`matej`, mode `755`) from an earlier, unrelated job — not something to
chmod or delete (not our data). Root cause, traced through the vendored `deep_gemm`
package: `vllm/utils/deep_gemm.py` only sets `DG_JIT_CACHE_DIR` to
`$VLLM_CACHE_ROOT/deep_gemm` **if it isn't already set**
(`if not os.environ.get(DEEP_GEMM_JIT_CACHE_ENV_NAME, None)`). But the vendored
`deep_gemm/__init__.py` itself sets `DG_JIT_CACHE_DIR=/tmp/.research_deep_gemm` (its own
hardcoded default, via `deep_gemm/envs.py`) via the *same* "if not already set" pattern,
and does so at the top of the module, *before* the CUDA-13 `_C` extension import that's
expected to fail on this cluster (the "expected `ImportError` for `deep_gemm`" the task
brief warns about). So the failed import still runs far enough to poison the process-wide
env var to the other user's path before it dies — and once poisoned, vLLM's own later
attempt to default it to its own cache root sees the var already set and backs off.

Fix: export `DG_JIT_CACHE_DIR=$HOME/.cache/vllm/deep_gemm` (matching vLLM's own intended
default) before launching the server, so the poisoned default never gets a chance to win
the race. This is a launch-time environment fix, not a code change — no repo files
touched for this one. Should probably get documented in the DeepSeek V4 skill/example
config for anyone else hitting this on a shared node.

## Bug found and fixed: CUTLASS scaled_mm doesn't support this checkpoint's UE8M0 weight scale

Past the DeepGEMM cache fix, weights loaded fully (20 GiB/rank, ~93s) and the engine
crashed during the memory-profiling dummy forward pass, in every worker at once:

```
RuntimeError: dispatch_scaled_mm, /workspace/csrc/libtorch_stable/quantization/w8a8/
cutlass/c3x/scaled_mm_helper.hpp:17,
```

via `Fp8LinearMethod.apply -> BlockScaledMMLinearKernel.apply_block_scaled_mm ->
ops.cutlass_scaled_mm -> torch.ops._C.cutlass_scaled_mm` — i.e. the dense FP8
block-quantized linear layers, not the MXFP4 experts. Root cause: this checkpoint's
`quantization_config.scale_fmt = "ue8m0"` (confirmed on disk), a less-common packed scale
format vLLM already special-cases for DeepGEMM (logged at startup: "Detected
quantization_config.scale_fmt=ue8m0; enabling UE8M0 for DeepGEMM") but
`Fp8BlockScaledMMLinearKernel` (the CUTLASS-backed kernel in
`vllm/model_executor/kernels/linear/scaled_mm/BlockScaledMMLinearKernel.py`) hardcodes
`use_ue8m0=False` for its activation quantizer regardless, so CUTLASS's low-level kernel
dispatch rejects the checkpoint's actual (UE8M0) weight-scale layout.

Which kernel gets picked is controlled by `VLLM_USE_DEEP_GEMM`, which vLLM itself
defaults to `1`/on — but prime-rl's own launcher (`src/prime_rl/inference/server.py`)
forces it to `0` unless `InferenceConfig.use_deep_gemm` is set (defaults `False`), which
is exactly the toggle the task brief's step 3A treats as an optional later experiment.
For this checkpoint it isn't optional: the CUTLASS fallback path prime-rl defaults to is
outright broken for UE8M0 weights, so DeepGEMM is required just to boot, not merely a
throughput experiment. Restarted with the top-level `--use-deep-gemm` flag (not
`--vllm.use-deep-gemm`), which flips `VLLM_USE_DEEP_GEMM=1` and should route these layers
through the already UE8M0-aware DeepGEMM kernel instead of CUTLASS.

This is worth flagging back: the config may need `use_deep_gemm = true` set as the
default for this checkpoint (not just available via flag), since the alternative doesn't
work at all here — a different conclusion than the brief anticipated (it expected both
paths to at least run, differing only in possible silent numerical drift above 1024
tokens on SM120).

## Result: server booted and healthy

Full command: `export DG_JIT_CACHE_DIR=$HOME/.cache/vllm/deep_gemm; uv run inference @
examples/advanced/deepseek-v4-flash/inference.toml --use-deep-gemm`, in a detached tmux
session (`ds-v4-inference`).

- Boot wall time (launch to `/health` succeeding): **897s (~15 min)**. Faster than the
  brief's up-to-70-min estimate for a first boot, most likely because the DeepGEMM JIT
  kernel cache (`~/.cache/vllm/deep_gemm`) was already partially warm from the earlier
  crashed attempts (compilation for a given kernel/shape happened before those runs died
  on later, unrelated errors, and DeepGEMM's on-disk cache persists across process
  restarts).
- Coherence check (`"Say hello in one sentence."`):
  ```json
  {"role": "assistant", "content": "Hello! 😊", "reasoning": null, "finish_reason": "stop", ...}
  ```
  Coherent, non-garbled, natural `stop` (not length-truncated). No `<think>` reasoning by
  default for this prompt.
- Per-GPU memory once loaded and warm: **124,259 MiB / 143,771 MiB** per H200 (matches
  `gpu_memory_utilization=0.85`: 0.85 × 143771 ≈ 122205 + some KV cache growth from serving
  the two prior requests).

## Follow-up 3A: coherence above 1024 tokens with the now-required `--use-deep-gemm`

Sent a 2015-prompt-token request (well past the 1024-token threshold TODO.md flags as where
the non-deep_gemm TileLang mHC kernel goes silently wrong on SM120). Response: coherent
`"Hello!"`, clean `finish_reason: "stop"`, no NaNs, no garbling. This is the missing Hopper
data point TODO.md's "Not yet checked on Hopper" note was waiting on — now confirmed good.
(Framing changed from the original task brief: deep_gemm turned out to be required just to
boot at all on this checkpoint, per the CUTLASS/UE8M0 bug above, so this was a single coherence
check under the only viable configuration rather than an A/B comparison.)

## Follow-up 3B: thinking-mode control key

`chat_template_kwargs: {"thinking": true}` and `{"thinking": false}` both work and are
respected:

- `{"thinking": true}`: response has a populated `reasoning` field (a scratch-style
  chain-of-thought) separate from `content`, e.g. `reasoning: "We need answer simple
  multiplication..."`, `content: "408"`.
- `{"thinking": false}`: `reasoning: null`, direct `content: "17 * 24 = 408."`.
- **Default (no `chat_template_kwargs` at all) behaves like `thinking: false`**: tested with
  "What is 17 * 24? Think step by step." and got a direct worked-out answer in `content`,
  `reasoning: null`. This matters for the planned RL run's 128-completion-token cap: the model
  does not spend the budget in a separate reasoning block unless explicitly asked to.

## Follow-up: propagated `use_deep_gemm = true` to sibling configs and corrected TODO.md

`examples/advanced/deepseek-v4-flash/rl.toml` and `kl-check.toml` name the same real
checkpoint and had the identical missing `use_deep_gemm = true` (only `rl-mini-smoke.toml`
already had it, with a comment already correctly calling it "mandatory, not an optimization").
Added the same setting + comment to `inference.toml`, `rl.toml`, and `kl-check.toml`; all three
still parse under `--dry-run`. Also updated `TODO.md`'s "DeepSeek V4: blockers for the real
cluster run" section: filled in the "not yet checked on Hopper" gap, added the newly-found
CUTLASS/UE8M0 crash as a second, more basic reason the flag is required, and corrected a stale
claim that `monkey_patch_deepseek_v4_compress_ratios` was "a no-op on the real checkpoint" (it
was not — confirmed by actually booting, not just reading the config statically).

Note: `rl.toml`/`kl-check.toml` still can't complete a *full* real-checkpoint RL run today per
TODO.md's separate, unrelated blocker ("The trainer cannot load the real checkpoint" — no fp8
dequantization path yet). This inference-side fix is still correct and forward-looking so
whoever unblocks the trainer side doesn't have to rediscover this.

## Follow-up: removed `monkey_patch_deepseek_v4_compress_ratios` entirely

Traced why the recipe's plain `vllm serve` would never hit this crash: pristine vLLM treats
`compress_ratios` as a plain attribute (no property), so the real checkpoint's raw list just
works. The crash was 100% self-inflicted by an *existing* prime-rl patch (predating this
session) that shadowed it with a getter-only property for prime-rl's own mini checkpoint
(modern schema, no raw `compress_ratios`). Confirmed `scripts/mini_moe.py` is the *only* place
in this repo that writes a DeepSeek V4 `config.json` (filesystem weight broadcast never writes
one at all — vLLM loads the config once at startup and only receives tensors afterward), and
confirmed `to_dict()` serializes arbitrary instance attributes, not just declared fields
(matching the existing `qk_rope_head_dim` precedent).

Fix: added `self.compress_ratios = [self.compress_rates.get(lt, 0) for lt in self.layer_types]`
to `src/prime_rl/trainer/models/deepseek_v4/configuration_deepseek_v4.py`, right after
`layer_types` is finalized. Removed `monkey_patch_deepseek_v4_compress_ratios` entirely from
`src/prime_rl/inference/patches.py`.

Verified:
- Rebuilt the mini checkpoint (`scripts/mini_moe.py --arch deepseek_v4`) — verify passed,
  `config.json` now has `compress_ratios: [0, 4, 128, 4, 0]`.
- CPU-only `get_config()` against both the mini and real checkpoints, monkeypatch removed:
  both resolve correctly.
- `uv run pytest tests/unit/train/models/test_deepseek_v4.py`: 15 passed.
- Full e2e: real vLLM server boot against the real checkpoint, monkeypatch removed. Healthy in
  375s (fastest yet — DeepGEMM JIT cache fully warm from prior runs). Coherence check:
  `"Say hello in one sentence."` → `"Hello!"`, clean `stop`.
