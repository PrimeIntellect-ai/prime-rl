---
name: preflight-lora-smoke
description: MUST run before the first GPU training run that uses a hot-swapped LoRA adapter concurrently with base-model requests on the same vLLM server (debate external-opponent, DPO-style ref-on-same-instance, any "learner vs frozen self" setup). Triggers on "LoRA", "external opponent", "first GPU run", "enable_lora", "load_lora_adapter", or any setup that combines trainer.model.lora + agent_bindings_fn that routes some seats to the base model name. Three fast probes catch the known vLLM 0.19 failure modes before a long training run hits them.
---

# Pre-flight: LoRA-self vs base on one vLLM

Before any first training run where a **single vLLM server** hosts:
- the learner behind a hot-swapped LoRA adapter, AND
- frozen requests (opponent, reference, judge) addressed as the base model name,

run `scripts/preflight_lora_smoke.py`. It takes ~5 minutes and catches three known-problematic codepaths upstream that are not covered by any existing prime-rl test.

## Why this gate exists

The LoRA-self-vs-base deployment pattern is architecturally supported by vLLM (documented in `docs/features/lora.md`, canonical example at `examples/offline_inference/multilora_inference.py`), but community-wide adoption for RL is thin. Most RL frameworks (SPIRAL, SPPO, verl, OpenRLHF, TRL) use **two separate vLLM instances** for this use case instead. Our prime-rl setup would be first in this particular stack to exercise the mixed base+LoRA batched path at training scale.

The search that motivated this gate surfaced three specific vLLM issues worth probing on pinned version before committing a training run:

| vLLM issue | Failure mode | What happens if it bites |
|---|---|---|
| [#18372](https://github.com/vllm-project/vllm/issues/18372) | 3rd+ sequential adapter hot-swap silently doesn't take effect; outputs keep following an older adapter | Training run proceeds but orchestrator's "updated weights" is a lie; gradients flow on stale rollouts |
| [#33791](https://github.com/vllm-project/vllm/issues/33791) | `/load_lora_adapter` with `load_inplace=True` can fall back to CPU execution — prime-rl's `monkey_patch_load_lora_adapter` uses exactly this path | Silent 10-100× slowdown on adapter load |
| [#7977](https://github.com/vllm-project/vllm/issues/7977) | `sgmv_shrink` atomics → LoRA-routed greedy outputs non-deterministic | Eval reproducibility broken; RL advantage-estimation signal noisier than expected |

## The three probes

Run `uv run scripts/preflight_lora_smoke.py --base-model <model> --adapter-a <path-to-adapter-a> --adapter-b <path-to-adapter-b>` **after** launching a vLLM server with `--enable-lora --max-loras 1 --enable-lora-runtime-update` (or prime-rl's equivalent `inference.enable_lora = true` TOML key). The script will:

1. **Mixed-batch correctness.** Fire the same prompt concurrently with `model=<base>` and `model=<adapter>`. Verify:
   - Both return 200 (no server error on the mixed batch).
   - Adapter-routed output's top-k token distribution differs from base-routed output by measurable KL. (If they match, either the adapter isn't loading, or mixed-batch routing is collapsing both to the same path.)

2. **Hot-swap idempotence (the #18372 probe).** Load adapter A, fire prompt P, record tokens $T_A$. Load adapter B (different weights, **same alias**), fire prompt P, record $T_B$. Load adapter A again, fire P, record $T_{A'}$. Assert:
   - $T_B \ne T_A$ (swap A → B actually took effect)
   - $T_{A'} = T_A$ (swap B → A took effect, not stuck on B)
   If any assertion fails, **stop** and switch to the two-instance topology — #18372 is live on this pin.

3. **Perf delta (the #10898 tax).** Microbench base-only throughput against the LoRA-enabled server with a second short run against a plain `--model` server (no `--enable-lora`). Report the ratio. If delta > 60%, budget accordingly and consider the two-instance topology for throughput-critical long runs.

## When to re-run

- After bumping the vLLM pin.
- After any change to `src/prime_rl/inference/vllm/patches.py` (the monkey-patches).
- After any change to `src/prime_rl/inference/vllm/worker/weight_transfer.py`.
- Before the first real training run on a new hardware configuration (e.g. different GPU family, different CUDA version).

## Escape hatch

If any probe fails and an upstream fix isn't immediately available, fall back to **two vLLM instances**: point `opponent_base_url` in the `gpqa_debate` env config at a second vLLM running the plain base model, no LoRA. The `agent_bindings_fn` accommodates both topologies without any env-pack change — this is the forward-compatible design that motivated that kwarg in the first place. See `docs/plans/2026-04-20-stage3-learner-vs-fixed.md`.

## Known probe footguns (fixed 2026-06-11)

- The load must hit prime-rl's root-level `/load_lora_adapter` (production route,
  applies the MoE key remap). vLLM's native `/v1/load_lora_adapter` returns 200 but
  silently no-ops for expert-targeted adapters.
- Probes must use raw `/v1/completions`: on reasoning-parsed servers, short chat
  responses land entirely in `reasoning_content`, making `message.content` empty on
  both sides and producing false "byte-identical" failures.
- Probe adapters must be strongly non-trivial: early-RL adapters (few steps, low lr)
  have ~1e-5-scale weights that cannot flip greedy tokens; synthesize ~0.02-scale
  random adapters for the probes.
