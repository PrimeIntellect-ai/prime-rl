# Routing replay integration plan (vLLM routed experts)

This plan covers integrating routed expert replay (as in https://arxiv.org/abs/2510.23027) so the
policy is conditioned on both tokens and the experts used at inference time. The goal is to keep
training-time routing consistent with inference-time routing while using importance sampling to close
performance gaps.

## 0) Current repo state / branches

* Only the `work` branch exists locally; no other branches were found.
* No merge conflicts were detected during the branch check.

## 1) vLLM: return routed experts for every token

* Enable vLLM’s `--enable-return-routed-experts` flag via `InferenceConfig` so the inference server
  can request expert indices + probabilities per token.
* Validate that the vLLM response for Qwen3 MoE includes routed expert indices and probabilities for
  each generated token (and prompt token where applicable), and document the exact schema.

## 2) Verifiers + orchestrator: transport routed expert metadata

* Extend `verifiers` response parsing to preserve routed expert indices/probabilities in the
  per-step `tokens` payload.
* Wire the new fields through:
  * `orchestrator` trajectory collection
  * `TrainingSample` / `MicroBatch` transport structs
  * `trainer` batching utilities
* Ensure alignment rules: the routing metadata must align to completion tokens, not masked prompt
  tokens, and must be padded consistently during packing.

## 3) Training-time routing replay (force same experts)

To make training tokens go to the **same experts as inference**:

* Update Qwen3 MoE forward to accept **optional routed expert indices + routing probabilities**.
  * Add a “forced routing” path that bypasses router top‑k selection and instead uses the provided
    expert indices/probs.
  * Validate tensor shapes: `[batch, seq, top_k]` for expert indices and the same for routing probs.
* Ensure numerical behavior matches vLLM routing:
  * Apply the same normalization (e.g., softmax/renorm within top‑k) as vLLM uses.
  * Confirm per-token scaling order (score‑before‑experts vs score‑after‑experts) matches vLLM.
* Add a configuration flag to toggle forced routing on/off (default off).

## 4) Importance sampling with joint token+expert policy

* Extend RL loss to treat the policy as **conditioned on tokens and experts**:
  * Compute a joint logprob per token that combines:
    * token logprob, and
    * routed expert logprob (from trainer) vs inference routing logprob (from vLLM).
  * Use the joint logprob in the importance ratio.
* Add masking rules so only completion tokens (and their routed experts) contribute to loss.
* Keep backward compatibility: if expert routing info is missing, fall back to token-only loss.

## 5) Bench: separate benchmark configuration

* Create a dedicated benchmark config that enables:
  * vLLM `--enable-return-routed-experts`
  * trainer forced-routing + joint importance sampling
* Add a short doc snippet describing how to run the new benchmark.

## 6) Validation / tests

* Add unit tests for:
  * routing replay path in Qwen3 MoE (forced expert indices/probs),
  * batch packing/unpacking of routing metadata,
  * loss computation with joint token+expert logprobs.
* Run a minimal integration test with a small Qwen3 MoE model to confirm:
  * inference routing metadata is captured,
  * training uses identical expert indices,
  * loss computation remains stable.

