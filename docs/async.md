### Step Definition

At each step `n`, all artifacts (e.g., checkpoint, rollout, gradient) are tagged with the same step number.
- Step 0:
  - Uses checkpoint 0 on rollout 0 to compute gradient 0.
  - Then computes checkpoint 1 as: `ckpt 1 = ckpt 0 - grad 0`

In general, the model used for generating rollouts at step `n` is from `ckpt[n - async_level]`.

- When async_level = 0, the rollout and gradient are based on the same model version.
  This is equivalent to synchronous on-policy training.
