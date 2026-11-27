# Analyze and compare training runs

## Establish the measurement

Read the resolved configs and source/image revisions for each attempt. Record
model, data/grader revision, sampling support and temperature, loss and masks,
baseline, pruning/admission, batch size, parallelism, hardware, and async limits.
A change to filtering can change effective training reward without improving the
policy; compare all-rollout reward, admission/error rates, and held-out eval too.

Keep run ID, attempt, component, timestamp, and policy/training step with each
observation. Trainer and orchestrator can both log `step` and `time/step` to the
same W&B run. The orchestrator's shipped batch is not a completed optimizer step.
W&B summary fields can come from different events; they are not one coherent
snapshot. Confirm progress from trainer completion logs and timestamped trainer
metrics. A platform `RUNNING` state or fresh rollout heartbeat is insufficient.

Inspect available keys in the run's revision. History queries requiring several
keys return only rows containing all of them: fetch separate series when that
would hide newer observations, and align by producer, attempt, and step. Preserve
raw exports. Do not fill missing metrics with zero or forward-fill across attempts.

## Restarts, checkpoints, and pipeline warmup

Build an attempt timeline from component logs, launch records, and restored
checkpoints. A step decrease is a clue, not proof of a restart: concurrent writers,
delayed uploads, and async eval can arrive out of order. Reconcile with attempt
identity before discarding data. Capture partial component restarts as well as
whole-run resumes; a restart need not change the run ID or roll back the trainer.

Maintain two views when needed:

- **Policy trajectory:** retain the checkpoint lineage leading to the current
  policy. For a resume from step 200, discard the parent's steps above 200 from
  this view, then append the resumed optimization steps according to that revision's
  step convention. Do not average repeated step numbers from different attempts.
  Preserve discarded work separately and mark restart/rollback boundaries.
- **Operational history:** retain all elapsed time, failed attempts, repeated
  work, and downtime when reporting cost or time to reach a checkpoint/score.

A checkpoint can restore optimizer progress without restoring the rollout queues,
in-flight episodes, inference cache, or compiled kernels. Verify what this revision
actually restores. A high training step does not imply a warmed-up pipeline.
Immediately after a restart, expect possible queue refill, inference/compile
warmup, weight synchronization, or startup eval contention. An initially empty
pipeline can temporarily lower staleness; the first completions can overrepresent
short/easy episodes. Both throughput and reward may shift before the steady-state
mix returns. These are hypotheses to check against logs and queue/length metrics,
not reasons to dismiss a loss or gradient spike.

Label startup/refill intervals using observable stabilization of in-flight counts,
queue depth, policy lag, step timings, and episode lengths; report the criterion
and duration. Do not silently drop a fixed number of steps, or select an exclusion
window because it removes unfavorable outcomes. Show warmup-inclusive operational
results alongside steady-state comparisons; if no stable interval exists, say so.
Reset rolling windows at confirmed attempt boundaries. Compare matched training
stages and comparable steady-state windows; do not mix one arm's post-restart
warmup with another arm's mature pipeline.

Eval results belong to the policy version evaluated, not their arrival time.
A late eval of policy 180 after policy 200 is not a rollback. Keep these points,
including separate repeated evaluations when relevant; report counts and errors.
If a discarded policy is plotted, label it as off the retained checkpoint lineage.

## Metrics to inspect together

| Question | Measurements and interpretation |
| --- | --- |
| Is learning improving? | Train reward for both `all` and `effective` populations, held-out eval by policy version, episode counts, pruning/admission, errors and truncation. An incomplete/error-heavy eval is not directly comparable to a full clean one. |
| Is the update stable? | Entropy, mismatch KL, loss, gradient norm, masked fraction, accepted/rejected-token KL, and SC correction metrics if available. In the loss metrics, `unmasked_mismatch_kl` means accepted tokens and `masked_mismatch_kl` means rejected tokens; SC may still correct rejected actions. Verify entropy's sampling support before comparing truncated and full-vocab runs. |
| Are rare weights dominating? | Importance-ratio mean/max and cumulative exceedance fractions, separately for all and accepted tokens. `above_5`, `above_10`, `above_100`, `above_10000` are tails, not disjoint bins. Their means are fractions. Inspect saturation flags; ordinary percentiles can miss extreme rare weights. |
| Is the data getting stale? | Available token/rollout staleness mean, tails/max, policy versions, queue depth, and in-flight counts. Check token versus rollout weighting and the matched batch. Mean alone can hide a long tail; do not invent unlogged percentiles. |
| Why is it slow? | Separate trainer batch wait, load/convert, forward/backward, broadcast, and checkpoint time from orchestrator policy wait and rollout/inference time. Inspect tokens per batch, output-length tails, truncation, errors, inference throughput, and eval contention. Normalize by actual processed tokens as well as steps. |

For overlapped pipelines, do not add trainer and orchestrator step times. A maximum
of comparable steady-state component times is a bottleneck approximation, not an
exact end-to-end measurement: waits can overlap and batches can differ. Prefer
elapsed time between confirmed trainer completions for realized training cadence.
For a token rate, divide total tokens by total elapsed time, not the unweighted
mean of per-step rates. State whether downtime and discarded work are included.

Check the aggregation semantics before combining metrics: token means need token
counts, episode means need episode counts, and averaging subgroup/batch means is
not generally a global mean. Global percentiles cannot be reconstructed by averaging
per-batch or per-rank percentiles. Entropy/staleness adjustment is a descriptive
comparison, not causal proof; show overlap and unmatched observations.

## Present the comparison

Show raw observations with any smoothing window stated, and mark confirmed
restarts and warmup intervals. Fit axes to the selected runs; label clipping and
outliers rather than hiding divergence. Use separate early-stage or per-arm panels
when runs differ greatly in length. State refresh time, latest completed trainer
step, eval policy version, sample coverage, and material config differences.
Distinguish observed facts from hypotheses; numerical deterioration does not by
itself establish a process crash. Analysis alone does not authorize stopping,
restarting, or launching runs.
