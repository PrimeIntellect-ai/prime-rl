# Issue: concurrent SLURM launches can share one namespace

## Status

Deferred from the TTT review. No implementation is present on `main` as part of this note.

## Summary

`prime_rl.entrypoints.rl` validates and may clean the run output directory before writing
resolved configs and the generated SLURM script. A queued job reads those files only when the
scheduler starts it. Two submissions targeting the same output or checkpoint directory can
therefore overlap before either job establishes durable ownership.

## Failure scenario

1. Submission A validates `outputs/run-x`, writes configs, and waits in the queue.
2. Submission B targets the same path before A starts.
3. B cleans or overwrites A's configs and SLURM script.
4. A starts later and reads files written by B, or both jobs write checkpoints and rollouts to
   the same namespace.

Checking only for existing checkpoints is insufficient because the dangerous interval begins
at submission time, before a queued job has written a checkpoint. A marker inside the output
tree is also fragile when launch validation is allowed to delete that tree.

## Change considered on the TTT branch

The TTT branch introduced a user-global registry under `~/.cache/prime-rl/submissions`. Each
canonical output/checkpoint path mapped to a hashed owner directory. Creation used an atomic,
non-empty directory rename, and generated job scripts removed only their unique owner child in
an EXIT trap. Job names and matching sandbox labels also received a submission UUID.

The non-empty owner-directory protocol was chosen to avoid a stale cleanup race: an old job
must not delete a marker that a new job acquired between an ownership check and removal.

## Why it was removed from TTT

The implementation changed **every** Prime-RL SLURM launch, including job names, dry-run
behavior, filesystem side effects, and cleanup semantics. The TTT experiment's immediate
collision was already addressed more locally by giving A0-A5 distinct output directories,
checkpoint roots, job names, and sandbox labels.

## Questions for a standalone change

- Should a dry run reserve a namespace indefinitely when its printed script may never run?
- How should operators remove a marker for a job cancelled before its script starts?
- Is user-global state appropriate on clusters with ephemeral or shared home directories?
- Should explicit resume be allowed to share a namespace with a queued/running job?
- Should unique job names be opt-in rather than rewriting all names?

## Suggested tests

- Two processes racing to reserve the same canonical path: exactly one succeeds.
- Independent git worktrees resolve to the same reservation key.
- Nested output/checkpoint paths do not place markers inside trees that launch cleanup removes.
- Old cleanup cannot remove a replacement owner's marker before or after its own child unlink.
- Failed config generation and failed `sbatch` release ownership.
- Dry-run, resume, and queued-cancellation behavior are specified explicitly.

## Relevant code

- `src/prime_rl/entrypoints/rl.py`
- `src/prime_rl/templates/multi_node_rl.sbatch.j2`
- output/checkpoint validation helpers in `src/prime_rl/utils/pathing.py`
