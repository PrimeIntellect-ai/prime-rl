---
name: in-alloc-launch
description: Run multi_node RL inside an already-held Slurm allocation (no sbatch) on Isambard-AI, and split one allocation into several concurrent runs ("lanes"). Use when launching training/eval from inside an allocation, when a multi_node run hangs before inference becomes reachable, when inference dies with "vllm-router: command not found", or when concurrent runs clobber each other's outputs.
---

# In-allocation multi_node launch

On Isambard-AI you hold a Slurm allocation as a **node pool** and run `rl` **inside** it. `rl` is the orchestrator: it `srun`s the trainer and inference onto the allocation's nodes itself. You do **not** wrap it in `srun` or `sbatch`.

A config runs in-allocation **iff it has no `[slurm]` block**. With a `[slurm]` block the same config submits a fresh `sbatch` job instead — use that only when you do *not* already hold nodes.

## Single run

```bash
# 1. Hold the pool (from the login node). Drops you into a shell INSIDE the allocation.
launch-script-mnode 4

# 2. Bind that shell to this checkout.
cd "$PROJECTDIR"/joanv.a6r/work/prime-rl-main
source scripts/env/activate-prime-rl.sh        # loads .env + venv + Slingshot fabric

# 3. Launch — plainly. No srun, no sbatch.
uv run rl @ configs/isambard/rl_2node_inalloc.toml
```

With no `--deployment.hosts`, the whole allocation is treated as one run. Topology comes from the config (`num_train_nodes`, `num_infer_nodes`, `gpus_per_node = 4`).

## Concurrent runs ("lanes")

Split the pool into disjoint slices, one full run on each. A lane needs **exactly** `num_train_nodes + num_infer_nodes` nodes (the config's own topology). Each lane needs a disjoint host slice, a distinct `--deployment.port-base` (≥100 apart), and a distinct `--deployment.lane-tag`:

```bash
mapfile -t n < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
uv run rl @ configs/isambard/rl_2node_inalloc.toml \
  --deployment.hosts "${n[0]},${n[1]}" --deployment.port-base 29500 --deployment.lane-tag "$SLURM_JOB_ID-l0" &
uv run rl @ configs/isambard/rl_2node_inalloc.toml \
  --deployment.hosts "${n[2]},${n[3]}" --deployment.port-base 29700 --deployment.lane-tag "$SLURM_JOB_ID-l1" &
wait
```

You pick the *shape* by picking the config: a 2-node config on a 32-node hold makes 16 lanes; an 8-node config on the same hold makes 4. See [`docs/launch.md`](../../docs/launch.md) for the full command surface and [`docs/scaling.md`](../../docs/scaling.md) for the pool model.

## Pitfalls (these bite consistently)

1. **Run `rl` plainly — never under `srun`/`sbatch`.** Inside an allocation `rl` *is* the head-node orchestrator and fans out via its own internal `srun --exact`. An outer `srun` nests step creation and **deadlocks**.
2. **In-alloc requires NO `[slurm]` block in the config.** A `[slurm]` block silently switches to the sbatch-submit path — wrong when you already hold nodes.
3. **`gpus_per_node = 4` on AIP2** (4 GH200/node). Upstream configs default to 8; leaving 8 breaks the GPU math.
4. **Set `weight_broadcast.type = "nccl"` for real runs.** It moves weights GPU→GPU over Slingshot (~10× faster than the `filesystem` default, which round-trips a checkpoint through Lustre). It needs the fabric, which `activate-prime-rl.sh` loads.
5. **Concurrent lanes must be fully disjoint:** distinct node slices, `port-base` ≥100 apart, distinct `lane-tag`. The `lane-tag` namespaces outputs/rollouts/caches; omit or repeat it and lanes overwrite each other's rollouts and one crashes (`FileNotFoundError` on `rollouts/step_N`).
6. **The `vllm-router` aarch64 wheel must be in `uv.lock`.** If multi_node inference dies with `vllm-router: command not found`, the lock is missing the `manylinux_2_28_aarch64` router wheel. `pyproject.toml` pins both-arch wheels; after any pyproject change run `uv lock` **and commit the lock** — committing the manifest without its lock is the trap.
7. **Slingshot is libfabric (`cxi`), not InfiniBand.** Never use `ibv_devinfo` / `NCCL_IB_HCA`. The fabric loads via `module load brics/nccl` (done by `activate-prime-rl.sh`); without it NCCL silently falls back to TCP sockets (~10× slower). A healthy fabric does ≥~23 GB/s on `nccl-tests` and prints `NET/OFI` in `NCCL_DEBUG=INFO`.
8. **`/tmp` and `/dev/shm` are a shared, RAM-backed tmpfs that Slurm does NOT wipe between jobs.** A lane can hit `OSError [Errno 28] ENOSPC` from *other users'* leftover caches on a node — an environment issue, not a launcher bug. Pick fresh nodes or clear space; don't chase it as a code bug.
