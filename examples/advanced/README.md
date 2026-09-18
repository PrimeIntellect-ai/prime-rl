# Advanced recipes

Advanced RL recipes are composed from an environment and a model-topology profile:

```bash
uv run rl \
  @ configs/advanced/envs/swe.toml \
  @ configs/advanced/models/glm-4.5-air/rl/6-nodes.toml
```

The environment file owns tasksets, harnesses, runtime settings, judges, and train/eval source
lists. The model profile owns the checkpoint, trainer and inference parallelism, memory strategy,
and model-specific runtime settings. Output paths, cache locations, partitions, and run identity
remain CLI overrides until they become a third cluster overlay. The Mooncake profile retains its
required RDMA device list until that overlay exists.

All checked-in environment and model profiles are composition-tested. Use any environment with any
model profile, then apply CLI overrides for run identity and local cluster settings:

```bash
uv run rl \
  @ configs/advanced/envs/math.toml \
  @ configs/advanced/models/qwen3-30b-a3b/rl/4-nodes.toml \
  --output-dir /shared/outputs \
  --run.name qwen3-math \
  --slurm.partition <partition>
```

## Environments

- [`agentic-mix.toml`](../../configs/advanced/envs/agentic-mix.toml): SWE, search, math, logic, and code.
- [`math.toml`](../../configs/advanced/envs/math.toml): `i3_math` with AIME evaluation.
- [`search.toml`](../../configs/advanced/envs/search.toml): OpenSeeker and RedSearcher with BrowseComp evaluation.
- [`swe.toml`](../../configs/advanced/envs/swe.toml): `r2e-gym` with SWE-Bench Verified evaluation.
- [`swe-scaleswe.toml`](../../configs/advanced/envs/swe-scaleswe.toml): ScaleSWE with SWE-Bench Verified evaluation.
- [`terminal.toml`](../../configs/advanced/envs/terminal.toml): Tmax with SWE-Bench Verified and Terminal-Bench 2 evaluation.
- [`tool.toml`](../../configs/advanced/envs/tool.toml): General-agent training with colocated Modal tools.

## Model profiles

- [`glm-4.5-air`](../../configs/advanced/models/glm-4.5-air/rl): 6-node Muon, plus 4- and 2-node full-offload profiles.
- [`glm-5.3`](../../configs/advanced/models/glm-5.3/rl): 32-node P/D and 32-node llm-d + Mooncake KV-offload profiles.
- [`intellect-3.1`](../../configs/advanced/models/intellect-3.1/rl): 16-node profile.
- [`qwen3-30b-a3b`](../../configs/advanced/models/qwen3-30b-a3b/rl): 4-node Thinking and 2-node Instruct profiles.

GLM-5.3's inference-only preflight configs and setup notes live in
[`configs/advanced/models/glm-5.3/infer.md`](../../configs/advanced/models/glm-5.3/infer.md).
