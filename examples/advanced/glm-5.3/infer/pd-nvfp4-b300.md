# GLM-5.3 NVFP4 P/D and Scale-SWE training on B300

This RL config fragment contains `[inference]` and the Scale-SWE training source
under `[orchestrator.train.source]`. Add trainer, training deployment, batching,
model/tokenizer and weight-broadcast settings before launching a full RL run.
The inference checkpoint is prequantized; training also needs an explicit choice
of trainer checkpoint and inference weight-reload/quantization settings.

To run only the inference portion from the repository root:

```bash
uv sync --all-extras --all-packages
uv run python - <<'PY'
import json
import tomllib
from pathlib import Path

config = tomllib.loads(Path("examples/advanced/glm-5.3/infer/pd-nvfp4-b300.toml").read_text())
inference = config["inference"] | {key: config[key] for key in ("slurm", "output_dir")}
Path("/tmp/glm53-pd-inference.json").write_text(json.dumps(inference))
PY
uv run inference @ /tmp/glm53-pd-inference.json --dry-run
uv run inference @ /tmp/glm53-pd-inference.json
```

The dry run writes resolved prefill/decode configs and the Slurm script without
allocating nodes. Remove `--dry-run` to submit two exclusive eight-GPU nodes,
with no job time limit. The router listens on port 8000 on the first allocated
node; use model name `glm53`. Each role is an independent DPEP8 group, TP1.

| Setting | Prefill | Decode |
|---|---|---|
| All-to-all backend | FlashInfer NVLink one-sided | DeepEP low latency |
| Batched tokens | 32,768 | 320 |
| Maximum sequences per GPU | 256 | 320 |
| GPU memory utilization | 0.75 | 0.90 |
| Graphs | None | Full + piecewise, breakable |
| Mooncake store RAM per node | 2,560 GiB | 0 |
| HiSparse host RAM per GPU | Disabled | 300 GiB |

Both roles use FP8 KV, a 32,768-token context limit, index cache with frequency
4, natural expert routing, and no EPLB, DBO or MTP. Routing is sticky least
loaded with retries disabled. Each rank binds to its configured NUMA node and
UCX NIC. These mappings and the host-memory capacities target this B300 cluster;
adjust them for a different machine topology.

The model is the prequantized `RadixArk/GLM-5.3-NVFP4` checkpoint at revision
`11af4cba759e6559eda70358a5778bd1bddddd78`, using its existing local download.
To load from HF instead, override `--vllm.model RadixArk/GLM-5.3-NVFP4`; keep
the revision pin. This config does not quantize a BF16 trainer checkpoint online.

Use the pinned HiSparse/NVFP4 wheel with the standard Python frontend and native
`NixlConnector`. No experimental connector module or worker class is required.
Mooncake master/storage processes belong to the Slurm job, so cancellation also
releases their memory. Both roles connect to the same store pool with separate
role cache-key prefixes; shared capacity does not imply cross-role key reuse.
Each connector also uses a 512 MiB RDMA staging buffer.

The training source matches the Scale-SWE eval: quarter-zero tasks selected by
the instance-ID filter, bash harness, automatic compaction, 720 sandbox creates
per minute, a six-hour rollout timeout, and no agentic judge. It retains 16
rollouts per task, `clear_thinking=false`, and a static environment pool of 20
workers with 128 concurrent slots each. Training sandboxes use the distinct
`glm53-pd-train` label alongside `int4-syn-gen-bash`. Pool capacity is 2,560;
the rollout concurrency cap is fixed at 2,048 by setting initial, minimum and
maximum in-flight episodes to the same value. Episodes doing tool work also
occupy slots, so this does not guarantee 2,048 simultaneous inference requests.

`orchestrator.batch_size` counts trainer-bound traces per optimizer step,
independently of concurrency. It is omitted here and resolves to 128 by default;
choose the training batch size when adding the trainer. With `group_size=16`,
it must be a multiple of 16. The default `constant_trainer_batch_size=true`
prunes zero-advantage RL samples before counting toward the batch and keeps
collecting until enough useful traces remain. For plain GRPO, groups with
identical rewards have zero advantage and do not fill that batch. Setting
`constant_trainer_batch_size=false` still prunes zero-advantage samples, but
after selecting the batch, so fewer useful traces can reach the trainer.

The inference-only command above does not start environments, training, evals,
persistent supervisors or profiling. The RL launcher uses the same role-resolution
and Mooncake provisioning paths.
