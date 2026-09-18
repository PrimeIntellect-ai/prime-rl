# GLM-5.3 inference preflight

These are standalone inference-only preflight configurations for the GLM-5 family. Run them
through the prime-rl inference entrypoint, not `vllm serve`.

```bash
uv run inference @ configs/advanced/models/glm-5.3/infer/pd.toml
uv run inference @ configs/advanced/models/glm-5.3/infer/pd-llmd.toml
```

`pd-llmd.toml` requires llm-d, Mooncake, and the RDMA device list appropriate for the cluster.
Set its output path and Slurm partition for your site before launching.
