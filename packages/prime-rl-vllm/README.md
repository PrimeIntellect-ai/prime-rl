# prime-rl-vllm

prime-rl's vLLM plugin. Installed next to a stock vLLM, it makes `vllm serve` behave like prime-rl's inference server:

- `vllm.general_plugins` entry point: prime-rl's vLLM patches, the `/inference/v1/generate` handler, and the admin routes (`/pause`, `/resume`, `/update_weights`, `/init_broadcaster`, `/liveness`, ...).
- Weight-update worker extensions, selected with `--worker-extension-cls`:
  - `prime_rl.inference.vllm.worker.filesystem.FileSystemWeightUpdateWorker`
  - `prime_rl.inference.vllm.worker.nccl.NCCLWeightUpdateWorker`
  - `prime_rl.inference.vllm.worker.modelexpress.ModelExpressWeightUpdateWorker` (install the `modelexpress` extra)
  - `prime_rl.inference.vllm.worker.nixl.NIXLWeightUpdateWorker` (needs the full `prime-rl` package)
- `prime_rl.model_conversion`: PrimeRL's model configs and the HF <-> PrimeRL state-dict conversion chains, shared with the trainer. The ModelExpress worker uses it to convert the PrimeRL-native tensors a trainer publishes.

Settings that `vllm serve` has no flag for are read from the environment:

| Variable | Default | Effect |
| --- | --- | --- |
| `PRIME_RL_LIVENESS_TIMEOUT_S` | `30` | Timeout of the `/liveness` engine probe |
| `PRIME_NO_MOE_LORA` | unset | `1` disables LoRA on MoE layers |
| `PRL_TOP_P_BEFORE_TOP_K` | unset | `1` applies top-p before top-k |

The plugin patches vLLM internals, so it supports exactly the vLLM version prime-rl pins.
