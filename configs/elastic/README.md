# Testing Elastic Inference

To test the elastic inference pool without having the `rl` entrypoint start the inference server:

1. **Run RL with the elastic config** (orchestrator discovers servers via DNS; it will find `localhost:8000` once the server is started):

   ```bash
   uv run rl @ examples/alphabet_sort/rl_elastic.toml
   ```

   The config uses `[orchestrator.client.elastic]` with `hostname = "localhost"` and `port = 8000`. The orchestrator will wait for at least one ready server before starting rollouts.

2. **Start the inference server manually** (in a separate terminal or on another machine):

   ```bash
   uv run inference --model.name Qwen/Qwen3-4B-Instruct-2507 --enable-lora --max-lora-rank 32
   ```