# Dynamo RL examples

These examples run Prime-RL against an external Python `dynamo.vllm` deployment.

- [`basic/`](basic/README.md) runs five Qwen3-0.6B MathEnv steps with one trainer GPU, one inference GPU, and NCCL weight transfer.
- [`extra/nemotron-3.5-super/`](extra/nemotron-3.5-super/README.md) runs Nemotron 3.5 Super VLM training with TP=4 inference, four trainer GPUs, inline routed-expert and sampling-mask replay, NCCL weight transfer, and W&B.

Prime-RL does not launch the external Dynamo frontend or worker for either recipe. Follow each recipe's README to start Dynamo before training.
