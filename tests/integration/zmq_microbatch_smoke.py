from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.distributed as dist

import prime_rl.trainer.runs as runs
from prime_rl.configs.shared import FileSystemTransportConfig, ZMQTransportConfig
from prime_rl.trainer.rl.data import DataLoader
from prime_rl.trainer.runs import setup_multi_run_manager
from prime_rl.trainer.world import get_world, reset_world


def main() -> None:
    output_dir = Path(sys.argv[1])
    zmq_base_port = int(sys.argv[2])

    dist.init_process_group("gloo", init_method="env://")
    reset_world()
    runs._MULTI_RUN_MANAGER = None

    world = get_world()
    setup_multi_run_manager(output_dir=output_dir, max_runs=1, device=torch.device("cpu"))

    loader = DataLoader(
        output_dir=output_dir,
        start_step=0,
        dp_world_size=world.world_size,
        seq_len=2,
        pad_to_multiple_of=1,
        tokenizer=None,
        config=FileSystemTransportConfig(),
        micro_batch_transport_config=ZMQTransportConfig(
            host="127.0.0.1",
            port=zmq_base_port,
            recv_timeout_seconds=5,
            ready_timeout_seconds=5,
            publish_timeout_seconds=20,
            publish_grace_ms=0,
        ),
    )

    loader.wait_for_batch()
    loader.synchronize_state()
    batch = loader.get_batch()
    assert len(batch) == 1
    micro_batch = batch[0]

    rank_output_dir = output_dir / "rank_outputs"
    rank_output_dir.mkdir(exist_ok=True)
    with open(rank_output_dir / f"rank_{world.rank}.json", "w") as f:
        json.dump(
            {
                "input_ids": micro_batch["input_ids"].tolist(),
                "position_ids": micro_batch["position_ids"].tolist(),
                "loss_mask": micro_batch["loss_mask"].tolist(),
                "advantages": micro_batch["advantages"].tolist(),
                "rewards": micro_batch["rewards"].tolist() if micro_batch["rewards"] is not None else None,
                "inference_logprobs": micro_batch["inference_logprobs"].tolist(),
                "temperatures": micro_batch["temperatures"].tolist(),
                "env_names": micro_batch["env_names"],
                "lora_num_tokens": micro_batch["lora_num_tokens"].tolist(),
                "training_mode": micro_batch["training_mode"],
                "mm_kwargs": micro_batch["mm_kwargs"],
                "mm_token_type_ids": micro_batch["mm_token_type_ids"],
            },
            f,
        )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
