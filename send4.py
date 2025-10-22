import argparse

import torch
from nixl._api import nixl_agent, nixl_agent_config

from prime_rl.trainer.rl.nixl_utils import SenderTensor
from prime_rl.utils.logger import setup_logger

logger = setup_logger("DEBUG")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5555)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    torch.set_default_device("cuda:0")

    config = nixl_agent_config(True, True, args.port)

    # Allocate memory and register with NIXL
    agent = nixl_agent("target", config)
    all_tensors = [torch.ones(10, dtype=torch.float32) for _ in range(2)]

    logger.info("Running test with %s tensors", all_tensors)

    for tensor in all_tensors:
        sender_tensor = SenderTensor(agent, [tensor])
        sender_tensor.send()

    logger.info("Test Complete.")
