import argparse

import torch
from nixl._api import nixl_agent, nixl_agent_config

from prime_rl.trainer.rl.nixl_utils import ReceiverTensor
from prime_rl.utils.logger import setup_logger

logger = setup_logger("DEBUG")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, required=True)
    parser.add_argument("--port", type=int, default=5555)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    torch.set_default_device("cuda:0")

    config = nixl_agent_config(True, True, 0)

    # Allocate memory and register with NIXL
    agent = nixl_agent("initiator", config)
    all_tensors = [torch.zeros(10, dtype=torch.float32) for _ in range(2)]

    logger.info("Running test with %s tensors", all_tensors)

    for tensor in all_tensors:
        receiver_tensor = ReceiverTensor(agent, [tensor], args.ip, args.port)
        receiver_tensor.receive()

    # Verify data after read
    # Verify data after read
    for i, tensor in enumerate(all_tensors):
        if not torch.allclose(tensor, torch.ones(10)):
            logger.error("Data verification failed for tensor %d.", i)
            exit()
    logger.info("Data verification passed")
    logger.info("Test Complete.")
