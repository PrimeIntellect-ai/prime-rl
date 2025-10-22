import torch
from nixl._api import nixl_agent, nixl_agent_config
from torch import nn
from torch.distributed.tensor import DTensor

from prime_rl.trainer.rl.config import NixlBroadcastConfig
from prime_rl.trainer.weights import _convert_tt_moe_to_hf_, _has_tt_moe_layers
from prime_rl.utils.logger import get_logger


class NixlBroadcastManager:
    """Utility class to broadcast the weight checkpoint using Nixl."""

    def __init__(self, config: NixlBroadcastConfig):
        self.logger = get_logger()

        agent_config = nixl_agent_config(True, True, config.port)
        self.agent = nixl_agent("target", agent_config)

    def broadcast(self, model: nn.Module, dtype: torch.dtype = torch.bfloat16):
        """Broadcast the weight checkpoint using Nixl."""

        model_state_dict = model.state_dict()
        if _has_tt_moe_layers(model_state_dict):
            _convert_tt_moe_to_hf_(model_state_dict)

        for key, value in model.state_dict().items():
            if isinstance(value, DTensor):
                value = value.to(dtype)
                value = value.full_tensor()

            # todo: use a better bucketing strategy
            # todo: make sure both communication can be done in parallel
            sender = SenderTensor(self.agent, [value])
            sender.send()


class SenderTensor:
    def __init__(self, agent: nixl_agent, tensors: list[torch.Tensor]):
        self.agent = agent
        self.tensors = tensors

    def send(self):
        reg_descs = self.agent.register_memory(self.tensors)
        if not reg_descs:
            raise RuntimeError("Memory registration failed.")

        ready = False

        target_descs = reg_descs.trim()
        target_desc_str = self.agent.get_serialized_descs(target_descs)

        # Send desc list to initiator when metadata is ready
        while not ready:
            ready = self.agent.check_remote_metadata("initiator")

        self.agent.send_notif("initiator", target_desc_str)

        get_logger().debug("Waiting for transfer")

        # Waiting for transfer
        while not self.agent.check_remote_xfer_done("initiator", b"UUID"):
            continue

        self.agent.deregister_memory(reg_descs)


class ReceiverTensor:
    def __init__(self, agent: nixl_agent, tensors: list[torch.Tensor], ip: str, port: int):
        self.agent = agent
        self.tensors = tensors
        self.ip = ip
        self.port = port

    def receive(self):
        reg_descs = self.agent.register_memory(self.tensors)
        if not reg_descs:
            raise RuntimeError("Memory registration failed.")

        get_logger().info("Initiator sending to %s", self.ip)
        self.agent.fetch_remote_metadata("target", self.ip, self.port)
        self.agent.send_local_metadata(self.ip, self.port)

        notifs = self.agent.get_new_notifs()

        while len(notifs) == 0:
            notifs = self.agent.get_new_notifs()

        target_descs = self.agent.deserialize_descs(notifs["target"][0])
        initiator_descs = reg_descs.trim()

        # Ensure remote metadata has arrived from fetch
        ready = False
        while not ready:
            ready = self.agent.check_remote_metadata("target")

        get_logger().info("Ready for transfer")

        xfer_handle = self.agent.initialize_xfer("READ", initiator_descs, target_descs, "target", "UUID")

        if not xfer_handle:
            get_logger().error("Creating transfer failed.")
            exit()

        state = self.agent.transfer(xfer_handle)
        if state == "ERR":
            get_logger().error("Posting transfer failed.")
            exit()
        while True:
            state = self.agent.check_xfer_state(xfer_handle)
            if state == "ERR":
                get_logger().error("Transfer got to Error state.")
                exit()
            elif state == "DONE":
                break

        self.agent.remove_remote_agent("target")
        self.agent.release_xfer_handle(xfer_handle)
        self.agent.invalidate_local_metadata(self.ip, self.port)

        self.agent.deregister_memory(reg_descs)
