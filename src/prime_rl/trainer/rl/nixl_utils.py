import torch
from nixl._api import nixl_agent

from prime_rl.utils.logger import get_logger


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
