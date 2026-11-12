from vllm.distributed.kv_transfer.kv_connector.v1.example_hidden_states_connector import ExampleHiddenStatesConnector


class TailHiddenStatesConnector(ExampleHiddenStatesConnector):
    """vLLM's hidden-states connector, writing only each prompt's last ``tail_tokens`` positions.

    Draft windows deep inside long trajectories need the target to prefill the whole prefix, but
    only the window's hidden states. Set ``tail_tokens`` in ``kv_connector_extra_config``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tail_tokens = self._kv_transfer_config.get_from_extra_config("tail_tokens", None)

    def _write_tensors(self, tensors, event, filename, lock_fd) -> None:
        if self.tail_tokens is not None:
            # Views of the pending host copies; the base class waits for the copy before writing.
            tensors = {key: value[-self.tail_tokens :] for key, value in tensors.items()}
        super()._write_tensors(tensors, event, filename, lock_fd)
