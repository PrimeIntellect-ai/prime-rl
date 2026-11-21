"""Score an immutable initial-policy model using the trainer's packed inputs."""

import torch

from prime_rl.trainer.model import forward
from prime_rl.trainer.rl.loss import selective_log_softmax, shift_tensor_right


@torch.no_grad()
def reference_logprobs(model, input_ids, position_ids, labels, temperatures, seq_lens):
    out = forward(model, input_ids, position_ids, labels=labels, temperature=temperatures, seq_lens=seq_lens)
    values = out.get("logprobs")
    if values is None:
        values = selective_log_softmax(out["logits"] / temperatures.unsqueeze(-1), labels)
    return shift_tensor_right(values, pad_value=0.0).detach()
