from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import Tensor, nn
from transformers import PreTrainedModel

SEQ_LEN = 100


def hf_checkpoint_state_dict(hf_model: PreTrainedModel, path: Path) -> dict[str, Tensor]:
    """The state dict as saved on disk, which is the layout the PrimeRL conversion chain loads from."""
    hf_model.save_pretrained(path)
    return {name: tensor.cuda() for name, tensor in load_file(path / "model.safetensors").items()}


def assert_relative_close(actual: Tensor, expected: Tensor, tol: float) -> None:
    """Relative L2 error, so one bf16 tolerance holds at any tensor scale."""
    error = ((actual.float() - expected.float()).norm() / expected.float().norm()).item()
    assert error < tol, f"relative error {error:.4f} >= {tol}"


def assert_matches_hf(hf_model: nn.Module, prime_model: nn.Module, tol: float) -> None:
    """Compare logits and input-embedding gradients on one random sequence."""
    input_ids = torch.randint(0, hf_model.config.vocab_size, (1, SEQ_LEN), device="cuda")
    position_ids = torch.arange(1, SEQ_LEN + 1, device="cuda").unsqueeze(0)
    seq_lens = torch.tensor([SEQ_LEN], device="cuda")
    hf_logits = hf_model(input_ids, position_ids=position_ids).logits
    prime_logits = prime_model(input_ids, position_ids=position_ids, seq_lens=seq_lens)["logits"]
    hf_logits.sum().backward()
    prime_logits.sum().backward()

    assert_relative_close(prime_logits, hf_logits, tol)
    assert_relative_close(
        prime_model.get_input_embeddings().weight.grad, hf_model.get_input_embeddings().weight.grad, tol
    )
