"""Two-GPU FSDP training smoke tests for every PrimeRL architecture.

Each case writes a tiny HF-format checkpoint, builds the model through the real trainer path
(`setup_model`: meta init, FSDP2, DCP load with HF->PrimeRL conversion, activation checkpointing),
then runs a few forward/backward/optimizer steps on a fixed packed batch. Parallel layouts
(CP, EP) are checked against a single-GPU reference in `test_parallel_equivalence.py`.
Run on a 2-GPU node (e.g. 2x H200):

    uv run torchrun --nproc-per-node=2 -m pytest tests/unit/train/models/test_fsdp_training.py -v
"""

import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from prime_rl.configs.trainer import ModelConfig
from prime_rl.trainer.model import forward, setup_model
from prime_rl.trainer.models.layers.lm_head import IGNORE_INDEX
from prime_rl.trainer.models.llama import LlamaConfig
from prime_rl.trainer.parallel_dims import get_parallel_dims
from prime_rl.trainer.utils import clip_grad_norm_, setup_torch_distributed
from prime_rl.utils.weights import save_state_dict
from tests.unit.train.models.arch_configs import ARCH_CONFIGS, build_model, reference_state_dict

NUM_STEPS = 5
SEQ_LEN = 256
# Several documents per packed row so per-document boundaries (and the CP shard cut) are exercised.
DOC_LENS = [96, 112, 48]
LR = 1e-3


@pytest.fixture(scope="module")
def distributed():
    if int(os.environ.get("WORLD_SIZE", 1)) != 2:
        pytest.skip("run with torchrun --nproc-per-node=2")
    setup_torch_distributed()
    yield
    dist.barrier()
    if dist.get_rank() == 0:
        for arch in ARCH_CONFIGS:
            if arch in _MODEL_DIRS:
                shutil.rmtree(_MODEL_DIRS[arch], ignore_errors=True)
    dist.destroy_process_group()


def _write_checkpoint(arch: str, model_dir: Path) -> None:
    """Write `config.json` plus a HF-format safetensors checkpoint for the arch."""
    (model_dir / "config.json").write_text(json.dumps(ARCH_CONFIGS[arch]))
    save_state_dict(build_model(arch).convert_to_hf(reference_state_dict(arch)), model_dir)


_MODEL_DIRS: dict[str, Path] = {}


def _model_dir(arch: str) -> Path:
    """A checkpoint directory shared by both ranks: rank 0 writes it, every rank gets its path."""
    if arch not in _MODEL_DIRS:
        path = [tempfile.mkdtemp(prefix=f"prime-rl-{arch}-") if dist.get_rank() == 0 else None]
        dist.broadcast_object_list(path, src=0)
        model_dir = Path(path[0])
        if dist.get_rank() == 0:
            _write_checkpoint(arch, model_dir)
        dist.barrier()
        _MODEL_DIRS[arch] = model_dir
    return _MODEL_DIRS[arch]


def _fake_batch(vocab_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The same packed row on every rank: input ids, per-document position ids, labels, seq_lens."""
    generator = torch.Generator().manual_seed(0)
    input_ids = torch.randint(0, vocab_size, (1, SEQ_LEN), generator=generator)
    position_ids = torch.cat([torch.arange(length) for length in DOC_LENS]).unsqueeze(0)
    labels = torch.roll(input_ids, shifts=-1, dims=1)
    # No next-token target across a document boundary.
    labels[0, torch.cumsum(torch.tensor(DOC_LENS), 0) - 1] = IGNORE_INDEX
    seq_lens = torch.tensor(DOC_LENS)
    return input_ids.cuda(), position_ids.cuda(), labels.cuda(), seq_lens.cuda()


def _train(arch: str) -> list[float]:
    """Run NUM_STEPS steps and return the global mean loss before each optimizer step."""
    config = ModelConfig(name=str(_model_dir(arch)), ep=1, compile=None)
    parallel_dims = get_parallel_dims(config, SEQ_LEN)
    model = setup_model(config, parallel_dims)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)

    text_config = getattr(model.config, "text_config", model.config)
    input_ids, position_ids, labels, seq_lens = _fake_batch(text_config.vocab_size)

    dp_group = parallel_dims.get_mesh("dp_cp").get_group()
    global_tokens = (labels != IGNORE_INDEX).sum()
    dist.all_reduce(global_tokens, group=dp_group)
    # Same normalization as the SFT trainer: FSDP averages gradients over every data-parallel rank.
    loss_scale = parallel_dims.fsdp_gradient_divide_factor / global_tokens.item()

    losses = []
    for step in range(NUM_STEPS):
        out = forward(model, input_ids, position_ids, seq_lens=seq_lens, labels=labels)
        loss_sum = out["loss"]
        (loss_sum * loss_scale).backward()

        grad_norm = clip_grad_norm_(None, model, max_norm=1.0)
        assert torch.isfinite(grad_norm), f"{arch} step {step}: non-finite grad norm {grad_norm.item()}"
        assert grad_norm > 0, f"{arch} step {step}: zero gradients"
        optimizer.step()
        optimizer.zero_grad()

        global_loss = loss_sum.detach().clone()
        dist.all_reduce(global_loss, group=dp_group)
        losses.append(global_loss.item() / global_tokens.item())

    for param in model.parameters():
        local = param.to_local() if isinstance(param, DTensor) else param
        assert torch.isfinite(local).all(), f"{arch}: non-finite parameters after {NUM_STEPS} steps"

    del model, optimizer, out
    torch.cuda.empty_cache()
    return losses


@pytest.mark.gpu
@pytest.mark.parametrize("arch", list(ARCH_CONFIGS))
def test_fsdp_training(arch: str, distributed):
    losses = _train(arch)
    assert all(torch.isfinite(torch.tensor(loss)) for loss in losses), f"{arch}: losses {losses}"
    assert losses[-1] < losses[0], f"{arch}: loss did not decrease over {NUM_STEPS} steps: {losses}"


def test_tied_word_embeddings_are_rejected():
    with pytest.raises(ValueError, match="tie_word_embeddings"):
        LlamaConfig(hidden_size=64, num_attention_heads=4, vocab_size=128, tie_word_embeddings=True)
