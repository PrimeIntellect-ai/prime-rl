import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from pydantic import ValidationError
from torch import nn
from torch.distributed.tensor import DTensor
from transformers import PretrainedConfig

import prime_rl.trainer.parallel_dims as parallel_dims_module
from prime_rl.configs.sft import SFTConfig, SFTCustomLossConfig
from prime_rl.configs.trainer import ModelConfig
from prime_rl.trainer.model import forward, reshard_module, setup_fsdp
from prime_rl.trainer.models.base import PreTrainedModelPrimeRL
from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head, token_logprobs
from prime_rl.trainer.parallel_dims import get_parallel_dims
from prime_rl.trainer.sft.loss import LossInputs, LossOutputs, run_loss_fn, setup_sft_loss_fn


def scaled_ce(inputs: LossInputs, scale: float = 1.0) -> LossOutputs:
    """Stock SFT's token cross-entropy, times ``scale``."""
    batch = inputs.batch
    logprobs = token_logprobs(inputs.model.lm_head, inputs.hidden_states, batch["target_ids"])
    ce = -(logprobs * batch["loss_mask"]).sum()
    return LossOutputs(loss=scale * ce, count=batch["loss_mask"].sum(), metrics={"ce": ce.detach()})


class TinyCausalLM(nn.Module):
    def __init__(self, vocab: int = 11, hidden: int = 4):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.model = self.backbone
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def backbone(self, input_ids, position_ids, **kwargs):
        return SimpleNamespace(last_hidden_state=self.norm(self.embed(input_ids)))


def tiny_causal_lm(chunk_size: int | None = None) -> TinyCausalLM:
    torch.manual_seed(0)
    model = TinyCausalLM()
    model.model = nn.Module()  # inject_prime_lm_head expects a backbone module at model.model
    model.model.forward = model.backbone
    inject_prime_lm_head(model, chunk_size=chunk_size)
    return model


def test_forward_runs_a_function_on_the_hidden_states():
    model = tiny_causal_lm()
    input_ids, position_ids, seq_lens = torch.tensor([[1, 2, 3]]), torch.tensor([[0, 1, 2]]), torch.tensor([3])

    out = forward(model, input_ids, position_ids, seq_lens=seq_lens, hidden_states_fn=lambda h: {"h": h})
    torch.testing.assert_close(out["h"], model.backbone(input_ids, position_ids).last_hidden_state)

    out = forward(model, input_ids, position_ids, seq_lens=seq_lens)
    assert out["logits"].shape == (1, 3, 11)


def test_custom_loss_calls_the_function_with_its_kwargs():
    loss_fn = setup_sft_loss_fn(SFTCustomLossConfig(import_path=f"{__name__}.scaled_ce", kwargs={"scale": 2.0}))
    model = tiny_causal_lm()
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    batch = {
        "input_ids": input_ids,
        "position_ids": torch.tensor([[0, 1, 2, 0, 1]]),
        "seq_lens": torch.tensor([3, 2]),
        "target_ids": torch.tensor([[2, 3, 4, 5, 6]]),
        "loss_mask": torch.tensor([[True, True, False, True, True]]),
    }
    outputs = run_loss_fn(loss_fn, model, batch)

    # stock SFT's summed cross-entropy on the same micro-batch
    stock = forward(
        model, input_ids, batch["position_ids"], seq_lens=batch["seq_lens"],
        labels=batch["target_ids"].masked_fill(~batch["loss_mask"], -100),
    )["loss"]  # fmt: skip
    torch.testing.assert_close(outputs.loss, 2.0 * stock)
    torch.testing.assert_close(outputs.metrics["ce"], stock.detach())
    assert outputs.count == 4


@pytest.mark.parametrize("chunk_size", [None, 4])  # the vanilla and the fused LM head
def test_token_logprobs(chunk_size):
    model = tiny_causal_lm(chunk_size)
    hidden_states, labels = torch.randn(1, 5, 4, requires_grad=True), torch.tensor([[1, 0, 10, 3, 3]])
    logprobs = token_logprobs(model.lm_head, hidden_states, labels)
    expected = (hidden_states @ model.lm_head.weight.T).log_softmax(-1).gather(-1, labels[..., None])[..., 0]
    torch.testing.assert_close(logprobs, expected)
    logprobs.sum().backward()
    assert hidden_states.grad.abs().sum() > 0 and model.lm_head.weight.grad.abs().sum() > 0


def test_custom_loss_config_rejects_unsupported_setups():
    assert SFTConfig().loss.type == "ce"
    loss = {"type": "custom", "import_path": f"{__name__}.scaled_ce"}
    assert SFTConfig(loss=loss).loss.import_path.endswith("scaled_ce")
    with pytest.raises(ValidationError, match="not supported with context parallelism"):
        SFTConfig(loss=loss, model={"cp": 2})


class HeadsConfig(PretrainedConfig):
    model_type = "custom_loss_heads_test"


class Block(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.mix = nn.Linear(hidden, hidden)

    def forward(self, x):
        return x + torch.tanh(self.mix(x))


class Backbone(nn.Module):
    def __init__(self, vocab: int, hidden: int):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList([Block(hidden), Block(hidden)])
        self.norm = nn.LayerNorm(hidden)

    def forward(self, input_ids, position_ids, *, seq_lens, seq_lens_are_pre_shard=False):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        return SimpleNamespace(last_hidden_state=self.norm(x))


class HeadsModel(PreTrainedModelPrimeRL):
    """A causal LM with a head of its own, its LM head tied to the embeddings or not. Under prime-rl's FSDP setup the
    score head and a tied LM head belong to the root unit; an untied LM head shares a unit with the final norm."""

    config_class = HeadsConfig

    def __init__(self, config, vocab: int = 32, hidden: int = 8):
        super().__init__(config)
        self.model = Backbone(vocab, hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.score_head = nn.Linear(hidden, 1)

    def init_buffers_post_meta(self):
        pass


def heads_loss(inputs: LossInputs) -> LossOutputs:
    """Calls the LM head twice (the summed cross-entropy, then per-token log-probabilities) and the score head."""
    batch, head = inputs.batch, inputs.model.score_head
    ce = inputs.model.lm_head(inputs.hidden_states, batch["target_ids"].masked_fill(~batch["loss_mask"], -100))
    logprobs = token_logprobs(inputs.model.lm_head, inputs.hidden_states, batch["target_ids"])
    scores = head(inputs.hidden_states[batch["loss_mask"]].to(head.weight.dtype)).float()
    loss = ce["loss"] - (logprobs * batch["loss_mask"]).sum() + scores.square().sum()
    return LossOutputs(loss=loss, count=batch["loss_mask"].sum())


def heads_model(tied: bool) -> HeadsModel:
    torch.manual_seed(0)
    model = HeadsModel(HeadsConfig(tie_word_embeddings=tied))
    inject_prime_lm_head(model)
    return model


def heads_batch(rank: int) -> dict:
    generator = torch.Generator().manual_seed(rank)
    input_ids = torch.randint(0, 32, (1, 10), generator=generator)
    return {
        "input_ids": input_ids,
        "position_ids": torch.tensor([[*range(6), *range(4)]]),
        "seq_lens": torch.tensor([6, 4]),
        "target_ids": torch.roll(input_ids, -1, dims=1),
        "loss_mask": torch.tensor([[False, True, True, False, True, True, False, True, True, True]]),
    }


def fsdp_worker(rank: int, world: int, tied: bool, init_file: str, out_dir: str) -> None:
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world), LOCAL_RANK=str(rank), LOCAL_WORLD_SIZE=str(world))
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world)
    parallel_dims_module.device_type = "cpu"  # the mesh is built on CUDA by default
    config = ModelConfig(name="heads", ep=1, compile=None)
    model = heads_model(tied)
    setup_fsdp(model, config, get_parallel_dims(config))  # prime-rl's real FSDP setup

    run_loss_fn(heads_loss, model, heads_batch(rank)).loss.backward()
    grads = {name: p.grad.full_tensor() for name, p in model.named_parameters()}
    with torch.no_grad():  # as in validation, which the trainer follows with reshard_module before clipping
        run_loss_fn(heads_loss, model, heads_batch(rank))
    reshard_module(model)
    resharded = all(isinstance(p, DTensor) for p in model.parameters())
    torch.save({"grads": grads, "resharded": resharded}, os.path.join(out_dir, f"{rank}.pt"))
    dist.destroy_process_group()


@pytest.mark.parametrize("tied", [True, False])
def test_custom_loss_under_fsdp(tied, tmp_path):
    world = 2
    mp.spawn(fsdp_worker, args=(world, tied, str(tmp_path / "rendezvous"), str(tmp_path)), nprocs=world, join=True)

    # The same micro-batches on one process without FSDP, in bf16 as FSDP computes; FSDP averages over ranks.
    model = heads_model(tied).to(torch.bfloat16)
    sum(run_loss_fn(heads_loss, model, heads_batch(rank)).loss for rank in range(world)).div(world).backward()
    for rank in range(world):
        result = torch.load(tmp_path / f"{rank}.pt")
        assert result["resharded"]  # after a loss without a backward, every FSDP unit reshards
        for name, param in model.named_parameters():
            torch.testing.assert_close(result["grads"][name], param.grad.float(), rtol=0.05, atol=0.02)
