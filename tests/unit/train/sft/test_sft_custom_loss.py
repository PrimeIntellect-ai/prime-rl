import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from pydantic import ValidationError
from torch import nn
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
    ce = -(inputs.target_logprobs() * inputs.loss_mask).sum()
    return LossOutputs(loss=scale * ce, count=inputs.loss_mask.sum(), metrics={"ce": ce.detach()})


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


def padded_batch() -> dict:
    """Two samples of 3 and 2 tokens, then 2 tokens of padding."""
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 1, 2]])
    return {
        "input_ids": input_ids,
        "position_ids": torch.tensor([[0, 1, 2, 0, 1, 0, 1]]),
        "seq_lens": torch.tensor([3, 2, 2]),
        "sample_lens": torch.tensor([3, 2]),
        "truncated": torch.tensor([False, False]),
        "target_ids": torch.tensor([[2, 3, 4, 5, 6, 0, 0]]),
        "loss_mask": torch.tensor([[True, True, False, True, True, False, False]]),
    }


def test_custom_loss_calls_the_function_with_its_kwargs():
    loss_fn = setup_sft_loss_fn(SFTCustomLossConfig(import_path=f"{__name__}.scaled_ce", kwargs={"scale": 2.0}))
    model = tiny_causal_lm()
    batch = padded_batch()
    lm_head_calls = []
    model.lm_head.register_forward_hook(lambda *args: lm_head_calls.append(1))
    outputs = run_loss_fn(loss_fn, model, batch)

    # stock SFT's summed cross-entropy on the same micro-batch
    stock = forward(
        model, batch["input_ids"], batch["position_ids"], seq_lens=batch["seq_lens"],
        labels=batch["target_ids"].masked_fill(~batch["loss_mask"], -100),
    )["loss"]  # fmt: skip
    torch.testing.assert_close(outputs.loss, 2.0 * stock)
    assert outputs.metrics["ce"].shape == (2,)  # one value per sample
    torch.testing.assert_close(outputs.metrics["ce"].sum(), stock.detach())
    assert outputs.count == 4
    assert len(lm_head_calls) == 2  # once for the micro-batch's two samples, once for the stock loss


def test_custom_loss_is_called_on_each_sample_without_the_padding():
    model = tiny_causal_lm()
    batch = {**padded_batch(), "truncated": torch.tensor([False, True]), "sample_data": ["a", "b"]}
    hidden_states = model.backbone(batch["input_ids"], batch["position_ids"]).last_hidden_state[0]
    samples = []

    def record(inputs: LossInputs) -> LossOutputs:
        samples.append(inputs)
        return LossOutputs(loss=inputs.hidden_states.sum(), count=1)

    run_loss_fn(record, model, batch)
    assert [sample.input_ids.tolist() for sample in samples] == [[1, 2, 3], [4, 5]]
    assert [sample.position_ids.tolist() for sample in samples] == [[0, 1, 2], [0, 1]]
    assert [sample.target_ids.tolist() for sample in samples] == [[2, 3, 4], [5, 6]]
    assert [sample.loss_mask.tolist() for sample in samples] == [[True, True, False], [True, True]]
    assert [sample.truncated for sample in samples] == [False, True]
    assert [sample.sample_data for sample in samples] == ["a", "b"]
    torch.testing.assert_close(samples[1].hidden_states, hidden_states[3:5])
    # a slice of the micro-batch's log-probabilities, equal to those of the sample on its own
    sample = samples[1]
    torch.testing.assert_close(
        sample.target_logprobs(), token_logprobs(model.lm_head, sample.hidden_states[None], sample.target_ids[None])[0]
    )


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
    with pytest.raises(ValidationError, match="offload"):
        SFTConfig(loss=loss, model={"offload": "full"})


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
    """Calls the LM head on the sample (its summed cross-entropy), through ``target_logprobs`` (once per micro-batch)
    and the score head."""
    mask, head = inputs.loss_mask, inputs.model.score_head
    ce = inputs.model.lm_head(inputs.hidden_states[None], inputs.target_ids.masked_fill(~mask, -100)[None])
    scores = head(inputs.hidden_states[mask].to(head.weight.dtype)).float()
    loss = ce["loss"] - (inputs.target_logprobs() * mask).sum() + scores.square().sum()
    return LossOutputs(loss=loss, count=mask.sum())


def score_loss(inputs: LossInputs) -> LossOutputs:
    """Calls only the score head, not the LM head."""
    head = inputs.model.score_head
    scores = head(inputs.hidden_states[inputs.loss_mask].to(head.weight.dtype)).float()
    return LossOutputs(loss=scores.square().sum(), count=inputs.loss_mask.sum())


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
        "sample_lens": torch.tensor([6, 4]),
        "truncated": torch.tensor([False, False]),
        "target_ids": torch.roll(input_ids, -1, dims=1),
        "loss_mask": torch.tensor([[False, True, True, False, True, True, False, True, True, True]]),
    }


def fsdp_worker(rank: int, world: int, tied: bool, loss_fn, init_file: str, out_dir: str) -> None:
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world), LOCAL_RANK=str(rank), LOCAL_WORLD_SIZE=str(world))
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world)
    parallel_dims_module.device_type = "cpu"  # the mesh is built on CUDA by default
    config = ModelConfig(name="heads", ep=1, compile=None)
    model = heads_model(tied)
    setup_fsdp(model, config, get_parallel_dims(config))  # prime-rl's real FSDP setup

    def grads() -> dict:
        return {name: p.grad.full_tensor() for name, p in model.named_parameters() if p.grad is not None}

    # As in the trainer: a step's backward, then validation (a no-grad forward, then reshard_module before clipping),
    # then the next step's backward on cleared gradients
    run_loss_fn(loss_fn, model, heads_batch(rank)).loss.backward()
    with torch.no_grad():
        run_loss_fn(loss_fn, model, heads_batch(rank))
    reshard_module(model)
    steps = [grads()]
    model.zero_grad(set_to_none=True)
    run_loss_fn(loss_fn, model, heads_batch(rank)).loss.backward()
    steps.append(grads())
    torch.save(steps, os.path.join(out_dir, f"{rank}.pt"))
    dist.destroy_process_group()


@pytest.mark.parametrize("loss_fn", [heads_loss, score_loss])
@pytest.mark.parametrize("tied", [True, False])
def test_custom_loss_under_fsdp(tied, loss_fn, tmp_path):
    world = 2
    args = (world, tied, loss_fn, str(tmp_path / "rendezvous"), str(tmp_path))
    mp.spawn(fsdp_worker, args=args, nprocs=world, join=True)

    # The same micro-batches on one process without FSDP, in bf16 as FSDP computes; FSDP averages over ranks.
    model = heads_model(tied).to(torch.bfloat16)
    sum(run_loss_fn(loss_fn, model, heads_batch(rank)).loss for rank in range(world)).div(world).backward()
    expected = {name: param.grad.float() for name, param in model.named_parameters() if param.grad is not None}
    if not tied:  # a tied LM head's weight is listed as the embeddings'
        assert ("lm_head.weight" in expected) == (loss_fn is heads_loss)
    for rank in range(world):
        for grads in torch.load(tmp_path / f"{rank}.pt"):
            assert grads.keys() == expected.keys()
            for name, grad in expected.items():
                torch.testing.assert_close(grads[name], grad, rtol=0.05, atol=0.02)
