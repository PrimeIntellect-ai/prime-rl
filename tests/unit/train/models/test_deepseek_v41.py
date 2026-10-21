import os

import pytest
import torch
import torch.distributed as dist
from torch import nn

from prime_rl.trainer.models.deepseek_v41 import DeepseekV41Config, DeepseekV41ForCausalLM, DeepseekV41TextConfig
from prime_rl.trainer.models.deepseek_v41 import attention as dsv41_attention
from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from prime_rl.utils.cp import CPContext
from prime_rl.utils.utils import default_dtype
from tests.unit.train.models.test_deepseek_v41_cpu import PUBLISHED_CONFIG, _published_keys

# Every layer is built through `DeepseekV41Attention.__init__`, which refuses to construct without
# the kernel it would dispatch to, so the whole file needs tilelang even though not every test calls it.
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        dsv41_attention.dsv4_sparse_attn is None,
        reason="the fused sparse attention kernel did not import; tilelang ships in the `gpu` extra, on linux only",
    ),
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
        reason="the indexer kernel quantizes to Triton fp8e4nv (e4m3), only supported on Hopper (SM90) and newer",
    ),
]

# One layer of every kind: a sliding-window layer, a ratio-2 KV and index source with an engram in
# front of it, a ratio-2 consumer, a ratio-1 source that also ranks candidate blocks, a second
# engram, and a ratio-1 index source scoring inside those candidates. The window is narrow and the
# candidate blocks few, so the long-range paths and the candidate filter all bind at `SEQ` tokens.
MODEL = dict(
    vocab_size=64,
    hidden_size=128,
    moe_intermediate_size=64,
    num_hidden_layers=6,
    # The smallest head count the sparse attention kernel's backward accepts.
    num_attention_heads=32,
    head_dim=32,
    q_lora_rank=64,
    o_lora_rank=16,
    o_groups=2,
    partial_rotary_factor=0.5,
    max_position_embeddings=512,
    sliding_window=6,
    compress_ratios=[0, 2, 2, 1, 1, 1],
    kv_source_layer_ids=[1, 3],
    index_source_layer_ids=[1, 3, 5],
    candidate_source_layer_id=3,
    candidate_topk_blocks=2,
    candidate_block_size=4,
    # `index_head_dim` must be a power of two for the fp8 scoring kernel.
    index_n_heads=16,
    index_head_dim=64,
    index_topk=4,
    n_routed_experts=8,
    num_experts_per_tok=2,
    engram_layer_ids=[1, 4],
    engram_num_embeddings=[240, 306],
    engram_max_ngram_size=3,
    engram_vocab_size=50,
    engram_n_heads=2,
    engram_head_dim=16,
    engram_compressed_vocab_size=64,
    rms_norm_eps=1e-6,
    rope_scaling={
        "rope_type": "yarn",
        "factor": 4,
        "beta_fast": 32,
        "beta_slow": 1,
        "original_max_position_embeddings": 128,
    },
)
MODEL_CONFIG = DeepseekV41TextConfig(**MODEL)
SEQ = 96


@pytest.fixture(autouse=True)
def _seed_rng():
    torch.manual_seed(0)


def _randomize(module: nn.Module) -> None:
    """Non-degenerate values for every parameter: the init values (unit gains, zero sinks and
    biases) would leave the paths they control indistinguishable from no-ops."""
    with torch.no_grad():
        for name, param in module.named_parameters():
            if name.endswith("scale") or name.endswith("norm.weight"):
                param.uniform_(0.5, 1.5)
            elif name.endswith("sinks") or name.endswith("base"):
                param.normal_(mean=0.0, std=0.5)
            else:
                param.normal_(mean=0.0, std=0.02)
        for name, buffer in module.named_buffers():
            if name.endswith("selection_bias"):
                buffer.normal_(mean=0.0, std=0.1)


def get_model() -> nn.Module:
    with torch.device("cuda"), default_dtype(torch.bfloat16):
        model = DeepseekV41ForCausalLM(MODEL_CONFIG)
    _randomize(model)
    model.model.engram_hasher.set_token_map(list(range(MODEL["vocab_size"])), MODEL["vocab_size"])
    inject_prime_lm_head(model)
    return model


def _packed(doc_lens: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    position_ids = torch.cat([torch.arange(length) for length in doc_lens])[None].cuda()
    return position_ids, torch.tensor(doc_lens, device="cuda")


def test_deepseek_v41_backward():
    """Every parameter that can train does, and the Lightning Indexers' still cannot."""
    model = get_model()
    input_ids = torch.randint(0, MODEL["vocab_size"], (1, SEQ), device="cuda")
    position_ids, seq_lens = _packed((SEQ,))
    output = model(input_ids=input_ids, position_ids=position_ids, seq_lens=seq_lens)
    assert output["logits"].shape == (1, SEQ, MODEL["vocab_size"])
    assert torch.isfinite(output["logits"]).all()
    output["logits"].float().square().mean().backward()

    dead, unexpectedly_alive = [], []
    for name, param in model.named_parameters():
        has_grad = param.grad is not None and param.grad.norm().item() > 0
        # The indexer reaches the loss only through integer top-k indices.
        if ".indexer." in name:
            if has_grad:
                unexpectedly_alive.append(name)
        elif not has_grad:
            dead.append(name)
    assert not dead, f"Parameters with zero/no gradients: {dead}"
    assert not unexpectedly_alive, f"Lightning Indexer parameters received a gradient: {unexpectedly_alive}"


def test_deepseek_v41_packed_matches_unpacked():
    """Two packed documents see exactly what each sees alone: window, entries, picks, candidate
    blocks and engram look-back all stop at the document boundary."""
    model = get_model()
    doc_lens = (SEQ - 40, 40)
    input_ids = torch.randint(0, MODEL["vocab_size"], (1, SEQ), device="cuda")

    with torch.no_grad():
        packed = model(input_ids=input_ids, seq_lens=_packed(doc_lens)[1], position_ids=_packed(doc_lens)[0])
        start = 0
        for length in doc_lens:
            doc_ids = input_ids[:, start : start + length]
            position_ids, seq_lens = _packed((length,))
            alone = model(input_ids=doc_ids, position_ids=position_ids, seq_lens=seq_lens)
            torch.testing.assert_close(
                packed["logits"][:, start : start + length], alone["logits"], rtol=2e-2, atol=2e-2
            )
            start += length


def test_deepseek_v41_rejects_positions_that_disagree_with_seq_lens():
    model = get_model()
    input_ids = torch.randint(0, MODEL["vocab_size"], (1, SEQ), device="cuda")
    _, seq_lens = _packed((SEQ // 2, SEQ // 2))
    position_ids = torch.arange(SEQ, device="cuda")[None]
    with pytest.raises(ValueError, match="restart at 0"):
        model(input_ids=input_ids, position_ids=position_ids, seq_lens=seq_lens)


def test_deepseek_v41_weight_conversion_roundtrip():
    with torch.device("cuda"):
        model = DeepseekV41ForCausalLM(MODEL_CONFIG)
    original = {name: tensor.clone() for name, tensor in model.state_dict().items()}

    state_dict = model.state_dict()
    model.convert_to_hf(state_dict)
    assert DeepseekV41ForCausalLM.is_hf_state_dict(state_dict)
    assert not DeepseekV41ForCausalLM.is_prime_state_dict(state_dict)
    model.convert_to_prime(state_dict)
    assert DeepseekV41ForCausalLM.is_prime_state_dict(state_dict)

    assert set(state_dict) == set(original)
    for name, tensor in original.items():
        assert torch.equal(state_dict[name], tensor), f"Value mismatch for {name}"


def test_deepseek_v41_model_names_are_the_published_ones():
    """The published model's state dict, converted to on-disk names, is exactly the checkpoint's
    trained keys: nothing the model holds is missing from the file, and vice versa."""
    config = DeepseekV41Config(**PUBLISHED_CONFIG)
    with torch.device("meta"):
        model = DeepseekV41ForCausalLM(config)
    state_dict = model.state_dict()
    model.convert_to_hf(state_dict)

    dropped = ("mtp.", "vision.", "image_")
    expected = {key for key in _published_keys(config.text_config) if not key.startswith(dropped)}
    expected = {key for key in expected if not key.endswith("bias_vl")}
    assert set(state_dict) == expected


@pytest.fixture(scope="module")
def cp_process_group():
    if int(os.environ.get("WORLD_SIZE", 1)) not in (2, 4):
        pytest.skip("run with torchrun --nproc-per-node=2 (or 4)")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    yield dist.group.WORLD
    dist.destroy_process_group()


def test_deepseek_v41_context_parallel_matches_the_whole_row(cp_process_group):
    """Every CP rank holding a contiguous shard of the queries reproduces the unsharded loss and
    gradients, on a packed row whose documents cross the shard boundaries: the window, the shared
    entries and index keys, the picks, the candidate blocks and the engram look-back all reach
    back into the previous rank's tokens and stop at document boundaries."""
    cp_rank, cp_world_size = dist.get_rank(), dist.get_world_size()
    model = get_model()
    with torch.no_grad():
        for tensor in [*model.parameters(), *model.buffers()]:
            dist.broadcast(tensor, src=0)

    # 192 tokens: at cp = 2 the cut at 96 falls inside the second document, at cp = 4 the cuts at
    # 48 and 144 fall inside the first and second; every document is long enough for candidates.
    doc_lens = (70, 90, 32)
    seq_len = sum(doc_lens)
    input_ids = torch.randint(0, MODEL["vocab_size"], (1, seq_len), device="cuda")
    labels = torch.randint(0, MODEL["vocab_size"], (1, seq_len), device="cuda")
    dist.broadcast(input_ids, src=0)
    dist.broadcast(labels, src=0)
    position_ids, seq_lens = _packed(doc_lens)

    with torch.no_grad():
        reference_logits = model(input_ids=input_ids, position_ids=position_ids, seq_lens=seq_lens)["logits"]
    reference_loss = model(input_ids=input_ids, position_ids=position_ids, seq_lens=seq_lens, labels=labels)["loss"]
    reference_loss.backward()
    reference_grads = {name: param.grad.clone() for name, param in model.named_parameters() if param.grad is not None}
    model.zero_grad(set_to_none=True)

    cp_context = CPContext(cp_process_group, cp_rank, cp_world_size)
    for module in model.modules():
        if hasattr(module, "cp_context"):
            module.cp_context = cp_context
    shard = slice(cp_rank * seq_len // cp_world_size, (cp_rank + 1) * seq_len // cp_world_size)
    kwargs = dict(
        input_ids=input_ids[:, shard],
        position_ids=position_ids[:, shard],
        seq_lens=seq_lens,
        seq_lens_are_pre_shard=True,
    )

    with torch.no_grad():
        logits = model(**kwargs)["logits"]
    torch.testing.assert_close(logits, reference_logits[:, shard], rtol=1e-3, atol=1e-3)

    loss = model(**kwargs, labels=labels[:, shard])["loss"]
    loss.backward()
    total_loss = loss.detach().clone()
    dist.all_reduce(total_loss)
    torch.testing.assert_close(total_loss, reference_loss.detach(), rtol=1e-5, atol=1e-5)

    # The forward is exact, but each rank rounds its partial bf16 gradients before they are summed.
    # Splitting the unsharded loss into the same token shards, with no CP at all, moves the
    # gradients just as much: ~0.4% overall and up to ~10% on the 3-element mHC scales, whose
    # gradient sums many cancelling per-token terms. A rank missing another rank's contribution
    # would be off by order `1 / cp_world_size` (at least 25% here), which the per-parameter bound catches.
    diff_sq, reference_sq = 0.0, 0.0
    for name, param in model.named_parameters():
        assert (param.grad is None) == (name not in reference_grads), name
        if param.grad is None:
            continue
        grad = param.grad.float()
        dist.all_reduce(grad)
        reference = reference_grads[name].float()
        diff_sq += (grad - reference).square().sum().item()
        reference_sq += reference.square().sum().item()
        relative = (grad - reference).norm() / reference.norm().clamp_min(1e-12)
        assert relative < 0.2, f"{name}: gradient differs by {relative:.3e} relative to the whole row"
    assert (diff_sq / reference_sq) ** 0.5 < 1e-2
