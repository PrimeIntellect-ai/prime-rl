"""Two-GPU FSDP training smoke tests for every PrimeRL architecture.

Each case writes a tiny HF-format checkpoint, builds the model through the real trainer path
(`setup_model`: meta init, FSDP2, DCP load with HF->PrimeRL conversion, activation checkpointing),
then runs a few forward/backward/optimizer steps on a fixed packed batch, with and without
context parallelism. Run on a 2-GPU node (e.g. 2x H200):

    uv run torchrun --nproc-per-node=2 -m pytest tests/unit/train/models/test_fsdp_training.py -v
"""

import functools
import json
import os
import shutil
import tempfile
import zlib
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from prime_rl.configs.trainer import ModelConfig
from prime_rl.trainer.model import forward, setup_model
from prime_rl.trainer.models.afmoe.modeling_afmoe import AfmoeFlashAttention
from prime_rl.trainer.models.gpt_oss.attention import GptOssAttention
from prime_rl.trainer.models.layers.attn import FlashAttention
from prime_rl.trainer.models.layers.lm_head import IGNORE_INDEX
from prime_rl.trainer.models.llama import LlamaConfig, LlamaForCausalLM
from prime_rl.trainer.models.registry import build_model_config, get_model_cls
from prime_rl.trainer.parallel_dims import get_parallel_dims
from prime_rl.trainer.utils import clip_grad_norm_, setup_torch_distributed
from prime_rl.utils.cp import setup_context_parallel, setup_cp_params, shard_for_cp
from prime_rl.utils.utils import default_dtype
from prime_rl.utils.weights import save_state_dict

NUM_STEPS = 5
SEQ_LEN = 256
# Several documents per packed row so per-document boundaries (and the CP shard cut) are exercised.
DOC_LENS = [96, 112, 48]
LR = 1e-3

_ATTN = dict(num_attention_heads=4, num_key_value_heads=2, head_dim=64)
_MOE = dict(num_experts=8, num_experts_per_tok=2, moe_intermediate_size=128)
_QWEN3_5_TEXT = dict(
    vocab_size=512,
    hidden_size=256,
    intermediate_size=512,
    num_hidden_layers=4,
    layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    max_position_embeddings=1024,
    linear_key_head_dim=32,
    linear_value_head_dim=32,
    linear_num_key_heads=4,
    linear_num_value_heads=8,
    **_ATTN,
)

# Tiny `config.json`s, one per architecture. Shapes respect each family's kernel constraints
# (e.g. DSA sparse MLA needs kv_lora_rank + qk_rope_head_dim == 576 and index_topk % 64 == 0;
# the DeepSeek V4 attention kernel needs >= 32 heads).
ARCH_CONFIGS: dict[str, dict] = {
    "llama": dict(
        model_type="llama",
        vocab_size=512,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        max_position_embeddings=1024,
        rope_theta=500000.0,
        **_ATTN,
    ),
    "qwen3": dict(
        model_type="qwen3",
        vocab_size=512,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        max_position_embeddings=1024,
        **_ATTN,
    ),
    "qwen3_moe": dict(
        model_type="qwen3_moe",
        vocab_size=512,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        max_position_embeddings=1024,
        norm_topk_prob=True,
        **_ATTN,
        **_MOE,
    ),
    "afmoe": dict(
        model_type="afmoe",
        vocab_size=512,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_dense_layers=1,
        max_position_embeddings=1024,
        global_attn_every_n_layers=2,
        sliding_window=64,
        num_shared_experts=1,
        **_ATTN,
        **_MOE,
    ),
    "minimax_m2": dict(
        model_type="minimax_m2",
        vocab_size=512,
        hidden_size=256,
        intermediate_size=128,
        num_hidden_layers=2,
        max_position_embeddings=1024,
        num_local_experts=8,
        num_experts_per_tok=2,
        rotary_dim=32,
        use_qk_norm=True,
        **_ATTN,
    ),
    "glm4_moe": dict(
        model_type="glm4_moe",
        vocab_size=512,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=3,
        first_k_dense_replace=1,
        max_position_embeddings=1024,
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        use_qk_norm=True,
        **_ATTN,
    ),
    "glm_moe_dsa": dict(
        model_type="glm_moe_dsa",
        vocab_size=512,
        pad_token_id=0,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=3,
        first_k_dense_replace=1,
        max_position_embeddings=1024,
        num_attention_heads=16,
        num_key_value_heads=16,
        q_lora_rank=128,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        qk_nope_head_dim=64,
        v_head_dim=64,
        index_n_heads=8,
        index_head_dim=128,
        index_topk=64,
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
    ),
    "gpt_oss": dict(
        model_type="gpt_oss",
        vocab_size=512,
        hidden_size=256,
        intermediate_size=128,
        num_hidden_layers=2,
        max_position_embeddings=1024,
        num_local_experts=8,
        num_experts_per_tok=2,
        sliding_window=64,
        **_ATTN,
    ),
    "laguna": dict(
        model_type="laguna",
        vocab_size=512,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_attention_heads_per_layer=[4] * 4,
        num_key_value_heads=2,
        head_dim=64,
        max_position_embeddings=1024,
        layer_types=["full_attention", "sliding_attention", "sliding_attention", "sliding_attention"],
        sliding_window=64,
        mlp_layer_types=["dense", "sparse", "sparse", "sparse"],
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=128,
        rope_parameters={
            "full_attention": {
                "rope_type": "yarn",
                "rope_theta": 500000.0,
                "factor": 4.0,
                "original_max_position_embeddings": 256,
                "partial_rotary_factor": 0.5,
            },
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
        },
    ),
    "nemotron_h": dict(
        model_type="nemotron_h",
        vocab_size=512,
        hidden_size=256,
        hybrid_override_pattern="ME*E",
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        max_position_embeddings=1024,
        intermediate_size=512,
        expand=2,
        mamba_num_heads=8,
        mamba_head_dim=64,
        ssm_state_size=64,
        n_groups=1,
        conv_kernel=4,
        chunk_size=64,
        n_routed_experts=8,
        n_shared_experts=1,
        moe_intermediate_size=128,
        moe_shared_expert_intermediate_size=128,
        num_experts_per_tok=2,
        norm_topk_prob=True,
    ),
    "qwen3_5": dict(model_type="qwen3_5_text", **_QWEN3_5_TEXT),
    "qwen3_5_moe": dict(model_type="qwen3_5_moe_text", shared_expert_intermediate_size=128, **_QWEN3_5_TEXT, **_MOE),
    # The VLM body on text-only data: the vision tower still runs on dummy pixels.
    "qwen3_5_vlm": dict(
        model_type="qwen3_5",
        text_config=dict(model_type="qwen3_5_text", **_QWEN3_5_TEXT),
        vision_config=dict(depth=1, hidden_size=64, intermediate_size=128, num_heads=4, out_hidden_size=256),
        image_token_id=500,
    ),
    "qwen3_8_flash_next": dict(
        model_type="qwen4_exp_text",
        vocab_size=512,
        eos_token_id=511,
        hidden_size=256,
        num_hidden_layers=4,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        indexer_n_heads=4,
        indexer_head_dim=64,
        indexer_budget=128,
        indexer_compress_ratio=4,
        hc_count=2,
        hc_lowrank=32,
        ple_layer_ids=[2],
        ple_embed_dim=64,
        heads_per_ngram=2,
        ngram_vocab_size_base=1024,
        make_ngram_vocab_size_divisible_by=8,
        split_ngram_parts=4,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=128,
    ),
    "deepseek_v4": dict(
        model_type="deepseek_v4",
        vocab_size=512,
        hidden_size=128,
        moe_intermediate_size=64,
        num_hidden_layers=5,
        num_attention_heads=32,
        num_key_value_heads=1,
        head_dim=32,
        q_lora_rank=64,
        partial_rotary_factor=0.5,
        rope_theta=10000.0,
        compress_rope_theta=160000.0,
        max_position_embeddings=1024,
        sliding_window=6,
        o_groups=2,
        o_lora_rank=16,
        layer_types=[
            "sliding_attention",
            "compressed_sparse_attention",
            "heavily_compressed_attention",
            "compressed_sparse_attention",
            "sliding_attention",
        ],
        compress_rates={"compressed_sparse_attention": 4, "heavily_compressed_attention": 8},
        index_n_heads=64,
        index_head_dim=128,
        index_topk=2,
        n_routed_experts=8,
        num_experts_per_tok=3,
        n_shared_experts=1,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.5,
        swiglu_limit=10.0,
        num_hash_layers=2,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        hc_eps=1e-6,
    ),
}

# (cp, cp_style); cp=1 ignores the style.
PARALLEL_MODES = {"cp1": (1, "ring"), "cp2-ring": (2, "ring"), "cp2-ulysses": (2, "ulysses")}


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


@pytest.fixture(autouse=True)
def restore_cp_attention():
    """Context parallelism rebinds attention methods on the classes; undo it between cases."""
    patched = [
        (FlashAttention, "_compute_attention"),
        (AfmoeFlashAttention, "_compute_attention"),
        (GptOssAttention, "compute_attention"),
    ]
    originals = [(cls, name, cls.__dict__[name]) for cls, name in patched]
    yield
    for cls, name, original in originals:
        setattr(cls, name, original)


def _seeded_generator(name: str) -> torch.Generator:
    return torch.Generator().manual_seed(zlib.crc32(name.encode()))


def _init_tensor(name: str, tensor: torch.Tensor, config_dict: dict) -> torch.Tensor:
    """A deterministic, well-conditioned value for every parameter and persistent buffer."""
    generator = _seeded_generator(name)
    if name.endswith("tid2eid"):
        # DeepSeek V4 hash routing: each token goes to `top_k` distinct experts.
        num_experts = config_dict["n_routed_experts"]
        rows = [torch.randperm(num_experts, generator=generator)[: tensor.shape[1]] for _ in range(tensor.shape[0])]
        return torch.stack(rows).to(tensor.dtype)
    if not tensor.is_floating_point():
        return tensor  # constants computed at construction (e.g. n-gram hash tables)
    if tensor.ndim >= 2:
        return (torch.randn(tensor.shape, generator=generator) * 0.02).to(tensor.dtype)
    if "norm" in name:
        return torch.ones(tensor.shape, dtype=tensor.dtype)
    return torch.zeros(tensor.shape, dtype=tensor.dtype)


def _write_checkpoint(arch: str, model_dir: Path) -> None:
    """Write `config.json` plus a HF-format safetensors checkpoint for the arch."""
    config_dict = ARCH_CONFIGS[arch]
    (model_dir / "config.json").write_text(json.dumps(config_dict))
    model_config = build_model_config(config_dict)
    # Built on CPU so buffers computed at construction hold real values.
    with default_dtype(torch.bfloat16):
        model = get_model_cls(model_config.model_type)(model_config)
    state_dict = {name: _init_tensor(name, tensor, config_dict) for name, tensor in model.state_dict().items()}
    save_state_dict(model.convert_to_hf(state_dict), model_dir)


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


def _train(arch: str, cp: int, cp_style: str) -> list[float]:
    """Run NUM_STEPS steps and return the global mean loss before each optimizer step."""
    model_dir = _model_dir(arch)
    config = ModelConfig(
        name=str(model_dir),
        cp=cp,
        cp_style=cp_style,
        ep=1,
        compile=None,
    )
    parallel_dims = get_parallel_dims(config, SEQ_LEN)
    model = setup_model(config, parallel_dims)
    if parallel_dims.cp_enabled:
        setup_context_parallel(model, config, parallel_dims)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)

    text_config = getattr(model.config, "text_config", model.config)
    input_ids, position_ids, labels, seq_lens = _fake_batch(text_config.vocab_size)
    if parallel_dims.cp_enabled:
        cp_mesh = parallel_dims.world_mesh["cp"]
        cp_rank, cp_group = cp_mesh.get_local_rank(), cp_mesh.get_group()
        input_ids, position_ids = setup_cp_params(
            input_ids, position_ids, cp_rank, cp, cp_group, seq_lens=seq_lens, cp_style=cp_style
        )
        labels = shard_for_cp(labels, cp_rank=cp_rank, cp_world_size=cp)

    dp_cp_group = parallel_dims.get_mesh("dp_cp").get_group()
    global_tokens = (labels != IGNORE_INDEX).sum()
    dist.all_reduce(global_tokens, group=dp_cp_group)
    # Same normalization as the SFT trainer: FSDP averages gradients over every dp/cp rank.
    loss_scale = parallel_dims.fsdp_gradient_divide_factor / global_tokens.item()

    losses = []
    for step in range(NUM_STEPS):
        out = forward(
            model,
            input_ids,
            position_ids,
            seq_lens=seq_lens,
            labels=labels,
            seq_lens_are_pre_shard=parallel_dims.cp_enabled,
        )
        loss_sum = out["loss"]
        (loss_sum * loss_scale).backward()

        grad_norm = clip_grad_norm_(None, model, max_norm=1.0)
        assert torch.isfinite(grad_norm), f"{arch} step {step}: non-finite grad norm {grad_norm.item()}"
        assert grad_norm > 0, f"{arch} step {step}: zero gradients"
        optimizer.step()
        optimizer.zero_grad()

        global_loss = loss_sum.detach().clone()
        dist.all_reduce(global_loss, group=dp_cp_group)
        losses.append(global_loss.item() / global_tokens.item())

    for param in model.parameters():
        local = param.to_local() if isinstance(param, DTensor) else param
        assert torch.isfinite(local).all(), f"{arch}: non-finite parameters after {NUM_STEPS} steps"

    del model, optimizer, out
    torch.cuda.empty_cache()
    return losses


@functools.cache
def _reference_losses(arch: str) -> tuple[float, ...]:
    return tuple(_train(arch, cp=1, cp_style="ring"))


def _supported_cp_styles(arch: str) -> frozenset[str]:
    config = build_model_config(ARCH_CONFIGS[arch])
    return get_model_cls(config.model_type).cp_support(config).styles


@pytest.mark.gpu
@pytest.mark.parametrize("mode", list(PARALLEL_MODES))
@pytest.mark.parametrize("arch", list(ARCH_CONFIGS))
def test_fsdp_training(arch: str, mode: str, distributed):
    cp, cp_style = PARALLEL_MODES[mode]
    if cp > 1 and cp_style not in _supported_cp_styles(arch):
        pytest.skip(f"{arch} does not support cp_style={cp_style!r}")

    reference = _reference_losses(arch)
    losses = reference if cp == 1 else tuple(_train(arch, cp=cp, cp_style=cp_style))

    assert all(torch.isfinite(torch.tensor(loss)) for loss in losses), f"{arch} {mode}: losses {losses}"
    assert losses[-1] < losses[0], f"{arch} {mode}: loss did not decrease over {NUM_STEPS} steps: {losses}"
    if cp > 1:
        # Sharding the sequence must not change the math: same losses as cp=1, up to bf16 noise.
        torch.testing.assert_close(
            torch.tensor(losses),
            torch.tensor(reference),
            rtol=2e-2,
            atol=0.0,
            msg=lambda m: f"{arch} {mode} losses {losses} diverge from cp1 {reference}: {m}",
        )


def test_tied_word_embeddings_are_rejected():
    config = LlamaConfig(hidden_size=64, num_attention_heads=4, vocab_size=128, tie_word_embeddings=True)
    with pytest.raises(ValueError, match="tie_word_embeddings"):
        with torch.device("meta"):
            LlamaForCausalLM(config)
