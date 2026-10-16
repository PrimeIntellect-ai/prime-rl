import copy

import pytest
import torch
from torch import nn

from prime_rl.configs.trainer import FP8Config
from prime_rl.trainer.distributed.token_dispatcher import LocalTokenDispatcher
from prime_rl.trainer.model import forward
from prime_rl.trainer.models.fp8_rounding import FROZEN_STEP, seed_for_step, set_fp8_rounding_step, setup_fp8_rounding
from prime_rl.trainer.models.kernels.fp8_utils import per_block_cast_to_fp8_triton
from prime_rl.trainer.models.layers.expert_compute import BF16ExpertCompute, DeepGemmFP8ExpertCompute
from prime_rl.trainer.models.layers.fp8_grouped_gemm import grouped_fp8_gemm
from prime_rl.trainer.models.layers.fp8_linear import (
    Float8BlockwiseLinear,
    _fp8_blockwise_mm,
    bf16_compute,
    bf16_compute_enabled,
    replace_linear_with_fp8_blockwise_linear,
)
from prime_rl.trainer.models.layers.lm_head import IGNORE_INDEX, inject_prime_lm_head
from prime_rl.trainer.models.layers.moe import GroupedExperts, MoE
from prime_rl.trainer.models.qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM
from prime_rl.trainer.sft.master_cast_probe import (
    FP8_COVERED_PARAM,
    FP8FlipRate,
    MasterCastProbe,
    ue8m0_cast,
    ue8m0_rtn,
)
from prime_rl.utils.utils import default_dtype

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]

SEQ_LEN = 256
BASE_SEED = 7
KINDS = ("bf16", "rtn", "sr", "fixed")


@pytest.fixture(autouse=True)
def _ue8m0_weight_casts(monkeypatch):
    monkeypatch.setenv("PRIME_RL_FP8_WEIGHT_UE8M0", "1")


def make_model(kind: str) -> nn.Module:
    config = Qwen3MoeConfig(
        head_dim=128,
        hidden_size=256,
        num_attention_heads=2,
        num_key_value_heads=1,
        moe_intermediate_size=128,
        num_experts=4,
        num_experts_per_tok=2,
        num_hidden_layers=2,
        vocab_size=512,
        max_position_embeddings=1024,
        use_qk_norm=True,
        norm_topk_prob=True,
    )
    config._attn_implementation = "flash_attention_3"
    torch.manual_seed(0)
    with torch.device("cuda"), default_dtype(torch.bfloat16):
        model = Qwen3MoeForCausalLM._from_config(config)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "norm" in name:
                param.fill_(1.0)
            else:
                nn.init.normal_(param, std=0.05)
    inject_prime_lm_head(model, chunk_size=SEQ_LEN)
    if kind != "bf16":
        stochastic = kind in ("sr", "fixed")
        replace_linear_with_fp8_blockwise_linear(model, FP8Config().ignore_patterns, stochastic_rounding=stochastic)
        for moe in (module for module in model.modules() if isinstance(module, MoE)):
            compute = DeepGemmFP8ExpertCompute(stochastic_rounding=stochastic)
            moe.experts.set_compute(compute)
            moe.token_dispatcher = LocalTokenDispatcher(
                num_experts=moe.experts.num_experts,
                top_k=moe.router.top_k,
                token_group_alignment=compute.token_group_alignment,
            )
    setup_fp8_rounding(model, BASE_SEED, frozen=kind == "fixed")
    return model


def make_batches(count: int, seed: int) -> list[dict]:
    generator = torch.Generator().manual_seed(seed)
    seq_lens = torch.tensor([100, SEQ_LEN - 100])
    batches = []
    for _ in range(count):
        batches.append(
            {
                "input_ids": torch.randint(0, 512, (1, SEQ_LEN), generator=generator),
                "position_ids": torch.cat([torch.arange(n) for n in seq_lens.tolist()]).unsqueeze(0),
                "target_ids": torch.randint(0, 512, (1, SEQ_LEN), generator=generator),
                "loss_mask": torch.rand(1, SEQ_LEN, generator=generator) < 0.7,
                "seq_lens": seq_lens,
            }
        )
    return [{key: value.cuda() for key, value in batch.items()} for batch in batches]


def compute_loss(model: nn.Module, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    labels = batch["target_ids"].masked_fill(~batch["loss_mask"], IGNORE_INDEX)
    out = forward(model, batch["input_ids"], batch["position_ids"], seq_lens=batch["seq_lens"], labels=labels)
    return out["loss"], batch["loss_mask"].sum()


def covered(model: nn.Module) -> list[nn.Parameter]:
    return [param for name, param in model.named_parameters() if FP8_COVERED_PARAM.search(name)]


def loss_and_grads(model: nn.Module, batches: list[dict]) -> tuple[float, list[torch.Tensor]]:
    params = covered(model)
    loss_sum, tokens = 0.0, 0
    # Accumulate in the parameter dtype, as autograd accumulates into .grad.
    grad_sums = [torch.zeros_like(param) for param in params]
    with bf16_compute():
        for batch in batches:
            loss, count = compute_loss(model, batch)
            for grad_sum, grad in zip(grad_sums, torch.autograd.grad(loss, params)):
                grad_sum += grad
            loss_sum += loss.item()
            tokens += count.item()
    return loss_sum / tokens, [grad_sum.double() / tokens for grad_sum in grad_sums]


def own_salts(model: nn.Module) -> list[torch.Tensor | None]:
    salts = {}
    for module in model.modules():
        if isinstance(module, Float8BlockwiseLinear) and module.weight is not None:
            salts[id(module.weight)] = module.rounding_salt
        elif isinstance(module, GroupedExperts):
            for name, param in module.named_parameters(recurse=False):
                salts[id(param)] = getattr(module, "rounding_salts", {}).get(name)
    return [salts[id(param)] for param in covered(model)]


def dot(a: list[torch.Tensor], b: list[torch.Tensor]) -> float:
    return sum((x * y).sum().item() for x, y in zip(a, b))


def reference_cast_metrics(model: nn.Module, batches: list[dict], start: list[torch.Tensor], casts: list) -> dict:
    """Probe metrics for one cast, from a deep copy whose covered weights are replaced by ``casts[i](weight)``."""
    loss_master, grads_master = loss_and_grads(model, batches)
    cast_model = copy.deepcopy(model)
    with torch.no_grad():
        for cast, param in zip(casts, covered(cast_model)):
            param.copy_(cast(param))
    loss_cast, grads_cast = loss_and_grads(cast_model, batches)
    masters = [master.detach() for master in covered(model)]
    displacement = [master.double() - s.double() for master, s in zip(masters, start)]
    cast_displacement = [cast(m).double() - cast(s).double() for cast, m, s in zip(casts, masters, start)]
    d_norm = dot(displacement, displacement) ** 0.5
    grad_change = [cast - master for cast, master in zip(grads_cast, grads_master)]
    return {
        "loss_master": loss_master,
        "loss_cast": loss_cast,
        "gap": loss_cast - loss_master,
        "displacement_norm": d_norm,
        "grad_norm_master": dot(grads_master, grads_master) ** 0.5,
        "grad_norm_cast": dot(grads_cast, grads_cast) ** 0.5,
        "descent_master": -dot(grads_master, displacement) / d_norm,
        "descent_cast": -dot(grads_cast, displacement) / d_norm,
        "grad_change_cos_displacement": dot(grad_change, displacement)
        / (dot(grad_change, grad_change) ** 0.5 * d_norm),
        "forward_visible_fraction": dot(cast_displacement, displacement) / d_norm**2,
        "cast_changed_fraction": sum((cast(m) != m).sum().item() for cast, m in zip(casts, masters))
        / sum(m.numel() for m in masters),
    }


def train_step(model: nn.Module, optimizer: torch.optim.Optimizer, batch: dict) -> None:
    loss, count = compute_loss(model, batch)
    (loss / count).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def test_ue8m0_rtn_matches_the_forward_cast_and_is_idempotent():
    torch.manual_seed(1)
    weight = torch.randn(256, 384, device="cuda", dtype=torch.bfloat16)
    fp8, scale = per_block_cast_to_fp8_triton(weight, True)
    expected = (fp8.float() * scale.repeat_interleave(128, 0).repeat_interleave(128, 1)).bfloat16()
    assert torch.equal(ue8m0_rtn(weight), expected)
    assert torch.equal(ue8m0_rtn(expected), expected)
    assert torch.equal(ue8m0_rtn(weight.float()), expected)
    assert torch.all(torch.log2(scale) == torch.log2(scale).round())
    experts = torch.randn(3, 256, 128, device="cuda", dtype=torch.bfloat16)
    assert torch.equal(ue8m0_rtn(experts)[1], ue8m0_rtn(experts[1]))


def test_bf16_compute_bypasses_fp8_and_restores_it_on_exit():
    torch.manual_seed(2)
    x = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16)

    dense = nn.Sequential(nn.Linear(128, 256, bias=False, device="cuda", dtype=torch.bfloat16))
    replace_linear_with_fp8_blockwise_linear(dense, [])
    assert isinstance(dense[0], Float8BlockwiseLinear)
    bf16_out = nn.functional.linear(x, dense[0].weight)
    fp8_out = dense(x)
    assert not torch.equal(fp8_out, bf16_out)
    with bf16_compute():
        assert torch.equal(dense(x), bf16_out)
    assert torch.equal(dense(x), fp8_out)

    experts = GroupedExperts(128, 256, 3, compute=DeepGemmFP8ExpertCompute()).to("cuda", torch.bfloat16)
    for param in experts.parameters():
        nn.init.normal_(param, std=0.05)
    reference = copy.deepcopy(experts)
    reference.set_compute(BF16ExpertCompute())
    counts = torch.tensor([128, 0, 128], device="cuda")
    fp8_out = experts(x, counts)
    assert not torch.equal(fp8_out, reference(x, counts))
    with bf16_compute():
        assert torch.equal(experts(x, counts), reference(x, counts))
    assert torch.equal(experts(x, counts), fp8_out)

    with pytest.raises(RuntimeError), bf16_compute():
        raise RuntimeError
    assert not bf16_compute_enabled()


@pytest.mark.parametrize("kind", KINDS)
def test_probe_matches_reference_and_leaves_training_state_unchanged(kind):
    model = make_model(kind)
    batches = make_batches(2, seed=3)
    start = [param.detach().clone() for param in covered(model)]
    probe = MasterCastProbe(model, batches, lambda batch: compute_loss(model, batch), None, 1)
    assert probe.kind == kind
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    for step, batch in enumerate(make_batches(2, seed=4), start=1):
        set_fp8_rounding_step(model, step)
        train_step(model, optimizer, batch)

    params = {name: param.detach().clone() for name, param in model.named_parameters()}
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    probe_input = make_batches(1, seed=5)[0]
    with torch.no_grad():
        fp8_out = compute_loss(model, probe_input)[0]

    metrics = probe.measure(4)

    for name, param in model.named_parameters():
        assert torch.equal(param, params[name]), name
        assert param.grad is None, name
    for key, value in optimizer.state_dict()["state"].items():
        for field, tensor in value.items():
            assert torch.equal(tensor, optimizer_state["state"][key][field]), (key, field)
    assert all(module.tokens_per_expert.count_nonzero() == 0 for module in model.modules() if isinstance(module, MoE))
    with torch.no_grad():
        assert torch.equal(compute_loss(model, probe_input)[0], fp8_out)

    n_covered = len(start)
    expected = reference_cast_metrics(model, batches, start, [ue8m0_rtn] * n_covered)
    assert expected["displacement_norm"] > 0 and expected["gap"] != 0
    for key, value in expected.items():
        assert metrics[f"master_cast_probe/{key}"] == pytest.approx(value, rel=1e-3, abs=1e-6), key

    own = {key.removeprefix("master_cast_probe/own/"): value for key, value in metrics.items() if "/own/" in key}
    if kind == "rtn":
        for key, value in own.items():
            assert value == metrics[f"master_cast_probe/{key}"], key
    elif kind == "bf16":
        assert own["gap"] == 0 and own["forward_visible_fraction"] == 1
        assert own["descent_cast"] == metrics["master_cast_probe/descent_master"]
    else:
        seed = torch.tensor(seed_for_step(BASE_SEED, 5 if kind == "sr" else FROZEN_STEP), device="cuda")
        casts = [lambda weight, salt=salt: ue8m0_cast(weight, seed, salt) for salt in own_salts(model)]
        expected_own = reference_cast_metrics(model, batches, start, casts)
        if kind == "sr":
            assert own.pop("forward_visible_fraction") == 1
            own["forward_visible_fraction"] = own.pop("forward_visible_fraction_realized")
        assert expected_own["gap"] != expected["gap"]
        for key in (
            "loss_cast",
            "gap",
            "grad_norm_cast",
            "descent_cast",
            "grad_change_cos_displacement",
            "forward_visible_fraction",
            "cast_changed_fraction",
        ):
            assert own[key] == pytest.approx(expected_own[key], rel=1e-3, abs=1e-6), key


def test_own_cast_reproduces_the_stochastic_forward():
    torch.manual_seed(8)
    seed = torch.tensor(seed_for_step(BASE_SEED, 3), device="cuda")
    salt = torch.tensor(1234, device="cuda")
    x = torch.randn(200, 256, device="cuda", dtype=torch.bfloat16)
    linear = nn.Sequential(nn.Linear(256, 512, bias=False, device="cuda", dtype=torch.bfloat16))
    replace_linear_with_fp8_blockwise_linear(linear, [], stochastic_rounding=True)
    setup_fp8_rounding(linear, BASE_SEED)
    set_fp8_rounding_step(linear, 3)
    weight = linear[0].weight.detach()
    stochastic = linear(x)
    assert not torch.equal(stochastic, _fp8_blockwise_mm(x, weight, 128))
    own = ue8m0_cast(weight, seed, linear[0].rounding_salt)
    assert torch.equal(stochastic, _fp8_blockwise_mm(x, own, 128))

    x = torch.randn(384, 128, device="cuda", dtype=torch.bfloat16)
    stored = torch.randn(3, 256, 128, device="cuda", dtype=torch.bfloat16)
    offsets = torch.tensor([128, 128, 384], device="cuda", dtype=torch.int32)
    offset = 4 * 256 * 128
    stochastic = grouped_fp8_gemm(x, stored.transpose(1, 2), offsets, seed, salt, offset)
    own = ue8m0_cast(stored, seed, salt, offset)
    assert torch.equal(stochastic, grouped_fp8_gemm(x, own.transpose(1, 2), offsets))


@pytest.mark.parametrize("kind", ("rtn", "sr", "fixed"))
def test_flip_rate_tracks_forward_and_master_changes(kind):
    model = make_model(kind)
    probe = MasterCastProbe(model, make_batches(2, seed=3), lambda batch: compute_loss(model, batch), None, 1)
    flip = FP8FlipRate(probe)
    set_fp8_rounding_step(model, 1)
    first = flip.step(1)
    assert first["fp8_flip/forward_changed_vs_first"] == 0 and first["fp8_flip/master_changed_vs_start"] == 0
    assert "fp8_flip/forward_changed_vs_prev" not in first
    set_fp8_rounding_step(model, 2)
    unchanged = flip.step(2)
    assert unchanged["fp8_flip/master_changed_vs_prev"] == 0
    if kind == "sr":
        # Each element keeps its value with probability 1 - 2p(1 - p), so about a third of them re-roll.
        assert 0.2 < unchanged["fp8_flip/forward_changed_vs_prev"] < 0.45
    else:
        assert unchanged["fp8_flip/forward_changed_vs_prev"] == 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train_step(model, optimizer, make_batches(1, seed=4)[0])
    set_fp8_rounding_step(model, 3)
    moved = flip.step(3)
    assert 0 < moved["fp8_flip/master_changed_vs_prev"] == moved["fp8_flip/master_changed_vs_start"]
    if kind != "sr":
        assert 0 < moved["fp8_flip/forward_changed_vs_prev"] < moved["fp8_flip/master_changed_vs_prev"]


def test_fixed_and_stochastic_own_casts_agree_at_step_zero():
    metrics = {}
    for kind in ("sr", "fixed"):
        model = make_model(kind)
        probe = MasterCastProbe(model, make_batches(2, seed=3), lambda batch: compute_loss(model, batch), None, 1)
        metrics[kind] = probe.measure(0)
    for key in ("own/loss_cast", "own/cast_changed_fraction", "own/grad_norm_cast"):
        assert metrics["sr"][f"master_cast_probe/{key}"] == pytest.approx(
            metrics["fixed"][f"master_cast_probe/{key}"], rel=1e-4
        ), key


def test_on_grid_start_has_zero_gap():
    model = make_model("rtn")
    with torch.no_grad():
        for param in covered(model):
            param.copy_(ue8m0_rtn(param))
    probe = MasterCastProbe(model, make_batches(2, seed=6), lambda batch: compute_loss(model, batch), None, 1)
    metrics = probe.measure(4)
    assert metrics["master_cast_probe/gap"] == 0
    assert metrics["master_cast_probe/displacement_norm"] == 0
    assert metrics["master_cast_probe/cast_changed_fraction"] == 0
    assert "master_cast_probe/descent_master" not in metrics
