"""Run on four GPUs with ``uv run torchrun --standalone --nproc-per-node=4 -m pytest -s``."""

import copy
import datetime
import json
import os

import pytest
import torch
import torch.distributed as dist

from prime_rl.trainer.distributed.collectives import all_gather_cp
from prime_rl.trainer.models.layers.ulysses_attn import (
    _all_to_all_head_to_seq,
    _all_to_all_seq_to_head,
)
from prime_rl.utils.sequence import CPPartition


@pytest.fixture(autouse=True)
def restore_attention(monkeypatch):
    from prime_rl.trainer.models.layers.attn import FlashAttention

    monkeypatch.setattr(FlashAttention, "_compute_attention", FlashAttention._compute_attention)


@pytest.fixture(scope="module")
def cp_group():
    if int(os.environ.get("WORLD_SIZE", "1")) != 4:
        pytest.skip("requires four distributed GPU ranks")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch._dynamo.config.recompile_limit = 64
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=180))
    yield dist.group.WORLD
    dist.destroy_process_group()


@pytest.mark.gpu
@pytest.mark.parametrize("compiled", [False, True])
def test_cp_exchange_and_gather_gradients(cp_group, compiled):
    rank, degree = cp_group.rank(), cp_group.size()

    def exchange(x, shape):
        total = shape.shape[0]
        heads = _all_to_all_seq_to_head(x, degree, cp_group, total)
        restored = _all_to_all_head_to_seq(heads, degree, cp_group, total)
        return restored, all_gather_cp(x, 0, total, cp_group)

    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    call = torch.compile(exchange, fullgraph=True, dynamic=True) if compiled else exchange
    for total in [8, 9, 10, 11, 12, 13, 9, 8, 1, 2, 3, 0]:
        print(f"collectives rank={rank} compiled={compiled} total={total}", flush=True)
        partition = CPPartition(total, degree)
        full = torch.arange(total * 8 * 3, device="cuda", dtype=torch.float64).reshape(total, 8, 3)
        local = partition.shard(full, rank, 0).detach().requires_grad_()
        shape = torch.empty(total, 0, device="cuda")
        restored, gathered = call(local, shape)
        torch.testing.assert_close(restored, local, rtol=0, atol=0)
        torch.testing.assert_close(gathered, full, rtol=0, atol=0)
        upstream = torch.arange(total, device="cuda", dtype=torch.float64).reshape(total, 1, 1) + 1
        (restored.sum() * (rank + 1) + (gathered * upstream).sum() * (rank + 1)).backward()
        expected = partition.shard(upstream, rank, 0) * (degree * (degree + 1) // 2) + rank + 1
        torch.testing.assert_close(local.grad, expected.expand_as(local), rtol=0, atol=0)
    print(
        json.dumps(
            {
                "test": "collectives",
                "rank": rank,
                "compiled": compiled,
                "graphs": dict(torch._dynamo.utils.counters["stats"]),
                "breaks": dict(torch._dynamo.utils.counters["graph_break"]),
            }
        ),
        flush=True,
    )


@pytest.mark.gpu
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("style", ["ulysses", "ring"])
@pytest.mark.parametrize("flash_attn_version", [2, 3, 4])
def test_uneven_attention_matches_unsharded(cp_group, compiled, style, flash_attn_version):
    if flash_attn_version == 4:
        from flash_attn.cute import flash_attn_varlen_func as flash
    elif flash_attn_version == 3:
        from flash_attn_interface import flash_attn_varlen_func as flash
    else:
        from flash_attn import flash_attn_varlen_func as flash

    rank, degree = cp_group.rank(), cp_group.size()

    from prime_rl.trainer.models.layers.cp_attn import context_parallel_attention
    from prime_rl.utils.cp import CPContext

    context = CPContext(cp_group, rank, degree, style, True)

    def attention(q, k, v, cu, shape, maximum):
        return context_parallel_attention(flash, q, k, v, cu, maximum, shape.shape[0], context, flash_attn_version)

    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    call = torch.compile(attention, fullgraph=flash_attn_version != 4) if compiled else attention
    errors = torch.zeros(4, device="cuda")
    for total in [8, 9, 10, 11, 12, 13, 9, 8, 1, 2, 3]:
        print(f"attention rank={rank} compiled={compiled} total={total}", flush=True)
        torch.manual_seed(42 + total)
        partition = CPPartition(total, degree)
        full = [
            torch.randn(total, heads, 64, device="cuda", dtype=torch.bfloat16).requires_grad_() for heads in [8, 2, 2]
        ]
        boundaries = [0, 5, total] if total > 5 else [0, total]
        maximum = max(b - a for a, b in zip(boundaries, boundaries[1:]))
        cu = torch.tensor(boundaries, device="cuda", dtype=torch.int32)
        if flash_attn_version == 4:
            reference, _ = flash(
                *full, cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=maximum, max_seqlen_k=maximum, causal=True
            )
        else:
            reference = flash(*full, cu, cu, maximum, maximum, causal=True)
        reference.float().square().sum().backward()
        local = [partition.shard(x, rank, 0).detach().clone().requires_grad_() for x in full]
        actual = call(*local, cu, torch.empty(total, 0, device="cuda"), maximum)
        actual.float().square().sum().backward()
        torch.testing.assert_close(actual, partition.shard(reference, rank, 0), rtol=0.03, atol=0.03)
        pairs = [(actual.detach(), partition.shard(reference.detach(), rank, 0))]
        pairs.extend((x.grad, partition.shard(ref.grad, rank, 0)) for x, ref in zip(local, full))
        for index, (observed, expected) in enumerate(pairs):
            torch.testing.assert_close(observed, expected, rtol=0.05, atol=0.05)
            if observed.numel():
                errors[index] = torch.maximum(errors[index], (observed.float() - expected.float()).abs().max())
    dist.all_reduce(errors, op=dist.ReduceOp.MAX)
    print(
        json.dumps(
            {
                "test": "attention",
                "version": flash_attn_version,
                "rank": rank,
                "compiled": compiled,
                "max_abs_errors": errors.tolist(),
                "graphs": dict(torch._dynamo.utils.counters["stats"]),
                "breaks": dict(torch._dynamo.utils.counters["graph_break"]),
            }
        ),
        flush=True,
    )


@pytest.mark.gpu
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("objective", ["rl", "sft"])
def test_qwen3_cp_model_and_checkpoint_gradients(cp_group, compiled, objective):
    from transformers import Qwen3Config

    from prime_rl.configs.trainer import ActivationCheckpointConfig, CompileConfig, ModelConfig
    from prime_rl.trainer.model import apply_ac, apply_compile
    from prime_rl.trainer.models.layers.attn import FlashAttention
    from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
    from prime_rl.trainer.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
    from prime_rl.trainer.parallel_dims import ParallelDims
    from prime_rl.utils.cp import setup_context_parallel

    rank, degree = cp_group.rank(), cp_group.size()
    torch.manual_seed(7)
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=128,
        attn_implementation="flash_attention_3",
        use_cache=False,
    )
    model = Qwen3ForCausalLM(config).cuda().to(torch.bfloat16)
    inject_prime_lm_head(model, chunk_size=8)
    reference = copy.deepcopy(model)
    original_attention = FlashAttention._compute_attention
    for module in reference.modules():
        if isinstance(module, FlashAttention):
            module._compute_attention = original_attention.__get__(module)
    dims = ParallelDims(dp_replicate=1, dp_shard=1, cp=degree, pp=1, ep=1, world_size=degree)
    setup_context_parallel(
        model,
        ModelConfig(cp=degree, cp_style="ulysses", cp_unpadded=True, impl="custom", attn="flash_attention_3"),
        dims,
    )
    apply_ac(model, ActivationCheckpointConfig(mode="full"))
    if compiled:
        apply_compile(model, CompileConfig(fullgraph=True))
    for total in [9, 10, 1, 3, 12]:
        print(f"model rank={rank} compiled={compiled} total={total}", flush=True)
        model.zero_grad(set_to_none=True)
        reference.zero_grad(set_to_none=True)
        partition = CPPartition(total, degree)
        ids = (torch.arange(total, device="cuda").reshape(1, total) + 3) % 64
        lengths = [5, total - 5] if total > 5 else [total]
        seq_lens = torch.tensor(lengths, device="cuda")
        positions = torch.cat([torch.arange(n, device="cuda") for n in lengths]).unsqueeze(0)
        labels = ids.roll(-1, 1)
        temperature = torch.ones_like(ids, dtype=torch.float32)
        ref = reference(ids, position_ids=positions, labels=labels, temperature=temperature, seq_lens=seq_lens)
        if objective == "sft":
            mask = positions.remainder(3) != 0
            reference_loss = -ref["logprobs"][mask].sum()
            labels = labels.masked_fill(~mask, -100)
            reference_loss.backward()
        else:
            ref["logprobs"].sum().backward()
        actual = model(
            partition.shard(ids, rank),
            position_ids=partition.shard(positions, rank),
            labels=partition.shard(labels, rank),
            temperature=partition.shard(temperature, rank) if objective == "rl" else None,
            seq_lens=seq_lens,
            seq_lens_are_pre_shard=True,
            cp_total_tokens=total,
        )
        if objective == "sft":
            global_loss = actual["loss"].detach().clone()
            dist.all_reduce(global_loss)
            torch.testing.assert_close(global_loss, reference_loss, rtol=0.03, atol=0.03)
            actual["loss"].backward()
        else:
            gathered = all_gather_cp(actual["logprobs"], 1, total, cp_group)
            torch.testing.assert_close(gathered, ref["logprobs"], rtol=0.03, atol=0.03)
            (gathered.sum() / degree).backward()
        max_error = 0.0
        for (name, param), (ref_name, ref_param) in zip(model.named_parameters(), reference.named_parameters()):
            assert name.replace("_checkpoint_wrapped_module.", "") == ref_name
            assert param.grad is not None, name
            dist.all_reduce(param.grad, group=cp_group)
            torch.testing.assert_close(param.grad, ref_param.grad, rtol=0.05, atol=0.05, msg=name)
            max_error = max(max_error, float((param.grad.float() - ref_param.grad.float()).abs().max()))
        print(
            json.dumps(
                {"test": "model", "rank": rank, "compiled": compiled, "total": total, "max_param_grad_error": max_error}
            ),
            flush=True,
        )
    FlashAttention._compute_attention = original_attention


@pytest.mark.gpu
@pytest.mark.parametrize(
    "moe,compiled,ep",
    [(False, False, 1), (False, True, 1), (True, False, 1), (True, True, 1), (True, False, 4), (True, True, 4)],
)
def test_fsdp_inactive_lane_preserves_global_gradient(cp_group, moe, compiled, ep):
    from transformers import Qwen3Config

    from prime_rl.configs.trainer import ActivationCheckpointConfig, CompileConfig, ModelConfig
    from prime_rl.trainer.model import apply_ac, apply_compile, apply_fp32_moe_router, get_global_moe_stats, setup_fsdp
    from prime_rl.trainer.models.layers.attn import FlashAttention
    from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
    from prime_rl.trainer.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
    from prime_rl.trainer.moe_runtime import configure_moe_runtime
    from prime_rl.trainer.parallel_dims import ParallelDims
    from prime_rl.utils.cp import setup_context_parallel

    torch.manual_seed(17)
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=128,
        attn_implementation="flash_attention_3",
        use_cache=False,
    )
    if moe:
        from prime_rl.trainer.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
        from prime_rl.trainer.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM

        config = Qwen3MoeConfig(
            **(config.to_dict() | {"num_experts": 4, "num_experts_per_tok": 2, "moe_intermediate_size": 256})
        )
        model = Qwen3MoeForCausalLM(config).cuda()
        for layer in model.model.layers:
            layer.mlp.experts.init_weights(config.initializer_range)
    else:
        model = Qwen3ForCausalLM(config).cuda()
    inject_prime_lm_head(model, chunk_size=8)
    reference = copy.deepcopy(model).to(torch.bfloat16)
    if moe:
        apply_fp32_moe_router(model)
        apply_fp32_moe_router(reference)
    original_attention = FlashAttention._compute_attention
    for module in reference.modules():
        if isinstance(module, FlashAttention):
            module._compute_attention = original_attention.__get__(module)
    degree = ep
    dims = ParallelDims(dp_replicate=1, dp_shard=2, cp=2, pp=1, ep=degree, world_size=4)
    model_config = ModelConfig(
        cp=2,
        cp_style="ulysses",
        cp_unpadded=True,
        ep=degree,
        impl="custom",
        attn="flash_attention_3",
        optim_cpu_offload=False,
    )
    configure_moe_runtime(model, model_config, dims)
    setup_context_parallel(model, model_config, dims)
    apply_ac(model, ActivationCheckpointConfig(mode="full"))
    if compiled:
        apply_compile(model, CompileConfig(fullgraph=not moe))
    setup_fsdp(model, model_config, dims)
    group = dims.get_mesh("cp").get_group()
    cp_rank = group.rank()
    lane = dist.get_rank() // 2

    for lengths in json.loads(os.environ.get("CP_TEST_LENGTHS", "[[9,5],[9,0],[0,7],[3,1]]")):
        print(f"fsdp rank={dist.get_rank()} lengths={lengths} moe={moe} compiled={compiled}", flush=True)
        model.zero_grad(set_to_none=True)
        reference.zero_grad(set_to_none=True)
        denominator = sum(lengths)
        for total in lengths:
            if total == 0:
                continue
            ids = (torch.arange(total, device="cuda").reshape(1, total) + 3) % 64
            ref = reference(
                ids,
                labels=ids.roll(-1, 1),
                temperature=torch.ones_like(ids, dtype=torch.float32),
                seq_lens=torch.tensor([total], device="cuda"),
            )
            (ref["logprobs"].sum() / denominator).backward()

        total = lengths[lane]
        partition = CPPartition(total, 2)
        ids = (torch.arange(total, device="cuda").reshape(1, total) + 3) % 64
        labels = ids.roll(-1, 1)
        positions = torch.arange(total, device="cuda").reshape(1, total)
        seq_lens = torch.tensor([total] if total else [], device="cuda", dtype=torch.int64)
        actual = model(
            partition.shard(ids, cp_rank),
            position_ids=partition.shard(positions, cp_rank),
            labels=partition.shard(labels, cp_rank),
            temperature=torch.ones(1, partition.lengths[cp_rank], device="cuda"),
            seq_lens=seq_lens,
            seq_lens_are_pre_shard=True,
        )
        assert torch.isfinite(actual["logprobs"]).all()
        gathered = all_gather_cp(actual["logprobs"], 1, total, group)
        (gathered.sum() / (2 * denominator)).backward()
        if moe:
            metrics = get_global_moe_stats(
                model,
                dims.get_mesh("ep").get_group() if ep > 1 else None,
                dims.get_mesh("dp_cp").get_group(),
            )
            assert metrics
            assert all(torch.isfinite(value) for value in metrics.values())
        max_error = 0.0
        for (name, param), (_, ref_param) in zip(model.named_parameters(), reference.named_parameters()):
            assert param.grad is not None, name
            gradient = param.grad.full_tensor() * dims.fsdp_gradient_divide_factor
            torch.testing.assert_close(
                gradient,
                ref_param.grad.float(),
                rtol=0.05,
                atol=0.025 if moe else 0.015,
                msg=lambda msg: f"{name}: {msg}",
            )
            max_error = max(max_error, float((gradient - ref_param.grad.float()).abs().max()))
        print(
            json.dumps(
                {
                    "test": "fsdp-inactive",
                    "moe": moe,
                    "compiled": compiled,
                    "rank": dist.get_rank(),
                    "lengths": lengths,
                    "max_param_grad_error": max_error,
                }
            ),
            flush=True,
        )
    FlashAttention._compute_attention = original_attention


@pytest.mark.gpu
@pytest.mark.parametrize("kind", ["mamba", "delta", "delta_gather"])
def test_uneven_recurrence_matches_unsharded(cp_group, kind):
    from prime_rl.utils.cp import CPContext

    if kind == "mamba":
        from prime_rl.trainer.models.nemotron_h import NemotronHConfig
        from prime_rl.trainer.models.nemotron_h.mamba import NemotronHMamba2

        config = NemotronHConfig(
            hidden_size=256,
            hybrid_override_pattern="M",
            mamba_num_heads=8,
            mamba_head_dim=64,
            n_groups=4,
            ssm_state_size=64,
            conv_kernel=4,
            chunk_size=64,
        )
        model = NemotronHMamba2(config)
    else:
        from prime_rl.trainer.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        from prime_rl.trainer.models.qwen3_5.gated_delta_net import Qwen3_5GatedDeltaNet

        config = Qwen3_5TextConfig(
            hidden_size=256,
            linear_key_head_dim=32,
            linear_value_head_dim=32,
            linear_num_key_heads=1 if kind == "delta_gather" else 4,
            linear_num_value_heads=2 if kind == "delta_gather" else 8,
            linear_conv_kernel_dim=4,
        )
        model = Qwen3_5GatedDeltaNet(config)
    model = model.cuda().to(torch.bfloat16)
    for parameter in model.parameters():
        dist.broadcast(parameter.data, 0)
    reference = copy.deepcopy(model)
    rank, degree = cp_group.rank(), cp_group.size()
    model.cp_context = CPContext(cp_group, rank, degree, "ulysses", True)
    for total in [9, 10, 1, 3, 0, 12]:
        model.zero_grad(set_to_none=True)
        reference.zero_grad(set_to_none=True)
        torch.manual_seed(42 + total)
        partition = CPPartition(total, degree)
        full = torch.randn(1, total, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        local = partition.shard(full, rank).detach().clone().requires_grad_()
        boundaries = [0, 5, total] if total > 5 else ([0, total] if total else [0])
        cu = torch.tensor(boundaries, device="cuda", dtype=torch.int32)
        expected = reference(full, cu)
        actual = model(local, cu, cp_total_tokens=total)
        expected.float().square().sum().backward()
        actual.float().square().sum().backward()
        torch.testing.assert_close(actual, partition.shard(expected, rank), rtol=0.03, atol=0.03)
        torch.testing.assert_close(local.grad, partition.shard(full.grad, rank), rtol=0.05, atol=0.05)
        for (name, parameter), (_, ref_parameter) in zip(model.named_parameters(), reference.named_parameters()):
            assert parameter.grad is not None, name
            dist.all_reduce(parameter.grad)
            torch.testing.assert_close(
                parameter.grad, ref_parameter.grad, rtol=0.05, atol=0.25, msg=lambda msg: f"{name}: {msg}"
            )
        print(json.dumps({"test": "recurrence", "kind": kind, "rank": rank, "total": total}), flush=True)


@pytest.mark.gpu
def test_vlm_images_and_inactive_lanes_with_ep(cp_group):
    from prime_rl.configs.shared import VLMConfig
    from prime_rl.configs.trainer import ActivationCheckpointConfig, CompileConfig, ModelConfig
    from prime_rl.trainer.model import apply_ac, apply_compile, apply_fp32_moe_router, setup_fsdp
    from prime_rl.trainer.models import AutoModelForCausalLMPrimeRL
    from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
    from prime_rl.trainer.moe_runtime import configure_moe_runtime
    from prime_rl.trainer.parallel_dims import ParallelDims
    from prime_rl.utils.cp import gather_for_cp, setup_context_parallel
    from tests.unit.train.models.qwen3_5.test_vlm import get_image_inputs, get_vlm_config

    torch.manual_seed(7)
    config = get_vlm_config()
    model = AutoModelForCausalLMPrimeRL.from_config(config, attn_implementation="flash_attention_3").cuda()
    inject_prime_lm_head(model, chunk_size=8)
    reference = copy.deepcopy(model).to(torch.bfloat16)
    apply_fp32_moe_router(model)
    apply_fp32_moe_router(reference)
    dims = ParallelDims(dp_replicate=1, dp_shard=2, cp=2, pp=1, ep=4, world_size=4)
    runtime = ModelConfig(
        impl="custom",
        attn="flash_attention_3",
        cp=2,
        cp_style="ulysses",
        cp_unpadded=True,
        inactive_micro_batches=True,
        ep=4,
        optim_cpu_offload=False,
        vlm=VLMConfig(
            freeze_vision_encoder=False, vision_encoder_attr="model.visual", language_model_attr="model.language_model"
        ),
    )
    configure_moe_runtime(model, runtime, dims)
    setup_context_parallel(model, runtime, dims)
    apply_ac(model, ActivationCheckpointConfig(mode="full"))
    apply_compile(model, CompileConfig(fullgraph=False))
    setup_fsdp(model, runtime, dims)
    group = dims.get_mesh("cp").get_group()
    rank = group.rank()
    lane = dist.get_rank() // 2

    def inputs(total):
        ids = (torch.arange(total, device="cuda").reshape(1, total) + 3) % 64
        extras = {}
        if total:
            ids[0, 1] = config.image_token_id
            torch.manual_seed(17 + total)
            pixels, grid, _ = get_image_inputs(config)
            extras = dict(
                pixel_values=pixels, image_grid_thw=grid, mm_token_type_ids=(ids == config.image_token_id).long()
            )
        return ids, extras

    for totals in [[9, 0], [0, 7], [3, 5]]:
        model.zero_grad(set_to_none=True)
        reference.zero_grad(set_to_none=True)
        for total in totals:
            if not total:
                continue
            ids, extras = inputs(total)
            out = reference(
                ids,
                labels=ids.roll(-1, 1),
                temperature=torch.ones_like(ids, dtype=torch.float32),
                seq_lens=torch.tensor([total], device="cuda"),
                **extras,
            )
            (out["logprobs"].sum() / sum(totals)).backward()
        total = totals[lane]
        ids, extras = inputs(total)
        partition = CPPartition(total, 2)
        out = model(
            ids,
            labels=partition.shard(ids.roll(-1, 1), rank),
            temperature=torch.ones(1, partition.lengths[rank], device="cuda"),
            seq_lens=torch.tensor([total] if total else [], device="cuda", dtype=torch.int64),
            seq_lens_are_pre_shard=True,
            cp_total_tokens=total,
            **extras,
        )
        gathered = gather_for_cp(out["logprobs"], group, total)
        (gathered.sum() / (2 * sum(totals))).backward()
        maximum = 0.0
        for (name, p), rp in zip(model.named_parameters(), reference.parameters()):
            assert p.grad is not None, name
            g = p.grad.full_tensor() * dims.fsdp_gradient_divide_factor
            torch.testing.assert_close(g, rp.grad.float(), atol=0.02, rtol=0.06, msg=lambda msg: f"{name}: {msg}")
            maximum = max(maximum, float((g - rp.grad.float()).abs().max()))
        print(
            json.dumps({"test": "vlm-inactive", "totals": totals, "rank": dist.get_rank(), "max_grad_error": maximum}),
            flush=True,
        )
