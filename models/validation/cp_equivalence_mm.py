"""cp=1 vs cp=8 equivalence for the Nemotron-VL image path (GPU, torchrun).

The text-only equivalence gate exercised FSDP+CP on the real 120B model; this test covers
the piece that has never run on GPU: the scatter-then-shard multimodal CP path (ulysses).
A tiny random NemotronVL model (mamba + attention + moe layers, real kernels) processes a
packed 2-document batch where each document contains one image and one image-token run
deliberately CROSSES a CP shard boundary (tokens 60..66 vs 64-token shards at cp=8).

Run 1 (torchrun nproc=1, --cp-size 1): no CP; saves init weights, full post-scatter embeds
(captured via a pre-hook on language_model), summed loss, and gradients.
Run 2 (torchrun nproc=8, --cp-size 8): loads the same weights, runs the trainer's deferral
path (full input_ids/pixel_values into the model, target/mask sharded outside), all-gathers
the sharded embeds, sum-all-reduces loss and grads (what FSDP's hsdp mesh does up to the
normalization both runs share), and compares against run 1:

  - post-scatter embeds: expected bitwise-equal (same weights, same deterministic ops)
  - loss + grads (mlp1 / vision / LM): bf16-noise tolerances

Launch via models/validation/cp_equivalence_mm.sbatch.
"""

import argparse
import os

import torch
import torch.distributed as dist

from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from prime_rl.trainer.models.nemotron_vl import NemotronVLConfig, NemotronVLForCausalLM
from prime_rl.utils.cp import setup_model_cp, shard_for_cp

SEQ_LEN = 512
DOC_LEN = 256
IMG_TOKEN_ID = 18
IMAGE_RUNS = [(60, 66), (376, 382)]  # 6 tokens each; first run crosses the 64-token cp=8 boundary
TILE_H, TILE_W = 64, 96  # 4x6 patch grid -> 6 tokens per tile after pixel shuffle


def tiny_config() -> NemotronVLConfig:
    return NemotronVLConfig(
        text_config=dict(
            vocab_size=256,
            hidden_size=256,
            layers_block_type=["mamba", "attention", "moe", "mamba"],
            num_attention_heads=8,  # ulysses all-to-all needs heads % cp_size == 0
            num_key_value_heads=8,
            head_dim=32,
            max_position_embeddings=SEQ_LEN,
            intermediate_size=512,
            mamba_expand=2,
            mamba_num_heads=8,
            mamba_head_dim=64,
            ssm_state_size=64,
            mamba_n_groups=8,  # matches the real model's cp<=8 constraint
            mamba_d_conv=4,
            mamba_chunk_size=64,
            n_routed_experts=4,
            n_shared_experts=1,
            moe_intermediate_size=256,
            moe_shared_expert_intermediate_size=256,
            moe_latent_size=128,
            num_experts_per_tok=2,
            n_group=1,
            topk_group=1,
            norm_topk_prob=True,
            routed_scaling_factor=1.0,
            attn_implementation="flash_attention_3",
        ),
        vision_config=dict(
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=64,
            patch_size=16,
            max_img_size=128,
            num_cls_tokens=2,
        ),
        vit_hidden_size=32,
        projector_hidden_size=48,
        img_context_token_id=IMG_TOKEN_ID,
    )


def build_batch() -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(123)
    input_ids = torch.randint(32, 200, (1, SEQ_LEN), generator=g)
    for start, end in IMAGE_RUNS:
        input_ids[0, start:end] = IMG_TOKEN_ID
    target_ids = torch.randint(32, 200, (1, SEQ_LEN), generator=g)
    loss_mask = torch.zeros(1, SEQ_LEN, dtype=torch.bool)
    loss_mask[0, 128:DOC_LEN] = True  # doc 0 answer span
    loss_mask[0, 384:SEQ_LEN] = True  # doc 1 answer span
    return {
        "input_ids": input_ids,
        "target_ids": target_ids,
        "loss_mask": loss_mask,
        "position_ids": torch.cat([torch.arange(DOC_LEN), torch.arange(DOC_LEN)]).unsqueeze(0),
        "pixel_values": torch.randn(len(IMAGE_RUNS), 3, TILE_H, TILE_W, generator=g),
        "seq_lens": torch.tensor([DOC_LEN, DOC_LEN]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cp-size", type=int, required=True, choices=[1, 8])
    parser.add_argument("--scratch", type=str, required=True, help="node-local dir shared by both runs")
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == args.cp_size
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    torch.manual_seed(0)
    model = NemotronVLForCausalLM(tiny_config())
    state_path = os.path.join(args.scratch, "cp_equiv_init.pt")
    ref_path = os.path.join(args.scratch, "cp_equiv_ref.pt")
    if args.cp_size == 1:
        torch.save(model.state_dict(), state_path)
    else:
        model.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=True))
    model = model.to(torch.bfloat16).cuda()
    inject_prime_lm_head(model, chunk_size=64)

    captured = {}
    model.model.language_model.register_forward_pre_hook(
        lambda module, a, kw: captured.update(embeds=kw["inputs_embeds"].detach().clone()), with_kwargs=True
    )

    batch = {k: v.cuda() for k, v in build_batch().items()}
    pixel_values = batch["pixel_values"].to(torch.bfloat16)

    if args.cp_size == 1:
        target_ids, loss_mask = batch["target_ids"], batch["loss_mask"]
    else:
        # Same CP setup the SFT trainer does for cp_style="ulysses" (sft/train.py):
        # patch FlashAttention._compute_attention AND hand CP attrs to the mamba layers.
        from prime_rl.trainer.models.layers.ulysses_attn import substitute_ulysses_attn

        substitute_ulysses_attn(dist.group.WORLD, attn_impl="flash_attention_3")
        setup_model_cp(model, dist.group.WORLD, rank, world)
        target_ids = shard_for_cp(batch["target_ids"], cp_rank=rank, cp_world_size=world)
        loss_mask = shard_for_cp(batch["loss_mask"], cp_rank=rank, cp_world_size=world)

    out = model(
        input_ids=batch["input_ids"],
        position_ids=batch["position_ids"],
        labels=target_ids,
        temperature=torch.ones_like(target_ids, dtype=torch.float32),
        pixel_values=pixel_values,
        seq_lens=batch["seq_lens"],
        seq_lens_are_pre_shard=args.cp_size > 1,
    )
    loss_sum = -out["logprobs"][loss_mask].sum()
    loss_sum.backward()

    grads = {}
    for name, param in model.named_parameters():
        if param.numel() == 0:  # experts.w3 grouped-mm dummy
            continue
        grad = param.grad if param.grad is not None else torch.zeros_like(param)
        grad = grad.float()
        if args.cp_size > 1:
            dist.all_reduce(grad, op=dist.ReduceOp.SUM)
        grads[name] = grad.cpu()
    total_loss = loss_sum.detach().float()
    if args.cp_size > 1:
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)

    if args.cp_size == 1:
        torch.save(
            {"loss_sum": total_loss.cpu(), "grads": grads, "embeds": captured["embeds"].cpu()},
            ref_path,
        )
        print(f"[cp=1] loss_sum={total_loss.item():.6f}; reference saved to {ref_path}", flush=True)
        dist.destroy_process_group()
        return

    # Gather the sharded post-scatter embeds and compare to the cp=1 full-sequence embeds.
    shards = [torch.empty_like(captured["embeds"]) for _ in range(world)]
    dist.all_gather(shards, captured["embeds"])
    embeds_full = torch.cat(shards, dim=1).cpu()

    if rank == 0:
        ref = torch.load(ref_path, weights_only=True)
        embeds_exact = torch.equal(embeds_full, ref["embeds"])
        embeds_diff = (embeds_full.float() - ref["embeds"].float()).abs().max().item()
        loss_rel = abs(total_loss.item() - ref["loss_sum"].item()) / abs(ref["loss_sum"].item())

        print(f"[cp=8] post-scatter embeds: exact={embeds_exact} max|d|={embeds_diff:.3e}")
        print(f"[cp=8] loss_sum={total_loss.item():.6f} ref={ref['loss_sum'].item():.6f} rel={loss_rel:.3e}")

        worst, nan_params = [], []
        for name, grad in grads.items():
            ref_grad = ref["grads"][name]
            ref_nan = not torch.isfinite(ref_grad).all().item()
            our_nan = not torch.isfinite(grad).all().item()
            if ref_nan or our_nan:
                nan_params.append((name, "cp1" if ref_nan else "", "cp8" if our_nan else ""))
                continue
            scale = ref_grad.abs().max().item()
            if scale < 1e-12 and grad.abs().max().item() < 1e-12:
                continue
            max_diff = (grad - ref_grad).abs().max().item()
            cos = torch.nn.functional.cosine_similarity(grad.flatten(), ref_grad.flatten(), dim=0).item()
            worst.append((max_diff / max(scale, 1e-12), cos, name))
        worst.sort(reverse=True)
        print(f"[cp=8] non-finite grads on {len(nan_params)} params (side with NaN/inf marked):")
        for name, r, o in nan_params[:15]:
            print(f"    {r}{'+' if r and o else ''}{o} {name}")
        print("[cp=8] worst-10 finite grad rel diffs:")
        for rel, cos, name in worst[:10]:
            print(f"    rel={rel:.3e} cos={cos:.6f} {name}")
        print("[cp=8] projector grads (the only trainable params in phase 1):")
        for rel, cos, name in worst:
            if ".mlp1." in name:
                print(f"    rel={rel:.3e} cos={cos:.6f} {name}")

        # Vision-tower grads are unused in every planned stage (tower frozen in phase-1 SFT;
        # LoRA stage excludes the vision encoder) — NaNs there are reported but not gating.
        gating_nan = [n for n, _, _ in nan_params if ".visual." not in n]
        min_cos = min(cos for _, cos, _ in worst)
        ok = embeds_diff < 1e-6 and loss_rel < 5e-3 and min_cos > 0.99 and not gating_nan
        print(
            f"\n{'PASS' if ok else 'FAIL'}: embeds_max|d|={embeds_diff:.1e} loss_rel={loss_rel:.1e} "
            f"min_grad_cos={min_cos:.4f} nan_params={len(nan_params)} (gating: {len(gating_nan)})"
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
