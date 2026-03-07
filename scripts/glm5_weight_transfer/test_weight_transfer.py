"""Test weight transfer: PrimeRL bf16 → vLLM FP8 for GLM-5 (GlmMoeDsa).

Uses reload_weights(is_checkpoint_format=False) for CUDA-graph safe in-place transfer.
"""
import os
os.environ["VLLM_USE_DEEP_GEMM"] = "1"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import torch
import torch.nn.functional as F
from vllm import LLM, SamplingParams

BF16_DIR = "/home/matej/dev/prime-rl/scripts/glm5_weight_transfer/checkpoints/glm5-tiny-bf16"
FP8_DIR = "/home/matej/dev/prime-rl/scripts/glm5_weight_transfer/checkpoints/glm5-tiny-fp8"
VOCAB_SIZE = 1024


def quantize_to_fp8_blockwise(weight, block_size=128):
    rows, cols = weight.shape
    br = bc = block_size
    pad_r = (br - rows % br) % br
    pad_c = (bc - cols % bc) % bc
    if pad_r > 0 or pad_c > 0:
        padded = torch.zeros(rows + pad_r, cols + pad_c, dtype=weight.dtype, device=weight.device)
        padded[:rows, :cols] = weight
    else:
        padded = weight.clone()
    pr, pc = padded.shape
    blocks = padded.reshape(pr // br, br, pc // bc, bc).permute(0, 2, 1, 3)
    max_abs = blocks.float().abs().amax(dim=(2, 3))
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    scale = (max_abs / fp8_max).clamp(min=1e-12)
    blocks_fp8 = (blocks.float() / scale[:, :, None, None]).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
    return blocks_fp8.permute(0, 2, 1, 3).reshape(pr, pc)[:rows, :cols].contiguous(), scale.float().contiguous()


def build_kernel_weights(prime_sd, config, quantize_fp8=True):
    """Convert PrimeRL state dict to vLLM kernel-format weights."""
    weights = {}

    def add(name, tensor):
        weights[name] = tensor.cpu()

    def add_fp8(name, tensor):
        if quantize_fp8 and tensor.ndim == 2:
            t = tensor.cuda()
            fp8_w, scale = quantize_to_fp8_blockwise(t)
            weights[name] = fp8_w.cpu()
            scale_name = name[:-len(".weight")] + ".weight_scale_inv" if name.endswith(".weight") else name + "_scale_inv"
            weights[scale_name] = scale.cpu()
        else:
            add(name, tensor)

    add("model.embed_tokens.weight", prime_sd["model.embed_tokens.weight"])
    add("model.norm.weight", prime_sd["model.norm.weight"])
    add("lm_head.weight", prime_sd["lm_head.weight"])

    for i in range(config.num_hidden_layers):
        p = f"model.layers.{i}"

        # Norms
        add(f"{p}.input_layernorm.weight", prime_sd[f"{p}.input_layernorm.weight"])
        add(f"{p}.post_attention_layernorm.weight", prime_sd[f"{p}.post_attention_layernorm.weight"])

        # Attention: fused_qkv_a_proj = cat(q_a_proj, kv_a_proj_with_mqa)
        fused = torch.cat([prime_sd[f"{p}.self_attn.q_a_proj.weight"], prime_sd[f"{p}.self_attn.kv_a_proj_with_mqa.weight"]], dim=0)
        add_fp8(f"{p}.self_attn.fused_qkv_a_proj.weight", fused)
        add(f"{p}.self_attn.q_a_layernorm.weight", prime_sd[f"{p}.self_attn.q_a_layernorm.weight"])
        add_fp8(f"{p}.self_attn.q_b_proj.weight", prime_sd[f"{p}.self_attn.q_b_proj.weight"])
        add(f"{p}.self_attn.kv_a_layernorm.weight", prime_sd[f"{p}.self_attn.kv_a_layernorm.weight"])
        add_fp8(f"{p}.self_attn.kv_b_proj.weight", prime_sd[f"{p}.self_attn.kv_b_proj.weight"])
        add_fp8(f"{p}.self_attn.o_proj.weight", prime_sd[f"{p}.self_attn.o_proj.weight"])

        # Indexer
        add_fp8(f"{p}.self_attn.indexer.wq_b.weight", prime_sd[f"{p}.self_attn.indexer.wq_b.weight"])
        add_fp8(f"{p}.self_attn.indexer.wk.weight", prime_sd[f"{p}.self_attn.indexer.wk.weight"])
        add(f"{p}.self_attn.indexer.k_norm.weight", prime_sd[f"{p}.self_attn.indexer.k_norm.weight"])
        add(f"{p}.self_attn.indexer.k_norm.bias", prime_sd[f"{p}.self_attn.indexer.k_norm.bias"])
        add(f"{p}.self_attn.indexer.weights_proj.weight", prime_sd[f"{p}.self_attn.indexer.weights_proj.weight"])

        if i < config.first_k_dense_replace:
            # Dense MLP
            gate_up = torch.cat([prime_sd[f"{p}.mlp.gate_proj.weight"], prime_sd[f"{p}.mlp.up_proj.weight"]], dim=0)
            add_fp8(f"{p}.mlp.gate_up_proj.weight", gate_up)
            add_fp8(f"{p}.mlp.down_proj.weight", prime_sd[f"{p}.mlp.down_proj.weight"])
        else:
            # MoE
            add(f"{p}.mlp.gate.weight", prime_sd[f"{p}.mlp.router.gate.weight"])
            add(f"{p}.mlp.gate.e_score_correction_bias", prime_sd[f"{p}.mlp.expert_bias"])

            w1 = prime_sd[f"{p}.mlp.experts.w1"].cuda()
            w3 = prime_sd[f"{p}.mlp.experts.w3"].cuda()
            w2 = prime_sd[f"{p}.mlp.experts.w2"].cuda()
            w13 = torch.cat([w1, w3], dim=1)

            if quantize_fp8:
                w13_fp8, w13_s, w2_fp8, w2_s = [], [], [], []
                for j in range(config.n_routed_experts):
                    f8, s = quantize_to_fp8_blockwise(w13[j])
                    w13_fp8.append(f8); w13_s.append(s)
                    f8, s = quantize_to_fp8_blockwise(w2[j])
                    w2_fp8.append(f8); w2_s.append(s)
                weights[f"{p}.mlp.experts.w13_weight"] = torch.stack(w13_fp8).cpu()
                weights[f"{p}.mlp.experts.w13_weight_scale_inv"] = torch.stack(w13_s).cpu()
                weights[f"{p}.mlp.experts.w2_weight"] = torch.stack(w2_fp8).cpu()
                weights[f"{p}.mlp.experts.w2_weight_scale_inv"] = torch.stack(w2_s).cpu()
            else:
                weights[f"{p}.mlp.experts.w13_weight"] = w13.cpu()
                weights[f"{p}.mlp.experts.w2_weight"] = w2.cpu()

            sw1 = prime_sd[f"{p}.mlp.shared_expert.w1"].cuda()
            sw3 = prime_sd[f"{p}.mlp.shared_expert.w3"].cuda()
            sw2 = prime_sd[f"{p}.mlp.shared_expert.w2"].cuda()
            if sw1.dim() == 3: sw1, sw3, sw2 = sw1.squeeze(0), sw3.squeeze(0), sw2.squeeze(0)
            add_fp8(f"{p}.mlp.shared_experts.gate_up_proj.weight", torch.cat([sw1, sw3], dim=0))
            add_fp8(f"{p}.mlp.shared_experts.down_proj.weight", sw2)

    return weights


def main():
    print("=== Step 1: Load FP8 model in vLLM ===")
    llm = LLM(
        model=FP8_DIR,
        dtype="bfloat16",
        enforce_eager=True,
        gpu_memory_utilization=0.5,
        max_model_len=256,
        trust_remote_code=True,
        worker_extension_cls="kernel_reload_worker.KernelReloadWorker",
    )
    print("FP8 vLLM model loaded.")

    torch.manual_seed(42)
    prompt_ids = torch.randint(1, VOCAB_SIZE, (32,)).tolist()
    sampling = SamplingParams(max_tokens=1, temperature=1.0, logprobs=20, prompt_logprobs=20)

    out = llm.generate([{"prompt_token_ids": prompt_ids}], sampling)
    print(f"FP8 generation test OK: {out[0].outputs[0].token_ids}")

    # ── 2. Load PrimeRL bf16 model ──────────────────────────────────────────

    print("\n=== Step 2: Load PrimeRL bf16 model ===")
    from prime_rl.trainer.models import AutoModelForCausalLMPrimeRL
    from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
    from prime_rl.trainer.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig

    prime_config = GlmMoeDsaConfig.from_pretrained(BF16_DIR)
    prime_config.use_cache = False
    print(f"PrimeRL config mlp_layer_types: {prime_config.mlp_layer_types}")
    prime_model = AutoModelForCausalLMPrimeRL.from_pretrained(BF16_DIR, config=prime_config, dtype=torch.bfloat16)
    prime_model = prime_model.cuda().eval()
    inject_prime_lm_head(prime_model, chunk_size=None)
    print("PrimeRL bf16 model loaded.")

    input_ids = torch.tensor([prompt_ids], device="cuda")
    position_ids = torch.arange(len(prompt_ids), device="cuda").unsqueeze(0)
    with torch.no_grad():
        prime_out = prime_model(input_ids=input_ids, position_ids=position_ids)
        prime_logits = prime_out["logits"] if isinstance(prime_out, dict) else prime_out.logits
        prime_logits = prime_logits.float().squeeze(0)

    print(f"PrimeRL logits: {prime_logits.shape}")
    print(f"PrimeRL logits stats: min={prime_logits.min():.4f}, max={prime_logits.max():.4f}, "
          f"mean={prime_logits.mean():.4f}, has_nan={prime_logits.isnan().any()}, has_inf={prime_logits.isinf().any()}")

    if prime_logits.isnan().any():
        # Debug: check each layer's output
        print("DEBUG: Checking per-layer outputs for NaN source...")
        with torch.no_grad():
            x = prime_model.model.embed_tokens(input_ids)
            print(f"  After embed: nan={x.isnan().any()}")
            pos_emb = prime_model.model.rotary_emb(x, position_ids)

            flat_pos = position_ids.view(-1)
            S = flat_pos.shape[0]
            ks = torch.arange(S, dtype=torch.int32, device=flat_pos.device) - flat_pos.to(torch.int32)
            ke = torch.arange(1, S + 1, dtype=torch.int32, device=flat_pos.device)

            for layer_idx, layer in enumerate(prime_model.model.layers):
                residual = x
                h = layer.input_layernorm(x)
                print(f"  Layer {layer_idx} after layernorm: nan={h.isnan().any()}")
                h, _ = layer.self_attn(h, position_embeddings=pos_emb, ks=ks, ke=ke)
                print(f"  Layer {layer_idx} after attn: nan={h.isnan().any()}")
                x = residual + h
                residual = x
                h = layer.post_attention_layernorm(x)
                h = layer.mlp(h)
                print(f"  Layer {layer_idx} after mlp: nan={h.isnan().any()}")
                x = residual + h
            x = prime_model.model.norm(x)
            print(f"  After final norm: nan={x.isnan().any()}")

    # ── 3. Transfer weights: PrimeRL bf16 → vLLM FP8 ───────────────────────

    print("\n=== Step 3: Convert & reload weights (is_checkpoint_format=False) ===")
    prime_sd = prime_model.state_dict()
    fp8_weights = build_kernel_weights(prime_sd, prime_config, quantize_fp8=True)
    print(f"Built {len(fp8_weights)} FP8 kernel-format weights")

    # Save kernel-format weights to shared filesystem
    from safetensors.torch import save_file
    kernel_path = "/home/matej/dev/prime-rl/scripts/glm5_weight_transfer/checkpoints/kernel_weights.safetensors"
    save_file(fp8_weights, kernel_path)
    print(f"Saved kernel weights to {kernel_path}")

    # Call reload on worker via collective_rpc - pass path as string (serializable)
    engine_core = llm.llm_engine.engine_core
    engine_core.collective_rpc(
        "reload_kernel_weights_from_path",
        args=(kernel_path,),
    )
    print("Weights reloaded into vLLM FP8 model.")

    engine_core.reset_prefix_cache()

    # ── 4. Get vLLM logprobs after weight transfer ──────────────────────────

    print("\n=== Step 4: Get vLLM logprobs after FP8 weight transfer ===")

    sampling = SamplingParams(max_tokens=1, temperature=1.0, logprobs=20, prompt_logprobs=20)
    out_after = llm.generate([{"prompt_token_ids": prompt_ids}], sampling)

    vllm_top20 = []
    for pos_lp in out_after[0].prompt_logprobs:
        if pos_lp is None:
            continue
        vllm_top20.append(pos_lp)

    # Debug: print first few positions' top predictions from both sides
    print(f"\nvLLM has {len(vllm_top20)} positions of logprobs")
    prime_log_probs = F.log_softmax(prime_logits.float(), dim=-1)

    # PrimeRL logits[i] predicts token i+1, vLLM prompt_logprobs[i] predicts token i
    # So prime_logits[0..N-2] aligns with vllm prompt_logprobs[1..N-1]
    # But wait - vllm prompt_logprobs skips position 0 (it's None), so
    # vllm_top20[0] is position 1's logprobs (predicting token at position 1 given 0..0)
    # prime_logits[0] predicts token at position 1 given token at position 0
    # So they should align: prime_logits[i] <-> vllm_top20[i]

    print("\nDebug: first 5 positions comparison")
    for pos_idx in range(min(5, len(vllm_top20))):
        vllm_lp = vllm_top20[pos_idx]
        vllm_top1_id = max(vllm_lp, key=lambda k: vllm_lp[k].logprob)
        vllm_top1_logprob = vllm_lp[vllm_top1_id].logprob

        if pos_idx < prime_logits.shape[0] - 1:
            prime_top1_id = prime_log_probs[pos_idx].argmax().item()
            prime_top1_logprob = prime_log_probs[pos_idx, prime_top1_id].item()
            prime_at_vllm_top1 = prime_log_probs[pos_idx, vllm_top1_id].item()
        else:
            prime_top1_id = -1
            prime_top1_logprob = float("nan")
            prime_at_vllm_top1 = float("nan")

        print(f"  pos {pos_idx}: vLLM top1={vllm_top1_id} (lp={vllm_top1_logprob:.4f}), "
              f"PrimeRL top1={prime_top1_id} (lp={prime_top1_logprob:.4f}), "
              f"PrimeRL@vLLM_top1={prime_at_vllm_top1:.4f}")

    # Compute KL properly using top-20 approximation
    n_match = 0
    n_total = 0
    kl_values = []

    for pos_idx, vllm_lp in enumerate(vllm_top20):
        if pos_idx >= prime_logits.shape[0] - 1:
            break

        vllm_top1 = max(vllm_lp, key=lambda k: vllm_lp[k].logprob)
        prime_top1 = prime_log_probs[pos_idx].argmax().item()
        if prime_top1 == vllm_top1:
            n_match += 1
        n_total += 1

        # KL using top-20 tokens (where most probability mass is)
        kl = 0.0
        for tok_id, lp_obj in vllm_lp.items():
            p = prime_log_probs[pos_idx, tok_id].exp().item()
            if p > 1e-10:
                log_p = prime_log_probs[pos_idx, tok_id].item()
                log_q = lp_obj.logprob
                kl += p * (log_p - log_q)
        kl_values.append(kl)

    top1_agreement = n_match / n_total if n_total > 0 else 0
    mean_kl = sum(kl_values) / len(kl_values) if kl_values else float("inf")

    print(f"\n{'='*60}")
    print(f"Top-1 prediction agreement: {top1_agreement:.4f} ({n_match}/{n_total})")
    print(f"Approximate KL (top-20): {mean_kl:.6f}")

    if mean_kl < 0.05:
        print(f"\nSUCCESS: KL < 0.05")
    else:
        print(f"\nFAIL: KL = {mean_kl:.6f} >= 0.05")

    print("\nDone.")


if __name__ == "__main__":
    main()
