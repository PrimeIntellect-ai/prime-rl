import torch
from flash_attn.cute import flash_attn_varlen_func

from prime_rl.trainer.models.gpt_oss import GptOssConfig
from prime_rl.trainer.models.gpt_oss.attention import GptOssAttention

for layer_type in ["sliding_attention", "full_attention"]:
    torch.manual_seed(0)
    config = GptOssConfig(
        num_hidden_layers=1,
        num_local_experts=4,
        vocab_size=128,
        hidden_size=512,
        intermediate_size=32,
        head_dim=64,
        num_attention_heads=8,
        num_key_value_heads=2,
        max_position_embeddings=4096,
        num_experts_per_tok=2,
        sliding_window=128,
        layer_types=[layer_type],
        rope_parameters={"rope_type": "default", "rope_theta": 150000.0},
        use_cache=False,
    )
    attention = GptOssAttention(config, 0).cuda().to(torch.bfloat16)
    with torch.no_grad():
        attention.sinks.normal_(0, 1)
    seq = [700, 37, 1290, 21]
    cu = torch.tensor([0, *torch.tensor(seq).cumsum(0).tolist()], device="cuda", dtype=torch.int32)
    T, M = sum(seq), max(seq)
    inputs = [torch.randn(T, h, 64, device="cuda", dtype=torch.bfloat16) for h in (8, 2, 2)]
    w = (attention.sliding_window - 1, 0) if attention.sliding_window is not None else (None, None)
    g = torch.randn(T, 8, 64, device="cuda", dtype=torch.bfloat16)

    def raw(deterministic):
        ri = [t.clone().requires_grad_() for t in inputs]
        rs = attention.sinks.detach().clone().requires_grad_()
        out, _ = flash_attn_varlen_func(
            *ri,
            cu_seqlens_q=cu,
            cu_seqlens_k=cu,
            max_seqlen_q=M,
            max_seqlen_k=M,
            softmax_scale=attention.scaling,
            causal=True,
            window_size=w,
            learnable_sink=rs,
            deterministic=deterministic,
        )
        out.backward(g)
        return [out, *(t.grad for t in ri), rs.grad]

    def op():
        attention.sinks.grad = None
        ai = [t.clone().requires_grad_() for t in inputs]
        out = attention.compute_attention(*ai, cu, M)
        out.backward(g)
        return [out, *(t.grad for t in ai), attention.sinks.grad]

    def diff(a, b):
        return {
            n: (x.float() - y.float()).abs().max().item() for n, x, y in zip(["out", "dq", "dk", "dv", "dsink"], a, b)
        }

    r1, r2, d1, d2, o1, o2 = raw(False), raw(False), raw(True), raw(True), op(), op()
    print(layer_type, "raw vs raw          ", diff(r1, r2))
    print(layer_type, "raw det vs raw det  ", diff(d1, d2))
    print(layer_type, "op vs op            ", diff(o1, o2))
    print(layer_type, "op vs raw           ", diff(o1, r1))
    print(layer_type, "raw vs raw det      ", diff(r1, d1))
    print(layer_type, "dk/dv max abs value ", inputs and (r1[2].abs().max().item(), r1[3].abs().max().item()))
