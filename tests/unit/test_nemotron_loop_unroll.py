import importlib.util
from pathlib import Path

import numpy as np


def _load_loop_unroll():
    path = Path(__file__).resolve().parents[2] / "src/prime_rl/trainer/models/nemotron_h/loop_unroll.py"
    spec = importlib.util.spec_from_file_location("loop_unroll", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


LU = _load_loop_unroll()


def _rmsnorm(x: np.ndarray, w: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return (x / rms) * w


def _linear(x: np.ndarray, w: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    y = x @ w.T
    return y + b if b is not None else y


def _loop_linear(
    x: np.ndarray, sd: dict[str, np.ndarray], prefix: str, loop_idx: int, scaling: float
) -> np.ndarray:
    y = _linear(x, sd[f"{prefix}.weight"], sd.get(f"{prefix}.bias"))
    if loop_idx > 0 and f"{prefix}.loop_lora_A.{loop_idx - 1}" in sd:
        a = sd[f"{prefix}.loop_lora_A.{loop_idx - 1}"]
        b = sd[f"{prefix}.loop_lora_B.{loop_idx - 1}"]
        y = y + scaling * ((x @ a.T) @ b.T)
    return y


def _run_layer(
    x: np.ndarray,
    sd: dict[str, np.ndarray],
    layer_idx: int,
    layer_char: str,
    loop_idx: int,
    scaling: float,
    latent_moe: bool = True,
) -> np.ndarray:
    p = f"model.layers.{layer_idx}"
    norm_w = sd[f"{p}.norm.weight"] if loop_idx == 0 else sd[f"{p}.loop_norms.{loop_idx - 1}.weight"]
    h = _rmsnorm(x, norm_w)

    if layer_char == "M":
        mid = np.tanh(_loop_linear(h, sd, f"{p}.mamba.in_proj", loop_idx, scaling))
        out = _loop_linear(mid, sd, f"{p}.mamba.out_proj", loop_idx, scaling)
    elif layer_char == "*":
        q = _loop_linear(h, sd, f"{p}.self_attn.q_proj", loop_idx, scaling)
        k = _loop_linear(h, sd, f"{p}.self_attn.k_proj", loop_idx, scaling)
        v = _loop_linear(h, sd, f"{p}.self_attn.v_proj", loop_idx, scaling)
        attn = np.tanh(q + k) * v
        out = _loop_linear(attn, sd, f"{p}.self_attn.o_proj", loop_idx, scaling)
    elif layer_char == "E":
        rg = (
            sd[f"{p}.mlp.router.gate.weight"]
            if loop_idx == 0
            else sd[f"{p}.loop_routers.{loop_idx - 1}.gate.weight"]
        )
        rb = (
            sd[f"{p}.mlp.router.selection_bias"]
            if loop_idx == 0
            else sd[f"{p}.loop_routers.{loop_idx - 1}.selection_bias"]
        )
        scores = 1.0 / (1.0 + np.exp(-(h @ rg.T)))
        sel = np.argmax(scores + rb, axis=-1)
        if latent_moe:
            e_in = _loop_linear(h, sd, f"{p}.mlp.fc1_latent_proj", loop_idx, scaling)
            e_out = np.zeros_like(e_in)
            for t in range(h.shape[0]):
                e = sel[t]
                up = np.maximum(e_in[t] @ sd[f"{p}.mlp.experts.up_proj"][e].T, 0.0) ** 2
                e_out[t] = (up @ sd[f"{p}.mlp.experts.down_proj"][e].T) * scores[t, e]
            routed = _loop_linear(e_out, sd, f"{p}.mlp.fc2_latent_proj", loop_idx, scaling)
        else:
            routed = np.zeros_like(h)
            for t in range(h.shape[0]):
                e = sel[t]
                up = np.maximum(h[t] @ sd[f"{p}.mlp.experts.up_proj"][e].T, 0.0) ** 2
                routed[t] = (up @ sd[f"{p}.mlp.experts.down_proj"][e].T) * scores[t, e]
        sh_up = np.maximum(_loop_linear(h, sd, f"{p}.mlp.shared_expert.up_proj", loop_idx, scaling), 0.0) ** 2
        shared = _loop_linear(sh_up, sd, f"{p}.mlp.shared_expert.down_proj", loop_idx, scaling)
        out = routed + shared
    else:
        raise AssertionError(layer_char)

    if loop_idx > 0:
        out = out * sd[f"{p}.loop_gammas.{loop_idx - 1}"]
    return x + out


def _make_synthetic_looped(pattern: str, start: int, end: int, num_loops: int, rank: int, gamma_scale: float, latent_moe: bool):
    rng = np.random.default_rng(42)
    d, mid_d, lat_d, n_exp = 12, 16, 8, 4
    sd: dict[str, np.ndarray] = {"model.embed_tokens.weight": rng.standard_normal((32, d)).astype(np.float32)}
    num_extra = num_loops - 1

    def add_lin(key: str, out_f: int, in_f: int, with_bias: bool, in_loop: bool):
        sd[f"{key}.weight"] = (rng.standard_normal((out_f, in_f)) * 0.2).astype(np.float32)
        if with_bias:
            sd[f"{key}.bias"] = (rng.standard_normal((out_f,)) * 0.1).astype(np.float32)
        if in_loop and rank > 0:
            for r in range(num_extra):
                sd[f"{key}.loop_lora_A.{r}"] = (rng.standard_normal((rank, in_f)) * 0.15).astype(np.float32)
                sd[f"{key}.loop_lora_B.{r}"] = (rng.standard_normal((out_f, rank)) * 0.15).astype(np.float32)

    for i, ch in enumerate(pattern):
        p = f"model.layers.{i}"
        in_loop = start <= i < end
        sd[f"{p}.norm.weight"] = (1.0 + rng.standard_normal((d,)) * 0.1).astype(np.float32)
        if in_loop:
            for r in range(num_extra):
                sd[f"{p}.loop_norms.{r}.weight"] = (1.0 + rng.standard_normal((d,)) * 0.1).astype(np.float32)
                sd[f"{p}.loop_gammas.{r}"] = (rng.standard_normal((d,)) * gamma_scale).astype(np.float32)
        if ch == "M":
            add_lin(f"{p}.mamba.in_proj", mid_d, d, with_bias=True, in_loop=in_loop)
            add_lin(f"{p}.mamba.out_proj", d, mid_d, with_bias=True, in_loop=in_loop)
        elif ch == "*":
            for proj in ("q_proj", "k_proj", "v_proj"):
                add_lin(f"{p}.self_attn.{proj}", d, d, with_bias=True, in_loop=in_loop)
            add_lin(f"{p}.self_attn.o_proj", d, d, with_bias=True, in_loop=in_loop)
        elif ch == "E":
            sd[f"{p}.mlp.router.gate.weight"] = (rng.standard_normal((n_exp, d)) * 0.2).astype(np.float32)
            sd[f"{p}.mlp.router.selection_bias"] = (rng.standard_normal((n_exp,)) * 0.05).astype(np.float32)
            if in_loop:
                for r in range(num_extra):
                    sd[f"{p}.loop_routers.{r}.gate.weight"] = (rng.standard_normal((n_exp, d)) * 0.2).astype(np.float32)
                    sd[f"{p}.loop_routers.{r}.selection_bias"] = (rng.standard_normal((n_exp,)) * 0.05).astype(np.float32)
            e_in_d = lat_d if latent_moe else d
            sd[f"{p}.mlp.experts.up_proj"] = (rng.standard_normal((n_exp, mid_d, e_in_d)) * 0.2).astype(np.float32)
            sd[f"{p}.mlp.experts.down_proj"] = (rng.standard_normal((n_exp, e_in_d, mid_d)) * 0.2).astype(np.float32)
            if latent_moe:
                add_lin(f"{p}.mlp.fc1_latent_proj", lat_d, d, with_bias=False, in_loop=in_loop)
                add_lin(f"{p}.mlp.fc2_latent_proj", d, lat_d, with_bias=True, in_loop=in_loop)
            add_lin(f"{p}.mlp.shared_expert.up_proj", mid_d, d, with_bias=False, in_loop=in_loop)
            add_lin(f"{p}.mlp.shared_expert.down_proj", d, mid_d, with_bias=True, in_loop=in_loop)

    return sd, d


def test_strip_loop_from_state_dict():
    sd, _ = _make_synthetic_looped("M*EM*E", 1, 5, 2, 4, 0.3, latent_moe=True)
    stripped = LU.strip_loop_from_state_dict(sd)
    assert all(not LU.is_loop_param_name(k) for k in stripped)
    assert len(stripped) < len(sd)
    assert "model.layers.1.self_attn.o_proj.weight" in stripped
    assert "model.layers.2.mlp.router.gate.weight" in stripped


def test_step0_zero_gamma_is_exact_identity():
    pattern = "M*EM*E"
    start, end, num_loops, rank, alpha = 1, 5, 2, 4, 8.0
    sd, d = _make_synthetic_looped(pattern, start, end, num_loops, rank, gamma_scale=0.0, latent_moe=True)
    x = np.random.default_rng(7).standard_normal((5, d)).astype(np.float32)

    h_base = x.copy()
    for i, ch in enumerate(pattern):
        h_base = _run_layer(h_base, sd, i, ch, loop_idx=0, scaling=alpha / rank)

    h_loop = x.copy()
    for i in range(0, start):
        h_loop = _run_layer(h_loop, sd, i, pattern[i], loop_idx=0, scaling=alpha / rank)
    for loop_idx in range(num_loops):
        for i in range(start, end):
            h_loop = _run_layer(h_loop, sd, i, pattern[i], loop_idx=loop_idx, scaling=alpha / rank)
    for i in range(end, len(pattern)):
        h_loop = _run_layer(h_loop, sd, i, pattern[i], loop_idx=0, scaling=alpha / rank)

    assert np.array_equal(h_base, h_loop)


def test_unroll_exact_equivalence_latent_and_non_latent_moe():
    pattern = "M*EM*E"
    start, end, num_loops, rank, alpha = 1, 5, 3, 4, 8.0
    loop_cfg = {"num_loops": num_loops, "start_layer": start, "end_layer": end, "lora_rank": rank, "lora_alpha": alpha}
    cfg = {"hybrid_override_pattern": pattern, "num_hidden_layers": len(pattern), "loop_config": loop_cfg}

    for latent_moe in (True, False):
        sd, d = _make_synthetic_looped(pattern, start, end, num_loops, rank, gamma_scale=0.4, latent_moe=latent_moe)
        x = np.random.default_rng(13).standard_normal((6, d)).astype(np.float32)

        h_loop = x.copy()
        for i in range(0, start):
            h_loop = _run_layer(h_loop, sd, i, pattern[i], loop_idx=0, scaling=alpha / rank, latent_moe=latent_moe)
        for loop_idx in range(num_loops):
            for i in range(start, end):
                h_loop = _run_layer(
                    h_loop, sd, i, pattern[i], loop_idx=loop_idx, scaling=alpha / rank, latent_moe=latent_moe
                )
        for i in range(end, len(pattern)):
            h_loop = _run_layer(h_loop, sd, i, pattern[i], loop_idx=0, scaling=alpha / rank, latent_moe=latent_moe)

        unrolled_sd, new_cfg = LU.unroll_looped_state_dict(sd, cfg, loop_cfg)
        assert "loop_config" not in new_cfg
        assert new_cfg["hybrid_override_pattern"] == "M" + "*EM*" * 3 + "E"
        assert new_cfg["num_hidden_layers"] == 14
        assert all(not LU.is_loop_param_name(k) for k in unrolled_sd)

        h_seq = x.copy()
        for i, ch in enumerate(new_cfg["hybrid_override_pattern"]):
            h_seq = _run_layer(h_seq, unrolled_sd, i, ch, loop_idx=0, scaling=0.0, latent_moe=latent_moe)

        max_diff = float(np.max(np.abs(h_loop - h_seq)))
        assert max_diff < 1e-5, f"latent_moe={latent_moe}: max_diff={max_diff}"
