import dataclasses
import os
import sys
from pathlib import Path

sys.path.insert(0, "tests/unit/train")
import tiny_models as tm
import test_compile as tc

tm.ROW_LEN = int(os.environ.get("PROD_ROW_LEN", "4096"))
SHAPES = {
    "ctrl": ("glm4_moe", dict()),
    "nemo32": ("nemotron_h", dict(hybrid_override_pattern="*E", hidden_size=1024, num_attention_heads=32, num_key_value_heads=2, head_dim=128)),
    "nemo64": ("nemotron_h", dict(hybrid_override_pattern="*E", hidden_size=1024, num_attention_heads=64, num_key_value_heads=2, head_dim=128)),
    "muse32": ("muse_glimmer", dict(hidden_size=1024, num_attention_heads=32, num_key_value_heads=2, head_dim=128)),
    "q3m64": ("qwen3_moe", dict(hidden_size=2048, num_attention_heads=64, num_key_value_heads=4, head_dim=128)),
    "glm05b": (
        "glm4_moe",
        dict(hidden_size=1024, num_attention_heads=16, num_key_value_heads=4, head_dim=64, partial_rotary_factor=0.5, attention_bias=True, use_qk_norm=False),
    ),
    "mm48": ("minimax_m2", dict(hidden_size=1024, num_attention_heads=48, num_key_value_heads=8, head_dim=128, rotary_dim=64)),
}
name, overrides = SHAPES[os.environ["SHAPE"]]
for configs in (tm.TINY_MODELS, tm.TINY_PRIME_RL_MODELS):
    if name in configs:
        old = configs[name].to_dict()
        if "partial_rotary_factor" in overrides and isinstance(old.get("rope_parameters"), dict):
            old["rope_parameters"] = {**old["rope_parameters"], "partial_rotary_factor": overrides["partial_rotary_factor"]}
        if "hybrid_override_pattern" in overrides:
            old.pop("layers_block_type", None)
            old.pop("layer_types", None)
            old.pop("num_hidden_layers", None)
        if name == "minimax_m2":
            old.pop("partial_rotary_factor", None)
        configs[name] = type(configs[name])(**{**old, **overrides})
        cfg = configs[name]
        print(name, {k: getattr(cfg, k, None) for k in overrides}, getattr(cfg, "rope_parameters", None), getattr(cfg, "layer_types", None), getattr(cfg, "partial_rotary_factor", None), flush=True)
tc.main(Path(sys.argv[1]), [dataclasses.replace(case, fullgraph=False) for case in tc.select_cases([f"{name}-dp2"])])
