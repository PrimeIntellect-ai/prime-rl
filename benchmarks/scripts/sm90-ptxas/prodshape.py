import os
import sys
from pathlib import Path

sys.path.insert(0, "tests/unit/train")
import tiny_models as tm
import test_compile as tc

tm.ROW_LEN = int(os.environ.get("PROD_ROW_LEN", "4096"))
OVERRIDES = {
    "glm4_moe": dict(hidden_size=4096, num_attention_heads=96, num_key_value_heads=8, head_dim=128, partial_rotary_factor=0.5, attention_bias=True),
    "qwen3_moe": dict(hidden_size=2048, num_attention_heads=32, num_key_value_heads=4, head_dim=128),
    "qwen3": dict(hidden_size=1024, num_attention_heads=16, num_key_value_heads=8, head_dim=128),
    "llama": dict(hidden_size=2048, num_attention_heads=32, num_key_value_heads=8, head_dim=64),
    "deepseek_v4": dict(hidden_size=4096, num_attention_heads=64, head_dim=512, q_lora_rank=1024, o_lora_rank=1024, o_groups=8, qk_rope_head_dim=64, index_n_heads=64, index_head_dim=128, index_topk=512, sliding_window=128),
}
for name, overrides in OVERRIDES.items():
    for configs in (tm.TINY_MODELS, tm.TINY_PRIME_RL_MODELS):
        if name in configs:
            old = configs[name].to_dict()
            if "partial_rotary_factor" in overrides and isinstance(old.get("rope_parameters"), dict):
                old["rope_parameters"] = {**old["rope_parameters"], "partial_rotary_factor": overrides["partial_rotary_factor"]}
            configs[name] = type(configs[name])(**{**old, **overrides})
            print(name, {k: getattr(configs[name], k, None) for k in overrides}, getattr(configs[name], "rope_parameters", None), flush=True)
tc.main(Path(sys.argv[1]), tc.select_cases(sys.argv[2:]))
