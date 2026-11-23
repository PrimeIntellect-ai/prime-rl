import json, sys
import torch
sys.path.insert(0, "tests/unit/train")
sys.path.insert(0, "tests/unit/train/models")
from test_golden_drophf import gpu_step_record

out, runs = sys.argv[1], int(sys.argv[2])
models = sys.argv[3].split(",") if len(sys.argv) > 3 else ["deepseek_v4", "glm5_next", "glm_moe_dsa", "laguna"]
torch.use_deterministic_algorithms(True, warn_only=True)
samples = {}
for name in models:
    samples[name] = [gpu_step_record(name, "flash_attention_3") for _ in range(runs)]
    print("DONE", name, flush=True)
json.dump(samples, open(out, "w"))
