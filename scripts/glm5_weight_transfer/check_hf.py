from transformers import AutoModelForCausalLM, AutoConfig
import traceback

BF16_DIR = "/home/matej/dev/prime-rl/scripts/glm5_weight_transfer/checkpoints/glm5-tiny-bf16"
config = AutoConfig.from_pretrained(BF16_DIR)
print("model_type:", config.model_type)
print("architectures:", config.architectures)

try:
    model = AutoModelForCausalLM.from_config(config)
    print("HF model class:", type(model).__name__)
    for name, p in model.named_parameters():
        print(f"  {name} {list(p.shape)}")
        if "layers.1" in name:
            break
except Exception:
    traceback.print_exc()
