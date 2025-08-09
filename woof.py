from prime_rl.trainer.model import get_model
from prime_rl.trainer.config import ModelConfig

model_config = ModelConfig(
    name="Qwen/Qwen3-30B-A3B",
    attn="sdpa",
    trust_remote_code=True,
    ep_mode=1,
)

model = get_model(model_config)
print(model)
