from prime_rl.trainer.custom_models.glm import Glm4MoeForCausalLM
from prime_rl.trainer.custom_models.llama import LlamaForCausalLM


def get_model_cls(architectures: list[str]):
    match architectures[0]:
        case "LlamaForCausalLM":
            return LlamaForCausalLM
        case "Glm4MoeForCausalLM":
            return Glm4MoeForCausalLM
        case _:
            raise ValueError(f"Unsupported architecture: {architectures}")


_all = [get_model_cls]
