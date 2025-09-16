import torch.nn as nn

from prime_rl.trainer.models.qwen3 import Qwen3Model, qwen3_configs


def init_model(pretrained_model_name_or_path: str) -> nn.Module:
    if pretrained_model_name_or_path not in qwen3_configs.keys():
        raise NotImplementedError(
            f"Model {pretrained_model_name_or_path} is not implemented with Prime Rl AutoModel, please use --model-impl hf instead"
        )

    args = qwen3_configs[pretrained_model_name_or_path]

    model = Qwen3Model(args)
    return model
