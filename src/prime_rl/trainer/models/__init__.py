from typing import Any

import torch.distributed.checkpoint as dcp
import torch.nn as nn
from huggingface_hub import snapshot_download
from torch.distributed.checkpoint import HuggingFaceStorageReader
from torch.distributed.checkpoint.state_dict import get_model_state_dict

from prime_rl.trainer.models.qwen3 import Qwen3Model, Qwen3ModelArgs, Qwen3StateDictAdapter, qwen3_configs

all_configs: dict[str, tuple[Qwen3ModelArgs, type[Qwen3Model], type[Qwen3StateDictAdapter]]] = {
    key: (value, Qwen3Model, Qwen3StateDictAdapter) for key, value in qwen3_configs.items()
}


# I am sorry but I cannot implement a class that return another class via init. It's simply against my religion
# sorry hf folks, sorry jackmin but Guido is looking at me
def from_pretrained(pretrained_model_name_or_path: str, config: Any | None = None) -> nn.Module:
    if pretrained_model_name_or_path not in qwen3_configs:
        raise NotImplementedError(
            f"Model {pretrained_model_name_or_path} is not implemented with Prime Rl AutoModel, please use --model-impl hf instead"
        )

    # base_model_state_dict = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, config=config, trust_remote_code=trust_remote_code).state_dict()

    args, model_class, state_dict_adapter = all_configs[pretrained_model_name_or_path]

    model = model_class(args)
    sd_adapter = state_dict_adapter(args, pretrained_model_name_or_path)

    model_state_dict = get_model_state_dict(model)

    hf_state_dict = sd_adapter.to_hf(model_state_dict)
    path_snapshot = snapshot_download(repo_id=pretrained_model_name_or_path, repo_type="model")

    # assert HuggingFaceStorageReader require torch nighlty 2.9 to run
    dcp.load(hf_state_dict, storage_reader=HuggingFaceStorageReader(path=path_snapshot))

    state_dict = sd_adapter.from_hf(hf_state_dict)
    model.load_state_dict(state_dict=state_dict)

    return model
