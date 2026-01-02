import torch
from safetensors.torch import save_file

tensors = {}

scale = 0.01
for i in range(128):
    tensors[f"model.layers.18.mlp.experts.{i}.down_proj.lora_A.weight"] = torch.randn((8, 768)) * scale
    tensors[f"model.layers.18.mlp.experts.{i}.down_proj.lora_B.weight"] = torch.randn((2048, 8)) * scale
    tensors[f"model.layers.18.mlp.experts.{i}.up_proj.lora_A.weight"] = torch.randn((8, 2048)) * scale
    tensors[f"model.layers.18.mlp.experts.{i}.up_proj.lora_B.weight"] = torch.randn((768, 8)) * scale
    tensors[f"model.layers.18.mlp.experts.{i}.gate_proj.lora_A.weight"] = torch.randn((8, 2048)) * scale
    tensors[f"model.layers.18.mlp.experts.{i}.gate_proj.lora_B.weight"] = torch.randn((768, 8)) * scale

for i in range(128):
    tensors[f"model.layers.0.mlp.experts.{i}.down_proj.lora_A.weight"] = torch.randn((8, 768)) * scale
    tensors[f"model.layers.0.mlp.experts.{i}.down_proj.lora_B.weight"] = torch.randn((2048, 8)) * scale
    tensors[f"model.layers.0.mlp.experts.{i}.up_proj.lora_A.weight"] = torch.randn((8, 2048)) * scale
    tensors[f"model.layers.0.mlp.experts.{i}.up_proj.lora_B.weight"] = torch.randn((768, 8)) * scale
    tensors[f"model.layers.0.mlp.experts.{i}.gate_proj.lora_A.weight"] = torch.randn((8, 2048)) * scale
    tensors[f"model.layers.0.mlp.experts.{i}.gate_proj.lora_B.weight"] = torch.randn((768, 8)) * scale

save_file(tensors, "trial_broadcast/adapter_model.safetensors")

