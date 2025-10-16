from pathlib import Path
import json
from typing import TYPE_CHECKING

import torch
from torch.nn import Module
from vllm.model_executor.model_loader import DefaultModelLoader, get_model_loader
from vllm.model_executor.model_loader.utils import process_weights_after_loading

# This is to get type hints for the Worker class but not actually extend it at runtime as this is required by vLLM worker extension
if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

    Worker = Worker
else:
    Worker = object


class CheckpointWorker(Worker):
    """
    This is an extension of a vLLM worker that allows for loading checkpoints
    from a specified directory via RPC calls from the AsyncLLMEngine class, exposed
    by the vLLM server. This is useful in RL training, where we want to load the
    recent policy model from a checkpoint directory.
    """

    def update_weights(self, weight_path: str) -> None:
        """Update weights from a specified path, detecting LoRA adapters automatically."""
        weight_path_obj = Path(weight_path)
        step_dir = weight_path_obj.parent
        lora_dir = step_dir / "lora_adapters"
        
        # Check if this is an adapter-based checkpoint
        if lora_dir.exists() and lora_dir.is_dir():
            self._update_weights_with_lora(weight_path)
        else:
            self._update_weights_native_vllm(weight_path)

    def _update_weights_with_lora(self, model_path: str) -> None:
        """Update weights using LoRA adapter merging approach."""
        model_path = Path(model_path)
        vllm_dtype = next(self.model_runner.model.parameters()).dtype
        
        step_dir = model_path.parent
        lora_dir = step_dir / "lora_adapters"
        
        if not hasattr(self, 'cpu_state'):
            self.cpu_state = {k: v.cpu().clone() for k, v in self.model_runner.model.state_dict().items()}
            self.previous_lora_deltas = {}
        
        previous_lora_deltas = self.previous_lora_deltas
        
        if previous_lora_deltas:
            for key, prev_delta in previous_lora_deltas.items():
                if key in self.cpu_state:
                    self.cpu_state[key] = self.cpu_state[key] - prev_delta
        
        lora_files = list(lora_dir.glob("*.bin"))
        if lora_files:
            config_path = lora_dir / "adapter_config.json"
            modules_to_save = []
            
            if config_path.exists():
                with open(config_path) as f:
                    adapter_config = json.load(f)
                    alpha = adapter_config["lora_alpha"]
                    rank = adapter_config["r"]
                    scaling = alpha / rank
                    modules_to_save = adapter_config.get("modules_to_save", [])
            else:
                lora_files = []
            
            new_lora_deltas = {}
            lora_adapted_keys = set()
            modules_to_save_weights = {}
            
            for lora_file in lora_files:
                lora_state = torch.load(lora_file, map_location="cpu", mmap=True)
                lora_state = {k: v.to(dtype=vllm_dtype) for k, v in lora_state.items()}
                
                lora_adapters = {}
                for key in lora_state.keys():
                    if key.startswith("base_model.model."):
                        clean_key = key.replace("base_model.model.", "")
                        
                        if ".lora_A.weight" in clean_key:
                            base_key = clean_key.replace(".lora_A.weight", "")
                            if base_key not in lora_adapters:
                                lora_adapters[base_key] = {}
                            lora_adapters[base_key]["A"] = lora_state[key]
                        elif ".lora_B.weight" in clean_key:
                            base_key = clean_key.replace(".lora_B.weight", "")
                            if base_key not in lora_adapters:
                                lora_adapters[base_key] = {}
                            lora_adapters[base_key]["B"] = lora_state[key]
                        else:
                            is_module_to_save = False
                            for module_name in modules_to_save:
                                if f".{module_name}." in clean_key or clean_key.endswith(f".{module_name}.weight") or clean_key.endswith(f".{module_name}.bias"):
                                    is_module_to_save = True
                                    break
                            
                            if is_module_to_save:
                                modules_to_save_weights[clean_key] = lora_state[key]
                
                merged_count = 0
                qkv_sizes = {}
                
                for layer_key, adapter in lora_adapters.items():
                    if "A" not in adapter or "B" not in adapter:
                        continue
                    
                    delta = (adapter["B"] @ adapter["A"]) * scaling
                    
                    if ".self_attn.q_proj" in layer_key:
                        layer_prefix = layer_key.replace(".self_attn.q_proj", "")
                        k_key = f"{layer_prefix}.self_attn.k_proj"
                        v_key = f"{layer_prefix}.self_attn.v_proj"
                        
                        if k_key in lora_adapters and v_key in lora_adapters:
                            if "A" in lora_adapters[k_key] and "B" in lora_adapters[k_key]:
                                if "A" in lora_adapters[v_key] and "B" in lora_adapters[v_key]:
                                    delta_q = delta
                                    delta_k = (lora_adapters[k_key]["B"] @ lora_adapters[k_key]["A"]) * scaling
                                    delta_v = (lora_adapters[v_key]["B"] @ lora_adapters[v_key]["A"]) * scaling
                                    
                                    q_size = delta_q.shape[0]
                                    k_size = delta_k.shape[0]
                                    v_size = delta_v.shape[0]
                                    qkv_sizes[layer_prefix] = (q_size, k_size, v_size)
                                    
                                    fused_delta = torch.cat([delta_q, delta_k, delta_v], dim=0)
                                    
                                    fused_key = f"{layer_prefix}.self_attn.qkv_proj.weight"
                                    if fused_key in self.cpu_state:
                                        self.cpu_state[fused_key] = self.cpu_state[fused_key] + fused_delta
                                        new_lora_deltas[fused_key] = fused_delta
                                        lora_adapted_keys.add(fused_key)
                                        merged_count += 1
                    
                    elif ".self_attn.k_proj" in layer_key or ".self_attn.v_proj" in layer_key:
                        continue
                    
                    elif ".mlp.gate_proj" in layer_key:
                        layer_prefix = layer_key.replace(".mlp.gate_proj", "")
                        up_key = f"{layer_prefix}.mlp.up_proj"
                        
                        if up_key in lora_adapters:
                            if "A" in lora_adapters[up_key] and "B" in lora_adapters[up_key]:
                                delta_gate = delta
                                delta_up = (lora_adapters[up_key]["B"] @ lora_adapters[up_key]["A"]) * scaling
                                
                                fused_delta = torch.cat([delta_gate, delta_up], dim=0)
                                
                                fused_key = f"{layer_prefix}.mlp.gate_up_proj.weight"
                                if fused_key in self.cpu_state:
                                    self.cpu_state[fused_key] = self.cpu_state[fused_key] + fused_delta
                                    new_lora_deltas[fused_key] = fused_delta
                                    lora_adapted_keys.add(fused_key)
                                    merged_count += 1
                    
                    elif ".mlp.up_proj" in layer_key:
                        continue
                    
                    else:
                        weight_key = f"{layer_key}.weight"
                        if weight_key in self.cpu_state:
                            self.cpu_state[weight_key] = self.cpu_state[weight_key] + delta
                            new_lora_deltas[weight_key] = delta
                            lora_adapted_keys.add(weight_key)
                            merged_count += 1
            
            self.previous_lora_deltas = new_lora_deltas
            
            state_dict = {}
            for key, value in self.cpu_state.items():
                if ".self_attn.qkv_proj.weight" in key:
                    layer_prefix = key.replace(".self_attn.qkv_proj.weight", "")
                    
                    if layer_prefix in qkv_sizes:
                        q_size, k_size, v_size = qkv_sizes[layer_prefix]
                        
                        q_weight = value[:q_size, :]
                        k_weight = value[q_size:q_size+k_size, :]
                        v_weight = value[q_size+k_size:, :]
                        
                        state_dict[f"{layer_prefix}.self_attn.q_proj.weight"] = q_weight
                        state_dict[f"{layer_prefix}.self_attn.k_proj.weight"] = k_weight
                        state_dict[f"{layer_prefix}.self_attn.v_proj.weight"] = v_weight
                    else:
                        state_dict[key] = value
                
                elif ".mlp.gate_up_proj.weight" in key:
                    layer_prefix = key.replace(".mlp.gate_up_proj.weight", "")
                    mid = value.shape[0] // 2
                    
                    gate_weight = value[:mid, :]
                    up_weight = value[mid:, :]
                    
                    state_dict[f"{layer_prefix}.mlp.gate_proj.weight"] = gate_weight
                    state_dict[f"{layer_prefix}.mlp.up_proj.weight"] = up_weight
                
                else:
                    state_dict[key] = value
            
            for module_key, module_weight in modules_to_save_weights.items():
                if module_key in state_dict:
                    state_dict[module_key] = module_weight
        else:
            state_dict = self.cpu_state
        
        # Load the merged state dict into the model
        self.model_runner.model.load_state_dict(state_dict, strict=False)

    def _update_weights_native_vllm(self, weight_path: str) -> None:
        """Update weights using vLLM's native model loader."""
        # Get vLLM model runner and model
        model_runner = self.model_runner
        model = model_runner.model
        assert isinstance(model, Module)

        # Get vLLM model loader
        model_loader = get_model_loader(self.load_config)
        assert isinstance(model_loader, DefaultModelLoader)
        local_source = DefaultModelLoader.Source(
            weight_path,
            revision=None,  # TODO: Check that this is correct or if we should use the default (model_config.revision)
            prefix="",
            fall_back_to_pt=getattr(model, "fall_back_to_pt_during_load", True),
            allow_patterns_overrides=getattr(model, "allow_patterns_overrides", None),
        )
        weights_iterator = model_loader._get_weights_iterator(local_source)
        model.load_weights(weights_iterator)  # type: ignore

        # Process weights after loading (important for some models)
        device = next(model.parameters()).device
        process_weights_after_loading(model, self.model_runner.model_config, device)
        
        # Clean up LoRA state if we're switching back to full checkpoints
        if hasattr(self, 'cpu_state'):
            delattr(self, 'cpu_state')
        if hasattr(self, 'previous_lora_deltas'):
            delattr(self, 'previous_lora_deltas')