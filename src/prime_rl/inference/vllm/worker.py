from pathlib import Path
import json

import torch


class CheckpointWorker:
    """
    This is an extension of a vLLM worker that allows for loading checkpoints
    from a specified directory via RPC calls from the AsyncLLMEngine class, exposed
    by the vLLM server. This is useful in RL training, where we want to load the
    recent policy model from a checkpoint directory.
    """

    def update_weights(self, model_path: Path) -> None:
        """Update weights from a specified path pointing to a .pt file."""
        model_path = Path(model_path)
        vllm_dtype = next(self.model_runner.model.parameters()).dtype
        
        step_dir = model_path.parent
        lora_dir = step_dir / "lora_adapters"
        
        if lora_dir.exists() and lora_dir.is_dir():
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
        else:
            state_dict = torch.load(model_path, map_location="cpu", mmap=True)
            state_dict = {k: v.to(dtype=vllm_dtype) for k, v in state_dict.items()}
            
            if hasattr(self, 'cpu_state'):
                delattr(self, 'cpu_state')
            if hasattr(self, 'previous_lora_deltas'):
                delattr(self, 'previous_lora_deltas')

        def weights_iterator():
            for key, value in state_dict.items():
                if not key:
                    continue
                yield key, value

        self.model_runner.model.load_weights(weights_iterator())

        # Process weights after loading (important for some models)
        from vllm.model_executor.model_loader.utils import process_weights_after_loading

        device = next(self.model_runner.model.parameters()).device
        process_weights_after_loading(self.model_runner.model, self.model_runner.model_config, device)
