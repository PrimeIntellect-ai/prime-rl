# vLLM Exploration Notes

This file contains findings from exploring the vLLM library codebase to understand its implementation details.

---

## LoRA Module Naming and Validation - 2026-01-02

### Question
How does vLLM determine expected target modules for LoRA? Why don't the expected FQNs include layer numbers (e.g., `experts.0.gate_proj` instead of `model.layers.0.mlp.experts.0.gate_proj`)? Is the same LoRA adapter shared across all layers, or does each layer have its own LoRA weights? How does vLLM parse LoRA adapter safetensor files and map the keys to model modules?

### Investigation Path
- Files examined:
  - `/vllm/lora/worker_manager.py` (lines 85-141)
  - `/vllm/lora/models.py` (lines 153-286, 493-573, 674-681)
  - `/vllm/lora/utils.py` (lines 127-168, 213-234)
  - `/vllm/lora/lora_weights.py`

- Key classes/functions:
  - `WorkerLoRAManager._load_adapter()` - Constructs expected module list
  - `get_supported_lora_modules()` - Extracts supported modules from model
  - `parse_fine_tuned_lora_name()` - Parses safetensor keys to module names
  - `LoRAModel.from_local_checkpoint()` - Loads and validates LoRA weights
  - `LoRAModelManager._match_target_modules()` - Matches modules to LoRA layers

### Findings

#### 1. How Expected Target Modules Are Determined

The expected target modules are built in `WorkerLoRAManager._load_adapter()` (worker_manager.py:87-97):

```python
supported_lora_modules = self._adapter_manager.supported_lora_modules
packed_modules_mapping = self._adapter_manager.packed_modules_mapping
expected_lora_lst: list[str] = []
for module in supported_lora_modules:
    if module in packed_modules_mapping:
        expected_lora_lst.extend(packed_modules_mapping[module])
    else:
        expected_lora_lst.append(module)
    if module == "experts":
        expected_lora_lst.append(module)
expected_lora_modules = set(expected_lora_lst)
```

The `supported_lora_modules` come from `get_supported_lora_modules()` (utils.py:213-234):

```python
def get_supported_lora_modules(model: nn.Module) -> list[str]:
    supported_lora_modules: set[str] = set()
    for name, module in model.named_modules():
        # get the embedding modules
        embedding_modules = getattr(module, "embedding_modules", None)
        if embedding_modules is not None:
            for name in embedding_modules:
                supported_lora_modules.add(name)

        # get all the linear suffixes
        if isinstance(module, (LinearBase,)):
            supported_lora_modules.add(name.split(".")[-1])  # ONLY LAST COMPONENT!

        if isinstance(module, (FusedMoE,)):
            supported_lora_modules.add(name.split(".")[-1])

    return list(supported_lora_modules)
```

**Key insight:** Only the **last component** of the module name is extracted (e.g., `"q_proj"`, `"experts"`). Layer numbers are intentionally omitted.

#### 2. LoRA Weight Sharing Across Layers

**YES - The same LoRA adapter is shared across ALL layers.** This is by design.

Evidence from `LoRAModelManager._create_lora_modules()` (models.py:493-573):

```python
for module_name, module in self.model.named_modules(remove_duplicate=False):
    if isinstance(module, PPMissingLayer):
        continue

    if not self._match_target_modules(module_name):
        continue

    # Creates LoRA wrapper for this module
    new_module = replace_submodule(
        self.model,
        module_name,
        from_layer(module, self.lora_slots, self.lora_config, ...)
    )
    self.register_module(module_name, new_module)
```

The `_match_target_modules()` method (models.py:674-681):

```python
def _match_target_modules(self, module_name: str):
    return any(
        re.match(
            r".*\.{target_module}$".format(target_module=target_module),
            module_name
        )
        or target_module == module_name
        for target_module in self.supported_lora_modules
    )
```

This matches **all modules** whose name ends with a supported suffix. For example:
- If `supported_lora_modules` contains `"q_proj"`
- It matches: `model.layers.0.self_attn.q_proj`, `model.layers.1.self_attn.q_proj`, etc.
- All of these modules share the **same LoRA weights**

**Why this design?**
- Memory efficiency: Only one copy of LoRA weights in memory
- Typical use case: Fine-tuning adapts the model uniformly across layers
- Alternative: If you need per-layer LoRA, vLLM doesn't support this

#### 3. Safetensor Parsing and Validation

**Step 1: Load safetensor file** (models.py:230-242):

```python
with safetensors.safe_open(lora_tensor_path, framework="pt") as f:
    check_unexpected_modules(f)
    for module in f.keys():
        tensors[module] = f.get_tensor(module)
```

**Step 2: Parse each key** via `parse_fine_tuned_lora_name()` (utils.py:127-168):

```python
def parse_fine_tuned_lora_name(
    name: str, weights_mapper: Optional["WeightsMapper"] = None
) -> tuple[str, bool]:
    # Handle optional base_model.model. prefix
    if name.startswith("base_model.model."):
        name = name.replace("base_model.model.", "")
        name = weights_mapper._map_name(name) if weights_mapper else name
        name = "base_model.model." + name
    else:
        name = weights_mapper._map_name(name) if weights_mapper else name

    # Determine start index for slicing
    start_index = 2 if name.startswith("base_model.model.") else 0

    parts = name.split(".")
    # Remove .lora_A.weight or .lora_B.weight suffix
    if parts[-1] == "weight" and (parts[-2] == "lora_A" or parts[-2] == "lora_B"):
        new_name = ".".join(parts[start_index:-2])
        return new_name, parts[-2] == "lora_A"

    # Handle .lora_embedding_A or .lora_embedding_B
    if parts[-1] == "lora_embedding_A" or parts[-1] == "lora_embedding_B":
        new_name = ".".join(parts[start_index:-1])
        return new_name, parts[-1] == "lora_embedding_A"
```

**Example transformations:**

| Input Key | After Parsing | Notes |
|-----------|--------------|-------|
| `base_model.model.layers.0.self_attn.q_proj.lora_A.weight` | `layers.0.self_attn.q_proj` | Standard case |
| `model.layers.0.self_attn.q_proj.lora_A.weight` | `model.layers.0.self_attn.q_proj` | No base_model prefix |
| `base_model.model.layers.0.mlp.experts.0.gate_proj.lora_A.weight` | `layers.0.mlp.experts.0.gate_proj` | MoE expert |

**Step 3: Validate against expected modules** (models.py:188-213):

```python
def check_unexpected_modules(modules: dict):
    for lora_module in modules.keys():
        if is_base_embeddding_weights(lora_module):
            continue
        if "base_layer" in lora_module:
            continue  # Special PEFT format handling

        module_name, _ = parse_fine_tuned_lora_name(lora_module, weights_mapper)

        # Special handling for expert modules
        if ".experts" in module_name:
            expert_idx = module_name.find(".experts")
            expert_suffix = module_name[expert_idx + 1:]
            # For "layers.0.mlp.experts.0.gate_proj", extracts "experts.0.gate_proj"
            if expert_suffix not in expected_lora_modules:
                unexpected_modules.append(module_name)

        # Standard modules: check if last component is expected
        elif module_name.rsplit(".", 1)[-1] not in expected_lora_modules:
            unexpected_modules.append(module_name)

    if unexpected_modules:
        raise ValueError(
            f"While loading {lora_dir}, expected"
            f" target modules in {expected_lora_modules}"
            f" but received {unexpected_modules}."
        )
```

**For MoE expert modules:**
- Finds `.experts` in the parsed name
- Extracts everything after `.experts` as the "expert suffix"
- Example: `layers.0.mlp.experts.0.gate_proj` → suffix is `experts.0.gate_proj`
- This suffix must be in `expected_lora_modules`

#### 4. Common Naming Issues

**Issue: Duplicate "experts" in path**

If your safetensor keys are:
```
base_model.model.layers.0.mlp.experts.experts.0.gate_proj.lora_A.weight
```

After parsing:
```
layers.0.mlp.experts.experts.0.gate_proj
```

After finding `.experts`:
```
expert_suffix = "experts.0.gate_proj"
```

But vLLM expects: `experts.0.gate_proj` in the expected modules, which it finds. However, the double `experts` indicates a mismatch in how the model was structured vs. how the LoRA was saved.

**Correct naming should be:**
```
base_model.model.layers.0.mlp.experts.0.gate_proj.lora_A.weight
```

This parses to `layers.0.mlp.experts.0.gate_proj`, with suffix `experts.0.gate_proj`.

### Key Insights

1. **Module suffixes, not full paths**: vLLM expects only the suffix of module names (e.g., `"q_proj"`, `"experts.0.gate_proj"`) because LoRA weights are shared across all matching modules.

2. **Shared LoRA across layers**: The same LoRA adapter is applied to all layers that match the target module pattern. This is memory-efficient but means you cannot have per-layer LoRA weights.

3. **Flexible prefix handling**: The `base_model.model.` prefix is optional and automatically handled by the parser.

4. **Expert-specific validation**: For MoE models, vLLM extracts the suffix after `.experts` and validates it against expected modules.

5. **Match by regex suffix**: Module matching uses `r".*\.{target_module}$"` regex, so `"q_proj"` matches any module ending in `.q_proj`.

6. **No per-layer LoRA support**: vLLM's architecture doesn't support different LoRA weights for different layers - all layers share the same adapter weights.

### Critical Code Locations

| Component | File Path | Lines |
|-----------|-----------|-------|
| Expected module construction | `/vllm/lora/worker_manager.py` | 85-141 |
| Supported module extraction | `/vllm/lora/utils.py` | 213-234 |
| LoRA checkpoint loading | `/vllm/lora/models.py` | 153-286 |
| Name parsing | `/vllm/lora/utils.py` | 127-168 |
| Module matching | `/vllm/lora/models.py` | 674-681 |
| LoRA layer creation | `/vllm/lora/models.py` | 493-573 |
| Validation logic | `/vllm/lora/models.py` | 188-213 |

---

## MoE LoRA Loading and Layer Support - 2026-01-01

### Question
How does vLLM's LoRA manager handle loading LoRAs into MoE (Mixture of Experts) layers? What FQN format does it expect? Which layers within MoE does it support LoRA for?

### Investigation Path
- Files examined:
  - `/vllm/lora/layers/fused_moe.py` (lines 45-748)
  - `/vllm/lora/models.py` (lines 289-904)
  - `/vllm/lora/utils.py` (lines 70-75, 127-168, 283-307)
  - `/vllm/lora/lora_weights.py` (lines 156-206)

- Key classes/functions:
  - `FusedMoEWithLoRA` - Standard MoE LoRA implementation
  - `FusedMoE3DWithLoRA` - 3D fused weight MoE LoRA implementation
  - `parse_fine_tuned_lora_name()` - Name parsing logic
  - `PackedLoRALayerWeights.pack_moe()` - Weight packing for MoE

### Findings

#### 1. LoRA Loading Mechanism
vLLM handles MoE LoRA loading through specialized classes in `fused_moe.py`:
- **FusedMoEWithLoRA**: For models where w1 (gate_proj) and w3 (up_proj) are stored separately
  - Expects 3 tensors per expert: w1, w2, w3
- **FusedMoE3DWithLoRA**: For models where w1 and w3 are fused on disk
  - Expects 2 tensors: w13 (fused), w2

LoRA weights are stacked per expert with shape: `(max_loras, num_experts, rank, hidden_size)`

#### 2. FQN (Fully Qualified Name) Format

**IMPORTANT: The `base_model.` prefix is OPTIONAL**

The parser handles both cases via `parse_fine_tuned_lora_name()`:
```python
# Dynamically determines start index based on prefix presence
start_index = 2 if name.startswith("base_model.model.") else 0
```

**Accepted formats:**

✅ **Standard format (separate w1/w3):**
```
base_model.model.layers.{layer_id}.mlp.experts.{expert_id}.gate_proj.lora_A.weight
base_model.model.layers.{layer_id}.mlp.experts.{expert_id}.gate_proj.lora_B.weight
base_model.model.layers.{layer_id}.mlp.experts.{expert_id}.down_proj.lora_A.weight
base_model.model.layers.{layer_id}.mlp.experts.{expert_id}.down_proj.lora_B.weight
base_model.model.layers.{layer_id}.mlp.experts.{expert_id}.up_proj.lora_A.weight
base_model.model.layers.{layer_id}.mlp.experts.{expert_id}.up_proj.lora_B.weight
```

✅ **Without base_model prefix (also works):**
```
model.layers.{layer_id}.mlp.experts.{expert_id}.gate_proj.lora_A.weight
layers.{layer_id}.mlp.experts.{expert_id}.gate_proj.lora_A.weight
```

✅ **3D format (fused w1+w3):**
```
base_model.model.layers.{layer_id}.mlp.experts.base_layer.lora_A.weight  # w13
base_model.model.layers.{layer_id}.mlp.experts.base_layer.lora_B.weight  # w13
base_model.model.layers.{layer_id}.mlp.experts.lora_A.weight              # w2
base_model.model.layers.{layer_id}.mlp.experts.lora_B.weight              # w2
```

The only requirement: After stripping LoRA suffixes and optional prefix, the resulting module name must match the model's actual module hierarchy from `model.named_modules()`.

#### 3. Supported Layers

vLLM supports LoRA **only for expert FFN projections**, NOT the router/gate:

**Supported:**
- ✅ `gate_proj` (w1) - First projection in FFN
- ✅ `down_proj` (w2) - Down projection in FFN
- ✅ `up_proj` (w3) - Up projection in FFN

**NOT supported:**
- ❌ Router/gate layer (the layer that routes tokens to experts)
- ❌ Shared experts (if present)

#### 4. Special Handling and Restrictions

**Restrictions:**

1. **Expert Parallelism Not Supported** (`fused_moe.py:50-52`):
   ```python
   assert not self.base_layer.use_ep, (
       "EP support for Fused MoE LoRA is not implemented yet."
   )
   ```

2. **All experts must have LoRA** (`lora_weights.py:175-182`):
   - Currently assumes all experts in a layer have LoRA weights
   - No support for sparse LoRA (some experts without LoRA)

3. **Shared LoRA ranks**: All experts in a layer share the same LoRA rank

**Special Handling:**

1. **Tensor Parallel Slicing** (`fused_moe.py:421-474`):
   - `w13_lora_a`: Sliced along rank dimension when `fully_sharded=True`
   - `w13_lora_b`: Sliced along intermediate_size dimension
   - `w2_lora_a`: Sliced along intermediate_size dimension
   - `w2_lora_b`: Sliced along hidden_size dimension when `fully_sharded=True`

2. **Kernel Integration** (`fused_moe.py:119-312`):
   - LoRA computation injected via decorators into base FusedMoE forward pass
   - Uses custom Triton kernels for efficient computation
   - Applies LoRA during activation and down-projection stages

3. **Weight Packing** (`lora_weights.py:156-206`):
   - MoE LoRAs use `PackedLoRALayerWeights.pack_moe()` instead of regular `pack()`
   - Expects 3N weights (w1, w2, w3 for each of N experts)
   - Stacks expert weights along dimension 0

4. **Model Detection** (`utils.py:70-75`):
   ```python
   def is_moe_model(model: nn.Module) -> bool:
       if any(isinstance(module, FusedMoE) for module in model.modules()):
           logger.info_once("MoE model detected. Using fused MoE LoRA implementation.")
           return True
   ```

5. **3D vs Standard MoE Detection** (`models.py:348`):
   - Checks `model.is_3d_moe_weight` attribute to determine which LoRA class to use

### Key Insights

- vLLM's MoE LoRA support is sophisticated with specialized tensor stacking and kernel integration
- The `base_model.model.` prefix is a convention from PEFT but is NOT required by vLLM
- Expert Parallelism (EP) is not yet supported for MoE LoRA - this is a known limitation
- All experts in a layer must have LoRA weights (no sparse LoRA support yet)
- LoRA is only applied to the expert FFN projections, not to the routing mechanism
- Two implementation classes exist to handle different weight storage formats (separate vs fused w1/w3)
- The implementation efficiently handles per-expert LoRA through 4D tensor stacking: `(max_loras, num_experts, rank, hidden_size)`

### Critical Code Locations

| Component | File Path | Lines |
|-----------|-----------|-------|
| MoE LoRA Layer | `/vllm/lora/layers/fused_moe.py` | 45-748 |
| LoRA Model Manager | `/vllm/lora/models.py` | 289-904 |
| MoE Detection | `/vllm/lora/utils.py` | 70-75, 283-307 |
| Weight Packing | `/vllm/lora/lora_weights.py` | 156-206 |
| Name Parsing | `/vllm/lora/utils.py` | 127-168 |
| Activation Logic | `/vllm/lora/models.py` | 393-456 |

---
