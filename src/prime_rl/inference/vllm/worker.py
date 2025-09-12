from pathlib import Path
import torch
from typing import Dict, Tuple, Iterator, Any
from concurrent.futures import ThreadPoolExecutor
import threading


def _unpack_5bit_values(packed_data: torch.Tensor, original_shape: tuple) -> torch.Tensor:
    """
    Unpack 6-bit values from packed bytes using vectorized operations.
    (Function name kept as 5bit for compatibility, but actually handles 6-bit)
    """
    # Ensure we have complete 3-byte chunks
    n_complete_chunks = len(packed_data) // 3
    if n_complete_chunks == 0:
        # Handle edge case of very small tensors
        return torch.zeros(original_shape, dtype=torch.int8, device=packed_data.device) - 31
    
    # Reshape to (N, 3) chunks for vectorized operations
    complete_data = packed_data[:n_complete_chunks * 3]
    chunks = complete_data.view(-1, 3)
    
    # Vectorized bit unpacking
    # Extract 4 6-bit values from 3 bytes
    val0 = (chunks[:, 0] >> 2) & 0x3F
    val1 = ((chunks[:, 0] & 0x3) << 4) | ((chunks[:, 1] >> 4) & 0xF)
    val2 = ((chunks[:, 1] & 0xF) << 2) | ((chunks[:, 2] >> 6) & 0x3)
    val3 = chunks[:, 2] & 0x3F
    
    # Stack and flatten to get unpacked values
    unpacked = torch.stack([val0, val1, val2, val3], dim=1).flatten()
    
    # Convert back to signed range [-31, 31] and trim to original size
    unpacked = unpacked.to(torch.int8) - 31
    total_elements = torch.prod(torch.tensor(original_shape)).item()
    unpacked = unpacked[:total_elements]
    
    return unpacked.reshape(original_shape)


def _dequantize_from_q5_0_per_channel_gpu(
    packed_tensor: torch.Tensor,
    scales: torch.Tensor,
    original_shape: tuple,
    target_dtype: torch.dtype = torch.bfloat16,
    target_device: torch.device = None,
    tensor_name: str = "unknown",
) -> torch.Tensor:
    """
    Dequantize Q6_0 tensor with per-channel scales back to target precision on GPU.
    Assumes packed_tensor and scales are already on target device.
    """
    # Validation
    assert packed_tensor.dtype == torch.uint8, f"Expected uint8 packed tensor, got {packed_tensor.dtype}"
    assert scales.dtype == torch.float32, f"Expected float32 scales, got {scales.dtype}"
    
    # Unpack the 6-bit values (now on GPU)
    quantized_tensor = _unpack_5bit_values(packed_tensor, original_shape)
    
    # Validate quantized tensor range
    quant_min = quantized_tensor.min().item()
    quant_max = quantized_tensor.max().item()
    assert -31 <= quant_min <= quant_max <= 31, f"Quantized tensor values out of expected range: [{quant_min}, {quant_max}]"
    
    # Convert to float32 for computation
    quantized_f32 = quantized_tensor.to(torch.float32)
    
    # Handle per-tensor case (single scale)
    if len(scales) == 1:
        scale = scales[0].item()
        dequantized = quantized_f32 * scale
        return dequantized.to(target_dtype)
    
    # Handle per-channel groups
    assert len(original_shape) >= 2, f"Multi-scale quantization requires 2D+ tensors, got {len(original_shape)}D"
    
    out_channels = original_shape[0]
    num_groups = len(scales)
    channels_per_group = max(1, out_channels // num_groups)
    
    # Verify we have the right number of groups
    expected_groups = (out_channels + channels_per_group - 1) // channels_per_group
    assert num_groups == expected_groups, \
        f"Scale count {num_groups} doesn't match expected groups {expected_groups} for {out_channels} channels"
    
    # Preallocate on target device with target dtype
    dequantized = torch.empty(original_shape, dtype=target_dtype, device=quantized_f32.device)
    
    # Apply scales per channel group
    for group_idx in range(num_groups):
        start_ch = group_idx * channels_per_group
        end_ch = min((group_idx + 1) * channels_per_group, out_channels)
        
        # Strict bounds checking
        assert start_ch < out_channels, f"Group {group_idx} start channel {start_ch} >= {out_channels}"
        assert end_ch <= out_channels, f"Group {group_idx} end channel {end_ch} > {out_channels}"
        
        # Get scale for this group
        scale = scales[group_idx].item()
        
        # Dequantize: x = q * scale (symmetric quantization)
        group_quantized = quantized_f32[start_ch:end_ch]
        group_dequantized = (group_quantized * scale).to(target_dtype)
        dequantized[start_ch:end_ch] = group_dequantized
    
    return dequantized


class ParallelDequantizationBatch:
    """Manages a batch of tensors for parallel GPU dequantization."""
    
    def __init__(self, target_device: torch.device, target_dtype: torch.dtype):
        self.target_device = target_device
        self.target_dtype = target_dtype
        self.quantized_tensors: Dict[str, torch.Tensor] = {}
        self.scales: Dict[str, torch.Tensor] = {}
        self.shapes: Dict[str, tuple] = {}
        self.regular_tensors: Dict[str, torch.Tensor] = {}
        
    def add_quantized_tensor(self, key: str, packed_tensor: torch.Tensor, scales: torch.Tensor, shape: tuple):
        """Add a quantized tensor to the batch."""
        self.quantized_tensors[key] = packed_tensor
        self.scales[key] = scales
        self.shapes[key] = shape
        
    def add_regular_tensor(self, key: str, tensor: torch.Tensor):
        """Add a regular (non-quantized) tensor to the batch."""
        self.regular_tensors[key] = tensor
        
    def transfer_to_gpu_and_dequantize(self) -> Dict[str, torch.Tensor]:
        """Transfer all tensors to GPU and dequantize in parallel."""
        results = {}
        
        # Handle regular tensors - transfer and convert dtype if needed
        for key, tensor in self.regular_tensors.items():
            gpu_tensor = tensor.to(device=self.target_device, non_blocking=True)
            if gpu_tensor.dtype != self.target_dtype and gpu_tensor.dtype in [torch.float32, torch.float16, torch.bfloat16]:
                gpu_tensor = gpu_tensor.to(self.target_dtype)
            results[key] = gpu_tensor
        
        if not self.quantized_tensors:
            return results
            
        print(f"Transferring {len(self.quantized_tensors)} quantized tensors to GPU for parallel dequantization...")
        
        # Transfer all quantized tensors and scales to GPU
        gpu_quantized = {}
        gpu_scales = {}
        
        for key in self.quantized_tensors.keys():
            gpu_quantized[key] = self.quantized_tensors[key].to(device=self.target_device, non_blocking=True)
            gpu_scales[key] = self.scales[key].to(device=self.target_device, non_blocking=True)
        
        # Synchronize to ensure all transfers are complete
        if self.target_device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Dequantize all tensors in parallel using ThreadPoolExecutor
        # PyTorch will handle the actual GPU parallelism
        def dequantize_single(key: str) -> Tuple[str, torch.Tensor]:
            try:
                dequantized = _dequantize_from_q5_0_per_channel_gpu(
                    gpu_quantized[key],
                    gpu_scales[key], 
                    self.shapes[key],
                    self.target_dtype,
                    self.target_device,
                    tensor_name=key
                )
                return key, dequantized
            except Exception as e:
                print(f"ERROR: Failed to dequantize tensor {key}: {e}")
                raise RuntimeError(f"Failed to dequantize tensor {key}: {e}")
        
        # Use ThreadPoolExecutor to launch dequantization tasks
        # The actual compute will be parallel on GPU
        with ThreadPoolExecutor(max_workers=min(8, len(self.quantized_tensors))) as executor:
            dequant_futures = [executor.submit(dequantize_single, key) for key in self.quantized_tensors.keys()]
            
            for future in dequant_futures:
                key, dequantized_tensor = future.result()
                
                # Validate dequantized tensor
                expected_shape = self.shapes[key]
                assert dequantized_tensor.shape == expected_shape, \
                    f"Shape mismatch after dequantization: {dequantized_tensor.shape} != {expected_shape}"
                assert dequantized_tensor.dtype == self.target_dtype, \
                    f"Dtype mismatch after dequantization: {dequantized_tensor.dtype} != {self.target_dtype}"
                
                results[key] = dequantized_tensor
        
        print(f"Parallel dequantization complete for {len(self.quantized_tensors)} tensors")
        return results


class CheckpointWorker:
    """
    This is an extension of a vLLM worker that allows for loading checkpoints
    from a specified directory via RPC calls from the AsyncLLMEngine class, exposed
    by the vLLM server. This is useful in RL training, where we want to load the
    recent policy model from a checkpoint directory.
    """

    def update_weights(self, model_path: Path) -> None:
        """Update weights from a specified path pointing to a .pt file."""
        print(f"Loading checkpoint from {model_path}")
        state_dict = torch.load(model_path, map_location="cpu", mmap=True)
        
        # Get target dtype and device from model parameters  
        target_dtype = next(self.model_runner.model.parameters()).dtype
        target_device = next(self.model_runner.model.parameters()).device

        def weights_iterator() -> Iterator[Tuple[str, torch.Tensor]]:
            # Phase 1: Collect all tensors into batches
            batch = ParallelDequantizationBatch(target_device, target_dtype)
            scales_dict = {}
            shapes_dict = {}
            quantized_count = 0
            total_count = 0
            
            print("Phase 1: Collecting scales and shapes...")
            # Collect all scales and shapes
            for key, value in state_dict.items():
                if key.endswith("_scales"):
                    original_key = key[:-7]  # Remove "_scales" suffix
                    scales_dict[original_key] = value
                elif key.endswith("_shape"):
                    original_key = key[:-6]  # Remove "_shape" suffix
                    shapes_dict[original_key] = tuple(value.tolist())
            
            print("Phase 2: Batching tensors for parallel processing...")
            # Batch all tensors
            for key, value in state_dict.items():
                if not key or key.endswith("_scales") or key.endswith("_shape"):
                    continue
                
                total_count += 1
                
                # Check if this tensor was quantized
                if key in scales_dict and key in shapes_dict:
                    batch.add_quantized_tensor(key, value, scales_dict[key], shapes_dict[key])
                    quantized_count += 1
                else:
                    batch.add_regular_tensor(key, value)
            
            print(f"Phase 3: Transferring to GPU and parallel dequantization...")
            print(f"Processing {quantized_count} quantized + {total_count - quantized_count} regular tensors")
            
            # Phase 3: Transfer to GPU and dequantize everything in parallel
            all_tensors = batch.transfer_to_gpu_and_dequantize()
            
            # Phase 4: Yield results in original order
            print("Phase 4: Yielding processed tensors...")
            for key, value in state_dict.items():
                if not key or key.endswith("_scales") or key.endswith("_shape"):
                    continue
                
                if key in all_tensors:
                    yield key, all_tensors[key]
            
            # Log final summary
            if quantized_count > 0:
                print(f"Successfully dequantized {quantized_count}/{total_count} tensors from Q6_0 to {target_dtype} in parallel on GPU")

        # Clean up state_dict from CPU memory early since we've moved everything to GPU
        self.model_runner.model.load_weights(weights_iterator())
        
        # Clear the state_dict to free CPU memory
        del state_dict

        # Process weights after loading (important for some models)
        from vllm.model_executor.model_loader.utils import process_weights_after_loading

        process_weights_after_loading(self.model_runner.model, self.model_runner.model_config, target_device)
        print("Parallel weight loading complete")