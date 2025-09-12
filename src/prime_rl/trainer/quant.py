"""
Symmetric Q6_0 per-channel quantization module with vectorized bit packing.

Provides functions for quantizing and dequantizing tensors using symmetric 
int6 quantization with per-channel grouping and vectorized bit operations.
"""

import torch


def validate_quantization_input(tensor: torch.Tensor, num_channels: int) -> None:
    """Validate tensor for quantization."""
    assert tensor.dtype in [torch.float32, torch.float16, torch.bfloat16], \
        f"Unsupported tensor dtype: {tensor.dtype}. Must be float32, float16, or bfloat16"
    assert tensor.numel() > 0, "Cannot quantize empty tensor"
    assert len(tensor.shape) >= 2, f"Tensor must be 2D+, got {len(tensor.shape)}D"
    assert tensor.shape[0] >= num_channels, f"Tensor has {tensor.shape[0]} channels, need >= {num_channels}"


def compute_group_ranges(out_channels: int, num_channels: int) -> tuple[int, int]:
    """Compute channel grouping parameters."""
    channels_per_group = max(1, out_channels // num_channels)
    actual_groups = (out_channels + channels_per_group - 1) // channels_per_group
    return channels_per_group, actual_groups


def quantize_group(group_tensor: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Quantize a single channel group using symmetric 6-bit quantization."""
    group_min = group_tensor.min()
    group_max = group_tensor.max()
    
    assert group_min != group_max, f"Group is constant (min=max={group_min:.6f}), cannot quantize"
    
    abs_max = group_tensor.abs().max()
    assert abs_max > 0, "Group has zero absolute maximum"
    
    # 6-bit symmetric quantization: range [-31, 31]
    scale = abs_max / 31.0
    quantized = (group_tensor / scale).round().clamp(-31, 31)
    
    # Validate output
    actual_min = quantized.min().item()
    actual_max = quantized.max().item()
    assert -31 <= actual_min <= actual_max <= 31, \
        f"Quantized values out of range: [{actual_min}, {actual_max}]"
    
    return quantized.to(torch.int8), scale


def pack_5bit_values_vectorized(values: torch.Tensor) -> torch.Tensor:
    """
    Pack 6-bit values into bytes using vectorized operations.
    4 values (6 bits each) -> 3 bytes (24 bits total)
    """
    # Convert to unsigned range [0, 62] for packing
    unsigned_values = (values + 31).flatten().to(torch.uint8)
    
    # Pad to multiple of 4 for packing
    remainder = len(unsigned_values) % 4
    if remainder != 0:
        padding = 4 - remainder
        unsigned_values = torch.cat([unsigned_values, torch.zeros(padding, dtype=torch.uint8, device=unsigned_values.device)])
    
    # Reshape to (N, 4) chunks for vectorized operations
    chunks = unsigned_values.view(-1, 4)
    
    # Vectorized bit packing using tensor operations
    # Pack 4 6-bit values into 3 bytes (24 bits)
    byte0 = (chunks[:, 0] << 2) | (chunks[:, 1] >> 4)
    byte1 = ((chunks[:, 1] & 0xF) << 4) | (chunks[:, 2] >> 2)
    byte2 = ((chunks[:, 2] & 0x3) << 6) | chunks[:, 3]
    
    # Stack and flatten to get final packed bytes
    packed = torch.stack([byte0, byte1, byte2], dim=1).flatten()
    
    return packed


def unpack_5bit_values_vectorized(packed_data: torch.Tensor, original_shape: tuple) -> torch.Tensor:
    """
    Unpack 6-bit values from packed bytes using vectorized operations.
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


def quantize_to_q5_0_per_channel(tensor: torch.Tensor, num_channels: int = 128) -> tuple[torch.Tensor, torch.Tensor, tuple]:
    """
    Quantize tensor to Q6_0 format using symmetric quantization with per-channel grouping and vectorized bit packing.
    
    Mathematical relationship:
    - quantized = round(tensor / scale).clamp(-31, 31)
    - dequantized = quantized * scale
    
    Args:
        tensor: Input tensor to quantize (must be float dtype and 2D+)
        num_channels: Number of channel groups for quantization
        
    Returns:
        Tuple of (packed_tensor as uint8, scales as float32, original_shape)
    """
    validate_quantization_input(tensor, num_channels)
    
    tensor_f32 = tensor.to(torch.float32)
    out_channels = tensor_f32.shape[0]
    original_shape = tensor_f32.shape
    
    channels_per_group, actual_groups = compute_group_ranges(out_channels, num_channels)
    
    scales = torch.zeros(actual_groups, dtype=torch.float32, device=tensor_f32.device)
    quantized = torch.zeros_like(tensor_f32, dtype=torch.int8)
    
    for group_idx in range(actual_groups):
        start_ch = group_idx * channels_per_group
        end_ch = min((group_idx + 1) * channels_per_group, out_channels)
        
        group_tensor = tensor_f32[start_ch:end_ch]
        group_quantized, scale = quantize_group(group_tensor)
        
        scales[group_idx] = scale
        quantized[start_ch:end_ch] = group_quantized
    
    # Pack the quantized values using vectorized operations
    packed_data = pack_5bit_values_vectorized(quantized)
    
    return packed_data, scales, original_shape


def dequantize_from_q5_0_per_channel(
    packed_tensor: torch.Tensor,
    scales: torch.Tensor,
    original_shape: tuple,
    target_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Dequantize Q6_0 tensor with per-channel scales back to target precision.
    
    Mathematical relationship:
    - dequantized = quantized * scale
    
    Args:
        packed_tensor: Uint8 packed tensor
        scales: Float32 scales tensor (one per channel group)
        original_shape: Original tensor shape before packing
        target_dtype: Target dtype for dequantized tensor
        
    Returns:
        Dequantized tensor in target dtype
    """
    assert packed_tensor.dtype == torch.uint8, f"Expected uint8 packed tensor, got {packed_tensor.dtype}"
    assert scales.dtype == torch.float32, f"Expected float32 scales, got {scales.dtype}"
    
    # Unpack the 6-bit values using vectorized operations
    quantized_tensor = unpack_5bit_values_vectorized(packed_tensor, original_shape)
    
    # Validate quantized tensor range
    quant_min = quantized_tensor.min().item()
    quant_max = quantized_tensor.max().item()
    assert -31 <= quant_min <= quant_max <= 31, f"Quantized tensor values out of expected range: [{quant_min}, {quant_max}]"
    
    # Convert to float32 for computation
    quantized_f32 = quantized_tensor.to(torch.float32)
    
    # Handle per-channel groups
    assert len(original_shape) >= 2, f"Multi-scale quantization requires 2D+ tensors, got {len(original_shape)}D"
    
    out_channels = original_shape[0]
    num_groups = len(scales)
    channels_per_group = max(1, out_channels // num_groups)
    
    # Verify we have the right number of groups
    expected_groups = (out_channels + channels_per_group - 1) // channels_per_group
    assert num_groups == expected_groups, \
        f"Scale count {num_groups} doesn't match expected groups {expected_groups} for {out_channels} channels"
    
    dequantized = torch.zeros_like(quantized_f32)
    
    # Apply scales per channel group
    for group_idx in range(num_groups):
        start_ch = group_idx * channels_per_group
        end_ch = min((group_idx + 1) * channels_per_group, out_channels)
        
        assert start_ch < out_channels, f"Group {group_idx} start channel {start_ch} >= {out_channels}"
        assert end_ch <= out_channels, f"Group {group_idx} end channel {end_ch} > {out_channels}"
        
        scale = scales[group_idx]
        dequantized[start_ch:end_ch] = quantized_f32[start_ch:end_ch] * scale
    
    return dequantized.to(target_dtype)


def should_quantize_tensor(key: str, tensor: torch.Tensor) -> bool:
    """
    Determine if a tensor should be quantized based on its key and properties.
    Skip small tensors, biases, embeddings, layer norms, and LoRA parameters.
    """
    if "lora_A" in key or "lora_B" in key:
        return False
    
    if tensor.numel() < 1024:
        return False
    
    if "bias" in key:
        return False
    
    skip_patterns = [
        "embed_tokens", "embed_positions", "embed_in", "embed_out",
        "norm", "ln_", "layer_norm", "rmsnorm",
        "pos_emb", "position_embedding"
    ]
    
    for pattern in skip_patterns:
        if pattern in key.lower():
            return False
    
    if tensor.dtype not in [torch.float32, torch.float16, torch.bfloat16]:
        return False
        
    return True