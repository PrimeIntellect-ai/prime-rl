"""
Symmetric Q8_0 per-channel quantization module.

Provides functions for quantizing and dequantizing tensors using symmetric 
int8 quantization with per-channel grouping.
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
    """Quantize a single channel group using symmetric 8-bit quantization."""
    group_min = group_tensor.min()
    group_max = group_tensor.max()
    
    assert group_min != group_max, f"Group is constant (min=max={group_min:.6f}), cannot quantize"
    
    abs_max = group_tensor.abs().max()
    assert abs_max > 0, "Group has zero absolute maximum"
    
    # 8-bit symmetric quantization: range [-127, 127]
    scale = abs_max / 127.0
    quantized = (group_tensor / scale).round().clamp(-127, 127)
    
    # Validate output
    actual_min = quantized.min().item()
    actual_max = quantized.max().item()
    assert -127 <= actual_min <= actual_max <= 127, \
        f"Quantized values out of range: [{actual_min}, {actual_max}]"
    
    return quantized.to(torch.int8), scale


def quantize_to_q8_0_per_channel(tensor: torch.Tensor, num_channels: int = 128) -> tuple[torch.Tensor, torch.Tensor, tuple]:
    """
    Quantize tensor to Q8_0 format using symmetric quantization with per-channel grouping.
    
    Mathematical relationship:
    - quantized = round(tensor / scale).clamp(-127, 127)
    - dequantized = quantized * scale
    
    Args:
        tensor: Input tensor to quantize (must be float dtype and 2D+)
        num_channels: Number of channel groups for quantization
        
    Returns:
        Tuple of (quantized_tensor as int8, scales as float32, original_shape)
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
    
    return quantized, scales, original_shape


def dequantize_from_q8_0_per_channel(
    quantized_tensor: torch.Tensor,
    scales: torch.Tensor,
    original_shape: tuple,
    target_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Dequantize Q8_0 tensor with per-channel scales back to target precision.
    
    Mathematical relationship:
    - dequantized = quantized * scale
    
    Args:
        quantized_tensor: Int8 quantized tensor
        scales: Float32 scales tensor (one per channel group)
        original_shape: Original tensor shape before quantization
        target_dtype: Target dtype for dequantized tensor
        
    Returns:
        Dequantized tensor in target dtype
    """
    assert quantized_tensor.dtype == torch.int8, f"Expected int8 quantized tensor, got {quantized_tensor.dtype}"
    assert scales.dtype == torch.float32, f"Expected float32 scales, got {scales.dtype}"
    
    # Validate quantized tensor range
    quant_min = quantized_tensor.min().item()
    quant_max = quantized_tensor.max().item()
    assert -127 <= quant_min <= quant_max <= 127, f"Quantized tensor values out of expected range: [{quant_min}, {quant_max}]"
    
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