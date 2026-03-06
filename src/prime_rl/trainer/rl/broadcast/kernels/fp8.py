import torch
import triton
import triton.language as tl

FP8_DTYPE = torch.float8_e4m3fn
_FP8_MAX = torch.finfo(FP8_DTYPE).max
_FP8_MIN = -_FP8_MAX
_FP8_BLOCK_SIZE = (128, 128)


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


@triton.jit
def _blockwise_cast_to_fp8_triton(
    x_ptr,
    y_ptr,
    s_ptr,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    stride_sm,
    stride_sn,
    m_size,
    n_size,
    eps,
    fp8_min_value,
    fp8_max_value,
    BLOCK_M: tl.constexpr = 32,
    BLOCK_N: tl.constexpr = 128,
):
    pid_m = tl.cast(tl.program_id(axis=0), tl.int64)
    pid_n = tl.cast(tl.program_id(axis=1), tl.int64)
    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = off_m < m_size
    mask_n = off_n < n_size
    mask = mask_m[:, None] & mask_n[None, :]

    x = tl.load(
        x_ptr + off_m[:, None] * stride_xm + off_n[None, :] * stride_xn,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    absmax = tl.maximum(tl.max(tl.abs(x)), eps)
    scale = absmax / fp8_max_value
    scale_inv = 1.0 / scale
    y_q = tl.clamp(x * scale_inv, fp8_min_value, fp8_max_value).to(
        y_ptr.dtype.element_ty
    )

    tl.store(
        y_ptr + off_m[:, None] * stride_ym + off_n[None, :] * stride_yn,
        y_q,
        mask=mask,
    )
    tl.store(s_ptr + pid_m * stride_sm + pid_n * stride_sn, scale)


def blockwise_cast_to_fp8_triton(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D tensor to blockwise FP8, returning (fp8_weight, scale_inv)."""
    block_m, block_n = _FP8_BLOCK_SIZE
    m_size, n_size = weight.shape
    qweight = torch.empty(m_size, n_size, device=weight.device, dtype=FP8_DTYPE)
    scale = torch.empty(
        _ceil_div(m_size, block_m),
        _ceil_div(n_size, block_n),
        dtype=torch.float32,
        device=weight.device,
    )

    if weight.is_contiguous():
        launch_kwargs = {
            "BLOCK_M": block_m,
            "BLOCK_N": block_n,
            "num_warps": 8,
            "num_stages": 2,
        }
    else:
        launch_kwargs = {
            "BLOCK_M": block_m,
            "BLOCK_N": block_n,
            "num_warps": 1,
            "num_stages": 4,
        }

    grid = (triton.cdiv(m_size, block_m), triton.cdiv(n_size, block_n))
    _blockwise_cast_to_fp8_triton[grid](
        weight,
        qweight,
        scale,
        *weight.stride(),
        *qweight.stride(),
        *scale.stride(),
        m_size,
        n_size,
        1e-10,
        _FP8_MIN,
        _FP8_MAX,
        **launch_kwargs,
    )
    return qweight, scale


def quantize_weight_to_block_fp8(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D weight tensor to blockwise FP8.

    Returns:
        (fp8_weight, weight_scale_inv) where weight_scale_inv has shape
        [ceil(out/128), ceil(in/128)] with dtype float32.
    """
    if weight.ndim != 2:
        raise ValueError(f"Expected a 2D weight tensor, got shape={tuple(weight.shape)}")
    if not weight.is_cuda:
        raise ValueError("FP8 blockwise quantization requires CUDA tensors")
    return blockwise_cast_to_fp8_triton(weight)
