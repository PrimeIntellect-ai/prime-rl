#include <ATen/ATen.h>
#include <torch/library.h>

#include <tuple>

namespace prime_rl_kernels::nvfp4 {

std::tuple<at::Tensor, at::Tensor, at::Tensor>
quantize_activations_cuda(
    const at::Tensor& matrix,
    const at::Tensor& offsets);

std::tuple<at::Tensor, at::Tensor, at::Tensor>
quantize_weights_cuda(const at::Tensor& weight_rows);

at::Tensor dequantize_activations_cuda(
    const at::Tensor& packed,
    const at::Tensor& block_scales,
    const at::Tensor& global_scales,
    const at::Tensor& offsets);

at::Tensor dequantize_weights_cuda(
    const at::Tensor& packed,
    const at::Tensor& block_scales,
    const at::Tensor& global_scales);

TORCH_LIBRARY_FRAGMENT(prime_rl_kernels_nvfp4, m) {
  m.def(
      "quantize_activations(Tensor matrix, Tensor offsets) -> "
      "(Tensor, Tensor, Tensor)");
  m.def(
      "quantize_weights(Tensor weight_rows) -> "
      "(Tensor, Tensor, Tensor)");
  m.def(
      "dequantize_activations(Tensor packed, Tensor block_scales, "
      "Tensor global_scales, Tensor offsets) -> Tensor");
  m.def(
      "dequantize_weights(Tensor packed, Tensor block_scales, "
      "Tensor global_scales) -> Tensor");
}

TORCH_LIBRARY_IMPL(prime_rl_kernels_nvfp4, CUDA, m) {
  m.impl("quantize_activations", quantize_activations_cuda);
  m.impl("quantize_weights", quantize_weights_cuda);
  m.impl("dequantize_activations", dequantize_activations_cuda);
  m.impl("dequantize_weights", dequantize_weights_cuda);
}

} // namespace prime_rl_kernels::nvfp4
