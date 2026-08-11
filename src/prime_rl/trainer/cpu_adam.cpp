#include <ATen/Parallel.h>
#include <ATen/cpu/vec/functional.h>
#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

struct TensorSpan {
  float* param;
  const void* grad;
  at::BFloat16* compute_param;
  float* exp_avg;
  float* exp_avg_sq;
  bool grad_bfloat16;
  int64_t begin;
  int64_t end;
  float step_size;
  float bias_correction2_sqrt;
};

void check_tensor(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.device().is_cpu(), name, " must be on CPU");
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, name, " must be float32");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_grad(const torch::Tensor& tensor) {
  TORCH_CHECK(tensor.device().is_cpu(), "grad must be on CPU");
  TORCH_CHECK(
      tensor.scalar_type() == torch::kFloat32 || tensor.scalar_type() == torch::kBFloat16,
      "grad must be float32 or bfloat16");
  TORCH_CHECK(tensor.is_contiguous(), "grad must be contiguous");
}

void add_bfloat16_(torch::Tensor destination, const torch::Tensor& source) {
  check_tensor(destination, "destination");
  TORCH_CHECK(source.device().is_cpu(), "source must be on CPU");
  TORCH_CHECK(source.scalar_type() == torch::kBFloat16, "source must be bfloat16");
  TORCH_CHECK(source.is_contiguous(), "source must be contiguous");
  TORCH_CHECK(destination.numel() == source.numel(), "source shape must match destination shape");

  float* destination_data = destination.data_ptr<float>();
  const at::BFloat16* source_data = source.const_data_ptr<at::BFloat16>();
  const int64_t numel = destination.numel();
  constexpr int64_t grain_size = 32 * 1024;
  at::parallel_for(0, numel, grain_size, [&](int64_t begin, int64_t end) {
    using Vec = at::vec::Vectorized<float>;
    using BVec = at::vec::Vectorized<at::BFloat16>;
    constexpr int64_t block_size = 2 * Vec::size();
    int64_t i = begin;
    for (; i <= end - block_size; i += block_size) {
      const BVec source_vec = BVec::loadu(source_data + i);
      auto [source0, source1] = at::vec::convert_to_float<at::BFloat16>(source_vec);
      (Vec::loadu(destination_data + i) + source0).store(destination_data + i);
      (Vec::loadu(destination_data + i + Vec::size()) + source1)
          .store(destination_data + i + Vec::size());
    }
    for (; i < end; ++i) {
      destination_data[i] += static_cast<float>(source_data[i]);
    }
  });
}

void adamw_span(
    const TensorSpan& span,
    int64_t begin,
    int64_t end,
    float beta1,
    float beta2,
    float weight_decay_factor,
    float eps,
    float gradient_scale) {
  using Vec = at::vec::Vectorized<float>;
  using BVec = at::vec::Vectorized<at::BFloat16>;

  float* param = span.param + begin;
  const float* float_grad = span.grad_bfloat16 ? nullptr : static_cast<const float*>(span.grad) + begin;
  const at::BFloat16* bfloat16_grad =
      span.grad_bfloat16 ? static_cast<const at::BFloat16*>(span.grad) + begin : nullptr;
  at::BFloat16* compute_param = span.compute_param == nullptr ? nullptr : span.compute_param + begin;
  float* exp_avg = span.exp_avg + begin;
  float* exp_avg_sq = span.exp_avg_sq + begin;
  const int64_t size = end - begin;

  const Vec beta1_vec(beta1);
  const Vec beta2_vec(beta2);
  const Vec one_minus_beta1(1.0f - beta1);
  const Vec one_minus_beta2(1.0f - beta2);
  const Vec weight_decay_vec(weight_decay_factor);
  const Vec step_size_vec(span.step_size);
  const Vec bias_correction2_sqrt_vec(span.bias_correction2_sqrt);
  const Vec eps_vec(eps);
  const Vec gradient_scale_vec(gradient_scale);

  auto update = [&](int64_t offset, const Vec& grad_vec) {
    Vec param_vec = Vec::loadu(param + offset) * weight_decay_vec;
    Vec exp_avg_vec = Vec::loadu(exp_avg + offset);
    Vec exp_avg_sq_vec = Vec::loadu(exp_avg_sq + offset);
    const Vec scaled_grad = grad_vec * gradient_scale_vec;

    exp_avg_vec = exp_avg_vec * beta1_vec + scaled_grad * one_minus_beta1;
    exp_avg_sq_vec = exp_avg_sq_vec * beta2_vec + scaled_grad * scaled_grad * one_minus_beta2;
    const Vec denominator = exp_avg_sq_vec.sqrt() / bias_correction2_sqrt_vec + eps_vec;
    param_vec = param_vec - step_size_vec * exp_avg_vec / denominator;

    param_vec.store(param + offset);
    exp_avg_vec.store(exp_avg + offset);
    exp_avg_sq_vec.store(exp_avg_sq + offset);
    return param_vec;
  };

  int64_t i = 0;
  constexpr int64_t block_size = 2 * Vec::size();
  for (; i <= size - block_size; i += block_size) {
    Vec grad0;
    Vec grad1;
    if (span.grad_bfloat16) {
      const BVec grad_bvec = BVec::loadu(bfloat16_grad + i);
      std::tie(grad0, grad1) = at::vec::convert_to_float<at::BFloat16>(grad_bvec);
    } else {
      grad0 = Vec::loadu(float_grad + i);
      grad1 = Vec::loadu(float_grad + i + Vec::size());
    }
    const Vec param0 = update(i, grad0);
    const Vec param1 = update(i + Vec::size(), grad1);
    if (compute_param != nullptr) {
      at::vec::convert_from_float<at::BFloat16>(param0, param1).store(compute_param + i);
    }
  }

  for (; i < size; ++i) {
    float param_value = param[i] * weight_decay_factor;
    const float grad_value =
        (span.grad_bfloat16 ? static_cast<float>(bfloat16_grad[i]) : float_grad[i]) * gradient_scale;
    const float exp_avg_value = beta1 * exp_avg[i] + (1.0f - beta1) * grad_value;
    const float exp_avg_sq_value =
        beta2 * exp_avg_sq[i] + (1.0f - beta2) * grad_value * grad_value;
    const float denominator = std::sqrt(exp_avg_sq_value) / span.bias_correction2_sqrt + eps;

    param_value -= span.step_size * exp_avg_value / denominator;
    param[i] = param_value;
    exp_avg[i] = exp_avg_value;
    exp_avg_sq[i] = exp_avg_sq_value;
    if (compute_param != nullptr) {
      compute_param[i] = at::BFloat16(param_value);
    }
  }
}

void adamw_step(
    const std::vector<torch::Tensor>& params,
    const std::vector<torch::Tensor>& grads,
    const std::vector<torch::Tensor>& exp_avgs,
    const std::vector<torch::Tensor>& exp_avg_sqs,
    const std::vector<torch::Tensor>& state_steps,
    const std::vector<torch::Tensor>& compute_params,
    double lr,
    double beta1,
    double beta2,
    double weight_decay,
    double eps,
    double gradient_scale) {
  const auto count = params.size();
  TORCH_CHECK(grads.size() == count, "grads and params must have the same length");
  TORCH_CHECK(exp_avgs.size() == count, "exp_avgs and params must have the same length");
  TORCH_CHECK(exp_avg_sqs.size() == count, "exp_avg_sqs and params must have the same length");
  TORCH_CHECK(state_steps.size() == count, "state_steps and params must have the same length");
  TORCH_CHECK(
      compute_params.empty() || compute_params.size() == count,
      "compute_params must be empty or have the same length as params");
  if (count == 0) {
    return;
  }

  std::vector<TensorSpan> spans;
  spans.reserve(count);
  int64_t total_numel = 0;
  for (size_t i = 0; i < count; ++i) {
    check_tensor(params[i], "param");
    check_grad(grads[i]);
    check_tensor(exp_avgs[i], "exp_avg");
    check_tensor(exp_avg_sqs[i], "exp_avg_sq");
    check_tensor(state_steps[i], "state_step");
    TORCH_CHECK(params[i].numel() == grads[i].numel(), "grad shape must match param shape");
    TORCH_CHECK(params[i].numel() == exp_avgs[i].numel(), "exp_avg shape must match param shape");
    TORCH_CHECK(params[i].numel() == exp_avg_sqs[i].numel(), "exp_avg_sq shape must match param shape");
    TORCH_CHECK(state_steps[i].numel() == 1, "state_step must contain one value");
    if (!compute_params.empty()) {
      TORCH_CHECK(compute_params[i].device().is_cpu(), "compute_param must be on CPU");
      TORCH_CHECK(compute_params[i].scalar_type() == torch::kBFloat16, "compute_param must be bfloat16");
      TORCH_CHECK(compute_params[i].is_contiguous(), "compute_param must be contiguous");
      TORCH_CHECK(params[i].numel() == compute_params[i].numel(), "compute_param shape must match param shape");
    }

    float* step = state_steps[i].data_ptr<float>();
    *step += 1.0f;
    const double bias_correction1 = 1.0 - std::pow(beta1, *step);
    const double bias_correction2 = 1.0 - std::pow(beta2, *step);
    const int64_t numel = params[i].numel();
    spans.push_back(TensorSpan{
        params[i].data_ptr<float>(),
        grads[i].const_data_ptr(),
        compute_params.empty() ? nullptr : compute_params[i].data_ptr<at::BFloat16>(),
        exp_avgs[i].data_ptr<float>(),
        exp_avg_sqs[i].data_ptr<float>(),
        grads[i].scalar_type() == torch::kBFloat16,
        total_numel,
        total_numel + numel,
        static_cast<float>(lr / bias_correction1),
        static_cast<float>(std::sqrt(bias_correction2)),
    });
    total_numel += numel;
  }

  const float beta1_value = static_cast<float>(beta1);
  const float beta2_value = static_cast<float>(beta2);
  const float weight_decay_factor = static_cast<float>(1.0 - lr * weight_decay);
  const float eps_value = static_cast<float>(eps);
  const float gradient_scale_value = static_cast<float>(gradient_scale);
  constexpr int64_t grain_size = 32 * 1024;

  at::parallel_for(0, total_numel, grain_size, [&](int64_t begin, int64_t end) {
    auto span_it = std::upper_bound(
        spans.begin(),
        spans.end(),
        begin,
        [](int64_t offset, const TensorSpan& span) { return offset < span.end; });
    while (begin < end) {
      TORCH_INTERNAL_ASSERT(span_it != spans.end());
      const int64_t span_begin = begin - span_it->begin;
      const int64_t span_end = std::min(end, span_it->end) - span_it->begin;
      adamw_span(
          *span_it,
          span_begin,
          span_end,
          beta1_value,
          beta2_value,
          weight_decay_factor,
          eps_value,
          gradient_scale_value);
      begin = std::min(end, span_it->end);
      ++span_it;
    }
  });
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "add_bfloat16_",
      &add_bfloat16_,
      "Add a BF16 tensor into an FP32 tensor",
      pybind11::call_guard<pybind11::gil_scoped_release>());
  module.def(
      "adamw_step",
      &adamw_step,
      "Read-only-gradient multi-tensor CPU AdamW",
      pybind11::call_guard<pybind11::gil_scoped_release>());
}
