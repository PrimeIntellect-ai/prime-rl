#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <torch/all.h>
#include <torch/library.h>
#include <torch/extension.h>

namespace pi {
    extern void fused_moe_bf16(
        const __nv_bfloat16 *x,
        const __nv_bfloat16 *w,
        const __nv_bfloat16 *w2,
        __nv_bfloat16 *out,
        const int32_t *sorted_token_ids,
        const int32_t *expert_idxs,
        const int32_t *num_tokens_post_padded,
        const float *topk_weights,
        int top_k,
        int M,
        int K,
        int N,
        int num_experts,
        int sorted_num,
        int block_m,
        int block_n,
        int warp_n,
        int stages,
        int bpc,
        int cpc,
        bool split,
        cudaStream_t stream
    );

    extern void fused_moe_mxfp8(
        const __nv_fp8_e4m3 *x,
        const __nv_fp8_e8m0 *x_scales,
        const __nv_fp8_e4m3 *w,
        const __nv_fp8_e8m0 *w_scales,
        const __nv_fp8_e4m3 *w2,
        const __nv_fp8_e8m0 *w2_scales,
        __nv_bfloat16 *out,
        const int32_t *sorted_token_ids,
        const int32_t *expert_idxs,
        const int32_t *num_tokens_post_padded,
        const float *topk_weights,
        int top_k,
        int M,
        int K,
        int N,
        int num_experts,
        int sorted_num,
        int block_m,
        int block_n,
        int warp_n,
        int stages,
        int bpc,
        bool split,
        cudaStream_t stream
    );

    extern void moe_align(
        const int32_t *topk_ids,
        int32_t *sorted_token_ids,
        int32_t *expert_ids,
        int32_t *num_tokens_post_padded,
        int n,
        int num_experts,
        int block_m,
        int pad_to,
        int max_padded,
        int max_blocks,
        cudaStream_t stream
    );

    static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> moe_align_torch_stub(
        const torch::Tensor &topk_ids,
        int64_t num_experts,
        int64_t block_m,
        int64_t bpc
    ) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        at::cuda::OptionalCUDAGuard device_guard {device_of(topk_ids)};
        TORCH_CHECK(topk_ids.is_contiguous());
        TORCH_CHECK(topk_ids.dtype() == torch::kInt32);
        TORCH_CHECK(block_m > 0 && bpc > 0 && num_experts > 0);
        int64_t n = topk_ids.numel();
        int64_t pad_to = block_m*bpc;
        int64_t max_padded = (n+num_experts*(pad_to-1)+pad_to-1) / pad_to*pad_to; // no bit tricks as pad can be a non power of 2 :()
        int64_t max_blocks = max_padded/block_m;
        auto opts = torch::TensorOptions().dtype(torch::kInt32).device(topk_ids.device());
        torch::Tensor sorted_token_ids = torch::empty({max_padded}, opts);
        torch::Tensor expert_ids = torch::empty({max_blocks}, opts);
        torch::Tensor num_tokens_post_padded = torch::empty({1}, opts);
        moe_align(
            static_cast<const int32_t *>(topk_ids.const_data_ptr()),
            static_cast<int32_t *>(sorted_token_ids.data_ptr()),
            static_cast<int32_t *>(expert_ids.data_ptr()),
            static_cast<int32_t *>(num_tokens_post_padded.data_ptr()),
            n,
            num_experts,
            block_m,
            pad_to,
            max_padded,
            max_blocks,
            stream
        );
        return {sorted_token_ids, expert_ids, num_tokens_post_padded};
    }

    static void fused_moe_bf16_torch_stub(
        const torch::Tensor &x,
        const torch::Tensor &w,
        const torch::Tensor &w2,
        const torch::Tensor &sorted_token_ids,
        const torch::Tensor &expert_ids,
        const torch::Tensor &num_tokens_post_padded,
        const torch::Tensor &topk_weights,
        torch::Tensor &out,
        int64_t top_k,
        int64_t block_m,
        int64_t block_n,
        int64_t warp_n,
        int64_t stages,
        int64_t bpc,
        int64_t cpc,
        bool split
    ) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        at::cuda::OptionalCUDAGuard device_guard {device_of(x)};
        TORCH_CHECK(x.size(0));
        TORCH_CHECK(x.size(1) == w.size(2));
        TORCH_CHECK((x.size(1)&127) == 0, "K dimension must be a multiple of 128");
        TORCH_CHECK((block_n == 64 && warp_n == 4) || (block_n == 32 && warp_n == 8));
        TORCH_CHECK(stages > 0 && stages < 6);
        TORCH_CHECK(bpc == 1 && cpc == 1, "bpc=cpc=1 is the only supported configuration");
        TORCH_CHECK(!split || stages <= 4, "split=True requires stages<=4, got stages="+std::to_string(stages));
        TORCH_CHECK(block_m == 128, "Blackwell tcgen05 fused MoE requires block_m=128.");
        TORCH_CHECK(x.is_contiguous());
        TORCH_CHECK(w.is_contiguous());
        TORCH_CHECK(w2.is_contiguous());
        TORCH_CHECK(sorted_token_ids.is_contiguous());
        TORCH_CHECK(expert_ids.is_contiguous());
        TORCH_CHECK(num_tokens_post_padded.is_contiguous());
        TORCH_CHECK(topk_weights.is_contiguous());
        TORCH_CHECK(out.is_contiguous());
        TORCH_CHECK(x.dtype() == torch::kBFloat16);
        TORCH_CHECK(w.dtype() == torch::kBFloat16);
        TORCH_CHECK(w2.dtype() == torch::kBFloat16);
        TORCH_CHECK(topk_weights.dtype() == torch::kFloat32);
        TORCH_CHECK(sorted_token_ids.dtype() == torch::kInt32);
        TORCH_CHECK(expert_ids.dtype() == torch::kInt32);
        TORCH_CHECK(num_tokens_post_padded.dtype() == torch::kInt32);
        TORCH_CHECK(out.dtype() == torch::kBFloat16);
        fused_moe_bf16(
            static_cast<const __nv_bfloat16 *>(x.const_data_ptr()),
            static_cast<const __nv_bfloat16 *>(w.const_data_ptr()),
            static_cast<const __nv_bfloat16 *>(w2.const_data_ptr()),
            static_cast<__nv_bfloat16 *>(out.data_ptr()),
            static_cast<const int32_t *>(sorted_token_ids.const_data_ptr()),
            static_cast<const int32_t *>(expert_ids.const_data_ptr()),
            static_cast<const int32_t *>(num_tokens_post_padded.const_data_ptr()),
            static_cast<const float *>(topk_weights.const_data_ptr()),
            top_k,
            x.size(0),
            x.size(1),
            w.size(1),
            w.size(0),
            sorted_token_ids.size(0),
            block_m,
            block_n,
            warp_n,
            stages,
            bpc,
            cpc,
            split,
            stream
        );
    }

    static void fused_moe_mxfp8_torch_stub(
        const torch::Tensor &x,
        const torch::Tensor &x_scales,
        const torch::Tensor &w,
        const torch::Tensor &w_scales,
        const torch::Tensor &w2,
        const torch::Tensor &w2_scales,
        const torch::Tensor &sorted_token_ids,
        const torch::Tensor &expert_ids,
        const torch::Tensor &num_tokens_post_padded,
        const torch::Tensor &topk_weights,
        torch::Tensor &out,
        int64_t top_k,
        int64_t block_m,
        int64_t block_n,
        int64_t warp_n,
        int64_t stages,
        int64_t bpc,
        bool split
    ) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        at::cuda::OptionalCUDAGuard device_guard {device_of(x)};
        const int64_t M = x.size(0);
        const int64_t K = x.size(1);
        const int64_t E = w.size(0);
        const int64_t N = w.size(1);
        TORCH_CHECK(M);
        TORCH_CHECK(K == w.size(2));
        TORCH_CHECK((K&255) == 0, "K dimension must be a multiple of 256 on the mxfp8 path");
        TORCH_CHECK((N&255) == 0, "N dimension must be a multiple of 256 on the mxfp8 path");
        TORCH_CHECK(w2.size(0) == E && w2.size(1) == K && w2.size(2) == N/2);
        TORCH_CHECK((block_n == 64 && warp_n == 4) || (block_n == 32 && warp_n == 8));
        TORCH_CHECK(stages > 0 && stages < 5, "mxfp8 supports stages in 1..4");
        TORCH_CHECK(bpc == 1, "mxfp8 supports bpc=1 only: tensor memory has no room for a second accumulator plus scale factors");
        TORCH_CHECK(block_m == 128, "Blackwell tcgen05 fused MoE requires block_m=128.");
        TORCH_CHECK(x_scales.numel() == M*(K/32), "x_scales must be a row major (M, K/32) e8m0 tensor");
        TORCH_CHECK(w_scales.numel() == E*N*(K/32), "w_scales must hold E*N*K/32 e8m0 values in the blocked layout");
        TORCH_CHECK(w2_scales.numel() == E*K*((N/2)/32), "w2_scales must hold E*K*(N/2)/32 e8m0 values in the blocked layout");
        TORCH_CHECK(x.is_contiguous());
        TORCH_CHECK(x_scales.is_contiguous());
        TORCH_CHECK(w.is_contiguous());
        TORCH_CHECK(w_scales.is_contiguous());
        TORCH_CHECK(w2.is_contiguous());
        TORCH_CHECK(w2_scales.is_contiguous());
        TORCH_CHECK(sorted_token_ids.is_contiguous());
        TORCH_CHECK(expert_ids.is_contiguous());
        TORCH_CHECK(num_tokens_post_padded.is_contiguous());
        TORCH_CHECK(topk_weights.is_contiguous());
        TORCH_CHECK(out.is_contiguous());
        TORCH_CHECK(x.dtype() == torch::kFloat8_e4m3fn);
        TORCH_CHECK(x_scales.dtype() == torch::kFloat8_e8m0fnu);
        TORCH_CHECK(w.dtype() == torch::kFloat8_e4m3fn);
        TORCH_CHECK(w_scales.dtype() == torch::kFloat8_e8m0fnu);
        TORCH_CHECK(w2.dtype() == torch::kFloat8_e4m3fn);
        TORCH_CHECK(w2_scales.dtype() == torch::kFloat8_e8m0fnu);
        TORCH_CHECK(topk_weights.dtype() == torch::kFloat32);
        TORCH_CHECK(sorted_token_ids.dtype() == torch::kInt32);
        TORCH_CHECK(expert_ids.dtype() == torch::kInt32);
        TORCH_CHECK(num_tokens_post_padded.dtype() == torch::kInt32);
        TORCH_CHECK(out.dtype() == torch::kBFloat16);
        fused_moe_mxfp8(
            static_cast<const __nv_fp8_e4m3 *>(x.const_data_ptr()),
            static_cast<const __nv_fp8_e8m0 *>(x_scales.const_data_ptr()),
            static_cast<const __nv_fp8_e4m3 *>(w.const_data_ptr()),
            static_cast<const __nv_fp8_e8m0 *>(w_scales.const_data_ptr()),
            static_cast<const __nv_fp8_e4m3 *>(w2.const_data_ptr()),
            static_cast<const __nv_fp8_e8m0 *>(w2_scales.const_data_ptr()),
            static_cast<__nv_bfloat16 *>(out.data_ptr()),
            static_cast<const int32_t *>(sorted_token_ids.const_data_ptr()),
            static_cast<const int32_t *>(expert_ids.const_data_ptr()),
            static_cast<const int32_t *>(num_tokens_post_padded.const_data_ptr()),
            static_cast<const float *>(topk_weights.const_data_ptr()),
            top_k,
            M,
            K,
            N,
            E,
            sorted_token_ids.size(0),
            block_m,
            block_n,
            warp_n,
            stages,
            bpc,
            split,
            stream
        );
    }
}

TORCH_LIBRARY_FRAGMENT(prime_moe, m) {
    m.def("fused_moe_bf16("
        "Tensor x, "
        "Tensor w, "
        "Tensor w2, "
        "Tensor sorted_token_ids, "
        "Tensor expert_ids, "
        "Tensor num_tokens_post_padded, "
        "Tensor topk_weights, "
        "Tensor(a!) out, "
        "int top_k, "
        "int block_m, "
        "int block_n, "
        "int warp_n, "
        "int stages, "
        "int bpc, "
        "int cpc, "
        "bool split=True"
        ") -> ()"
    );
    m.impl("fused_moe_bf16", torch::kCUDA, &pi::fused_moe_bf16_torch_stub);

    m.def("fused_moe_mxfp8("
        "Tensor x, "
        "Tensor x_scales, "
        "Tensor w, "
        "Tensor w_scales, "
        "Tensor w2, "
        "Tensor w2_scales, "
        "Tensor sorted_token_ids, "
        "Tensor expert_ids, "
        "Tensor num_tokens_post_padded, "
        "Tensor topk_weights, "
        "Tensor(a!) out, "
        "int top_k, "
        "int block_m, "
        "int block_n, "
        "int warp_n, "
        "int stages, "
        "int bpc, "
        "bool split=True"
        ") -> ()"
    );
    m.impl("fused_moe_mxfp8", torch::kCUDA, &pi::fused_moe_mxfp8_torch_stub);

    m.def("moe_align("
        "Tensor topk_ids, "
        "int num_experts, "
        "int block_m, "
        "int bpc"
        ") -> (Tensor, Tensor, Tensor)"
    );
    m.impl("moe_align", torch::kCUDA, &pi::moe_align_torch_stub);
}

PYBIND11_MODULE(_C, m) {}
