#include "tcgen05_prelude.cuh"

namespace pi {
    template <int STAGES, int WN, int BM,int BK, int BN, int BPC>
    struct smem_up_pod {
        alignas(1024) __nv_bfloat16 w[STAGES*WN*BK*BN];
        alignas(1024) __nv_bfloat16 x[STAGES*BK*BM*BPC];
    };

    template <int STAGES, int WN, int BM, int BK, int BN, int BPC>
    struct smem_down_pod {
        alignas(1024) __nv_bfloat16 w[STAGES*WN*BK*BN];
        alignas(1024) __nv_bfloat16 x[BM*WN*BN/2*BPC];
        alignas(16) __nv_bfloat16 out[BM*(BK*2+8)];
    };

    template <int BM, int BK, int BN, int WN, int STAGES, int PRODUCER_THREADS, int BPC>
    static __global__ __launch_bounds__(WN*32+PRODUCER_THREADS) void fused_moe_kernel(
        const __nv_bfloat16 *__restrict__ x,
        const __grid_constant__ CUtensorMap map_w,
        const __grid_constant__ CUtensorMap map_w2,
        __nv_bfloat16 *__restrict__ out,
        const int32_t *__restrict__ sorted_token_ids,
        const int32_t *__restrict__ expert_idxs,
        const int32_t *__restrict__ num_tokens_post_padded,
        const float *__restrict__ topk_weights,
        int32_t top_k,
        int M,
        int K,
        int N
    ) {
        static_assert(BM == 128, "tcgen05 path: BM=128 / Layout-D only.");
        static_assert(WN*BN == 256, "tcgen05 path asumes 256 output columns per MMA tile.");
        static_assert(256*BPC <= 512, "tmem holds at most 512 columns (BPC*256 acc).");

        static constexpr int CONSUMER_THREADS = WN<<5;
        static constexpr int BLOCK_SHAPE_K = BK;
        static constexpr int BK2 = (WN*BN)>>1;
        static constexpr int BN2 = BK<<1;
        static constexpr int WS = WN*BK*BN;
        static constexpr int XS = BK*BM;           // up x per block and stages
        static constexpr int DXS = BM*WN*BN/2;     // down x per block
        static constexpr int TB = 16;
        static constexpr int PAD = BN2+8;
        static constexpr int TC_N = WN*BN;
        static constexpr int ACC = 256;   // tmem columns per block accumulator

        alignas(1024) extern __shared__ uint8_t smem_raw[];
        alignas(8) __shared__ barrier bar_copy[STAGES];
        alignas(8) __shared__ barrier bar_recycle[STAGES];
        alignas(8) __shared__ barrier bar_mma[STAGES];
        alignas(4) __shared__ uint32_t tmem_acc;
        __shared__ float topk_scales[BM*BPC];
        __shared__ int32_t row_tok[BM*BPC];

        auto &smem_up = *reinterpret_cast<smem_up_pod<STAGES, WN, BM, BK, BN, BPC> *>(smem_raw);

        int block_base = blockIdx.y * BPC;
        int expert_idx = expert_idxs[block_base];
        if (BM*block_base >= *num_tokens_post_padded) return;
        int N2 = K;
        int lane_id = threadIdx.x&31;
        bool is_prod = threadIdx.x < PRODUCER_THREADS;
        int warp_id = is_prod ? threadIdx.x : threadIdx.x-PRODUCER_THREADS;
        warp_id >>= 5;
        if (!threadIdx.x) {
            for (int i=0; i < STAGES; ++i) {
                bar_copy[i].init(PRODUCER_THREADS+1);
                bar_recycle[i].init(CONSUMER_THREADS);
                bar_mma[i].init(1);
            }
            asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
        }
        __syncthreads();
        if (threadIdx.x >= PRODUCER_THREADS && threadIdx.x < PRODUCER_THREADS+32)
            tcgen05::tmem_alloc(&tmem_acc, ACC*BPC);
        __syncthreads();
        for (int r = threadIdx.x; r < BM*BPC; r += blockDim.x) {
            int tdest=sorted_token_ids[block_base*BM+r];
            int tok = -1;
            if (tdest >= 0) tok = tdest / top_k;
            row_tok[r] = tok;
            if (tdest >= 0 && tok < M) {
                uint32_t ptr = __cvta_generic_to_shared(topk_scales+r);
                cp_async::cg64<4>(ptr, topk_weights+tdest);
            }
        }
        __syncthreads();
        auto con_sync = [=]() -> void { asm volatile ("bar.sync 1, %0;\n" :: "n"(CONSUMER_THREADS)); };
        int n_stages_up = K / BLOCK_SHAPE_K;
        int n_stages_down = N2 / BN2;
        int phase = 0;
        const auto producer = [&] {
            static constexpr int32_t CHUNKS_PER_ROW = BK>>3;
            static_assert(!(CHUNKS_PER_ROW&(CHUNKS_PER_ROW-1)));
            static constexpr int32_t ROWS_PER_WAVE  = PRODUCER_THREADS / CHUNKS_PER_ROW;
            int smem_stage = 0;
            for (int stage=0; stage < n_stages_up; ++stage) {
                if (smem_stage == STAGES) {
                    phase^=1;
                    smem_stage = 0;
                }
                int offs = stage*BK;
                bar_recycle[smem_stage].await(phase);
                for (int wave=0; wave < (BM*BPC)/ROWS_PER_WAVE; ++wave) {
                    int r = wave*ROWS_PER_WAVE+(threadIdx.x/CHUNKS_PER_ROW);
                    int k8 = threadIdx.x&(CHUNKS_PER_ROW-1);
                    int b = r / BM;
                    int row_in_block = r - b*BM;
                    int tok = row_tok[r];
                    if (r < BM*BPC && tok >= 0 && tok < M) {
                        auto *dst = smem_up.x+smem_stage*XS*BPC+b*XS+((k8*BM + row_in_block)<<3);
                        const uint4 *src = reinterpret_cast<const uint4 *>(x+tok*K+offs+(k8<<3));
                        cp_async::cg256<TB>(static_cast<uint32_t>(__cvta_generic_to_shared(dst)), src);
                    }
                }
                bar_copy[smem_stage].arrive_cp_async_mem();
                if (!threadIdx.x) {
                    bar_copy[smem_stage].expect_nb(WS*sizeof(__nv_bfloat16));
                    cp_async::load5d(
                        smem_up.w+smem_stage*WS,
                        &map_w,
                        *bar_copy[smem_stage],
                        0,
                        0,
                        expert_idx<<1,
                        blockIdx.x*((WN*BN)>>8),
                        offs>>3
                    );
                }
                ++smem_stage;
            }
            auto &smem_down = *reinterpret_cast<smem_down_pod<STAGES, WN, BM, BK, BN, BPC> *>(smem_raw);
            for (int stage=0; stage < n_stages_down; ++stage) {
                if (smem_stage == STAGES) {
                    phase^=1;
                    smem_stage = 0;
                }
                bar_recycle[smem_stage].await(phase);
                if (!threadIdx.x) {
                    bar_copy[smem_stage].expect_nb(WS*sizeof(__nv_bfloat16));
                    cp_async::load3d(
                        smem_down.w+smem_stage*WS,
                        &map_w2,
                        *bar_copy[smem_stage],
                        0,
                        expert_idx*N2+stage*BN2,
                        (blockIdx.x*BK2) / 8
                    );
                }
                bar_copy[smem_stage].arrive_cp_async_mem();
                ++smem_stage;
            }
        };

        const auto consumer = [&] {
            int32_t tok_src[BPC];
            bool live[BPC];
            #pragma unroll
            for (int b=0; b < BPC; ++b) {
                live[b] = row_tok[b*BM] >= 0;
                tok_src[b] = -1;
                if (threadIdx.x >= PRODUCER_THREADS && threadIdx.x < PRODUCER_THREADS+BM) {
                    int32_t tdest = sorted_token_ids[(block_base+b)*BM+(threadIdx.x - PRODUCER_THREADS)];
                    if (tdest >= 0) tok_src[b] = tdest / top_k;
                }
            }
            for (int i=0; i < STAGES; ++i) bar_recycle[i].arrive();
            uint32_t idesc = tcgen05::encode_idesc_format_1(BM, TC_N);
            int smem_stage = 0;
            for (int stage=0; stage < n_stages_up; ++stage) {
                if (smem_stage == STAGES) {
                    phase^=1;
                    smem_stage = 0;
                }
                bar_copy[smem_stage].await(phase);
                if (warp_id == 0 && lane_id == 0) {
                    #pragma unroll
                    for (int b=0; b < BPC; ++b) {
                        if (!live[b]) continue;
                        #pragma unroll
                        for (int k16=0; k16 < BK>>4; ++k16) {
                            auto *__restrict__ a_ptr = smem_up.x+smem_stage*XS*BPC+b*XS+((k16*BM)<<4);
                            auto *__restrict__ b_ptr = smem_up.w+smem_stage*WS+((k16*TC_N)<<4);
                            uint64_t a_desc = tcgen05::encode_smem_desc(a_ptr, BM);
                            uint64_t b_desc = tcgen05::encode_smem_desc(b_ptr, TC_N);
                            tcgen05::mma_f16(tmem_acc+b*ACC, a_desc, b_desc, idesc, stage||k16);
                        }
                    }
                    tcgen05::commit_mbarrier(*bar_mma[smem_stage]);
                }
                bar_mma[smem_stage].await(phase);
                bar_recycle[smem_stage].arrive();
                ++smem_stage;
            }
            con_sync();
            tcgen05::after_thread_sync();
            auto &smem_down = *reinterpret_cast<smem_down_pod<STAGES, WN, BM, BK, BN, BPC> *>(smem_raw);
            if (warp_id < 4) {
                int row = (warp_id<<5)+lane_id;
                #pragma unroll
                for (int b = 0; b < BPC; ++b) {
                    if (!live[b]) continue;
                    #pragma unroll
                    for (int c8=0; c8 < BK2>>3; ++c8) {
                        float gate[8];
                        float up[8];
                        tcgen05::ld_32x32b_x8(gate, tcgen05::tmem_addr(tmem_acc+b*ACC, row, (c8<<3)));
                        tcgen05::ld_32x32b_x8(up, tcgen05::tmem_addr(tmem_acc+b*ACC, row, BK2+(c8<<3)));
                        tcgen05::await_ld();
                        auto *dst = smem_down.x+b*DXS+((c8*BM+row)<<3);
                        #pragma unroll
                        for (int e=0; e < 8; ++e)
                            dst[e] = __float2bfloat16(swiglu(gate[e], up[e]));
                    }
                }
            }
            con_sync();
            idesc = tcgen05::encode_idesc_format_1(BM, BN2);
            for (int stage=0; stage < n_stages_down; ++stage) {
                if (smem_stage == STAGES) {
                    phase^=1;
                    smem_stage = 0;
                }
                bar_copy[smem_stage].await(phase);
                if (warp_id == 0 && lane_id == 0) {
                    #pragma unroll
                    for (int b=0; b < BPC; ++b) {
                        if (!live[b]) continue;
                        #pragma unroll
                        for (int k16=0; k16 < BK2>>4; ++k16) {
                            __nv_bfloat16 *a_ptr = smem_down.x+b*DXS+((k16*BM)<<4);
                            __nv_bfloat16 *b_ptr = smem_down.w+smem_stage*WS+((k16*BN2)<<4);
                            uint64_t a_desc = tcgen05::encode_smem_desc(a_ptr, BM);
                            uint64_t b_desc = tcgen05::encode_smem_desc(b_ptr, BN2);
                            tcgen05::mma_f16(tmem_acc+b*ACC, a_desc, b_desc, idesc, k16 != 0);
                        }
                    }
                    tcgen05::commit_mbarrier(*bar_mma[smem_stage]);
                }
                bar_mma[smem_stage].await(phase);
                bar_recycle[smem_stage].arrive();
                tcgen05::after_thread_sync();
                #pragma unroll
                for (int b=0; b < BPC; ++b) {
                    asm volatile("cp.async.bulk.wait_group 0;\n");
                    cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
                    con_sync();
                    if (warp_id < 4 && live[b]) {
                        int row = (warp_id<<5)+lane_id;
                        float scale = topk_scales[b*BM+row];
                        #pragma unroll
                        for (int c8=0; c8 < BN2>>3; ++c8) {
                            float vals[8];
                            tcgen05::ld_32x32b_x8(vals, tcgen05::tmem_addr(tmem_acc+b*ACC, row, (c8<<3)));
                            tcgen05::await_ld();
                            __nv_bfloat16 *dst = smem_down.out+row*PAD+(c8<<3);
                            #pragma unroll
                            for (int e=0; e < 8; ++e)
                                dst[e] = __float2bfloat16(scale*vals[e]);
                        }
                    }
                    cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
                    con_sync();
                    if (threadIdx.x < PRODUCER_THREADS+BM) {
                        if (tok_src[b] >= 0 && tok_src[b] < M) {
                            cuda::ptx::cp_reduce_async_bulk(
                                cuda::ptx::space_global,
                                cuda::ptx::space_shared,
                                cuda::ptx::op_add,
                                out+tok_src[b]*N2+stage*BN2,
                                smem_down.out+PAD*(threadIdx.x - PRODUCER_THREADS),
                                BN2*sizeof(__nv_bfloat16)
                            );
                        }
                        cuda::ptx::cp_async_bulk_commit_group();
                    }
                }
                ++smem_stage;
            }
            asm volatile("cp.async.bulk.wait_group 0;\n");
            con_sync();
            if (threadIdx.x >= PRODUCER_THREADS && threadIdx.x < PRODUCER_THREADS+32)
                tcgen05::tmem_free(tmem_acc, ACC*BPC);
        };

        if (is_prod) producer();
        else consumer();
    }

    template <int BM, int BN, int WN, int STAGES, int BPC = 2>
    static void launch_fused_moe_kernel(
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
        cudaStream_t stream
    ) {
        static constexpr auto BK = 64;
        static constexpr auto PRODUCER_THREADS = 128;
        static constexpr size_t SMEM = std::max(sizeof(smem_up_pod<STAGES, WN, BM, BK, BN, BPC>), sizeof(smem_down_pod<STAGES, WN, BM, BK, BN, BPC>));
        dim3 block = {(WN<<5)+PRODUCER_THREADS, 1, 1};
        dim3 grid = {
            static_cast<uint32_t>(std::ceil(static_cast<double>(N)/static_cast<double>(BN*WN))),
            static_cast<uint32_t>(std::ceil(static_cast<double>(sorted_num)/static_cast<double>(block_m*BPC))),
            1
        };
        auto *kernel = fused_moe_kernel<BM, BK, BN, WN, STAGES, PRODUCER_THREADS, BPC>;
        if (cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM) != cudaSuccess)
            throw std::runtime_error {"Failed to set max dynamic shared memory size for fused_moe_kernel: "+std::to_string(SMEM)+" bytes"};

        CUtensorMap map_w = init_tmap_swiglu_kmajor_5d_bf16(
            "w",
            w,
            static_cast<uint64_t>(num_experts),
            static_cast<uint64_t>(N),
            static_cast<uint64_t>(K),
            BN*WN,
            BK
        );
        CUtensorMap map_w2 = init_tmap_kmajor_3d(
            "w2",
            w2,
            static_cast<uint64_t>(K*num_experts),
            static_cast<uint64_t>(N >> 1),
            BK << 1,
            (WN*BN) >> 1
        );
        kernel<<<grid, block, SMEM, stream>>>(
            x,
            map_w,
            map_w2,
            out,
            sorted_token_ids,
            expert_idxs,
            num_tokens_post_padded,
            topk_weights,
            top_k,
            M,
            K,
            N
        );
    }

    template <int BM, int BN, int WN>
    static void dispatch_stages(
        int stages,
        int bpc,
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
        cudaStream_t stream
    ) {
        switch (((bpc&0xf)<<4)|(stages&0xf)) {
            case (1<<4)+1: launch_fused_moe_kernel<BM, BN, WN, 1, 1>(x, w, w2, out, sorted_token_ids, expert_idxs, num_tokens_post_padded, topk_weights, top_k, M, K, N, num_experts, sorted_num, block_m, stream); break;
            case (1<<4)+2: launch_fused_moe_kernel<BM, BN, WN, 2, 1>(x, w, w2, out, sorted_token_ids, expert_idxs, num_tokens_post_padded, topk_weights, top_k, M, K, N, num_experts, sorted_num, block_m, stream); break;
            case (1<<4)+3: launch_fused_moe_kernel<BM, BN, WN, 3, 1>(x, w, w2, out, sorted_token_ids, expert_idxs, num_tokens_post_padded, topk_weights, top_k, M, K, N, num_experts, sorted_num, block_m, stream); break;
            case (1<<4)+4: launch_fused_moe_kernel<BM, BN, WN, 4, 1>(x, w, w2, out, sorted_token_ids, expert_idxs, num_tokens_post_padded, topk_weights, top_k, M, K, N, num_experts, sorted_num, block_m, stream); break;
            case (1<<4)+5: launch_fused_moe_kernel<BM, BN, WN, 5, 1>(x, w, w2, out, sorted_token_ids, expert_idxs, num_tokens_post_padded, topk_weights, top_k, M, K, N, num_experts, sorted_num, block_m, stream); break;
            case (2<<4)+1: launch_fused_moe_kernel<BM, BN, WN, 1, 2>(x, w, w2, out, sorted_token_ids, expert_idxs, num_tokens_post_padded, topk_weights, top_k, M, K, N, num_experts, sorted_num, block_m, stream); break;
            case (2<<4)+2: launch_fused_moe_kernel<BM, BN, WN, 2, 2>(x, w, w2, out, sorted_token_ids, expert_idxs, num_tokens_post_padded, topk_weights, top_k, M, K, N, num_experts, sorted_num, block_m, stream); break;
            case (2<<4)+3: launch_fused_moe_kernel<BM, BN, WN, 3, 2>(x, w, w2, out, sorted_token_ids, expert_idxs, num_tokens_post_padded, topk_weights, top_k, M, K, N, num_experts, sorted_num, block_m, stream); break;
            default: throw std::runtime_error {"Invalid stages/bpc combination: "+std::to_string(stages)+"/"+std::to_string(bpc)};
        }
    }

    template <int BM>
    static void dispatch_bn_wn(
        int bn,
        int wn,
        int stages,
        int bpc,
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
        cudaStream_t stream
    ) {
        switch (((bn&0xff)<<8)+(wn&0xff)) {
            case (32<<8)+8: dispatch_stages<BM, 32, 8>(stages, bpc, x, w, w2, out, sorted_token_ids, expert_idxs, num_tokens_post_padded, topk_weights, top_k, M, K, N, num_experts, sorted_num, block_m, stream); break;
            case (64<<8)+4: dispatch_stages<BM, 64, 4>(stages, bpc, x, w, w2, out, sorted_token_ids, expert_idxs, num_tokens_post_padded, topk_weights, top_k, M, K, N, num_experts, sorted_num, block_m, stream); break;
            default: throw std::runtime_error {"Invalid BN and WN: "+std::to_string(bn)+" and "+std::to_string(wn)};
        }
    }

    static __global__ void moe_align_kernel(
        const int32_t *__restrict__ topk_ids,
        int32_t *__restrict__ sorted_token_ids,
        int32_t *__restrict__ expert_ids,
        int32_t *__restrict__ num_tokens_post_padded,
        int n,
        int num_experts,
        int block_m,
        int pad_to,
        int max_padded,
        int max_blocks
    ) {
        extern __shared__ int32_t smem_align[];
        auto *offs = smem_align;
        auto *fill = smem_align+num_experts+1;
        for (int i=threadIdx.x; i <= num_experts; i += blockDim.x) {
            offs[i] = 0;
            if (i < num_experts) fill[i] = 0;
        }
        __syncthreads();
        for (int i=threadIdx.x; i < n; i += blockDim.x)
            atomicAdd(offs+(topk_ids[i]+1), 1);
        __syncthreads();
        if (!threadIdx.x) {
            for (int e=0; e < num_experts; ++e) {
                int32_t padded = (offs[e+1]+pad_to-1)/pad_to*pad_to;
                offs[e+1] = offs[e]+padded;
            }
            *num_tokens_post_padded = offs[num_experts];
        }
        __syncthreads();
        int total = offs[num_experts];
        for (int i=threadIdx.x; i < max_padded; i += blockDim.x)
            sorted_token_ids[i] = -1;
        for (int j=threadIdx.x; j < max_blocks; j += blockDim.x) {
            int row = j*block_m;
            if (row >= total) {
                expert_ids[j] = num_experts-1;
                continue;
            }
            int lo = 0, hi = num_experts-1;
            while (lo < hi) {
                int mid = (lo+hi+1)>>1;
                if (offs[mid] <= row) lo = mid;
                else hi = mid-1;
            }
            expert_ids[j] = lo;
        }
        __syncthreads();
        for (int i=threadIdx.x; i < n; i += blockDim.x) {
            int e = topk_ids[i];
            int pos = atomicAdd(fill+e, 1);
            sorted_token_ids[offs[e]+pos] = i;
        }
    }

    void moe_align(
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
    ) {
        if (num_experts > 0x400)
            throw std::runtime_error {"moe_align supports at most 1024 experts, got "+std::to_string(num_experts)};
        moe_align_kernel<<<1, 0x400, ((num_experts<<1)+1)*sizeof(int), stream>>>(
            topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padded,
            n, num_experts, block_m, pad_to, max_padded, max_blocks
        );
    }

    void fused_moe_bf16(
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
        cudaStream_t stream
    ) {
        switch (block_m) {
            case 128:
                dispatch_bn_wn<128>(block_n, warp_n, stages, bpc, x, w, w2, out, sorted_token_ids, expert_idxs, num_tokens_post_padded, topk_weights, top_k, M, K, N, num_experts, sorted_num, block_m, stream);
                break;
            default:
                throw std::runtime_error{
                    "tcgen05 Blackwell path currently supports block_m=128 only; "
                    "BM=64 needs Layout-F epilogue, BM=32 needs tcgen05.mma.ws, BM=8/16 are unsupported."
                };
        }
    }
}
