#include "tcgen05_prelude.cuh"

namespace pi {
    __device__ __forceinline__ void mx_block_scale(float amax, uint32_t &sf_byte, float &inv) {
        int32_t e = static_cast<int32_t>((__float_as_uint(amax)>>23)&0xff)-8;
        e = max(0, min(254, e));
        sf_byte = static_cast<uint32_t>(e);
        inv = __uint_as_float(static_cast<uint32_t>(254-e)<<23);
    }

    __device__ __forceinline__ void store_fp8x16(void *dst, const float *v, float inv) {
        uint32_t w[4];
        #pragma unroll
        for (int q=0; q < 4; ++q) {
            uint32_t lo = __nv_cvt_float2_to_fp8x2(make_float2(v[(q<<2)+0]*inv, v[(q<<2)+1]*inv), __NV_SATFINITE, __NV_E4M3);
            uint32_t hi = __nv_cvt_float2_to_fp8x2(make_float2(v[(q<<2)+2]*inv, v[(q<<2)+3]*inv), __NV_SATFINITE, __NV_E4M3);
            w[q] = lo|(hi<<16);
        }
        *static_cast<uint4 *>(dst) = make_uint4(w[0], w[1], w[2], w[3]);
    }

    template <int STAGES, int WN, int BM, int BK, int BN>
    struct smem_up_pod_mxfp8 {
        alignas(1024) __nv_fp8_e4m3 w[STAGES*WN*BK*BN];
        alignas(1024) uint8_t sfw[STAGES*(WN*BN/128)*512];
        alignas(1024) uint8_t sfx[STAGES*512];
        alignas(1024) __nv_fp8_e4m3 x[STAGES*BK*BM];
    };

    template <int STAGES, int WN, int BM, int BK, int BN>
    struct smem_down_pod_mxfp8 {
        alignas(1024) __nv_fp8_e4m3 w[STAGES*WN*BK*BN];
        alignas(1024) uint8_t sfw[STAGES*(BK*2/128)*512];
        alignas(1024) uint8_t sfx[512];
        alignas(1024) __nv_fp8_e4m3 x[BM*(WN*BN/2)];
        alignas(16) __nv_bfloat16 out[BM*(BK*2+8)];
    };

    template <typename Up, typename Down>
    [[nodiscard]] consteval bool mxfp8_pod_alias_impl() noexcept {
      return offsetof(Up, w) == offsetof(Down, w)
        && sizeof(Up::w) == sizeof(Down::w)
        && offsetof(Up, sfw) == offsetof(Down, sfw)
        && sizeof(Up::sfw) == sizeof(Down::sfw)
        && offsetof(Up, sfx) == offsetof(Down, sfx)
        && sizeof(Up::sfx) >= sizeof(Down::sfx);
    }

    template <int STAGES, int WN, int BM, int BK, int BN>
    concept mxfp8_pod_alias = mxfp8_pod_alias_impl<smem_up_pod_mxfp8<STAGES, WN, BM, BK, BN>, smem_down_pod_mxfp8<STAGES, WN, BM, BK, BN>>();

    static_assert(mxfp8_pod_alias<1, 8, 128, 128, 32>);
    static_assert(mxfp8_pod_alias<4, 8, 128, 128, 32>);
    static_assert(mxfp8_pod_alias<4, 4, 128, 128, 64>);

    template <int BM, int BK, int BN, int WN, int STAGES, int PRODUCER_THREADS>
    static __global__ __launch_bounds__(WN*32 + PRODUCER_THREADS) void fused_moe_mxfp8_kernel(
        const __nv_fp8_e4m3 *__restrict__ x,
        const uint8_t *__restrict__ x_scales,
        const __grid_constant__ CUtensorMap map_w,
        const uint8_t *__restrict__ w_scales,
        const __grid_constant__ CUtensorMap map_w2,
        const uint8_t *__restrict__ w2_scales,
        __nv_bfloat16 *__restrict__ out,
        const int32_t *__restrict__ sorted_token_ids,
        const int32_t *__restrict__ expert_idxs,
        const int32_t *__restrict__ num_tokens_post_padded,
        const float *__restrict__ topk_weights,
        int top_k,
        int M,
        int K,
        int N
    ) {
        static_assert(BM == 128, "tcgen05 path: BM=128 / Layout-D only.");
        static_assert(WN*BN == 256, "tcgen05 path assumes 256 output columns per MMA tile.");
        static_assert(BK == 128, "mxfp8 path: one stage must cover exactly one UTCCP group of 4 MMA-K blocks.");
        static constexpr int CONSUMER_THREADS = WN<<5;
        static constexpr int BK2 = (WN*BN)>>1;     // down gemm K
        static constexpr int BN2 = BK<<1;          // down gemm N
        static constexpr int WS = WN*BK*BN;        // weight tile, shared by up and down
        static constexpr int XS = BK*BM;           // up x per stage
        static constexpr int PAD = BN2+8;
        static constexpr int TC_N = WN*BN;
        static constexpr int ACC = 256;            // tmem columns of accumulator
        static constexpr int SFT = 512;            // bytes of one 128 MN x 128 K scale factor tile
        static constexpr int NSFW = TC_N/128;      // scale factor tiles of the up weight tile
        static constexpr int NSFW2 = BN2/128;      // scale factor tiles of the down weight tile
        static constexpr int TM_SFA = ACC;         // 4 tmem columns
        static constexpr int TM_SFB = ACC+4;       // 8 tmem columns
        static constexpr int TMEM_COLS = 512;
        static constexpr int KB = BK>>5;           // MMA-K blocks per stage
        static_assert(TM_SFB+(NSFW<<2) <= TMEM_COLS && TM_SFB+(NSFW2<<2) <= TMEM_COLS);

        alignas(1024) extern __shared__ uint8_t smem_raw[];
        alignas(8) __shared__ barrier bar_copy[STAGES];
        alignas(8) __shared__ barrier bar_recycle[STAGES];
        alignas(8) __shared__ barrier bar_mma[STAGES];
        alignas(4) __shared__ uint32_t tmem_acc;
        __shared__ float topk_scales[BM];
        __shared__ int32_t row_tok[BM];

        auto &smem_up = *reinterpret_cast<smem_up_pod_mxfp8<STAGES, WN, BM, BK, BN> *>(smem_raw);

        int block_base = blockIdx.y;
        int expert_idx = expert_idxs[block_base];
        if (BM*block_base >= *num_tokens_post_padded) return;
        int N2 = K;
        int lane_id = threadIdx.x&31;
        bool is_prod = threadIdx.x < PRODUCER_THREADS;
        int warp_id = is_prod ? threadIdx.x : threadIdx.x-PRODUCER_THREADS;
        warp_id >>= 5;
        if (!threadIdx.x) {
            for (int i = 0; i < STAGES; ++i) {
                bar_copy[i].init(PRODUCER_THREADS+1);
                bar_recycle[i].init(CONSUMER_THREADS);
                bar_mma[i].init(1);
            }
            asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
        }
        __syncthreads();
        if (threadIdx.x >= PRODUCER_THREADS && threadIdx.x < PRODUCER_THREADS+32)
            tcgen05::tmem_alloc(&tmem_acc, TMEM_COLS);
        __syncthreads();
        for (int r=threadIdx.x; r < BM; r += blockDim.x) {
            int32_t tdest = sorted_token_ids[block_base*BM+r];
            int tok = -1;
            if (tdest >= 0) tok = tdest / top_k;
            row_tok[r] = tok;
            if (tdest >= 0 && tok < M) {
                uint32_t ptr = __cvta_generic_to_shared(topk_scales+r);
                cp_async::cg64<4>(ptr, topk_weights+tdest);
            }
        }
        cp_async::commit_group();
        cp_async::await_group<0>();
        __syncthreads();
        auto con_sync = [=]() -> void { asm volatile ("bar.sync 1, %0;\n" :: "n"(CONSUMER_THREADS)); };
        int n_stages_up = K / BK;
        int n_stages_down = N2 / BN2;
        int phase = 0;
        const auto producer = [&] {
            static constexpr int CHUNKS_PER_ROW = BK>>4;   // 16 e4m3 per 16B chunk
            static_assert(!(CHUNKS_PER_ROW&(CHUNKS_PER_ROW-1)));
            static constexpr int ROWS_PER_WAVE = PRODUCER_THREADS / CHUNKS_PER_ROW;
            static constexpr int SFW_CHUNKS = (NSFW*SFT)>>3;
            static constexpr int SFW2_CHUNKS = (NSFW2*SFT)>>3;
            int smem_stage = 0;
            for (int stage=0; stage < n_stages_up; ++stage) {
                if (smem_stage == STAGES) {
                    phase^=1;
                    smem_stage = 0;
                }
                int offs = stage*BK;
                bar_recycle[smem_stage].await(phase);
                for (int wave=0; wave < BM/ROWS_PER_WAVE; ++wave) {
                    int r = wave*ROWS_PER_WAVE+(threadIdx.x/CHUNKS_PER_ROW);
                    int k16 = threadIdx.x&(CHUNKS_PER_ROW-1);
                    if (r < BM) {
                        int tok = row_tok[r];
                        if (tok < 0 || tok >= M) tok = 0;
                        auto *dst = smem_up.x+smem_stage*XS+((k16*BM + r)<<4);
                        const uint4 *src = reinterpret_cast<const uint4 *>(x+static_cast<int64_t>(tok)*K+offs+(k16<<4));
                        cp_async::cg256<16>(static_cast<uint32_t>(__cvta_generic_to_shared(dst)), src);
                    }
                }
                for (int r=threadIdx.x; r < BM; r += PRODUCER_THREADS) {
                    int tok = row_tok[r];
                    if (tok < 0 || tok >= M) tok = 0;
                    auto *dst = smem_up.sfx+smem_stage*SFT+((r&31)<<4)+((r>>5)<<2);
                    cp_async::cg64<4>(
                        static_cast<uint32_t>(__cvta_generic_to_shared(dst)),
                        x_scales+static_cast<int64_t>(tok)*(K>>5)+(offs>>5)
                    );
                }
                for (int i=threadIdx.x; i < SFW_CHUNKS; i += PRODUCER_THREADS) {
                    int j = i/(SFT>>3);
                    int o = (i-j*(SFT>>3))<<3;
                    int32_t pb = blockIdx.x*NSFW+j;
                    int32_t sb = (1&pb)*(N>>8)+(pb>>1);
                    int64_t tile = (static_cast<int64_t>(expert_idx)*(N>>7)+sb)*(K>>7)+stage;
                    cp_async::cg64<8>(
                        static_cast<uint32_t>(__cvta_generic_to_shared(smem_up.sfw+smem_stage*NSFW*SFT+j*SFT+o)),
                        w_scales+tile*SFT+o
                    );
                }
                bar_copy[smem_stage].arrive_cp_async_mem();
                if (!threadIdx.x) {
                    bar_copy[smem_stage].expect_nb(WS*sizeof(__nv_fp8_e4m3));
                    cp_async::load5d(
                        smem_up.w+smem_stage*WS,
                        &map_w,
                        *bar_copy[smem_stage],
                        0,
                        0,
                        expert_idx<<1,
                        blockIdx.x*(TC_N>>8),
                        offs>>4
                    );
                }
                ++smem_stage;
            }
            auto &smem_down = *reinterpret_cast<smem_down_pod_mxfp8<STAGES, WN, BM, BK, BN> *>(smem_raw);
            for (int stage=0; stage < n_stages_down; ++stage) {
                if (smem_stage == STAGES) {
                    phase^=1;
                    smem_stage = 0;
                }
                bar_recycle[smem_stage].await(phase);
                for (int i=threadIdx.x; i < SFW2_CHUNKS; i += PRODUCER_THREADS) {
                    int j=i/(SFT>>3);
                    int o=(i-j*(SFT>>3))<<3;
                    int64_t tile = (static_cast<int64_t>(expert_idx)*(N2>>7)+stage*NSFW2+j)*((N>>1)>>7)+blockIdx.x;
                    cp_async::cg64<8>(
                        static_cast<uint32_t>(__cvta_generic_to_shared(smem_down.sfw+smem_stage*NSFW2*SFT+j*SFT+o)),
                        w2_scales+tile*SFT+o
                    );
                }
                bar_copy[smem_stage].arrive_cp_async_mem();
                if (!threadIdx.x) {
                    bar_copy[smem_stage].expect_nb(WS*sizeof(__nv_fp8_e4m3));
                    cp_async::load3d(
                        smem_down.w+smem_stage*WS,
                        &map_w2,
                        *bar_copy[smem_stage],
                        0,
                        expert_idx*N2+stage*BN2,
                        (blockIdx.x*BK2)>>4
                    );
                }
                ++smem_stage;
            }
        };

        const auto consumer = [&] {
            int tok_src = -1;
            bool live = row_tok[0] >= 0;
            if (threadIdx.x >= PRODUCER_THREADS && threadIdx.x < PRODUCER_THREADS+BM) {
                int32_t tdest = sorted_token_ids[block_base*BM+(threadIdx.x - PRODUCER_THREADS)];
                if (tdest >= 0) tok_src = tdest / top_k;
            }
            for (int i=0; i < STAGES; ++i) bar_recycle[i].arrive();
            uint32_t idesc = tcgen05::encode_idesc_block_scaled_e4m3(BM, TC_N);
            int smem_stage = 0;
            for (int stage=0; stage < n_stages_up; ++stage) {
                if (smem_stage == STAGES) {
                    phase^=1;
                    smem_stage = 0;
                }
                bar_copy[smem_stage].await(phase);
                if (warp_id == 0 && lane_id == 0) {
                    if (live) {
                        tcgen05::cp_sf_32x128b(
                            tmem_acc+TM_SFA,
                            tcgen05::encode_sf_smem_desc(smem_up.sfx+smem_stage*SFT)
                        );
                        #pragma unroll
                        for (int j=0; j < NSFW; ++j)
                            tcgen05::cp_sf_32x128b(
                                tmem_acc+TM_SFB+(j<<2),
                                tcgen05::encode_sf_smem_desc(smem_up.sfw+smem_stage*NSFW*SFT+j*SFT)
                            );
                        #pragma unroll
                        for (int k32=0; k32 < KB; ++k32) {
                            auto *__restrict__ a_ptr = smem_up.x+smem_stage*XS+k32*BM*32;
                            auto *__restrict__ b_ptr = smem_up.w+smem_stage*WS+k32*TC_N*32;
                            uint64_t a_desc = tcgen05::encode_smem_desc(a_ptr, BM);
                            uint64_t b_desc = tcgen05::encode_smem_desc(b_ptr, TC_N);
                            tcgen05::mma_mxf8f6f4(
                                tmem_acc,
                                a_desc,
                                b_desc,
                                tcgen05::idesc_with_sf_id(idesc, k32),
                                tmem_acc+TM_SFA,
                                tmem_acc+TM_SFB,
                                stage||k32
                            );
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
            auto &smem_down = *reinterpret_cast<smem_down_pod_mxfp8<STAGES, WN, BM, BK, BN> *>(smem_raw);
            if (warp_id < 4 && live) {
                int row = (warp_id<<5)+lane_id;
                #pragma unroll
                for (int g=0; g < BK2>>5; ++g) {
                    float act[32];
                    float amax = 0.f;
                    #pragma unroll
                    for (int c=0; c < 4; ++c) {
                        float gate[8];
                        float up[8];
                        int col = (g<<5)+(c<<3);
                        tcgen05::ld_32x32b_x8(gate, tcgen05::tmem_addr(tmem_acc, row, col));
                        tcgen05::ld_32x32b_x8(up, tcgen05::tmem_addr(tmem_acc, row, BK2+col));
                        tcgen05::await_ld();
                        #pragma unroll
                        for (int e=0; e < 8; ++e) {
                            float v = swiglu(gate[e], up[e]);
                            act[(c<<3)+e] = v;
                            amax = fmaxf(amax, fabsf(v));
                        }
                    }
                    uint32_t sf;
                    float inv;
                    mx_block_scale(amax, sf, inv);
                    smem_down.sfx[((row&31)<<4)+((row>>5)<<2)+g] = static_cast<uint8_t>(sf);
                    #pragma unroll
                    for (int h=0; h < 2; ++h) {
                        int k16 = (g<<1)+h;
                        store_fp8x16(smem_down.x+((k16*BM+row)<<4), act+(h<<4), inv);
                    }
                }
            }
            cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
            con_sync();
            idesc = tcgen05::encode_idesc_block_scaled_e4m3(BM, BN2);
            for (int stage=0; stage < n_stages_down; ++stage) {
                if (smem_stage == STAGES) {
                    phase^=1;
                    smem_stage = 0;
                }
                bar_copy[smem_stage].await(phase);
                if (warp_id == 0 && lane_id == 0) {
                    if (live) {
                        tcgen05::cp_sf_32x128b(tmem_acc+TM_SFA, tcgen05::encode_sf_smem_desc(smem_down.sfx));
                        #pragma unroll
                        for (int j=0; j < NSFW2; ++j)
                            tcgen05::cp_sf_32x128b(
                                tmem_acc+TM_SFB+(j<<2),
                                tcgen05::encode_sf_smem_desc(smem_down.sfw+smem_stage*NSFW2*SFT+j*SFT)
                            );
                        #pragma unroll
                        for (int k32=0; k32 < BK2>>5; ++k32) {
                            auto *__restrict__ a_ptr = smem_down.x+k32*BM*32;
                            auto *__restrict__ b_ptr = smem_down.w+smem_stage*WS+k32*BN2*32;
                            uint64_t a_desc = tcgen05::encode_smem_desc(a_ptr, BM);
                            uint64_t b_desc = tcgen05::encode_smem_desc(b_ptr, BN2);
                            tcgen05::mma_mxf8f6f4(
                                tmem_acc,
                                a_desc,
                                b_desc,
                                tcgen05::idesc_with_sf_id(idesc, k32),
                                tmem_acc+TM_SFA,
                                tmem_acc+TM_SFB,
                                k32 != 0
                            );
                        }
                    }
                    tcgen05::commit_mbarrier(*bar_mma[smem_stage]);
                }
                bar_mma[smem_stage].await(phase);
                bar_recycle[smem_stage].arrive();
                tcgen05::after_thread_sync();
                asm volatile("cp.async.bulk.wait_group 0;\n");
                cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
                con_sync();
                if (warp_id < 4 && live) {
                    int row = (warp_id<<5)+lane_id;
                    float scale = topk_scales[row];
                    #pragma unroll
                    for (int c8=0; c8 < BN2>>3; ++c8) {
                        float vals[8];
                        tcgen05::ld_32x32b_x8(vals, tcgen05::tmem_addr(tmem_acc, row, (c8<<3)));
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
                    if (tok_src >= 0 && tok_src < M) {
                        cuda::ptx::cp_reduce_async_bulk(
                            cuda::ptx::space_global,
                            cuda::ptx::space_shared,
                            cuda::ptx::op_add,
                            out+static_cast<int64_t>(tok_src)*N2+stage*BN2,
                            smem_down.out+PAD*(threadIdx.x - PRODUCER_THREADS),
                            BN2*sizeof(__nv_bfloat16)
                        );
                    }
                    cuda::ptx::cp_async_bulk_commit_group();
                }
                ++smem_stage;
            }
            asm volatile("cp.async.bulk.wait_group 0;\n");
            con_sync();
            if (threadIdx.x >= PRODUCER_THREADS && threadIdx.x < PRODUCER_THREADS+32)
                tcgen05::tmem_free(tmem_acc, TMEM_COLS);
        };

        if (is_prod) producer();
        else consumer();
    }

    template <int BM, int BN, int WN, int STAGES>
    static void launch_fused_moe_mxfp8_kernel(
        const __nv_fp8_e4m3 *x,
        const uint8_t *x_scales,
        const __nv_fp8_e4m3 *w,
        const uint8_t *w_scales,
        const __nv_fp8_e4m3 *w2,
        const uint8_t *w2_scales,
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
        static constexpr auto BK = 128;
        static constexpr auto PRODUCER_THREADS = 128;
        static constexpr size_t SMEM = std::max(
            sizeof(smem_up_pod_mxfp8<STAGES, WN, BM, BK, BN>),
            sizeof(smem_down_pod_mxfp8<STAGES, WN, BM, BK, BN>)
        );
        dim3 block = {(WN<<5)+PRODUCER_THREADS, 1, 1};
        dim3 grid = {
            static_cast<uint32_t>(std::ceil(static_cast<double>(N)/static_cast<double>(BN*WN))),
            static_cast<uint32_t>(std::ceil(static_cast<double>(sorted_num)/static_cast<double>(block_m))),
            1
        };
        auto *kernel = fused_moe_mxfp8_kernel<BM, BK, BN, WN, STAGES, PRODUCER_THREADS>;
        if (cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM) != cudaSuccess)
            throw std::runtime_error {"Failed to set max dynamic shared memory size for fused_moe_mxfp8_kernel: "+std::to_string(SMEM)+" bytes"};

        CUtensorMap map_w = init_tmap_swiglu_kmajor_5d_fp8(
            "w",
            w,
            static_cast<uint64_t>(num_experts),
            static_cast<uint64_t>(N),
            static_cast<uint64_t>(K),
            BN*WN,
            BK
        );
        CUtensorMap map_w2 = init_tmap_kmajor_3d_fp8(
            "w2",
            w2,
            static_cast<uint64_t>(K)*static_cast<uint64_t>(num_experts),
            static_cast<uint64_t>(N >> 1),
            BK << 1,
            (WN*BN) >> 1
        );
        kernel<<<grid, block, SMEM, stream>>>(
            x,
            x_scales,
            map_w,
            w_scales,
            map_w2,
            w2_scales,
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
    static void dispatch_stages_mxfp8(
        int stages,
        const __nv_fp8_e4m3 *x,
        const uint8_t *x_scales,
        const __nv_fp8_e4m3 *w,
        const uint8_t *w_scales,
        const __nv_fp8_e4m3 *w2,
        const uint8_t *w2_scales,
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
        #define launch_mxfp8(s) \
            launch_fused_moe_mxfp8_kernel<BM, BN, WN, s>(x, x_scales, w, w_scales, w2, w2_scales, out, sorted_token_ids, expert_idxs, \
            num_tokens_post_padded, topk_weights, top_k, M, K, N, num_experts, sorted_num, block_m, stream)
        switch (stages) {
            case 1: launch_mxfp8(1); break;
            case 2: launch_mxfp8(2); break;
            case 3: launch_mxfp8(3); break;
            case 4: launch_mxfp8(4); break;
            default: throw std::runtime_error {"Invalid stages: "+std::to_string(stages)+" (mxfp8 supports 1..4)"};
        }
        #undef launch_mxfp8
    }

    template <int  BM>
    static void dispatch_bn_wn_mxfp8(
        int bn,
        int wn,
        int stages,
        const __nv_fp8_e4m3 *x,
        const uint8_t *x_scales,
        const __nv_fp8_e4m3 *w,
        const uint8_t *w_scales,
        const __nv_fp8_e4m3 *w2,
        const uint8_t *w2_scales,
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
            case (32<<8)+8:
                dispatch_stages_mxfp8<BM, 32, 8>(stages, x, x_scales, w, w_scales, w2, w2_scales, out, sorted_token_ids, expert_idxs, num_tokens_post_padded, topk_weights, top_k, M, K, N, num_experts, sorted_num, block_m, stream);
                break;
            case (64<<8)+4:
                dispatch_stages_mxfp8<BM, 64, 4>(stages, x, x_scales, w, w_scales, w2, w2_scales, out, sorted_token_ids, expert_idxs, num_tokens_post_padded, topk_weights, top_k, M, K, N, num_experts, sorted_num, block_m, stream);
                break;
            default: throw std::runtime_error {"Invalid BN and WN: "+std::to_string(bn)+" and "+std::to_string(wn)};
        }
    }

    void fused_moe_mxfp8(
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
        cudaStream_t stream
    ) {
        if (bpc != 1)
            throw std::runtime_error {
                "mxfp8 tcgen05 path supports bpc=1 only: a 256 column accumulator plus the "
                "scale factor columns already fill the 512 column tensor memory."
            };
        switch (block_m) {
            case 128:
                dispatch_bn_wn_mxfp8<128>(
                    block_n,
                    warp_n,
                    stages,
                    x,
                    reinterpret_cast<const uint8_t *>(x_scales),
                    w,
                    reinterpret_cast<const uint8_t *>(w_scales),
                    w2,
                    reinterpret_cast<const uint8_t *>(w2_scales),
                    out,
                    sorted_token_ids,
                    expert_idxs,
                    num_tokens_post_padded,
                    topk_weights,
                    top_k,
                    M,
                    K,
                    N,
                    num_experts,
                    sorted_num,
                    block_m,
                    stream
                );
                break;
            default:
                throw std::runtime_error{
                    "tcgen05 Blackwell path currently supports block_m=128 only; "
                    "BM=64 needs Layout-F epilogue, BM=32 needs tcgen05.mma.ws, BM=8/16 are unsupported."
                };
        }
    }
}
