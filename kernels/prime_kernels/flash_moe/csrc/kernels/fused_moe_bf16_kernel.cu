#include "tcgen05_prelude.cuh"

namespace pi {
    template <int STAGES, int WN, int BM, int BK, int BN, int BPC, int CPC>
    struct smem_up_pod {
        alignas(1024) __nv_bfloat16 w[STAGES*CPC*WN*BK*BN];
        alignas(1024) __nv_bfloat16 x[STAGES*BK*BM*BPC];
    };

    template <int STAGES, int WN, int BM, int BK, int BN, int BPC, int CPC>
    struct smem_down_pod {
        static constexpr int DND = CPC > 1 ? 32 : (BK<<1);
        alignas(1024) __nv_bfloat16 w[STAGES*CPC*DND*(WN*BN/2)];
        alignas(1024) __nv_bfloat16 x[BM*WN*BN/2*BPC*CPC];
        alignas(16) __nv_bfloat16 out[STAGES <= 2 ? 8 : BM*(BK*2+8)];
    };

    static cudaStream_t g_zstream = nullptr;
    static cudaEvent_t g_e_head = nullptr, g_e_zero = nullptr;

    template <int BM, int BK, int BN, int WN, int STAGES, int PRODUCER_THREADS, int BPC, int CPC, bool SPLIT_UP = false>
    static __global__ __launch_bounds__(WN*32+PRODUCER_THREADS, 2) void fused_moe_kernel(
        const __nv_bfloat16 *__restrict__ x,
        const __nv_bfloat16 *__restrict__ w_global,
        const __grid_constant__ CUtensorMap map_x,
        const __grid_constant__ CUtensorMap map_w,
        const __grid_constant__ CUtensorMap map_w2,
        const __grid_constant__ CUtensorMap map_act,
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
        static_assert(BPC*CPC <= 2, "tmem holds at most 512 columns == 2 groups of 256.");

        static constexpr int CONSUMER_THREADS = WN<<5;
        static constexpr int BLOCK_SHAPE_K = BK;
        static constexpr int BK2 = (WN*BN)>>1;
        static constexpr int BN2 = BK<<1;
        static constexpr int DN = CPC > 1 ? 32 : BN2;   // down out-cols per stage
        static constexpr int DWS = DN*((WN*BN)>>1);     // w2 elems per (stage, col group)
        static constexpr int WS = WN*BK*BN;
        static constexpr int TC_N = WN*BN;
        static constexpr int PAD = BN2+8;
        static constexpr int XS = BK*BM;           // up x per block and stage
        static constexpr int DXS = BM*WN*BN/2;     // down x per block and group
        static constexpr int ACC = 256;            // tmem columns per accumulator group

        alignas(1024) extern __shared__ uint8_t smem_raw[];
        alignas(8) __shared__ barrier bar_copy[STAGES];
        alignas(8) __shared__ barrier bar_recycle[STAGES];
        alignas(8) __shared__ barrier bar_mma[STAGES];
        alignas(4) __shared__ uint32_t tmem_acc;
        __shared__ __nv_bfloat16 topk_scales[BM*BPC];
        __shared__ int32_t row_tok[BM*BPC];

        auto &smem_up = *reinterpret_cast<smem_up_pod<STAGES, WN, BM, BK, BN, BPC, CPC> *>(smem_raw);

        const int by_row = static_cast<int>(blockIdx.y);
        const int bx_col = static_cast<int>(blockIdx.x);
        int block_base = by_row * BPC;
        int total_padded = *num_tokens_post_padded;
        if (BM*BPC*by_row >= total_padded) return;
        bool own_dead = BM*block_base >= total_padded;
        int expert_idx = own_dead ? 0 : expert_idxs[block_base];
        int N2 = K;
        int lane_id = threadIdx.x&31;
        bool is_prod = threadIdx.x < PRODUCER_THREADS;
        int warp_id = is_prod ? threadIdx.x : threadIdx.x-PRODUCER_THREADS;
        warp_id >>= 5;
        if (!threadIdx.x) {
            for (int i=0; i < STAGES; ++i) {
                bar_copy[i].init(PRODUCER_THREADS+1);
                bar_recycle[i].init(1);
                bar_mma[i].init(1);
            }
            asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
        }
        __syncthreads();
        if (threadIdx.x >= PRODUCER_THREADS && threadIdx.x < PRODUCER_THREADS+32) {
            tcgen05::tmem_alloc(&tmem_acc, ACC*BPC*CPC);
        }
        __syncthreads();
        for (int r = threadIdx.x; r < BM*BPC; r += blockDim.x) {
            int tdest = own_dead ? -1 : sorted_token_ids[block_base*BM+r];
            int tok = -1;
            if (tdest >= 0) tok = tdest / top_k;
            row_tok[r] = tok;
            if (tdest >= 0 && tok < M) {
                topk_scales[r] = __float2bfloat16(topk_weights[tdest]);
            }
        }
        __syncthreads();
        auto con_sync = [=]() -> void { asm volatile ("bar.sync 1, %0;\n" :: "n"(CONSUMER_THREADS)); };
        int n_stages_up = K / BLOCK_SHAPE_K;
        int n_stages_down = N2 / DN;
        int phase = 0;
        const auto producer = [&] {
            int g4_rows[4];
            uint32_t g4_dst = 0;
            {
                if (threadIdx.x < 32) {
                    #pragma unroll
                    for (int i=0; i < 4; ++i)
                        g4_rows[i] = max(row_tok[threadIdx.x*4+i], 0);
                    g4_dst = static_cast<uint32_t>(__cvta_generic_to_shared(smem_up.x))
                           + threadIdx.x*4*BK*sizeof(__nv_bfloat16);
                }
            }
            int smem_stage = 0;
            for (int stage=0; stage < n_stages_up; ++stage) {
                if (smem_stage == STAGES) {
                    phase^=1;
                    smem_stage = 0;
                }
                int offs = stage*BK;
                bar_recycle[smem_stage].await(phase);
                if (!threadIdx.x) {
                    bar_copy[smem_stage].expect_nb((CPC*WS + XS*BPC)*sizeof(__nv_bfloat16));
                    #pragma unroll
                    for (int cg=0; cg < CPC; ++cg) {
                        cp_async::load4d(
                            smem_up.w+smem_stage*CPC*WS+cg*WS,
                            &map_w,
                            *bar_copy[smem_stage],
                            offs, 0, expert_idx<<1, bx_col*CPC+cg
                        );
                        
                    }
                    
                }
                if constexpr (SPLIT_UP) {
                    if (!threadIdx.x)
                        cp_async::load(smem_up.x+smem_stage*XS, &map_x, *bar_copy[smem_stage], offs, block_base*BM);
                } else if (threadIdx.x < 32) {
                    cp_async::gather4(
                    g4_dst + static_cast<uint32_t>(smem_stage*XS*sizeof(__nv_bfloat16)), &map_x, *bar_copy[smem_stage], offs, g4_rows[0], g4_rows[1], g4_rows[2], g4_rows[3]);
                }
                bar_copy[smem_stage].arrive_cp_async_mem();
                ++smem_stage;
            }
            auto &smem_down = *reinterpret_cast<smem_down_pod<STAGES, WN, BM, BK, BN, BPC, CPC> *>(smem_raw);
            if constexpr (SPLIT_UP) return;
            for (int stage=0; stage < n_stages_down; ++stage) {
                if (smem_stage == STAGES) {
                    phase^=1;
                    smem_stage = 0;
                }
                bar_recycle[smem_stage].await(phase);
                if (!threadIdx.x) {
                    bar_copy[smem_stage].expect_nb(CPC*DWS*sizeof(__nv_bfloat16));
                    #pragma unroll
                    for (int cg=0; cg < CPC; ++cg) {
                        int col_group = bx_col*CPC+cg;
                        #pragma unroll
                        for (int half=0; half < BK2>>6; ++half) {
                            auto *dst = smem_down.w+smem_stage*CPC*DWS+cg*DWS+half*(DN<<6);
                            cp_async::load3d(
                                dst,
                                &map_w2,
                                *bar_copy[smem_stage],
                                col_group*BK2+(half<<6),
                                expert_idx*N2+stage*DN,
                                0
                            );
                            
                        }
                    }
                }
                bar_copy[smem_stage].arrive_cp_async_mem();
                ++smem_stage;
            }
        };

        const auto consumer = [&] {
            int32_t tok_src[BPC];
            bool live[BPC];
            int tmem_row = (((threadIdx.x>>5)&3)<<5)+lane_id;
            #pragma unroll
            for (int b=0; b < BPC; ++b) {
                live[b] = row_tok[b*BM] >= 0;
                tok_src[b] = -1;
                int32_t tdest = own_dead ? -1 : sorted_token_ids[(block_base+b)*BM+tmem_row];
                if (tdest >= 0) tok_src[b] = tdest / top_k;
            }
            const auto recycle_arrive = [&](int slot) { bar_recycle[slot].arrive(); };
            uint32_t idesc = tcgen05::encode_idesc_format_1(BM, TC_N);
            auto &smem_down = *reinterpret_cast<smem_down_pod<STAGES, WN, BM, BK, BN, BPC, CPC> *>(smem_raw);
            if (warp_id == 0 && lane_id == 0) {
                int smem_stage = 0;
                int pend_slot = -1, pend_phase = 0;
                for (int i=0; i < STAGES; ++i) recycle_arrive(i);
                for (int stage=0; stage < n_stages_up; ++stage) {
                    if (smem_stage == STAGES) {
                        phase^=1;
                        smem_stage = 0;
                    }
                    bar_copy[smem_stage].await(phase);
                    #pragma unroll
                    for (int b=0; b < BPC; ++b) {
                        if (!live[b]) continue;
                        #pragma unroll
                        for (int cg=0; cg < CPC; ++cg) {
                            int slot = b*CPC+cg;
                            #pragma unroll
                            for (int k16=0; k16 < BK>>4; ++k16) {
                                auto *__restrict__ a_ptr = smem_up.x+smem_stage*XS*BPC+b*XS+(k16<<4);
                                auto *__restrict__ b_ptr = smem_up.w+smem_stage*CPC*WS+cg*WS+(k16<<4);
                                uint64_t a_desc = tcgen05::encode_smem_desc_swz<BK*2>(a_ptr);
                                uint64_t b_desc = tcgen05::encode_smem_desc_swz<BK*2>(b_ptr);
                                tcgen05::mma_f16(tmem_acc+slot*ACC, a_desc, b_desc, idesc, stage||k16);
                            }
                        }
                    }
                    tcgen05::commit_mbarrier(*bar_recycle[smem_stage]);
                    tcgen05::commit_mbarrier(*bar_mma[smem_stage]);
                    pend_slot = smem_stage;
                    pend_phase = phase;
                    ++smem_stage;
                }
                bar_mma[pend_slot].await(pend_phase);
            }
            int smem_stage = n_stages_up % STAGES;
            phase = (n_stages_up / STAGES) & 1;
            con_sync();
            tcgen05::after_thread_sync();
            auto *act_smem = SPLIT_UP ? reinterpret_cast<__nv_bfloat16 *>(smem_raw) : smem_down.x;
            if (warp_id < 4) {
                #pragma unroll
                for (int b = 0; b < BPC; ++b) {
                    if (!live[b]) continue;
                    #pragma unroll
                    for (int cg=0; cg < CPC; ++cg) {
                        int slot = b*CPC+cg;
                        #pragma unroll
                        for (int c32=0; c32 < BK2>>5; ++c32) {
                            float gate[32];
                            float up[32];
                            tcgen05::ld_32x32b_x32(gate, tcgen05::tmem_addr(tmem_acc+slot*ACC, tmem_row, (c32<<5)));
                            tcgen05::ld_32x32b_x32(up, tcgen05::tmem_addr(tmem_acc+slot*ACC, tmem_row, BK2+(c32<<5)));
                            tcgen05::await_ld();
                            #pragma unroll
                            for (int i=0; i < 4; ++i) {
                                auto *dst = act_smem+slot*DXS+((((c32<<2)+i)*BM+tmem_row)<<3);
                                #pragma unroll
                                for (int e=0; e < 8; ++e)
                                    dst[e] = __float2bfloat16(swiglu(gate[(i<<3)+e], up[(i<<3)+e]));
                            }
                        }
                        if constexpr (SPLIT_UP) {
                            asm volatile("bar.sync 2, 128;\n");
                            if (warp_id == 0 && lane_id == 0) {
                                cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
                                cp_async_extra::store3d(&map_act, act_smem, 0, block_base*BM, bx_col*(TC_N>>4));
                                cuda::ptx::cp_async_bulk_commit_group();
                            }
                        }
                    }
                }
            }
            con_sync();
            if constexpr (SPLIT_UP) {
                if (warp_id == 0 && lane_id == 0) {
                    asm volatile("cp.async.bulk.wait_group 0;\n" ::: "memory");
                    cudaTriggerProgrammaticLaunchCompletion();
                }
                if (threadIdx.x >= PRODUCER_THREADS && threadIdx.x < PRODUCER_THREADS+32)
                    tcgen05::tmem_free(tmem_acc, ACC*BPC*CPC);
                return;
            }
            int q_rank = 0;
            idesc = tcgen05::encode_idesc_format_1(BM, DN);
            const auto down_epilogue = [&](int j) {
                tcgen05::after_thread_sync();
                int half = (j&1)*DN;
                if (warp_id < 4) {
                    #pragma unroll
                    for (int b=0; b < BPC; ++b) {
                        float scale = __bfloat162float(topk_scales[b*BM+tmem_row]);
                        bool row_valid = tok_src[b] >= 0 && tok_src[b] < M;
                        if constexpr (STAGES <= 2) {
                            __nv_bfloat16 *gdst = out+(row_valid ? tok_src[b] : 0)*N2+j*DN;
                            #pragma unroll
                            for (int c32=0; c32 < DN>>5; ++c32) {
                                float vals[32];
                                tcgen05::ld_32x32b_x32(vals, tcgen05::tmem_addr(tmem_acc+b*ACC, tmem_row, half+(c32<<5)));
                                tcgen05::await_ld();
                                if (row_valid) {
                                    #pragma unroll
                                    for (int i=0; i < 4; ++i)
                                        red_global_add_bf16x8(gdst+(((c32<<2)+i)<<3), vals+(i<<3), scale);
                                }
                            }
                            continue;
                        }
                        asm volatile("cp.async.bulk.wait_group.read 0;\n");
                        #pragma unroll
                        for (int c32=0; c32 < DN>>5; ++c32) {
                            float vals[32];
                            tcgen05::ld_32x32b_x32(vals, tcgen05::tmem_addr(tmem_acc+b*ACC, tmem_row, half+(c32<<5)));
                            tcgen05::await_ld();
                            __nv_bfloat16 *dst = smem_down.out+tmem_row*PAD+(c32<<5);
                            #pragma unroll
                            for (int e=0; e < 32; ++e)
                                dst[e] = __float2bfloat16(scale*vals[e]);
                        }
                        cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
                        if (row_valid) {
                            cuda::ptx::cp_reduce_async_bulk(
                                cuda::ptx::space_global,
                                cuda::ptx::space_shared,
                                cuda::ptx::op_add,
                                out+tok_src[b]*N2+j*DN,
                                smem_down.out+PAD*tmem_row,
                                DN*sizeof(__nv_bfloat16)
                            );
                        }
                        cuda::ptx::cp_async_bulk_commit_group();
                    }
                }
                asm volatile("bar.sync 2, 128;\n");
            };
            if (warp_id < 4) {
                for (int stage=0; stage < n_stages_down; ++stage) {
                    if (smem_stage == STAGES) {
                        phase^=1;
                        smem_stage = 0;
                    }
                    if (warp_id == 0 && lane_id == 0) {
                        bar_copy[smem_stage].await(phase);
                    #pragma unroll
                    for (int b=0; b < BPC; ++b) {
                        if (!live[b]) continue;
                        #pragma unroll
                        for (int cg=0; cg < CPC; ++cg) {
                            int slot = b*CPC+cg;
                            #pragma unroll
                            for (int k16=0; k16 < BK2>>4; ++k16) {
                                __nv_bfloat16 *a_ptr = smem_down.x+slot*DXS+((k16*BM)<<4);
                                __nv_bfloat16 *b_ptr = smem_down.w+smem_stage*CPC*DWS+cg*DWS+((k16>>2)*(DN<<6))+((k16&3)<<4);
                                uint64_t a_desc = tcgen05::encode_smem_desc(a_ptr, BM);
                                uint64_t b_desc = tcgen05::encode_smem_desc_sw128(b_ptr);
                                tcgen05::mma_f16(tmem_acc+b*ACC+((stage&1)*DN), a_desc, b_desc, idesc, cg != 0 || k16 != 0);
                            }
                        }
                    }
                    tcgen05::commit_mbarrier(*bar_mma[smem_stage]);
                        
                    }
                    if (stage) down_epilogue(stage-1);
                    bar_mma[smem_stage].await(phase);
                    if (warp_id == 0 && lane_id == 0) recycle_arrive(smem_stage);

                    ++smem_stage;
                }
                down_epilogue(n_stages_down-1);
                asm volatile("cp.async.bulk.wait_group 0;\n");
            }
            if (threadIdx.x >= PRODUCER_THREADS && threadIdx.x < PRODUCER_THREADS+32) {
                tcgen05::tmem_free(tmem_acc, ACC*BPC*CPC);
            }
        };

        if (is_prod) producer();
        else consumer();
    }

    static __global__ void gather_x_kernel(
        const __nv_bfloat16 *__restrict__ x,
        __nv_bfloat16 *__restrict__ x_sorted,
        const int32_t *__restrict__ sorted_token_ids,
        const int32_t *__restrict__ num_tokens_post_padded,
        int32_t top_k,
        int K
    ) {
        int row = static_cast<int>(blockIdx.x)*(blockDim.x>>5) + (static_cast<int>(threadIdx.x)>>5);
        if (row >= *num_tokens_post_padded) return;
        int tdest = sorted_token_ids[row];
        int tok = tdest >= 0 ? tdest/top_k : 0;
        auto *src = reinterpret_cast<const uint4 *>(x) + (static_cast<int64_t>(tok)*K>>3);
        auto *dst = reinterpret_cast<uint4 *>(x_sorted) + (static_cast<int64_t>(row)*K>>3);
        for (int i=static_cast<int>(threadIdx.x)&31; i < K>>3; i += 32)
            dst[i] = __ldg(src+i);
    }

    template <int BM, int CK, int TCN, int STAGES, int PRODUCER_THREADS, int PAIR>
    static __global__ __launch_bounds__(128+PRODUCER_THREADS, PAIR > 1 ? 1 : 2) void fused_moe_down_kernel(
        const __grid_constant__ CUtensorMap map_act,
        const __grid_constant__ CUtensorMap map_w2d,
        __nv_bfloat16 *__restrict__ out,
        const int32_t *__restrict__ sorted_token_ids,
        const int32_t *__restrict__ expert_idxs,
        const int32_t *__restrict__ num_tokens_post_padded,
        const float *__restrict__ topk_weights,
        int32_t top_k,
        int M,
        int K,
        int H
    ) {
        static constexpr int WSD = TCN*CK;   // w2 elems per stage
        static constexpr int XSD = BM*CK;    // act elems per block and stage
        struct pod_t {
            alignas(1024) __nv_bfloat16 w[STAGES*WSD];
            alignas(1024) __nv_bfloat16 x[STAGES*PAIR*XSD];
        };
        alignas(1024) extern __shared__ uint8_t smem_raw[];
        auto &smem = *reinterpret_cast<pod_t *>(smem_raw);
        alignas(8) __shared__ barrier bar_copy[STAGES];
        alignas(8) __shared__ barrier bar_recycle[STAGES];
        alignas(8) __shared__ barrier bar_mma[STAGES];
        alignas(4) __shared__ uint32_t tmem_acc;
        __shared__ __nv_bfloat16 topk_scales[BM*PAIR];
        __shared__ int32_t row_tok[BM*PAIR];

        const int by_row = static_cast<int>(blockIdx.y);
        const int bx_col = static_cast<int>(blockIdx.x);
        int total_padded = *num_tokens_post_padded;
        int rb0 = by_row*PAIR;
        if (BM*rb0 >= total_padded) return;
        int nb = min(PAIR, total_padded/BM - rb0);
        int expert0 = expert_idxs[rb0];
        bool same = nb < 2 || expert0 == expert_idxs[rb0+1];
        int lane_id = threadIdx.x&31;
        bool is_prod = threadIdx.x < PRODUCER_THREADS;
        int warp_id = (is_prod ? threadIdx.x : threadIdx.x-PRODUCER_THREADS) >> 5;
        if (!threadIdx.x) {
            for (int i=0; i < STAGES; ++i) {
                bar_copy[i].init(PRODUCER_THREADS+1);
                bar_recycle[i].init(1);
                bar_mma[i].init(1);
            }
            asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
        }
        __syncthreads();
        if (threadIdx.x >= PRODUCER_THREADS && threadIdx.x < PRODUCER_THREADS+32)
            tcgen05::tmem_alloc(&tmem_acc, TCN*PAIR);
        __syncthreads();
        for (int r = threadIdx.x; r < BM*nb; r += blockDim.x) {
            int tdest = sorted_token_ids[rb0*BM+r];
            row_tok[r] = tdest >= 0 ? tdest / top_k : -1;
            if (tdest >= 0 && row_tok[r] < M) topk_scales[r] = __float2bfloat16(topk_weights[tdest]);
        }
        __syncthreads();
        int n_stages = H / CK;
        int total_stages = same ? n_stages : nb*n_stages;
        if (is_prod) {
            int pre = min(STAGES, total_stages);
            for (int s=0; s < pre; ++s) {
                bar_recycle[s].await(0);
                if (!threadIdx.x) {
                    int p = same ? 0 : s/n_stages, ks = same ? s : s%n_stages;
                    int nact = same ? nb : 1;
                    bar_copy[s].expect_nb((WSD+nact*XSD)*sizeof(__nv_bfloat16));
                    cp_async::load3d(smem.w+s*WSD, &map_w2d, *bar_copy[s], ks*CK, expert_idxs[rb0+p]*K+bx_col*TCN, 0);
                }
            }
            cudaGridDependencySynchronize();
            int smem_stage = 0, phase = 0;
            for (int stage=0; stage < total_stages; ++stage) {
                if (smem_stage == STAGES) { phase^=1; smem_stage = 0; }
                int p = same ? 0 : stage/n_stages, ks = same ? stage : stage%n_stages;
                if (stage >= pre) {
                    bar_recycle[smem_stage].await(phase);
                    if (!threadIdx.x) {
                        int nact = same ? nb : 1;
                        bar_copy[smem_stage].expect_nb((WSD+nact*XSD)*sizeof(__nv_bfloat16));
                        cp_async::load3d(smem.w+smem_stage*WSD, &map_w2d, *bar_copy[smem_stage],
                                         ks*CK, expert_idxs[rb0+p]*K+bx_col*TCN, 0);
                    }
                }
                if (!threadIdx.x) {
                    if (same) {
                        for (int b=0; b < nb; ++b)
                            cp_async::load(smem.x+(smem_stage*PAIR+b)*XSD, &map_act, *bar_copy[smem_stage], ks*CK, (rb0+b)*BM);
                    } else {
                        cp_async::load(smem.x+smem_stage*PAIR*XSD, &map_act, *bar_copy[smem_stage], ks*CK, (rb0+p)*BM);
                    }
                }
                bar_copy[smem_stage].arrive_cp_async_mem();
                ++smem_stage;
            }
            return;
        }
        int tmem_row = (((threadIdx.x>>5)&3)<<5)+lane_id;
        uint32_t idesc = tcgen05::encode_idesc_format_1(BM, TCN);
        if (warp_id == 0 && lane_id == 0) {
            int smem_stage = 0, phase = 0;
            for (int i=0; i < STAGES; ++i) bar_recycle[i].arrive();
            for (int stage=0; stage < total_stages; ++stage) {
                if (smem_stage == STAGES) { phase^=1; smem_stage = 0; }
                int p = same ? 0 : stage/n_stages, ks = same ? stage : stage%n_stages;
                bar_copy[smem_stage].await(phase);
                #pragma unroll
                for (int k16=0; k16 < CK>>4; ++k16) {
                    auto *__restrict__ b_ptr = smem.w+smem_stage*WSD+(k16<<4);
                    uint64_t b_desc = tcgen05::encode_smem_desc_swz<CK*2>(b_ptr);
                    if (same) {
                        for (int b=0; b < nb; ++b) {
                            auto *__restrict__ a_ptr = smem.x+(smem_stage*PAIR+b)*XSD+(k16<<4);
                            uint64_t a_desc = tcgen05::encode_smem_desc_swz<CK*2>(a_ptr);
                            tcgen05::mma_f16(tmem_acc+b*TCN, a_desc, b_desc, idesc, ks||k16);
                        }
                    } else {
                        auto *__restrict__ a_ptr = smem.x+smem_stage*PAIR*XSD+(k16<<4);
                        uint64_t a_desc = tcgen05::encode_smem_desc_swz<CK*2>(a_ptr);
                        tcgen05::mma_f16(tmem_acc+p*TCN, a_desc, b_desc, idesc, ks||k16);
                    }
                }
                tcgen05::commit_mbarrier(*bar_recycle[smem_stage]);
                ++smem_stage;
            }
            tcgen05::commit_mbarrier(*bar_mma[0]);
            bar_mma[0].await(0);
        }
        asm volatile ("bar.sync 1, 128;\n");
        tcgen05::after_thread_sync();
        if (warp_id < 4) {
            for (int b=0; b < nb; ++b) {
                float scale = __bfloat162float(topk_scales[b*BM+tmem_row]);
                int tok = row_tok[b*BM+tmem_row];
                bool row_valid = tok >= 0 && tok < M;
                __nv_bfloat16 *gdst = out+(row_valid ? tok : 0)*K+bx_col*TCN;
                #pragma unroll
                for (int c32=0; c32 < TCN>>5; ++c32) {
                    float vals[32];
                    tcgen05::ld_32x32b_x32(vals, tcgen05::tmem_addr(tmem_acc+b*TCN, tmem_row, c32<<5));
                    tcgen05::await_ld();
                    if (row_valid) {
                        #pragma unroll
                        for (int i=0; i < 4; ++i)
                            red_global_add_bf16x8(gdst+(((c32<<2)+i)<<3), vals+(i<<3), scale);
                    }
                }
            }
        }
        asm volatile ("bar.sync 1, 128;\n");
        if (threadIdx.x >= PRODUCER_THREADS && threadIdx.x < PRODUCER_THREADS+32)
            tcgen05::tmem_free(tmem_acc, TCN*PAIR);
    }


    template <int BM, int BN, int WN, int STAGES, int BPC, int CPC, bool SPLIT = false>
    static void launch_fused_moe_kernel_impl(
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
        static constexpr auto PRODUCER_THREADS = 32;
        static constexpr bool SP = SPLIT && BPC == 1 && CPC == 1 && STAGES <= 4;
        static constexpr auto BK = (CPC > 1) ? 32 : 64;
        static constexpr size_t SMEM = SP
            ? sizeof(smem_up_pod<STAGES, WN, BM, BK, BN, BPC, CPC>)
            : std::max(
                sizeof(smem_up_pod<STAGES, WN, BM, BK, BN, BPC, CPC>),
                sizeof(smem_down_pod<STAGES, WN, BM, BK, BN, BPC, CPC>)
            );
        if ((N/(BN*WN)) % CPC)
            throw std::runtime_error {"cpc="+std::to_string(CPC)+" must divide N/256 ("+std::to_string(N/(BN*WN))+") evenly"};
        dim3 block = {(WN<<5)+PRODUCER_THREADS, 1, 1};
        dim3 grid = {
            static_cast<uint32_t>(std::ceil(static_cast<double>(N)/static_cast<double>(BN*WN*CPC))),
            static_cast<uint32_t>(std::ceil(static_cast<double>(sorted_num)/static_cast<double>(block_m*BPC))),
            1
        };
        auto *kernel = fused_moe_kernel<BM, BK, BN, WN, STAGES, PRODUCER_THREADS, BPC, CPC, SP>;
        if (cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM) != cudaSuccess)
            throw std::runtime_error {"Failed to set max dynamic shared memory size for fused_moe_kernel: "+std::to_string(SMEM)+" bytes"};

        constexpr auto W_SWZ = BK == 64 ? CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B : CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_64B;
        const uint64_t H = static_cast<uint64_t>(N) >> 1;
        CUtensorMap map_w = init_tmap<4>(
            "w",
            CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            w,
            {static_cast<uint64_t>(K), 128ull, static_cast<uint64_t>(num_experts)<<1, H>>7},
            {static_cast<uint64_t>(K)*sizeof(__nv_bfloat16), H*K*sizeof(__nv_bfloat16), 128ull*K*sizeof(__nv_bfloat16)},
            {static_cast<uint32_t>(BK), 128u, 2u, 1u},
            {1u, 1u, 1u, 1u},
            W_SWZ
        );
        const uint64_t HD = static_cast<uint64_t>(N) >> 1;
        __nv_bfloat16 *act_ws = nullptr;
        __nv_bfloat16 *x_sorted = nullptr;
        if (SPLIT && !SP)
            throw std::runtime_error {"split pipeline requires stages<=4, bpc=1, cpc=1"};
        if constexpr (SP) {
            if (cudaMallocAsync(&act_ws, static_cast<size_t>(sorted_num)*(HD+K)*sizeof(__nv_bfloat16), stream) != cudaSuccess)
                throw std::runtime_error {"failed to allocate the split-pipeline act workspace"};
            x_sorted = act_ws + static_cast<size_t>(sorted_num)*HD;
            if (!g_zstream) {
                cudaStreamCreateWithFlags(&g_zstream, cudaStreamNonBlocking);
                cudaEventCreateWithFlags(&g_e_head, cudaEventDisableTiming);
                cudaEventCreateWithFlags(&g_e_zero, cudaEventDisableTiming);
            }
            cudaEventRecord(g_e_head, stream);
            cudaStreamWaitEvent(g_zstream, g_e_head, 0);
            cudaMemsetAsync(out, 0, static_cast<size_t>(M)*K*sizeof(__nv_bfloat16), g_zstream);
            cudaEventRecord(g_e_zero, g_zstream);
        }
        CUtensorMap map_act_st = init_tmap<3>(
            "act_st",
            CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            SP ? act_ws : reinterpret_cast<__nv_bfloat16 *>(out),
            {8ull, static_cast<uint64_t>(sorted_num), HD>>3},
            {HD*sizeof(__nv_bfloat16), 8ull*sizeof(__nv_bfloat16)},
            {8u, static_cast<uint32_t>(BM), 16u},
            {1u, 1u, 1u},
            CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE
        );
        CUtensorMap map_x = init_tmap<2>(
            "x",
            CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            SP ? x_sorted : x,
            {static_cast<uint64_t>(K), SP ? static_cast<uint64_t>(sorted_num) : static_cast<uint64_t>(M)},
            {static_cast<uint64_t>(K)*sizeof(__nv_bfloat16)},
            {static_cast<uint32_t>(BK), SP ? 128u : 1u},
            {1u, 1u},
            BK == 64 ? CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B
                     : CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_64B
        );
        CUtensorMap map_w2 = init_tmap<3>(
            "w2",
            CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            w2,
            {H, static_cast<uint64_t>(K)*num_experts, 1ull},
            {H*sizeof(__nv_bfloat16), H*K*static_cast<uint64_t>(num_experts)*sizeof(__nv_bfloat16)},
            {64u, static_cast<uint32_t>(CPC > 1 ? 32 : (BK<<1)), 1u},
            {1u, 1u, 1u},
            CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B
        );
        if constexpr (SP) {
            gather_x_kernel<<<(sorted_num+7)/8, 256, 0, stream>>>(
                x, x_sorted, sorted_token_ids, num_tokens_post_padded,
                static_cast<int32_t>(top_k), K
            );
        }
        {
            kernel<<<grid, block, SMEM, stream>>>(
                x,
                w,
                map_x,
                map_w,
                map_w2,
                map_act_st,
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
        if constexpr (SP) {
            static constexpr int TCN = 256;
            static constexpr int CK = 64;
            static constexpr int DSTAGES = 3;
            static constexpr int DPROD = 32;
            static constexpr int DPAIR = 2;
            CUtensorMap map_act_ld = init_tmap<2>(
                "act_ld",
                CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                act_ws,
                {HD, static_cast<uint64_t>(sorted_num)},
                {HD*sizeof(__nv_bfloat16)},
                {static_cast<uint32_t>(CK), static_cast<uint32_t>(BM)},
                {1u, 1u},
                CK == 64 ? CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B
                         : CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_64B
            );
            constexpr auto DW_SWZ = CK == 64 ? CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B
                                             : CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_64B;
            CUtensorMap map_w2d = init_tmap<3>(
                "w2d",
                CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                w2,
                {HD, static_cast<uint64_t>(K)*num_experts, 1ull},
                {HD*sizeof(__nv_bfloat16), HD*static_cast<uint64_t>(K)*num_experts*sizeof(__nv_bfloat16)},
                {CK, TCN, 1u},
                {1u, 1u, 1u},
                DW_SWZ
            );
            cudaStreamWaitEvent(stream, g_e_zero, 0);   // out zeroed before the down reds
            auto *dkernel = fused_moe_down_kernel<BM, CK, TCN, DSTAGES, DPROD, DPAIR>;
            static constexpr size_t DSMEM_SZ = DSTAGES*(TCN*CK + DPAIR*BM*CK)*sizeof(__nv_bfloat16);
            if (cudaFuncSetAttribute(dkernel, cudaFuncAttributeMaxDynamicSharedMemorySize, DSMEM_SZ) != cudaSuccess)
                throw std::runtime_error {"failed to set smem for fused_moe_down_kernel"};
            dim3 dgrid = {static_cast<uint32_t>(K/TCN), static_cast<uint32_t>((sorted_num/BM+DPAIR-1)/DPAIR), 1};
            dim3 dblock = {128+DPROD, 1, 1};
            cudaLaunchConfig_t dcfg = {};
            dcfg.gridDim = dgrid;
            dcfg.blockDim = dblock;
            dcfg.dynamicSmemBytes = DSMEM_SZ;
            dcfg.stream = stream;
            cudaLaunchAttribute dattr[1];
            dattr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
            dattr[0].val.programmaticStreamSerializationAllowed = 1;
            dcfg.attrs = dattr;
            dcfg.numAttrs = 1;
            cudaError_t derr = cudaLaunchKernelEx(
                &dcfg, dkernel,
                map_act_ld, map_w2d, out,
                sorted_token_ids, expert_idxs, num_tokens_post_padded, topk_weights,
                static_cast<int32_t>(top_k), M, K, static_cast<int>(HD)
            );
            if (derr != cudaSuccess)
                throw std::runtime_error {"fused_moe_down_kernel launch failed: "+std::string{cudaGetErrorString(derr)}};
            if (cudaFreeAsync(act_ws, stream) != cudaSuccess)
                throw std::runtime_error {"failed to free the split-pipeline act workspace"};
        }
    }

    template <int BM, int BN, int WN, int STAGES, int BPC = 2, int CPC = 1>
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
        cudaStream_t stream,
        bool split = true
    ) {
        if (split)
            launch_fused_moe_kernel_impl<BM, BN, WN, STAGES, BPC, CPC, true>(x, w, w2, out, sorted_token_ids, expert_idxs, num_tokens_post_padded, topk_weights, top_k, M, K, N, num_experts, sorted_num, block_m, stream);
        else
            launch_fused_moe_kernel_impl<BM, BN, WN, STAGES, BPC, CPC>(x, w, w2, out, sorted_token_ids, expert_idxs, num_tokens_post_padded, topk_weights, top_k, M, K, N, num_experts, sorted_num, block_m, stream);
    }

    template <int BM, int BN, int WN, int STAGES>
    static void dispatch_cpc(
        int bpc,
        int cpc,
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
        cudaStream_t stream,
        bool split
    ) {
        if (bpc != 1 || cpc != 1)
            throw std::runtime_error {"bpc=cpc=1 is the only supported configuration (variant paths were removed)"};
        launch_fused_moe_kernel<BM, BN, WN, STAGES, 1, 1>(x, w, w2, out, sorted_token_ids, expert_idxs, num_tokens_post_padded, topk_weights, top_k, M, K, N, num_experts, sorted_num, block_m, stream, split);
    }

    template <int BM, int BN, int WN>
    static void dispatch_stages(
        int stages,
        int bpc,
        int cpc,
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
        cudaStream_t stream,
        bool split
    ) {
        switch (stages) {
            case 1: dispatch_cpc<BM, BN, WN, 1>(bpc, cpc, x, w, w2, out, sorted_token_ids, expert_idxs, num_tokens_post_padded, topk_weights, top_k, M, K, N, num_experts, sorted_num, block_m, stream, split); break;
            case 2: dispatch_cpc<BM, BN, WN, 2>(bpc, cpc, x, w, w2, out, sorted_token_ids, expert_idxs, num_tokens_post_padded, topk_weights, top_k, M, K, N, num_experts, sorted_num, block_m, stream, split); break;
            case 3: dispatch_cpc<BM, BN, WN, 3>(bpc, cpc, x, w, w2, out, sorted_token_ids, expert_idxs, num_tokens_post_padded, topk_weights, top_k, M, K, N, num_experts, sorted_num, block_m, stream, split); break;
            case 4: dispatch_cpc<BM, BN, WN, 4>(bpc, cpc, x, w, w2, out, sorted_token_ids, expert_idxs, num_tokens_post_padded, topk_weights, top_k, M, K, N, num_experts, sorted_num, block_m, stream, split); break;
            case 5: dispatch_cpc<BM, BN, WN, 5>(bpc, cpc, x, w, w2, out, sorted_token_ids, expert_idxs, num_tokens_post_padded, topk_weights, top_k, M, K, N, num_experts, sorted_num, block_m, stream, split); break;
            default: throw std::runtime_error {"Invalid stages: "+std::to_string(stages)};
        }
    }

    template <int BM>
    static void dispatch_bn_wn(
        int bn,
        int wn,
        int stages,
        int bpc,
        int cpc,
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
        cudaStream_t stream,
        bool split
    ) {
        switch (((bn&0xff)<<8)+(wn&0xff)) {
            case (32<<8)+8: dispatch_stages<BM, 32, 8>(stages, bpc, cpc, x, w, w2, out, sorted_token_ids, expert_idxs, num_tokens_post_padded, topk_weights, top_k, M, K, N, num_experts, sorted_num, block_m, stream, split); break;
            case (64<<8)+4: dispatch_stages<BM, 64, 4>(stages, bpc, cpc, x, w, w2, out, sorted_token_ids, expert_idxs, num_tokens_post_padded, topk_weights, top_k, M, K, N, num_experts, sorted_num, block_m, stream, split); break;
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
        int cpc,
        bool split,
        cudaStream_t stream
    ) {
        switch (block_m) {
            case 128:
                dispatch_bn_wn<128>(block_n, warp_n, stages, bpc, cpc, x, w, w2, out, sorted_token_ids, expert_idxs, num_tokens_post_padded, topk_weights, top_k, M, K, N, num_experts, sorted_num, block_m, stream, split);
                break;
            default:
                throw std::runtime_error{
                    "tcgen05 Blackwell path currently supports block_m=128 only; "
                    "BM=64 needs Layout-F epilogue, BM=32 needs tcgen05.mma.ws, BM=8/16 are unsupported."
                };
        }
    }
}
