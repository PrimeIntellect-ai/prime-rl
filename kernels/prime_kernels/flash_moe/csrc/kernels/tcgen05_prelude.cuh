#pragma once

#include <atomic>
#include <array>
#include <mutex>
#include <stdexcept>
#include <string>

#include <cuda/ptx>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace pi {
    inline std::once_flag g_dlsym_once;
    inline std::atomic<PFN_cuTensorMapEncodeTiled_v12000> g_tmap_encode_fn = nullptr;
    [[nodiscard]] inline PFN_cuTensorMapEncodeTiled_v12000 lookup_proc_address_encode_tmap() {
        std::call_once(g_dlsym_once, [] {
            cudaDriverEntryPointQueryResult stat;
            PFN_cuTensorMapEncodeTiled_v12000 pfn = nullptr;
            auto res = cudaGetDriverEntryPointByVersion(
                "cuTensorMapEncodeTiled",
                reinterpret_cast<void **>(&pfn),
                12000,
                cudaEnableDefault,
                &stat
            );
            if (res != cudaSuccess || stat != cudaDriverEntryPointSuccess)
                throw std::runtime_error {"Failed to get address of cuTensorMapEncodeTiled: " + std::string{cudaGetErrorString(res)}};
            g_tmap_encode_fn.store(pfn, std::memory_order_release);
        });
        return g_tmap_encode_fn.load(std::memory_order_acquire);
    }

    namespace tcgen05 {
        [[nodiscard]] __device__ __forceinline__ uint64_t encode_smem_desc(const void *p, int32_t height) {
            constexpr uint64_t stride_dim = 128;            // stride dim N
            constexpr uint64_t mat_base_offs = 0;           // mat base offset
            constexpr uint64_t lead_dim_stride_mod = 0;     // leading dim stride mode, 0=byte offs rel, 1=byte offs abs
            constexpr uint64_t swizzling_mode = 0;          // swizzle mode, 0=no swizzling, 1=128B with 32B atomic swizzling, 2=128B swizzling, 4=64B swizzling, 6=32B swizzling
            static_assert(!swizzling_mode || !(swizzling_mode&(swizzling_mode-1)));
            auto mat_start_addr = static_cast<uint32_t>(__cvta_generic_to_shared(p));
            return
                ((mat_start_addr>>4)&((1ull<<14ull)-1ull))
                | ((((static_cast<uint64_t>(height & ((1ull<<14ull)-1ull)))))<<16)  // mat desc
                | ((((stride_dim>>4))&((1ull<<14ull)-1ull))<<32)                    // stride dim byte offset
                | ((0b1ull&3)<<46)                                                  // fixed const
                | ((mat_base_offs & 7)<<49)
                | ((lead_dim_stride_mod & 1)<<52)
                | (0xb0ull<<53)                                                     // fixed const
                | ((swizzling_mode & 7)<<61);
        }

        [[nodiscard]] constexpr __device__ __forceinline__ uint32_t encode_idesc_format_1(int32_t m, int32_t n) {
            constexpr uint32_t sparsity_sel = 0;  // ignored
            constexpr uint32_t sparsity_mode = 0; // 0=dense,1=sparse
            constexpr uint32_t sat_mode = 0;      // 0=no sat, 1=sat
            constexpr uint32_t mtype = 1;         // 1=f32
            constexpr uint32_t atype = 1;         // 1=bf16
            constexpr uint32_t btype = 1;         // 1=bf16
            constexpr uint32_t aᵀ = 0;            // 1 = Aᵀ
            constexpr uint32_t bᵀ = 0;            // 1 = Bᵀ
            constexpr uint32_t an = 0;            // 1 = -A
            constexpr uint32_t bn = 0;            // 1 = -B
            constexpr uint32_t bws_shift = 0;     // no shift = 0 maximum shift of 8 = 1 maximum shift of 16 = 2 maximum shift of 32 = 3
            return
                (sparsity_sel & 3)
                | ((sparsity_mode & 1)<<2)
                | ((sat_mode & 1)<<3)
                | ((mtype & 3)<<4)
                | ((atype & 7)<<7)
                | ((btype & 7)<<10)
                | ((an & 1)<<13)
                | ((bn & 1)<<14)
                | ((aᵀ & 1)<<15)
                | ((bᵀ & 1)<<16)
                | (((static_cast<uint32_t>(n)>>3) & 63)<<17)
                | (((static_cast<uint32_t>(m)>>4) & 31)<<24)
                | ((bws_shift & 3)<<30);
        }

        [[nodiscard]] constexpr __device__ __forceinline__ uint32_t encode_idesc_block_scaled_e4m3(int32_t m, int32_t n) {
            constexpr uint32_t sparsity_sel = 0;  // ignored
            constexpr uint32_t sparsity_mode = 0; // 0=dense, 1=sparse
            constexpr uint32_t a_sf_id = 0;       // patched per MMA, see idesc_with_sf_id
            constexpr uint32_t b_sf_id = 0;       // patched per MMA, see idesc_with_sf_id
            constexpr uint32_t atype = 0;         // mxf8f6f4: 0=e4m3, 1=e5m2, 3=e2m3, 4=e3m2, 5=e2m1
            constexpr uint32_t btype = 0;
            constexpr uint32_t aᵀ = 0;            // 1 = Aᵀ (MN-major)
            constexpr uint32_t bᵀ = 0;            // 1 = Bᵀ (MN-major)
            constexpr uint32_t an = 0;            // 1 = -A
            constexpr uint32_t bn = 0;            // 1 = -B
            constexpr uint32_t scale_fmt = 1;     // 0=e4m3, 1=e8m0
            constexpr uint32_t k_size = 0;        // mxf8f6f4 dense: MMA-K = 32
            return
                (sparsity_sel & 3)
                | ((sparsity_mode & 1)<<2)
                | ((b_sf_id & 3)<<4)
                | ((atype & 7)<<7)
                | ((btype & 7)<<10)
                | ((an & 1)<<13)
                | ((bn & 1)<<14)
                | ((aᵀ & 1)<<15)
                | ((bᵀ & 1)<<16)
                | (((static_cast<uint32_t>(n)>>3) & 63)<<17)
                | ((scale_fmt & 1)<<23)
                | (((static_cast<uint32_t>(m)>>4) & 31)<<24)
                | ((a_sf_id & 3)<<29)
                | ((k_size & 1)<<31);
        }

        [[nodiscard]] constexpr __device__ __forceinline__ uint32_t idesc_with_sf_id(uint32_t idesc, uint32_t sf_id) {
            return (idesc & ~((3u<<4)|(3u<<29))) | ((sf_id & 3)<<4) | ((sf_id & 3)<<29);
        }

        [[nodiscard]] __device__ __forceinline__ uint32_t tmem_addr(uint32_t base, uint32_t row, uint32_t col) {
            return base+(row<<16)+col;
        }

        __device__ __forceinline__ void mma_f16(
            uint32_t d_tmem,
            uint64_t desc_a,
            uint64_t desc_b,
            uint32_t idesc,
            int32_t enable_input_d
        ) {
            constexpr uint32_t mask[4] = {};
            asm volatile(
                "{\n\t"
                ".reg .pred p;\n\t"
                "setp.ne.b32 p, %4, 0;\n\t"
                "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, {%5, %6, %7, %8}, p; \n\t"
                "}\n"
                :
                : "r"(d_tmem), "l"(desc_a), "l"(desc_b),
                  "r"(idesc),
                  "r"(enable_input_d),
                  "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3])
                : "memory"
            );
        }

        __device__ __forceinline__ void mma_mxf8f6f4(
            uint32_t d_tmem,
            uint64_t desc_a,
            uint64_t desc_b,
            uint32_t idesc,
            uint32_t sfa_tmem,
            uint32_t sfb_tmem,
            int32_t enable_input_d
        ) {
            asm volatile(
                "{\n\t"
                ".reg .pred p;\n\t"
                "setp.ne.b32 p, %4, 0;\n\t"
                "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0], %1, %2, %3, [%5], [%6], p; \n\t"
                "}\n"
                :
                : "r"(d_tmem), "l"(desc_a), "l"(desc_b),
                  "r"(idesc),
                  "r"(enable_input_d),
                  "r"(sfa_tmem), "r"(sfb_tmem)
                : "memory"
            );
        }

        __device__ __forceinline__ void cp_sf_32x128b(uint32_t dst_tmem, uint64_t smem_desc) {
            asm volatile(
                "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;\n"
                :
                : "r"(dst_tmem), "l"(smem_desc)
                : "memory"
            );
        }

        [[nodiscard]] __device__ __forceinline__ uint64_t encode_sf_smem_desc(const void *p) {
            return encode_smem_desc(p, 0);
        }

        __device__ __forceinline__ void commit_mbarrier(uint64_t *mbar) {
            asm volatile(
                "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n"
                :
                : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(mbar)))
                : "memory"
            );
        }

        __device__ __forceinline__ void after_thread_sync() {
            asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
        }

        __device__ __forceinline__ void await_ld() {
            asm volatile("tcgen05.wait::ld.sync.aligned;\n" ::: "memory");
        }

        __device__ __forceinline__ void tmem_alloc(void *dst_smem, int32_t cols) {
            asm volatile(
                "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n"
                :
                : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst_smem))), "r"(cols)
                : "memory"
            );
        }

        __device__ __forceinline__ void tmem_free(uint32_t taddr, int32_t cols) {
            asm volatile(
                "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n"
                :
                : "r"(taddr), "r"(cols)
                : "memory"
            );
        }

        __device__ __forceinline__ void ld_32x32b_x8(float (&dst)[8], uint32_t addr) {
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];\n"
                : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3]),
                  "=f"(dst[4]), "=f"(dst[5]), "=f"(dst[6]), "=f"(dst[7])
                : "r"(addr)
                : "memory"
            );
        }
    }

    namespace cp_async {
        __device__ __forceinline__ void load(void *dst, const void *tma_map, uint64_t *bar, int32_t row, int32_t col) {
            static_assert(sizeof(void *) == sizeof(uint64_t));
            auto ptma = reinterpret_cast<uint64_t>(tma_map);
            auto pmbar = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
            auto pdst = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];\n"
                :
                : "r"(pdst), "l"(ptma), "r"(pmbar), "r"(row), "r"(col)
                : "memory"
            );
        }

        __device__ __forceinline__ void load3d(
            void *dst,
            const void *tma_map,
            uint64_t *bar,
            int32_t c0,
            int32_t c1,
            int32_t c2
        ) {
            asm volatile(
                "cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes "
                "[%0], [%1, {%3, %4, %5}], [%2];\n"
                :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst))),
                    "l"(reinterpret_cast<uint64_t>(tma_map)),
                    "r"(static_cast<uint32_t>(__cvta_generic_to_shared(bar))),
                    "r"(c0), "r"(c1), "r"(c2) : "memory"
            );
        }

        __device__ __forceinline__ void load5d(
            void *dst,
            const void *tma_map,
            uint64_t *bar,
            int32_t c0,
            int32_t c1,
            int32_t c2,
            int32_t c3,
            int32_t c4
        ) {
            asm volatile(
                "cp.async.bulk.tensor.5d.shared::cta.global.tile.mbarrier::complete_tx::bytes "
                "[%0], [%1, {%3, %4, %5, %6, %7}], [%2];\n"
                :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst))),
                    "l"(reinterpret_cast<uint64_t>(tma_map)),
                    "r"(static_cast<uint32_t>(__cvta_generic_to_shared(bar))),
                    "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(c4) : "memory"
            );
        }

        template <const int32_t nb, typename Dst, typename Src>
        __device__ __forceinline__ void cg256(Dst dst, Src src) {
            asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], %2;\n" :: "r"(dst), "l"(src), "n"(nb));
        }

        template <const int32_t nb, typename Dst, typename Src>
        __device__ __forceinline__ void cg64(Dst dst, Src src) {
            asm volatile("cp.async.ca.shared.global.L2::64B [%0], [%1], %2;\n" :: "r"(dst), "l"(src), "n"(nb));
        }

        __device__ __forceinline__ void commit_group() {
            asm volatile("cp.async.commit_group;\n" ::: "memory");
        }

        template <const int32_t n>
        __device__ __forceinline__ void await_group() {
            asm volatile("cp.async.wait_group %0;\n" :: "n"(n));
        }
    }

    struct barrier final {
        __device__ __forceinline__ void init(int32_t threads, int32_t transactions=0) {
            asm volatile(
                "mbarrier.init.shared::cta.b64 [%0], %1;\n"
                :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(this))), "r"(threads+transactions)
            );
        }

        __device__ __forceinline__ void arrive_cp_async_mem() {
            asm volatile(
                "cp.async.mbarrier.arrive.noinc.shared.b64 [%0];\n"
                :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(this)))
            );
        }

        __device__ __forceinline__ void arrive(uint32_t num=1) {
            asm volatile(
                "mbarrier.arrive.shared.b64 _, [%0], %1;\n"
                :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(this))), "r"(num)
                : "memory"
            );
        }

        __device__ __forceinline__ void expect_nb(uint32_t nb) {
            asm volatile(
                "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;\n"
                :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(this))), "r"(nb)
            );
        }

        __device__ __forceinline__ void await(int32_t phase) {
            asm volatile(
                "{\n"
                ".reg .pred P1;\n"
                "AWAIT:\n"
                "mbarrier.try_wait.parity.shared.b64 P1, [%0], %1;\n"
                "@!P1 bra.uni AWAIT;\n"
                "}\n"
                :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(this))), "r"(phase)
            );
        }

        constexpr __device__ __host__ __forceinline__ uint64_t *operator*() noexcept { return &m_handle; }
        constexpr __device__ __host__ __forceinline__ const uint64_t *operator*() const noexcept { return &m_handle; }

    private:
        uint64_t m_handle;
    };
    static_assert(sizeof(barrier) == sizeof(uint64_t) && alignof(barrier) == alignof(uint64_t));

    [[nodiscard]] static __device__ __forceinline__ float swiglu(float x, float w) {
        return w*(x/(1.f+__expf(-x)));
    }

    template <const size_t Rank>
    [[nodiscard]] inline CUtensorMap init_tmap(
        const char *name,
        CUtensorMapDataType dtype,
        const void *ptr,
        const std::array<uint64_t, Rank> &shape,
        const std::array<uint64_t, Rank - 1> &strides,
        const std::array<uint32_t, Rank> &box,
        const std::array<uint32_t, Rank> &elem_strides,
        CUtensorMapSwizzle swizzle
    ) {
        CUtensorMap map = {};
        CUresult res = (*lookup_proc_address_encode_tmap())(
            &map,
            dtype,
            Rank,
            const_cast<void *>(ptr),
            shape.data(),
            strides.data(),
            box.data(),
            elem_strides.data(),
            CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
            swizzle,
            CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        );
        if (res != CUDA_SUCCESS) {
            throw std::runtime_error {"Failed to encode tensor map: "+std::string{name}+": "+std::to_string(res)};
        }
        return map;
    }

    [[nodiscard]] inline CUtensorMap init_tmap_kmajor_3d(
        const char *name,
        const void *ptr,
        uint64_t global_height,
        uint64_t global_width,
        uint32_t tile_height,
        uint32_t tile_width
    ) {
        if ((7&global_width) || (7&tile_width))
            throw std::runtime_error{std::string{name}+": K-like dim must be a multiple of 8 bf16"};
        return init_tmap<3>(
            name,
            CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            ptr,
            {8ull, global_height, global_width>>3},
            {global_width*sizeof(__nv_bfloat16), 16ull},
            {8u, tile_height, tile_width>>3},
            {1u, 1u, 1u},
            CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE
        );
    }

    [[nodiscard]] inline CUtensorMap init_tmap_kmajor_3d_fp8(
        const char *name,
        const void *ptr,
        uint64_t global_height,
        uint64_t global_width,
        uint32_t tile_height,
        uint32_t tile_width
    ) {
        if ((15&global_width) || (15&tile_width))
            throw std::runtime_error{std::string{name}+": K-like dim must be a multiple of 16 fp8"};
        return init_tmap<3>(
            name,
            CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8,
            ptr,
            {16ull, global_height, global_width>>4},
            {global_width, 16ull},
            {16u, tile_height, tile_width>>4},
            {1u, 1u, 1u},
            CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE
        );
    }

    [[nodiscard]] inline CUtensorMap init_tmap_swiglu_kmajor_5d(
        const char *name,
        CUtensorMapDataType dtype,
        uint64_t elem_size,
        const void *ptr,
        uint64_t num_experts,
        uint64_t global_height,    // N == 2H, rows per expert
        uint64_t global_width,     // K
        uint32_t tile_height,      // rows of the packed tile, must be a multiple of 256
        uint32_t tile_width        // BK
    ) {
        const uint64_t vec = 16/elem_size;                   // elements per 16B inner atom
        if ((global_width%vec) || (tile_width%vec))
            throw std::runtime_error{std::string{name}+": K-like dim must be a multiple of "+std::to_string(vec)};
        if (255&global_height)
            throw std::runtime_error{std::string{name}+": N must be a multiple of 256 to interleave gate/up"};
        if (255&tile_height)
            throw std::runtime_error{std::string{name}+": tile height must be a multiple of 256 to interleave gate/up"};
        const uint64_t half = global_height>>1;              // H
        const uint64_t row = global_width*elem_size;         // bytes per weight row
        return init_tmap<5>(
            name,
            dtype,
            ptr,
            {vec, 128ull, num_experts<<1, half>>7, global_width/vec},
            {row, half*row, 128ull*row, 16ull},
            {static_cast<uint32_t>(vec), 128u, 2u, tile_height>>8, static_cast<uint32_t>(tile_width/vec)},
            {1u, 1u, 1u, 1u, 1u},
            CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE
        );
    }

    [[nodiscard]] inline CUtensorMap init_tmap_swiglu_kmajor_5d_bf16(
        const char *name,
        const void *ptr,
        uint64_t num_experts,
        uint64_t global_height,
        uint64_t global_width,
        uint32_t tile_height,
        uint32_t tile_width
    ) {
        return init_tmap_swiglu_kmajor_5d(
            name,
            CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            sizeof(__nv_bfloat16),
            ptr, num_experts, global_height, global_width, tile_height, tile_width
        );
    }

    [[nodiscard]] inline CUtensorMap init_tmap_swiglu_kmajor_5d_fp8(
        const char *name,
        const void *ptr,
        uint64_t num_experts,
        uint64_t global_height,
        uint64_t global_width,
        uint32_t tile_height,
        uint32_t tile_width
    ) {
        return init_tmap_swiglu_kmajor_5d(
            name,
            CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8,
            sizeof(uint8_t),
            ptr, num_experts, global_height, global_width, tile_height, tile_width
        );
    }
}
