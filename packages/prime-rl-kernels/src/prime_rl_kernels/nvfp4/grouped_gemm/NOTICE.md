# NVFP4 grouped GEMM provenance

The CUTLASS grouped-GEMM implementation in `csrc/` is adapted from
[meta-pytorch/MSLK](https://github.com/meta-pytorch/MSLK) commit
[`e0870cca46356ce9bbb16c26fb0b5ab1f9435445`](https://github.com/meta-pytorch/MSLK/commit/e0870cca46356ce9bbb16c26fb0b5ab1f9435445),
which added the B200 per-token-scaled FP4 grouped GEMM.

The relevant MSLK source was copied into this package and then changed to:

- use the `prime_rl_kernels::nvfp4` C++ namespace and a private CUTLASS
  namespace;
- register a `prime_rl_kernels_nvfp4` PyTorch dispatcher operation;
- compile only the two SM100 specializations used here;
- accept per-token activation decode scales and per-expert weight decode
  scales directly from the prime-rl quantizers.

The adapted files remain under the BSD-style terms in `MSLK_LICENSE`.
