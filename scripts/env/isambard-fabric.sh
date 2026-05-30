#!/usr/bin/env bash
# Isambard-AI AIP2 Slingshot/Cassini NCCL fabric. Source before any cross-node NCCL.
# brics/nccl auto-loads aws-ofi-nccl+openmpi -> NCCL_NET="AWS Libfabric", NCCL_SOCKET_IFNAME=hsn,
# FI_CXI_* tuning, OFI plugin on LD_LIBRARY_PATH (validated: probe 4884052, GPUDirect RDMA over cxi0-3).
# Slingshot is libfabric, NOT InfiniBand -> do NOT use ibv_devinfo/NCCL_IB_HCA.
# Idempotent and side-effect-safe to source from `set -euo pipefail` scripts.
command -v module >/dev/null 2>&1 && { module load brics/nccl 2>/dev/null || true; unset NCCL_ALGO; }
