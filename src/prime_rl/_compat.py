"""Compatibility shims for upstream regressions.

Imported early (before any model code) by trainer and orchestrator entrypoints.
Each shim documents the upstream issue and removal condition.
"""

# ---------------------------------------------------------------------------
# transformers >= 5.5 hub_kernels offline regression
#
# lazy_load_kernel() resolves kernel versions via HfApi().list_repo_refs(),
# which raises OfflineModeIsEnabled when HF_HUB_OFFLINE=1 — even if the
# kernel is already cached locally. Only FileNotFoundError and AssertionError
# are caught; OfflineModeIsEnabled (a ConnectionError subclass) is not.
#
# This breaks Mamba-based models (NemotronH, Zamba2, Jamba, etc.) on SLURM
# worker nodes where HF_HUB_OFFLINE=1 is set.
#
# Fix: patch the except clause to also catch ConnectionError.
# Remove once huggingface/transformers fixes lazy_load_kernel offline handling.
# ---------------------------------------------------------------------------
try:
    import transformers.integrations.hub_kernels as _hub_kernels
except ImportError:
    _hub_kernels = None  # transformers < 5.5, no patch needed

if _hub_kernels is not None:
    from huggingface_hub.errors import OfflineModeIsEnabled

    _original_lazy_load_kernel = _hub_kernels.lazy_load_kernel

    def _patched_lazy_load_kernel(kernel_name, mapping=_hub_kernels._KERNEL_MODULE_MAPPING):
        try:
            return _original_lazy_load_kernel(kernel_name, mapping)
        except OfflineModeIsEnabled:
            # Return None so callers can use their non-hub implementation.
            mapping[kernel_name] = None
            return None

    _hub_kernels.lazy_load_kernel = _patched_lazy_load_kernel

# ---------------------------------------------------------------------------
# Inductor compile-worker fork deadlock (pytorch/pytorch#200848)
#
# Inductor shuts its compile-worker pool down when idle and re-forks it on the
# next compile; a fork during the old pool's teardown can deadlock the rank.
#
# Fix: keep the pool alive, and fail after 30 min instead of hanging. Both env
# vars still take precedence.
# Remove the quiesce line once the minimum torch contains the fix for
# pytorch/pytorch#200848.
# ---------------------------------------------------------------------------
import os

import torch._inductor.config

torch._inductor.config.quiesce_async_compile_pool = False
if "TORCHINDUCTOR_COMPILE_WORKER_WAIT_TIMEOUT" not in os.environ:
    torch._inductor.config.compile_worker_wait_timeout = 1800

# ---------------------------------------------------------------------------
# ptxas 12.8 to 13.0 miscompile on Hopper
#
# Triton compiles sm90 kernels with its bundled ptxas 12.9, whose optimizer
# drops the -256 from the packed 16-bit channel offset in the XBLOCK=1024,
# num_warps=4 config of Inductor's qkv-gradient cat kernel for partial-rotary
# attention (e.g. GLM-4 MoE). That config reads 512 bytes past the key
# gradient: wrong values, or an illegal memory access when the read crosses
# into unmapped memory, as during a cold-cache autotune. ptxas -O0, 13.3 and
# 13.4 compile it correctly.
#
# Fix: compile every architecture with Triton's bundled Blackwell ptxas
# (13.3). The env var still takes precedence.
# Remove once Triton bundles ptxas 13.3 or newer for sm90.
# ---------------------------------------------------------------------------
import triton.knobs

if "TRITON_PTXAS_PATH" not in os.environ:
    os.environ["TRITON_PTXAS_PATH"] = triton.knobs.nvidia.ptxas_blackwell.path

# ---------------------------------------------------------------------------
# Inductor sym_size lowering inside flex attention subgraphs
#
# The backward of a flex_attention score_mod that reads a captured tensor with a gradient (Inkling 2's
# relative bias) scatters into zeros of that tensor's shape. Under dynamic shapes the shape can be an
# expression of the graph's symbols, such as a context-parallel rank's whole row (cp times its shard
# length), and the score_mod's backward subgraph then reads it with aten.sym_size. Inductor lowers
# sym_size from the meta of V.graph.current_node, which inside the subgraph is the flex attention node,
# and int() of its tuple value raises.
#
# Fix: inside a flex attention subgraph, read the size from the lowered tensor.
# Remove once torch's sym_size lowering handles pointwise subgraphs.
# ---------------------------------------------------------------------------
import torch
from torch._inductor import lowering as _inductor_lowering
from torch._inductor.subgraph_lowering import PointwiseSubgraphLowering
from torch._inductor.virtualized import V

_sym_size_lowering = _inductor_lowering.lowerings[torch.ops.aten.sym_size.int]


def _sym_size_in_subgraphs(tensor, dim):
    if isinstance(V.graph, PointwiseSubgraphLowering):
        return tensor.get_size()[dim]
    return _sym_size_lowering(tensor, dim)


_inductor_lowering.lowerings[torch.ops.aten.sym_size.int] = _sym_size_in_subgraphs
