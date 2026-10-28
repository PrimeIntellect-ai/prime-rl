"""The fused per-rank expert-parallel dispatch: `kernels` holds the Triton kernels and their launches, `ops` the slot-map
custom ops with their autograd (`sum_slots`, `scatter_slots`) and `dispatcher` the `FusedPerRankTokenDispatcher`."""
