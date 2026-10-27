Runs:

| run   | label                        | head       | TORCH_LOGS              |
| ----- | ---------------------------- | ---------- | ----------------------- |
| run 0 | main/cell3-graph-breaks/r0   | 1810d5b9ab | recompiles,graph_breaks |
| run 1 | branch/cell3-graph-breaks/r0 | 6431ef77b0 | recompiles,graph_breaks |

Per rank. Breaks: graph break records. Dup: those logged with the user stack suppressed as a repeat. Touch frag.: breaks whose location or user stack touches a fragment below. Recompiles: 'Recompiling function' records. Seqlen guards: guard failures matching /cu_seqlens|seqlen|seq_len|cp_context|cp_group|cp_rank|cp_world/, the CP parameters but also any varlen recompile at cp=1. Limit hits: recompile_limit warnings. Fewer is better in every column.

| run   | rank | breaks | dup | touch frag. | recompiles | seqlen guards | limit hits |
| ----- | ---: | -----: | --: | ----------: | ---------: | ------------: | ---------: |
| run 0 |    0 |     34 |  24 |           8 |         22 |             4 |          0 |
| run 0 |    1 |     34 |  24 |           8 |         22 |             3 |          0 |
| run 0 |    2 |     34 |  24 |           8 |         22 |             4 |          0 |
| run 0 |    3 |     34 |  24 |           8 |         22 |             6 |          0 |
| run 0 |    4 |     34 |  24 |           8 |         22 |             4 |          0 |
| run 0 |    5 |     34 |  24 |           8 |         22 |             7 |          0 |
| run 0 |    6 |     34 |  24 |           8 |         22 |             6 |          0 |
| run 0 |    7 |     34 |  24 |           8 |         23 |             4 |          0 |
| run 1 |    0 |     27 |  19 |           0 |         17 |             3 |          0 |
| run 1 |    1 |     27 |  19 |           0 |         17 |             2 |          0 |
| run 1 |    2 |     27 |  19 |           0 |         17 |             3 |          0 |
| run 1 |    3 |     27 |  19 |           0 |         17 |             5 |          0 |
| run 1 |    4 |     27 |  19 |           0 |         17 |             3 |          0 |
| run 1 |    5 |     27 |  19 |           0 |         17 |             6 |          0 |
| run 1 |    6 |     27 |  19 |           0 |         17 |             5 |          0 |
| run 1 |    7 |     27 |  19 |           0 |         18 |             3 |          0 |

Break records whose location or user stack touches each fragment (all ranks):

| fragment                                | run 0 | run 1 |
| --------------------------------------- | ----: | ----: |
| qwen3_8_flash_next/position_learning.py |     8 |     0 |
| qwen3_8_flash_next/ngram_embedding.py   |    16 |     0 |
| distributed/embedding_parallel.py       |    32 |     0 |
| layers/ngram.py                         |     0 |     0 |
| distributed/ngram_parallel.py           |     0 |     0 |
| torch/distributed/fsdp                  |    32 |     0 |
| any fragment                            |    64 |     0 |

Graph break reasons (numbers as N), records over all ranks:

| reason                                                                                               | run 0 | run 1 |
| ---------------------------------------------------------------------------------------------------- | ----: | ----: |
| gb0036 Dynamic shape operator (aten.repeat_interleave.Tensor)                                        |    88 |    88 |
| gb0036 Dynamic shape operator in autograd.Function (aten.repeat_interleave.Tensor)                   |    32 |    32 |
| gb0098 Skip calling `torch.compiler.disable()`d function in autograd.Function (<function layer_norm_ |    24 |    24 |
| gb0007 Attempted to call function marked as skipped (module: torch._dynamo.decorators, qualname: dis |    32 |     0 |
| gb0098 Skip calling `torch.compiler.disable()`d function (<function causal_conv1d_fwd>)              |    16 |    16 |
| gb0099 Skip inlining `torch.compiler.disable()`d function (<function causal_conv1d_fwd>)             |    16 |    16 |
| gb0098 Skip calling `torch.compiler.disable()`d function (<function layer_norm_gated_fwd>)           |    16 |    16 |
| gb0036 Dynamic shape operator (aten.bincount.default)                                                |    32 |     0 |
| gb0099 Skip inlining `torch.compiler.disable()`d function in autograd.Function (<function causal_con |     8 |    16 |
| gb0098 Skip calling `torch.compiler.disable()`d function (<function chunk_gated_delta_rule>)         |     8 |     8 |

Graph break locations, records over all ranks:

| location                                               | run 0 | run 1 |
| ------------------------------------------------------ | ----: | ----: |
| fla/ops/utils/index.py:128                             |   120 |   120 |
| fla/modules/norm/fused_norm_gate.py:656                |    40 |    40 |
| fla/modules/causal_conv1d/ops.py:958                   |    24 |    32 |
| fla/utils/decorators.py:325                            |    16 |    16 |
| torch/distributed/fsdp/_fully_shard/_fsdp_common.py:30 |    32 |     0 |
| prime_rl/trainer/distributed/embedding_parallel.py:35  |    32 |     0 |
| prime_rl/trainer/models/layers/gated_delta_net.py:114  |     8 |     8 |

Recompile guard failures (numbers as N), records over all ranks:

| guard failure                                                                                        | run 0 | run 1 |
| ---------------------------------------------------------------------------------------------------- | ----: | ----: |
| len(G['prepare_lens'].__closure__[N].cell_contents) == N                                             |    62 |    62 |
| not G['prepare_lens'].__closure__[N].cell_contents                                                   |    36 |    36 |
| GLOBAL_STATE changed: grad_mode                                                                      |    32 |    32 |
| tensor 'cu_seqlens' size mismatch at index N. expected N, actual N                                   |    24 |    20 |
| tensor 'module._parameters['weight']._local_tensor' requires_grad mismatch. expected requires_grad=N |    32 |     0 |
| tensor '___stack0' size mismatch at index N. expected N, actual N                                    |    13 |    13 |
| N <= cu_seqlens.size()[N] <= N                                                                       |    14 |    10 |
| KeyError on self.ple                                                                                 |     8 |    16 |
| tensor 'module._parameters['weight']._local_tensor' size mismatch at index N. expected N, actual N   |    16 |     0 |
| tensor 'counts' size mismatch at index N. expected N, actual N                                       |     8 |     8 |
| tensor '___stack0[N]' size mismatch at index N. expected N, actual N                                 |     8 |     8 |
| ___check_type_id(___stack1, N), type=<class 'tuple'>                                                 |     8 |     8 |
| ___check_obj_id(fn.__code__, N), type=<class 'code'>                                                 |     8 |     8 |
| tensor 'kwargs['chunk_indices']' size mismatch at index N. expected N, actual N                      |     8 |     8 |
| not cached                                                                                           |     8 |     8 |
| N <= counts.size()[N] <= N                                                                           |     5 |     5 |
| start == N                                                                                           |     7 |     0 |
| tensor 'weight' size mismatch at index N. expected N, actual N                                       |     1 |     0 |

Sequence-length and CP guard failures (numbers as N), records over all ranks:

| guard failure                                                      | run 0 | run 1 |
| ------------------------------------------------------------------ | ----: | ----: |
| tensor 'cu_seqlens' size mismatch at index N. expected N, actual N |    24 |    20 |
| N <= cu_seqlens.size()[N] <= N                                     |    14 |    10 |

