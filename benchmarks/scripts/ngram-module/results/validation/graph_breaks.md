Runs:

| run   | label                                | head       | TORCH_LOGS              |
| ----- | ------------------------------------ | ---------- | ----------------------- |
| run 0 | main/cell5-4gpu-graph-breaks/r0      | 1810d5b9ab | recompiles,graph_breaks |
| run 1 | main/cell3-4gpu-graph-breaks/r0      | 1810d5b9ab | recompiles,graph_breaks |
| run 2 | main/cell8-4gpu-graph-breaks/r0      | 1810d5b9ab | recompiles,graph_breaks |
| run 3 | main/cell8-4gpu-graph-breaks-noac/r0 | 1810d5b9ab | recompiles,graph_breaks |

Per rank. Breaks: graph break records. Dup: those logged with the user stack suppressed as a repeat. Touch frag.: breaks whose location or user stack touches a fragment below. Recompiles: 'Recompiling function' records. Seqlen guards: guard failures matching /cu_seqlens|seqlen|seq_len|cp_context|cp_group|cp_rank|cp_world/, the CP parameters but also any varlen recompile at cp=1. Limit hits: recompile_limit warnings. Fewer is better in every column.

| run   | rank | breaks | dup | touch frag. | recompiles | seqlen guards | limit hits |
| ----- | ---: | -----: | --: | ----------: | ---------: | ------------: | ---------: |
| run 0 |    0 |      1 |   0 |           0 |          0 |             0 |          0 |
| run 0 |    1 |      1 |   0 |           0 |          0 |             0 |          0 |
| run 0 |    2 |      1 |   0 |           0 |          0 |             0 |          0 |
| run 0 |    3 |      1 |   0 |           0 |          0 |             0 |          0 |
| run 1 |    0 |     34 |  24 |           8 |         22 |             4 |          0 |
| run 1 |    1 |     34 |  24 |           8 |         22 |             3 |          0 |
| run 1 |    2 |     34 |  24 |           8 |         22 |             4 |          0 |
| run 1 |    3 |     34 |  24 |           8 |         22 |             6 |          0 |
| run 2 |    0 |      1 |   0 |           0 |          0 |             0 |          0 |
| run 2 |    1 |      1 |   0 |           0 |          0 |             0 |          0 |
| run 2 |    2 |      1 |   0 |           0 |          0 |             0 |          0 |
| run 2 |    3 |      1 |   0 |           0 |          0 |             0 |          0 |
| run 3 |    0 |     19 |  11 |           8 |          8 |             3 |          0 |
| run 3 |    1 |     19 |  11 |           8 |          8 |             3 |          0 |
| run 3 |    2 |     19 |  11 |           8 |          8 |             3 |          0 |
| run 3 |    3 |     19 |  11 |           8 |          8 |             3 |          0 |

Break records whose location or user stack touches each fragment (all ranks):

| fragment                                | run 0 | run 1 | run 2 | run 3 |
| --------------------------------------- | ----: | ----: | ----: | ----: |
| qwen3_8_flash_next/position_learning.py |     0 |     4 |     0 |     4 |
| qwen3_8_flash_next/ngram_embedding.py   |     0 |     8 |     0 |     8 |
| distributed/embedding_parallel.py       |     0 |    16 |     0 |    16 |
| layers/ngram.py                         |     0 |     0 |     0 |     0 |
| distributed/ngram_parallel.py           |     0 |     0 |     0 |     0 |
| torch/distributed/fsdp                  |     0 |    16 |     0 |    16 |
| any fragment                            |     0 |    32 |     0 |    32 |

Graph break reasons (numbers as N), records over all ranks:

| reason                                                                               | run 0 | run 1 | run 2 | run 3 |
| ------------------------------------------------------------------------------------ | ----: | ----: | ----: | ----: |
| gb0036 Dynamic shape operator (aten.repeat_interleave.Tensor)                        |     0 |    44 |     0 |     0 |
| gb0036 Dynamic shape operator (aten.bincount.default)                                |     0 |    16 |     0 |    16 |
| gb0007 Attempted to call function marked as skipped (module: torch._dynamo.decorator |     0 |    16 |     0 |    16 |
| gb0098 Skip calling `torch.compiler.disable()`d function in autograd.Function (<func |     0 |    12 |     0 |    12 |
| gb0036 Dynamic shape operator in autograd.Function (aten.repeat_interleave.Tensor)   |     4 |    16 |     0 |     0 |
| gb0098 Skip calling `torch.compiler.disable()`d function (<function layer_norm_gated |     0 |     8 |     0 |     8 |
| gb0098 Skip calling `torch.compiler.disable()`d function (<function build_cp_context |     0 |     0 |     0 |    12 |
| gb0098 Skip calling `torch.compiler.disable()`d function (<function chunk_gated_delt |     0 |     4 |     0 |     8 |
| gb0099 Skip inlining `torch.compiler.disable()`d function (<function causal_conv1d_f |     0 |     8 |     0 |     0 |
| gb0098 Skip calling `torch.compiler.disable()`d function (<function causal_conv1d_fw |     0 |     8 |     0 |     0 |
| gb0098 Skip calling `torch.compiler.disable()`d function in torch.utils.checkpoint.c |     0 |     0 |     4 |     0 |
| gb0099 Skip inlining `torch.compiler.disable()`d function in autograd.Function (<fun |     0 |     4 |     0 |     0 |
| gb0098 Skip calling `torch.compiler.disable()`d function (<function causal_conv1d>)  |     0 |     0 |     0 |     4 |

Graph break locations, records over all ranks:

| location                                               | run 0 | run 1 | run 2 | run 3 |
| ------------------------------------------------------ | ----: | ----: | ----: | ----: |
| fla/ops/utils/index.py:128                             |     4 |    60 |     0 |     0 |
| fla/modules/norm/fused_norm_gate.py:656                |     0 |    20 |     0 |    20 |
| prime_rl/trainer/distributed/embedding_parallel.py:35  |     0 |    16 |     0 |    16 |
| torch/distributed/fsdp/_fully_shard/_fsdp_common.py:30 |     0 |    16 |     0 |    16 |
| prime_rl/trainer/models/layers/gated_delta_net.py:88   |     0 |     0 |     4 |    12 |
| fla/modules/causal_conv1d/ops.py:958                   |     0 |    12 |     0 |     0 |
| prime_rl/trainer/models/layers/gated_delta_net.py:114  |     0 |     4 |     0 |     8 |
| fla/utils/decorators.py:325                            |     0 |     8 |     0 |     0 |
| prime_rl/trainer/models/layers/gated_delta_net.py:95   |     0 |     0 |     0 |     4 |

Recompile guard failures (numbers as N), records over all ranks:

| guard failure                                                                        | run 0 | run 1 | run 2 | run 3 |
| ------------------------------------------------------------------------------------ | ----: | ----: | ----: | ----: |
| tensor 'module._parameters['weight']._local_tensor' requires_grad mismatch. expected |     0 |    16 |     0 |    16 |
| len(G['prepare_lens'].__closure__[N].cell_contents) == N                             |     0 |    31 |     0 |     0 |
| not G['prepare_lens'].__closure__[N].cell_contents                                   |     0 |    20 |     0 |     0 |
| GLOBAL_STATE changed: grad_mode                                                      |     0 |    16 |     0 |     0 |
| tensor 'module._parameters['weight']._local_tensor' size mismatch at index N. expect |     0 |     8 |     0 |     8 |
| N <= cu_seqlens.size()[N] <= N                                                       |     0 |     7 |     0 |     8 |
| tensor 'cu_seqlens' size mismatch at index N. expected N, actual N                   |     0 |    10 |     0 |     0 |
| KeyError on self.ple                                                                 |     0 |     4 |     0 |     4 |
| tensor '___stack0' size mismatch at index N. expected N, actual N                    |     0 |     6 |     0 |     0 |
| start == N                                                                           |     0 |     3 |     0 |     3 |
| tensor 'kwargs['chunk_indices']' size mismatch at index N. expected N, actual N      |     0 |     4 |     0 |     0 |
| tensor 'counts' size mismatch at index N. expected N, actual N                       |     0 |     4 |     0 |     0 |
| not cached                                                                           |     0 |     4 |     0 |     0 |
| ___check_type_id(___stack1, N), type=<class 'tuple'>                                 |     0 |     4 |     0 |     0 |
| tensor '___stack0[N]' size mismatch at index N. expected N, actual N                 |     0 |     4 |     0 |     0 |
| tensor 'context.cu_seqlens' size mismatch at index N. expected N, actual N           |     0 |     0 |     0 |     4 |
| ___check_obj_id(fn.__code__, N), type=<class 'code'>                                 |     0 |     4 |     0 |     0 |
| N <= counts.size()[N] <= N                                                           |     0 |     2 |     0 |     0 |
| tensor 'weight' size mismatch at index N. expected N, actual N                       |     0 |     1 |     0 |     1 |

Sequence-length and CP guard failures (numbers as N), records over all ranks:

| guard failure                                                              | run 0 | run 1 | run 2 | run 3 |
| -------------------------------------------------------------------------- | ----: | ----: | ----: | ----: |
| N <= cu_seqlens.size()[N] <= N                                             |     0 |     7 |     0 |     8 |
| tensor 'cu_seqlens' size mismatch at index N. expected N, actual N         |     0 |    10 |     0 |     0 |
| tensor 'context.cu_seqlens' size mismatch at index N. expected N, actual N |     0 |     0 |     0 |     4 |

