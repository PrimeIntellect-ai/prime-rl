Runs:

| run   | label                             | head       | TORCH_LOGS              |
| ----- | --------------------------------- | ---------- | ----------------------- |
| run 0 | main/cell8-graph-breaks-noac/r0   | 1810d5b9ab | recompiles,graph_breaks |
| run 1 | branch/cell8-graph-breaks-noac/r0 | 6431ef77b0 | recompiles,graph_breaks |
| run 2 | main/cell8-noac/r0                | 1810d5b9ab | recompiles              |
| run 3 | main/cell8-noac/r1                | 1810d5b9ab | recompiles              |
| run 4 | branch/cell8-noac/r0              | 6431ef77b0 | recompiles              |
| run 5 | branch/cell8-noac/r1              | 6431ef77b0 | recompiles              |

Per rank. Breaks: graph break records. Dup: those logged with the user stack suppressed as a repeat. Touch frag.: breaks whose location or user stack touches a fragment below. Recompiles: 'Recompiling function' records. Seqlen guards: guard failures matching /cu_seqlens|seqlen|seq_len|cp_context|cp_group|cp_rank|cp_world/, the CP parameters but also any varlen recompile at cp=1. Limit hits: recompile_limit warnings. Fewer is better in every column.

| run   | rank | breaks | dup | touch frag. | recompiles | seqlen guards | limit hits |
| ----- | ---: | -----: | --: | ----------: | ---------: | ------------: | ---------: |
| run 0 |    0 |     19 |  11 |           8 |          8 |             3 |          0 |
| run 0 |    1 |     19 |  11 |           8 |          8 |             3 |          0 |
| run 0 |    2 |     19 |  11 |           8 |          8 |             3 |          0 |
| run 0 |    3 |     19 |  11 |           8 |          8 |             3 |          0 |
| run 0 |    4 |     19 |  11 |           8 |          8 |             3 |          0 |
| run 0 |    5 |     19 |  11 |           8 |          8 |             3 |          0 |
| run 0 |    6 |     19 |  11 |           8 |          8 |             3 |          0 |
| run 0 |    7 |     19 |  11 |           8 |          8 |             3 |          0 |
| run 1 |    0 |     12 |   6 |           0 |          3 |             2 |          0 |
| run 1 |    1 |     12 |   6 |           0 |          3 |             2 |          0 |
| run 1 |    2 |     12 |   6 |           0 |          3 |             2 |          0 |
| run 1 |    3 |     12 |   6 |           0 |          3 |             2 |          0 |
| run 1 |    4 |     12 |   6 |           0 |          3 |             2 |          0 |
| run 1 |    5 |     12 |   6 |           0 |          3 |             2 |          0 |
| run 1 |    6 |     12 |   6 |           0 |          3 |             2 |          0 |
| run 1 |    7 |     12 |   6 |           0 |          3 |             2 |          0 |
| run 2 |    0 |      0 |   0 |           0 |          8 |             3 |          0 |
| run 2 |    1 |      0 |   0 |           0 |          8 |             3 |          0 |
| run 2 |    2 |      0 |   0 |           0 |          8 |             3 |          0 |
| run 2 |    3 |      0 |   0 |           0 |          8 |             3 |          0 |
| run 2 |    4 |      0 |   0 |           0 |          8 |             3 |          0 |
| run 2 |    5 |      0 |   0 |           0 |          8 |             3 |          0 |
| run 2 |    6 |      0 |   0 |           0 |          8 |             3 |          0 |
| run 2 |    7 |      0 |   0 |           0 |          8 |             3 |          0 |
| run 3 |    0 |      0 |   0 |           0 |          8 |             3 |          0 |
| run 3 |    1 |      0 |   0 |           0 |          8 |             3 |          0 |
| run 3 |    2 |      0 |   0 |           0 |          8 |             3 |          0 |
| run 3 |    3 |      0 |   0 |           0 |          8 |             3 |          0 |
| run 3 |    4 |      0 |   0 |           0 |          8 |             3 |          0 |
| run 3 |    5 |      0 |   0 |           0 |          8 |             3 |          0 |
| run 3 |    6 |      0 |   0 |           0 |          8 |             3 |          0 |
| run 3 |    7 |      0 |   0 |           0 |          8 |             3 |          0 |
| run 4 |    0 |      0 |   0 |           0 |          3 |             2 |          0 |
| run 4 |    1 |      0 |   0 |           0 |          3 |             2 |          0 |
| run 4 |    2 |      0 |   0 |           0 |          3 |             2 |          0 |
| run 4 |    3 |      0 |   0 |           0 |          3 |             2 |          0 |
| run 4 |    4 |      0 |   0 |           0 |          3 |             2 |          0 |
| run 4 |    5 |      0 |   0 |           0 |          3 |             2 |          0 |
| run 4 |    6 |      0 |   0 |           0 |          3 |             2 |          0 |
| run 4 |    7 |      0 |   0 |           0 |          3 |             2 |          0 |
| run 5 |    0 |      0 |   0 |           0 |          3 |             2 |          0 |
| run 5 |    1 |      0 |   0 |           0 |          3 |             2 |          0 |
| run 5 |    2 |      0 |   0 |           0 |          3 |             2 |          0 |
| run 5 |    3 |      0 |   0 |           0 |          3 |             2 |          0 |
| run 5 |    4 |      0 |   0 |           0 |          3 |             2 |          0 |
| run 5 |    5 |      0 |   0 |           0 |          3 |             2 |          0 |
| run 5 |    6 |      0 |   0 |           0 |          3 |             2 |          0 |
| run 5 |    7 |      0 |   0 |           0 |          3 |             2 |          0 |

Break records whose location or user stack touches each fragment (all ranks):

| fragment                                | run 0 | run 1 | run 2 | run 3 | run 4 | run 5 |
| --------------------------------------- | ----: | ----: | ----: | ----: | ----: | ----: |
| qwen3_8_flash_next/position_learning.py |     8 |     0 |     0 |     0 |     0 |     0 |
| qwen3_8_flash_next/ngram_embedding.py   |    16 |     0 |     0 |     0 |     0 |     0 |
| distributed/embedding_parallel.py       |    32 |     0 |     0 |     0 |     0 |     0 |
| layers/ngram.py                         |     0 |     0 |     0 |     0 |     0 |     0 |
| distributed/ngram_parallel.py           |     0 |     0 |     0 |     0 |     0 |     0 |
| torch/distributed/fsdp                  |    32 |     0 |     0 |     0 |     0 |     0 |
| any fragment                            |    64 |     0 |     0 |     0 |     0 |     0 |

Graph break reasons (numbers as N), records over all ranks:

| reason                                                               | run 0 | run 1 | run 2 | run 3 | run 4 | run 5 |
| -------------------------------------------------------------------- | ----: | ----: | ----: | ----: | ----: | ----: |
| gb0098 Skip calling `torch.compiler.disable()`d function (<function  |    24 |    32 |     0 |     0 |     0 |     0 |
| gb0098 Skip calling `torch.compiler.disable()`d function in autograd |    24 |    24 |     0 |     0 |     0 |     0 |
| gb0036 Dynamic shape operator (aten.bincount.default)                |    32 |     0 |     0 |     0 |     0 |     0 |
| gb0007 Attempted to call function marked as skipped (module: torch._ |    32 |     0 |     0 |     0 |     0 |     0 |
| gb0098 Skip calling `torch.compiler.disable()`d function (<function  |    16 |    16 |     0 |     0 |     0 |     0 |
| gb0098 Skip calling `torch.compiler.disable()`d function (<function  |    16 |    16 |     0 |     0 |     0 |     0 |
| gb0098 Skip calling `torch.compiler.disable()`d function (<function  |     8 |     8 |     0 |     0 |     0 |     0 |

Graph break locations, records over all ranks:

| location                                               | run 0 | run 1 | run 2 | run 3 | run 4 | run 5 |
| ------------------------------------------------------ | ----: | ----: | ----: | ----: | ----: | ----: |
| fla/modules/norm/fused_norm_gate.py:656                |    40 |    40 |     0 |     0 |     0 |     0 |
| prime_rl/trainer/models/layers/gated_delta_net.py:88   |    24 |    32 |     0 |     0 |     0 |     0 |
| prime_rl/trainer/models/layers/gated_delta_net.py:114  |    16 |    16 |     0 |     0 |     0 |     0 |
| torch/distributed/fsdp/_fully_shard/_fsdp_common.py:30 |    32 |     0 |     0 |     0 |     0 |     0 |
| prime_rl/trainer/distributed/embedding_parallel.py:35  |    32 |     0 |     0 |     0 |     0 |     0 |
| prime_rl/trainer/models/layers/gated_delta_net.py:95   |     8 |     8 |     0 |     0 |     0 |     0 |

Recompile guard failures (numbers as N), records over all ranks:

| guard failure                                                        | run 0 | run 1 | run 2 | run 3 | run 4 | run 5 |
| -------------------------------------------------------------------- | ----: | ----: | ----: | ----: | ----: | ----: |
| tensor 'module._parameters['weight']._local_tensor' requires_grad mi |    32 |     0 |    32 |    32 |     0 |     0 |
| KeyError on self.ple                                                 |     8 |    16 |     8 |     8 |    16 |    16 |
| tensor 'module._parameters['weight']._local_tensor' size mismatch at |    16 |     0 |    16 |    16 |     0 |     0 |
| tensor 'context.cu_seqlens' size mismatch at index N. expected N, ac |     8 |     8 |     8 |     8 |     8 |     8 |
| tensor 'cu_seqlens' size mismatch at index N. expected N, actual N   |     8 |     4 |     8 |     8 |     4 |     4 |
| N <= cu_seqlens.size()[N] <= N                                       |     8 |     4 |     8 |     8 |     4 |     4 |
| start == N                                                           |     7 |     0 |     7 |     7 |     0 |     0 |
| tensor 'weight' size mismatch at index N. expected N, actual N       |     1 |     0 |     1 |     1 |     0 |     0 |

Sequence-length and CP guard failures (numbers as N), records over all ranks:

| guard failure                                                        | run 0 | run 1 | run 2 | run 3 | run 4 | run 5 |
| -------------------------------------------------------------------- | ----: | ----: | ----: | ----: | ----: | ----: |
| tensor 'context.cu_seqlens' size mismatch at index N. expected N, ac |     8 |     8 |     8 |     8 |     8 |     8 |
| N <= cu_seqlens.size()[N] <= N                                       |     8 |     4 |     8 |     8 |     4 |     4 |
| tensor 'cu_seqlens' size mismatch at index N. expected N, actual N   |     8 |     4 |     8 |     8 |     4 |     4 |

