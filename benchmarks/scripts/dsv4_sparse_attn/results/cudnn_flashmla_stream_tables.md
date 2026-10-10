Dynamic stream of 32 items, each called once; times in seconds, lower is better.
`compiles` and `loads` come from the compile-entry-point wrapper; `new files` from the cache dirs.

`first item` carries the per-process costs; `rest` sums the other items' first calls.

| backend | phase | first item s | rest s | process wall s | compiles | loads | new cache files |
|---|---|---|---|---|---|---|---|
| tilelang | cold | 21.01 | 0.618 | 74.3 | 4 | 0 | 21 |
| tilelang | warm | 0.30 | 0.616 | 54.3 | 0 | 4 | 0 |
| cudnn_flashmla | cold | 6.90 | 7.929 | 68.6 | 7 | 0 | 6 |
| cudnn_flashmla | warm | 3.46 | 7.826 | 65.2 | 6 | 1 | 0 |

Per-process startup breakdown in seconds (lower is better).

| backend | phase | stage | s |
|---|---|---|---|
| tilelang | cold | interpreter_and_torch_import | 3.32 |
| tilelang | cold | import tilelang | 2.01 |
| tilelang | cold | import prime_rl.trainer.models.kernels.deepseek_v4.dsv4_sparse_attn | 42.83 |
| tilelang | cold | cuda_context | 0.18 |
| tilelang | cold | first_item_jit_compile | 16.07 |
| tilelang | cold | first_item_jit_disk_load | 0.00 |
| tilelang | cold | first_item_jit_other | 2.57 |
| tilelang | cold | first_item_launch_and_run | 2.37 |
| tilelang | warm | interpreter_and_torch_import | 3.93 |
| tilelang | warm | import tilelang | 2.64 |
| tilelang | warm | import prime_rl.trainer.models.kernels.deepseek_v4.dsv4_sparse_attn | 42.37 |
| tilelang | warm | cuda_context | 0.15 |
| tilelang | warm | first_item_jit_compile | 0.00 |
| tilelang | warm | first_item_jit_disk_load | 0.03 |
| tilelang | warm | first_item_jit_other | 0.15 |
| tilelang | warm | first_item_launch_and_run | 0.11 |
| cudnn_flashmla | cold | interpreter_and_torch_import | 3.86 |
| cudnn_flashmla | cold | import flash_mla | 0.01 |
| cudnn_flashmla | cold | import cutlass.cute | 0.57 |
| cudnn_flashmla | cold | import cudnn.deepseek_sparse_attention.sparse_attention_backward._interface_sm90 | 0.08 |
| cudnn_flashmla | cold | import tilelang | 2.57 |
| cudnn_flashmla | cold | import prime_rl.trainer.models.kernels.deepseek_v4.dsv4_sparse_attn | 42.18 |
| cudnn_flashmla | cold | cuda_context | 0.14 |
| cudnn_flashmla | cold | first_item_jit_compile | 5.91 |
| cudnn_flashmla | cold | first_item_jit_disk_load | 0.00 |
| cudnn_flashmla | cold | first_item_jit_other | 0.50 |
| cudnn_flashmla | cold | first_item_launch_and_run | 0.49 |
| cudnn_flashmla | warm | interpreter_and_torch_import | 3.86 |
| cudnn_flashmla | warm | import flash_mla | 0.01 |
| cudnn_flashmla | warm | import cutlass.cute | 0.64 |
| cudnn_flashmla | warm | import cudnn.deepseek_sparse_attention.sparse_attention_backward._interface_sm90 | 0.09 |
| cudnn_flashmla | warm | import tilelang | 2.43 |
| cudnn_flashmla | warm | import prime_rl.trainer.models.kernels.deepseek_v4.dsv4_sparse_attn | 42.38 |
| cudnn_flashmla | warm | cuda_context | 0.14 |
| cudnn_flashmla | warm | first_item_jit_compile | 3.31 |
| cudnn_flashmla | warm | first_item_jit_disk_load | 0.01 |
| cudnn_flashmla | warm | first_item_jit_other | 0.04 |
| cudnn_flashmla | warm | first_item_launch_and_run | 0.10 |

Per-item first-call time in ms (lower is better) and compiles triggered by that item.

| # | item | backend | cold ms | cold compiles | warm ms | warm compiles |
|---|---|---|---|---|---|---|
| 0 | stream00-short-14224-csa-cp8r1 | tilelang | 21006.0 | 4 | 296.5 | 0 |
| 0 | stream00-short-14224-csa-cp8r1 | cudnn_flashmla | 6904.3 | 4 | 3464.8 | 3 |
| 1 | stream01-tiny-48096-hca-cp1 | tilelang | 17.5 | 0 | 17.2 | 0 |
| 1 | stream01-tiny-48096-hca-cp1 | cudnn_flashmla | 2503.8 | 1 | 2490.8 | 1 |
| 2 | stream02-heavy-38136-hca-cp8r6 | tilelang | 9.7 | 0 | 9.6 | 0 |
| 2 | stream02-heavy-38136-hca-cp8r6 | cudnn_flashmla | 2538.2 | 1 | 2515.1 | 1 |
| 3 | stream03-short-37152-sliding-cp8r1 | tilelang | 3.6 | 0 | 3.4 | 0 |
| 3 | stream03-short-37152-sliding-cp8r1 | cudnn_flashmla | 3.3 | 0 | 3.1 | 0 |
| 4 | stream04-short-34872-csa-cp8r5 | tilelang | 5.2 | 0 | 5.1 | 0 |
| 4 | stream04-short-34872-csa-cp8r5 | cudnn_flashmla | 3.3 | 0 | 3.2 | 0 |
| 5 | stream05-single-25016-csa-cp8r4 | tilelang | 5.7 | 0 | 5.6 | 0 |
| 5 | stream05-single-25016-csa-cp8r4 | cudnn_flashmla | 4.5 | 0 | 5.3 | 0 |
| 6 | stream06-heavy-55096-csa-cp1 | tilelang | 42.6 | 0 | 42.5 | 0 |
| 6 | stream06-heavy-55096-csa-cp1 | cudnn_flashmla | 25.3 | 0 | 25.2 | 0 |
| 7 | stream07-short-17664-csa-cp8r2 | tilelang | 3.3 | 0 | 3.4 | 0 |
| 7 | stream07-short-17664-csa-cp8r2 | cudnn_flashmla | 2.3 | 0 | 2.2 | 0 |
| 8 | stream08-single-28000-hca-cp1 | tilelang | 20.5 | 0 | 20.6 | 0 |
| 8 | stream08-single-28000-hca-cp1 | cudnn_flashmla | 2545.3 | 1 | 2478.0 | 1 |
| 9 | stream09-tiny-33656-hca-cp1 | tilelang | 11.9 | 0 | 12.3 | 0 |
| 9 | stream09-tiny-33656-hca-cp1 | cudnn_flashmla | 9.1 | 0 | 9.0 | 0 |
| 10 | stream10-heavy-48264-csa-cp1 | tilelang | 52.8 | 0 | 53.1 | 0 |
| 10 | stream10-heavy-48264-csa-cp1 | cudnn_flashmla | 28.9 | 0 | 28.9 | 0 |
| 11 | stream11-tiny-12904-hca-cp1 | tilelang | 5.1 | 0 | 5.2 | 0 |
| 11 | stream11-tiny-12904-hca-cp1 | cudnn_flashmla | 3.8 | 0 | 3.7 | 0 |
| 12 | stream12-tiny-39552-sliding-cp1 | tilelang | 13.8 | 0 | 13.8 | 0 |
| 12 | stream12-tiny-39552-sliding-cp1 | cudnn_flashmla | 10.4 | 0 | 10.3 | 0 |
| 13 | stream13-single-64144-csa-cp8r1 | tilelang | 12.7 | 0 | 12.7 | 0 |
| 13 | stream13-single-64144-csa-cp8r1 | cudnn_flashmla | 6.7 | 0 | 6.7 | 0 |
| 14 | stream14-heavy-15304-csa-cp1 | tilelang | 14.3 | 0 | 14.2 | 0 |
| 14 | stream14-heavy-15304-csa-cp1 | cudnn_flashmla | 8.0 | 0 | 7.9 | 0 |
| 15 | stream15-heavy-48400-sliding-cp1 | tilelang | 22.7 | 0 | 22.0 | 0 |
| 15 | stream15-heavy-48400-sliding-cp1 | cudnn_flashmla | 14.8 | 0 | 14.9 | 0 |
| 16 | stream16-tiny-65360-hca-cp1 | tilelang | 23.8 | 0 | 22.7 | 0 |
| 16 | stream16-tiny-65360-hca-cp1 | cudnn_flashmla | 17.6 | 0 | 17.5 | 0 |
| 17 | stream17-heavy-4848-sliding-cp8r5 | tilelang | 2.3 | 0 | 2.4 | 0 |
| 17 | stream17-heavy-4848-sliding-cp8r5 | cudnn_flashmla | 1.6 | 0 | 1.6 | 0 |
| 18 | stream18-short-24576-sliding-cp1 | tilelang | 11.9 | 0 | 11.9 | 0 |
| 18 | stream18-short-24576-sliding-cp1 | cudnn_flashmla | 7.8 | 0 | 7.8 | 0 |
| 19 | stream19-single-36016-hca-cp1 | tilelang | 28.2 | 0 | 28.2 | 0 |
| 19 | stream19-single-36016-hca-cp1 | cudnn_flashmla | 16.6 | 0 | 16.6 | 0 |
| 20 | stream20-short-24768-hca-cp1 | tilelang | 13.3 | 0 | 13.3 | 0 |
| 20 | stream20-short-24768-hca-cp1 | cudnn_flashmla | 8.7 | 0 | 8.8 | 0 |
| 21 | stream21-heavy-39048-sliding-cp1 | tilelang | 18.0 | 0 | 17.8 | 0 |
| 21 | stream21-heavy-39048-sliding-cp1 | cudnn_flashmla | 11.9 | 0 | 12.0 | 0 |
| 22 | stream22-single-60624-sliding-cp1 | tilelang | 28.2 | 0 | 28.1 | 0 |
| 22 | stream22-single-60624-sliding-cp1 | cudnn_flashmla | 18.7 | 0 | 18.8 | 0 |
| 23 | stream23-single-61784-csa-cp1 | tilelang | 90.7 | 0 | 90.7 | 0 |
| 23 | stream23-single-61784-csa-cp1 | cudnn_flashmla | 47.2 | 0 | 47.2 | 0 |
| 24 | stream24-single-6944-hca-cp8r6 | tilelang | 2.7 | 0 | 3.0 | 0 |
| 24 | stream24-single-6944-hca-cp8r6 | cudnn_flashmla | 1.8 | 0 | 2.2 | 0 |
| 25 | stream25-single-56616-hca-cp1 | tilelang | 52.8 | 0 | 52.7 | 0 |
| 25 | stream25-single-56616-hca-cp1 | cudnn_flashmla | 30.0 | 0 | 30.2 | 0 |
| 26 | stream26-tiny-60696-sliding-cp8r2 | tilelang | 3.6 | 0 | 3.6 | 0 |
| 26 | stream26-tiny-60696-sliding-cp8r2 | cudnn_flashmla | 2.6 | 0 | 2.6 | 0 |
| 27 | stream27-tiny-62600-sliding-cp8r4 | tilelang | 3.6 | 0 | 3.7 | 0 |
| 27 | stream27-tiny-62600-sliding-cp8r4 | cudnn_flashmla | 2.6 | 0 | 2.6 | 0 |
| 28 | stream28-single-42136-csa-cp1 | tilelang | 60.8 | 0 | 60.8 | 0 |
| 28 | stream28-single-42136-csa-cp1 | cudnn_flashmla | 32.0 | 0 | 32.0 | 0 |
| 29 | stream29-tiny-22480-sliding-cp8r4 | tilelang | 2.7 | 0 | 2.6 | 0 |
| 29 | stream29-tiny-22480-sliding-cp8r4 | cudnn_flashmla | 1.7 | 0 | 1.7 | 0 |
| 30 | stream30-heavy-38080-csa-cp1 | tilelang | 31.3 | 0 | 31.3 | 0 |
| 30 | stream30-heavy-38080-csa-cp1 | cudnn_flashmla | 18.1 | 0 | 18.2 | 0 |
| 31 | stream31-short-12824-hca-cp8r5 | tilelang | 3.0 | 0 | 2.8 | 0 |
| 31 | stream31-short-12824-hca-cp8r5 | cudnn_flashmla | 1.8 | 0 | 1.9 | 0 |
