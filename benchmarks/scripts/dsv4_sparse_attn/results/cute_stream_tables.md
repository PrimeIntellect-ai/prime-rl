Dynamic stream of 32 items, each called once; times in seconds, lower is better.
`compiles` and `loads` come from the compile-entry-point wrapper; `new files` from the cache dirs.

`first item` carries the per-process costs; `rest` sums the other items' first calls.

| backend | phase | first item s | rest s | process wall s | compiles | loads | new cache files |
|---|---|---|---|---|---|---|---|
| tilelang | cold | 20.94 | 0.617 | 73.6 | 4 | 0 | 21 |
| tilelang | warm | 0.29 | 0.614 | 53.2 | 0 | 4 | 0 |
| cute | cold | 14.33 | 0.583 | 67.2 | 4 | 0 | 16 |
| cute | warm | 1.44 | 0.590 | 54.2 | 1 | 3 | 0 |
| cute_ws | cold | 16.08 | 0.510 | 68.9 | 4 | 0 | 16 |
| cute_ws | warm | 3.18 | 0.511 | 56.0 | 1 | 3 | 0 |

Per-process startup breakdown in seconds (lower is better).

| backend | phase | stage | s |
|---|---|---|---|
| tilelang | cold | interpreter_and_torch_import | 3.42 |
| tilelang | cold | import tilelang | 2.16 |
| tilelang | cold | import prime_rl.trainer.models.kernels.deepseek_v4.dsv4_sparse_attn | 42.00 |
| tilelang | cold | cuda_context | 0.24 |
| tilelang | cold | first_item_jit_compile | 16.03 |
| tilelang | cold | first_item_jit_disk_load | 0.00 |
| tilelang | cold | first_item_jit_other | 2.56 |
| tilelang | cold | first_item_launch_and_run | 2.34 |
| tilelang | warm | interpreter_and_torch_import | 3.83 |
| tilelang | warm | import tilelang | 2.38 |
| tilelang | warm | import prime_rl.trainer.models.kernels.deepseek_v4.dsv4_sparse_attn | 41.64 |
| tilelang | warm | cuda_context | 0.16 |
| tilelang | warm | first_item_jit_compile | 0.00 |
| tilelang | warm | first_item_jit_disk_load | 0.03 |
| tilelang | warm | first_item_jit_other | 0.15 |
| tilelang | warm | first_item_launch_and_run | 0.11 |
| cute | cold | interpreter_and_torch_import | 3.78 |
| cute | cold | import tilelang | 2.34 |
| cute | cold | import cutlass.cute | 0.58 |
| cute | cold | import prime_rl.trainer.models.kernels.deepseek_v4.dsv4_sparse_attn | 41.30 |
| cute | cold | import prime_rl.trainer.models.kernels.deepseek_v4.dsv4_sparse_attn_fwd_cute | 0.00 |
| cute | cold | cuda_context | 0.15 |
| cute | cold | first_item_jit_compile | 9.77 |
| cute | cold | first_item_jit_disk_load | 0.00 |
| cute | cold | first_item_jit_other | 1.69 |
| cute | cold | first_item_launch_and_run | 2.87 |
| cute | warm | interpreter_and_torch_import | 3.88 |
| cute | warm | import tilelang | 2.40 |
| cute | warm | import cutlass.cute | 0.72 |
| cute | warm | import prime_rl.trainer.models.kernels.deepseek_v4.dsv4_sparse_attn | 40.69 |
| cute | warm | import prime_rl.trainer.models.kernels.deepseek_v4.dsv4_sparse_attn_fwd_cute | 0.00 |
| cute | warm | cuda_context | 0.25 |
| cute | warm | first_item_jit_compile | 0.00 |
| cute | warm | first_item_jit_disk_load | 0.02 |
| cute | warm | first_item_jit_other | 0.08 |
| cute | warm | first_item_launch_and_run | 1.34 |
| cute_ws | cold | interpreter_and_torch_import | 3.73 |
| cute_ws | cold | import tilelang | 2.42 |
| cute_ws | cold | import cutlass.cute | 0.69 |
| cute_ws | cold | import prime_rl.trainer.models.kernels.deepseek_v4.dsv4_sparse_attn | 41.06 |
| cute_ws | cold | import prime_rl.trainer.models.kernels.deepseek_v4.dsv4_sparse_attn_fwd_cute_ws | 0.00 |
| cute_ws | cold | cuda_context | 0.14 |
| cute_ws | cold | first_item_jit_compile | 9.77 |
| cute_ws | cold | first_item_jit_disk_load | 0.00 |
| cute_ws | cold | first_item_jit_other | 1.67 |
| cute_ws | cold | first_item_launch_and_run | 4.64 |
| cute_ws | warm | interpreter_and_torch_import | 3.78 |
| cute_ws | warm | import tilelang | 2.46 |
| cute_ws | warm | import cutlass.cute | 0.69 |
| cute_ws | warm | import prime_rl.trainer.models.kernels.deepseek_v4.dsv4_sparse_attn | 41.01 |
| cute_ws | warm | import prime_rl.trainer.models.kernels.deepseek_v4.dsv4_sparse_attn_fwd_cute_ws | 0.00 |
| cute_ws | warm | cuda_context | 0.27 |
| cute_ws | warm | first_item_jit_compile | 0.00 |
| cute_ws | warm | first_item_jit_disk_load | 0.02 |
| cute_ws | warm | first_item_jit_other | 0.08 |
| cute_ws | warm | first_item_launch_and_run | 3.07 |

Per-item first-call time in ms (lower is better) and compiles triggered by that item.

| # | item | backend | cold ms | cold compiles | warm ms | warm compiles |
|---|---|---|---|---|---|---|
| 0 | stream00-short-14224-csa-cp8r1 | tilelang | 20939.9 | 4 | 292.9 | 0 |
| 0 | stream00-short-14224-csa-cp8r1 | cute | 14327.7 | 4 | 1441.0 | 1 |
| 0 | stream00-short-14224-csa-cp8r1 | cute_ws | 16082.2 | 4 | 3176.6 | 1 |
| 1 | stream01-tiny-48096-hca-cp1 | tilelang | 17.7 | 0 | 17.3 | 0 |
| 1 | stream01-tiny-48096-hca-cp1 | cute | 16.3 | 0 | 16.0 | 0 |
| 1 | stream01-tiny-48096-hca-cp1 | cute_ws | 13.4 | 0 | 13.1 | 0 |
| 2 | stream02-heavy-38136-hca-cp8r6 | tilelang | 10.1 | 0 | 9.9 | 0 |
| 2 | stream02-heavy-38136-hca-cp8r6 | cute | 7.3 | 0 | 7.4 | 0 |
| 2 | stream02-heavy-38136-hca-cp8r6 | cute_ws | 11.3 | 0 | 12.4 | 0 |
| 3 | stream03-short-37152-sliding-cp8r1 | tilelang | 3.5 | 0 | 3.3 | 0 |
| 3 | stream03-short-37152-sliding-cp8r1 | cute | 2.6 | 0 | 2.6 | 0 |
| 3 | stream03-short-37152-sliding-cp8r1 | cute_ws | 2.6 | 0 | 2.7 | 0 |
| 4 | stream04-short-34872-csa-cp8r5 | tilelang | 5.2 | 0 | 5.1 | 0 |
| 4 | stream04-short-34872-csa-cp8r5 | cute | 4.5 | 0 | 4.5 | 0 |
| 4 | stream04-short-34872-csa-cp8r5 | cute_ws | 4.2 | 0 | 4.3 | 0 |
| 5 | stream05-single-25016-csa-cp8r4 | tilelang | 5.6 | 0 | 5.7 | 0 |
| 5 | stream05-single-25016-csa-cp8r4 | cute | 5.0 | 0 | 5.0 | 0 |
| 5 | stream05-single-25016-csa-cp8r4 | cute_ws | 4.7 | 0 | 4.9 | 0 |
| 6 | stream06-heavy-55096-csa-cp1 | tilelang | 42.5 | 0 | 42.6 | 0 |
| 6 | stream06-heavy-55096-csa-cp1 | cute | 41.2 | 0 | 41.1 | 0 |
| 6 | stream06-heavy-55096-csa-cp1 | cute_ws | 35.8 | 0 | 35.6 | 0 |
| 7 | stream07-short-17664-csa-cp8r2 | tilelang | 3.4 | 0 | 3.4 | 0 |
| 7 | stream07-short-17664-csa-cp8r2 | cute | 2.9 | 0 | 2.8 | 0 |
| 7 | stream07-short-17664-csa-cp8r2 | cute_ws | 2.6 | 0 | 2.6 | 0 |
| 8 | stream08-single-28000-hca-cp1 | tilelang | 20.6 | 0 | 20.6 | 0 |
| 8 | stream08-single-28000-hca-cp1 | cute | 19.6 | 0 | 19.6 | 0 |
| 8 | stream08-single-28000-hca-cp1 | cute_ws | 16.7 | 0 | 16.7 | 0 |
| 9 | stream09-tiny-33656-hca-cp1 | tilelang | 11.9 | 0 | 11.9 | 0 |
| 9 | stream09-tiny-33656-hca-cp1 | cute | 10.9 | 0 | 10.9 | 0 |
| 9 | stream09-tiny-33656-hca-cp1 | cute_ws | 8.8 | 0 | 8.8 | 0 |
| 10 | stream10-heavy-48264-csa-cp1 | tilelang | 52.8 | 0 | 52.6 | 0 |
| 10 | stream10-heavy-48264-csa-cp1 | cute | 51.3 | 0 | 51.4 | 0 |
| 10 | stream10-heavy-48264-csa-cp1 | cute_ws | 45.1 | 0 | 45.3 | 0 |
| 11 | stream11-tiny-12904-hca-cp1 | tilelang | 5.2 | 0 | 5.1 | 0 |
| 11 | stream11-tiny-12904-hca-cp1 | cute | 4.4 | 0 | 4.5 | 0 |
| 11 | stream11-tiny-12904-hca-cp1 | cute_ws | 3.6 | 0 | 3.6 | 0 |
| 12 | stream12-tiny-39552-sliding-cp1 | tilelang | 13.7 | 0 | 13.7 | 0 |
| 12 | stream12-tiny-39552-sliding-cp1 | cute | 12.7 | 0 | 12.8 | 0 |
| 12 | stream12-tiny-39552-sliding-cp1 | cute_ws | 10.4 | 0 | 10.4 | 0 |
| 13 | stream13-single-64144-csa-cp8r1 | tilelang | 12.7 | 0 | 12.7 | 0 |
| 13 | stream13-single-64144-csa-cp8r1 | cute | 11.9 | 0 | 12.0 | 0 |
| 13 | stream13-single-64144-csa-cp8r1 | cute_ws | 10.6 | 0 | 10.6 | 0 |
| 14 | stream14-heavy-15304-csa-cp1 | tilelang | 14.2 | 0 | 14.2 | 0 |
| 14 | stream14-heavy-15304-csa-cp1 | cute | 13.3 | 0 | 13.4 | 0 |
| 14 | stream14-heavy-15304-csa-cp1 | cute_ws | 11.6 | 0 | 11.6 | 0 |
| 15 | stream15-heavy-48400-sliding-cp1 | tilelang | 22.1 | 0 | 22.0 | 0 |
| 15 | stream15-heavy-48400-sliding-cp1 | cute | 20.9 | 0 | 20.9 | 0 |
| 15 | stream15-heavy-48400-sliding-cp1 | cute_ws | 17.0 | 0 | 17.1 | 0 |
| 16 | stream16-tiny-65360-hca-cp1 | tilelang | 22.8 | 0 | 22.7 | 0 |
| 16 | stream16-tiny-65360-hca-cp1 | cute | 21.4 | 0 | 27.5 | 0 |
| 16 | stream16-tiny-65360-hca-cp1 | cute_ws | 17.5 | 0 | 17.6 | 0 |
| 17 | stream17-heavy-4848-sliding-cp8r5 | tilelang | 2.2 | 0 | 2.2 | 0 |
| 17 | stream17-heavy-4848-sliding-cp8r5 | cute | 1.6 | 0 | 1.7 | 0 |
| 17 | stream17-heavy-4848-sliding-cp8r5 | cute_ws | 1.7 | 0 | 1.5 | 0 |
| 18 | stream18-short-24576-sliding-cp1 | tilelang | 11.9 | 0 | 12.0 | 0 |
| 18 | stream18-short-24576-sliding-cp1 | cute | 11.0 | 0 | 11.0 | 0 |
| 18 | stream18-short-24576-sliding-cp1 | cute_ws | 9.0 | 0 | 9.0 | 0 |
| 19 | stream19-single-36016-hca-cp1 | tilelang | 28.4 | 0 | 28.1 | 0 |
| 19 | stream19-single-36016-hca-cp1 | cute | 27.1 | 0 | 27.2 | 0 |
| 19 | stream19-single-36016-hca-cp1 | cute_ws | 23.4 | 0 | 23.4 | 0 |
| 20 | stream20-short-24768-hca-cp1 | tilelang | 13.4 | 0 | 13.3 | 0 |
| 20 | stream20-short-24768-hca-cp1 | cute | 12.4 | 0 | 12.5 | 0 |
| 20 | stream20-short-24768-hca-cp1 | cute_ws | 10.5 | 0 | 10.4 | 0 |
| 21 | stream21-heavy-39048-sliding-cp1 | tilelang | 17.9 | 0 | 17.9 | 0 |
| 21 | stream21-heavy-39048-sliding-cp1 | cute | 16.9 | 0 | 16.9 | 0 |
| 21 | stream21-heavy-39048-sliding-cp1 | cute_ws | 13.7 | 0 | 13.8 | 0 |
| 22 | stream22-single-60624-sliding-cp1 | tilelang | 28.2 | 0 | 28.0 | 0 |
| 22 | stream22-single-60624-sliding-cp1 | cute | 26.8 | 0 | 26.9 | 0 |
| 22 | stream22-single-60624-sliding-cp1 | cute_ws | 21.9 | 0 | 22.1 | 0 |
| 23 | stream23-single-61784-csa-cp1 | tilelang | 90.3 | 0 | 90.4 | 0 |
| 23 | stream23-single-61784-csa-cp1 | cute | 88.5 | 0 | 88.7 | 0 |
| 23 | stream23-single-61784-csa-cp1 | cute_ws | 78.9 | 0 | 79.0 | 0 |
| 24 | stream24-single-6944-hca-cp8r6 | tilelang | 2.9 | 0 | 2.7 | 0 |
| 24 | stream24-single-6944-hca-cp8r6 | cute | 2.1 | 0 | 2.1 | 0 |
| 24 | stream24-single-6944-hca-cp8r6 | cute_ws | 2.0 | 0 | 1.9 | 0 |
| 25 | stream25-single-56616-hca-cp1 | tilelang | 52.6 | 0 | 52.4 | 0 |
| 25 | stream25-single-56616-hca-cp1 | cute | 51.2 | 0 | 51.2 | 0 |
| 25 | stream25-single-56616-hca-cp1 | cute_ws | 44.5 | 0 | 44.6 | 0 |
| 26 | stream26-tiny-60696-sliding-cp8r2 | tilelang | 3.5 | 0 | 3.5 | 0 |
| 26 | stream26-tiny-60696-sliding-cp8r2 | cute | 2.8 | 0 | 2.8 | 0 |
| 26 | stream26-tiny-60696-sliding-cp8r2 | cute_ws | 2.7 | 0 | 2.6 | 0 |
| 27 | stream27-tiny-62600-sliding-cp8r4 | tilelang | 3.6 | 0 | 3.6 | 0 |
| 27 | stream27-tiny-62600-sliding-cp8r4 | cute | 2.9 | 0 | 2.9 | 0 |
| 27 | stream27-tiny-62600-sliding-cp8r4 | cute_ws | 2.4 | 0 | 2.4 | 0 |
| 28 | stream28-single-42136-csa-cp1 | tilelang | 60.8 | 0 | 60.8 | 0 |
| 28 | stream28-single-42136-csa-cp1 | cute | 59.6 | 0 | 59.5 | 0 |
| 28 | stream28-single-42136-csa-cp1 | cute_ws | 52.8 | 0 | 52.8 | 0 |
| 29 | stream29-tiny-22480-sliding-cp8r4 | tilelang | 2.7 | 0 | 2.7 | 0 |
| 29 | stream29-tiny-22480-sliding-cp8r4 | cute | 1.9 | 0 | 1.9 | 0 |
| 29 | stream29-tiny-22480-sliding-cp8r4 | cute_ws | 2.0 | 0 | 1.9 | 0 |
| 30 | stream30-heavy-38080-csa-cp1 | tilelang | 31.3 | 0 | 31.1 | 0 |
| 30 | stream30-heavy-38080-csa-cp1 | cute | 30.0 | 0 | 30.1 | 0 |
| 30 | stream30-heavy-38080-csa-cp1 | cute_ws | 26.1 | 0 | 26.1 | 0 |
| 31 | stream31-short-12824-hca-cp8r5 | tilelang | 2.9 | 0 | 2.9 | 0 |
| 31 | stream31-short-12824-hca-cp8r5 | cute | 2.1 | 0 | 2.1 | 0 |
| 31 | stream31-short-12824-hca-cp8r5 | cute_ws | 2.1 | 0 | 2.1 | 0 |
