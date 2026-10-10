Dynamic stream of 32 items, each called once; times in seconds, lower is better.
`compiles` and `loads` come from the compile-entry-point wrapper; `new files` from the cache dirs.

`first item` carries the per-process costs; `rest` sums the other items' first calls.

| backend | phase | first item s | rest s | process wall s | compiles | loads | new cache files |
|---|---|---|---|---|---|---|---|
| tilelang | cold | 20.87 | 0.614 | 73.8 | 4 | 0 | 21 |
| tilelang | warm | 0.29 | 0.614 | 53.8 | 0 | 4 | 0 |

Per-process startup breakdown in seconds (lower is better).

| backend | phase | stage | s |
|---|---|---|---|
| tilelang | cold | interpreter_and_torch_import | 3.35 |
| tilelang | cold | import tilelang | 2.01 |
| tilelang | cold | import prime_rl.trainer.models.kernels.deepseek_v4.dsv4_sparse_attn | 42.61 |
| tilelang | cold | cuda_context | 0.15 |
| tilelang | cold | first_item_jit_compile | 16.03 |
| tilelang | cold | first_item_jit_disk_load | 0.00 |
| tilelang | cold | first_item_jit_other | 2.50 |
| tilelang | cold | first_item_launch_and_run | 2.35 |
| tilelang | warm | interpreter_and_torch_import | 3.85 |
| tilelang | warm | import tilelang | 2.68 |
| tilelang | warm | import prime_rl.trainer.models.kernels.deepseek_v4.dsv4_sparse_attn | 42.08 |
| tilelang | warm | cuda_context | 0.15 |
| tilelang | warm | first_item_jit_compile | 0.00 |
| tilelang | warm | first_item_jit_disk_load | 0.03 |
| tilelang | warm | first_item_jit_other | 0.15 |
| tilelang | warm | first_item_launch_and_run | 0.11 |

Per-item first-call time in ms (lower is better) and compiles triggered by that item.

| # | item | backend | cold ms | cold compiles | warm ms | warm compiles |
|---|---|---|---|---|---|---|
| 0 | stream00-short-14224-csa-cp8r1 | tilelang | 20873.4 | 4 | 288.8 | 0 |
| 1 | stream01-tiny-48096-hca-cp1 | tilelang | 17.6 | 0 | 17.2 | 0 |
| 2 | stream02-heavy-38136-hca-cp8r6 | tilelang | 8.8 | 0 | 9.9 | 0 |
| 3 | stream03-short-37152-sliding-cp8r1 | tilelang | 3.5 | 0 | 3.2 | 0 |
| 4 | stream04-short-34872-csa-cp8r5 | tilelang | 5.1 | 0 | 5.1 | 0 |
| 5 | stream05-single-25016-csa-cp8r4 | tilelang | 5.6 | 0 | 5.6 | 0 |
| 6 | stream06-heavy-55096-csa-cp1 | tilelang | 42.7 | 0 | 42.4 | 0 |
| 7 | stream07-short-17664-csa-cp8r2 | tilelang | 3.5 | 0 | 3.3 | 0 |
| 8 | stream08-single-28000-hca-cp1 | tilelang | 20.5 | 0 | 20.5 | 0 |
| 9 | stream09-tiny-33656-hca-cp1 | tilelang | 11.9 | 0 | 11.9 | 0 |
| 10 | stream10-heavy-48264-csa-cp1 | tilelang | 52.9 | 0 | 52.6 | 0 |
| 11 | stream11-tiny-12904-hca-cp1 | tilelang | 5.3 | 0 | 5.1 | 0 |
| 12 | stream12-tiny-39552-sliding-cp1 | tilelang | 13.7 | 0 | 13.7 | 0 |
| 13 | stream13-single-64144-csa-cp8r1 | tilelang | 12.7 | 0 | 12.7 | 0 |
| 14 | stream14-heavy-15304-csa-cp1 | tilelang | 14.2 | 0 | 14.3 | 0 |
| 15 | stream15-heavy-48400-sliding-cp1 | tilelang | 22.2 | 0 | 22.0 | 0 |
| 16 | stream16-tiny-65360-hca-cp1 | tilelang | 22.8 | 0 | 22.8 | 0 |
| 17 | stream17-heavy-4848-sliding-cp8r5 | tilelang | 2.3 | 0 | 2.7 | 0 |
| 18 | stream18-short-24576-sliding-cp1 | tilelang | 11.9 | 0 | 11.8 | 0 |
| 19 | stream19-single-36016-hca-cp1 | tilelang | 28.2 | 0 | 28.2 | 0 |
| 20 | stream20-short-24768-hca-cp1 | tilelang | 13.3 | 0 | 13.2 | 0 |
| 21 | stream21-heavy-39048-sliding-cp1 | tilelang | 17.9 | 0 | 17.8 | 0 |
| 22 | stream22-single-60624-sliding-cp1 | tilelang | 28.1 | 0 | 28.1 | 0 |
| 23 | stream23-single-61784-csa-cp1 | tilelang | 90.4 | 0 | 89.9 | 0 |
| 24 | stream24-single-6944-hca-cp8r6 | tilelang | 2.7 | 0 | 2.9 | 0 |
| 25 | stream25-single-56616-hca-cp1 | tilelang | 52.4 | 0 | 52.4 | 0 |
| 26 | stream26-tiny-60696-sliding-cp8r2 | tilelang | 3.5 | 0 | 3.5 | 0 |
| 27 | stream27-tiny-62600-sliding-cp8r4 | tilelang | 3.6 | 0 | 3.7 | 0 |
| 28 | stream28-single-42136-csa-cp1 | tilelang | 60.7 | 0 | 60.7 | 0 |
| 29 | stream29-tiny-22480-sliding-cp8r4 | tilelang | 2.5 | 0 | 2.7 | 0 |
| 30 | stream30-heavy-38080-csa-cp1 | tilelang | 31.1 | 0 | 31.1 | 0 |
| 31 | stream31-short-12824-hca-cp8r5 | tilelang | 2.7 | 0 | 2.9 | 0 |
