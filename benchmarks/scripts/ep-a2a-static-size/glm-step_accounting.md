main rank 0: 1 of 4 steps, window 2969.8 ms = compute 2343.1 + exposed comm 540.9 + idle 85.7
main rank 1: 1 of 4 steps, window 2969.8 ms = compute 2046.7 + exposed comm 871.1 + idle 51.9
main rank 2: 1 of 4 steps, window 2969.6 ms = compute 2303.5 + exposed comm 604.6 + idle 61.6
main rank 3: 1 of 4 steps, window 2969.7 ms = compute 1665.7 + exposed comm 1211.9 + idle 92.1
main rank 4: 1 of 4 steps, window 2969.8 ms = compute 2468.5 + exposed comm 436.4 + idle 65.0
main rank 5: 1 of 4 steps, window 2969.8 ms = compute 1784.3 + exposed comm 1126.7 + idle 58.7
main rank 6: 1 of 4 steps, window 2969.8 ms = compute 1659.5 + exposed comm 1239.2 + idle 71.1
main rank 7: 1 of 4 steps, window 2969.7 ms = compute 1829.3 + exposed comm 1048.9 + idle 91.5
main: 8 ranks, mean window 2969.7 ms, 7247 GPU activities per step
branch rank 0: 1 of 4 steps, window 2538.0 ms = compute 1941.9 + exposed comm 526.1 + idle 70.0
branch rank 1: 1 of 4 steps, window 2538.1 ms = compute 1678.4 + exposed comm 810.2 + idle 49.5
branch rank 2: 1 of 4 steps, window 2537.8 ms = compute 1907.4 + exposed comm 592.5 + idle 37.9
branch rank 3: 1 of 4 steps, window 2538.0 ms = compute 1322.3 + exposed comm 1131.3 + idle 84.5
branch rank 4: 1 of 4 steps, window 2538.1 ms = compute 2065.8 + exposed comm 419.2 + idle 53.2
branch rank 5: 1 of 4 steps, window 2538.3 ms = compute 1459.9 + exposed comm 1037.7 + idle 40.7
branch rank 6: 1 of 4 steps, window 2538.0 ms = compute 1363.3 + exposed comm 1102.1 + idle 72.6
branch rank 7: 1 of 4 steps, window 2538.0 ms = compute 1487.9 + exposed comm 968.8 + idle 81.2
branch: 8 ranks, mean window 2538.0 ms, 4160 GPU activities per step

| row | main ms | branch ms | delta ms (branch - main) |
|---|---:|---:|---:|
| compute: eager elementwise / copy / index | 659.11 | 129.87 | -529.24 |
| compute: inductor | 0.00 | 141.31 | 141.31 |
| compute: flash attention | 512.16 | 532.64 | 20.48 |
| compute: dense GEMM | 454.22 | 465.46 | 11.24 |
| compute: other compute | 87.98 | 80.92 | -7.06 |
| compute: grouped GEMM | 299.09 | 303.15 | 4.06 |
| exposed comm: SendRecv/AllToAll | 605.24 | 514.47 | -90.77 |
| exposed comm: AllGather | 113.81 | 131.28 | 17.47 |
| exposed comm: ReduceScatter | 164.37 | 174.90 | 10.53 |
| exposed comm: AllReduce | 1.56 | 2.84 | 1.28 |
| idle, host late: aten::copy_ in forward | 16.17 | 0.00 | -16.17 |
| idle, host late: other sites | 10.19 | 22.75 | 12.56 |
| idle, host late: PythonDispatchMode in forward | 8.07 | 0.00 | -8.07 |
| idle, host late: aten::copy_ in ## Call CompiledFxGraph fngz4jxoc52nacwollb3llbihzho2eqibhj4ududt3mfyfr75huu ## | 0.00 | 4.03 | 4.03 |
| idle, host late: aten::copy_ | 3.81 | 0.00 | -3.81 |
| idle, host late: CompiledFunction in ## Call CompiledFxGraph fqxos22cg2ynztxnompewljucb6o4tasroncmcgpusaptte5hwjq ## | 0.00 | 2.91 | 2.91 |
| idle, host late: aten::copy_ in ## Call CompiledFxGraph fng6iqr3oevkyv326idlyuxnmfywldrkkpzkbdr3pzkhyy6zvmig ## | 0.00 | 2.76 | 2.76 |
| idle, host late: blocked in cudaStreamSynchronize: aten::_local_scalar_dense | 6.70 | 8.74 | 2.03 |
| idle, host late: aten::topk in ## Call CompiledFxGraph fr45t5kzni5p7jhn747bq4whuofht7fab7xjlpsmvzbiq6fn2g3w ## | 0.00 | 1.92 | 1.92 |
| idle, host late: aten::view in FSDP::all_gather_copy_out (model.layers.2) [pg=8] | 1.81 | 0.00 | -1.81 |
| idle, host late: aten::addmm in forward | 1.55 | 0.00 | -1.55 |
| idle, host late: aten::histc in ## Call CompiledFxGraph fr45t5kzni5p7jhn747bq4whuofht7fab7xjlpsmvzbiq6fn2g3w ## | 0.00 | 1.52 | 1.52 |
| idle, host late: CompiledFunction in forward | 0.00 | 1.46 | 1.46 |
| idle, host late: aten::sigmoid in forward | 1.40 | 0.00 | -1.40 |
| idle, host late: aten::div | 2.41 | 1.32 | -1.08 |
| idle, host late: no cpu_op in forward | 2.99 | 3.31 | 0.32 |
| idle, host late: no cpu_op | 3.00 | 2.96 | -0.04 |
| idle, queued (stream wait) | 14.09 | 7.52 | -6.57 |
| **traced step window** | **2969.74** | **2538.04** | **-431.70** |
| tracing overhead / residual (untraced minus traced window) | -12.74 | 5.96 | 18.70 |
| **untraced median step** | **2957.00** | **2544.00** | **-413.00** |
