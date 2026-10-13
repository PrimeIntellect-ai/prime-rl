main rank 0: 2 of 4 steps, window 6032.2 ms = compute 4878.5 + exposed comm 955.5 + idle 198.2
main rank 1: 2 of 4 steps, window 6032.7 ms = compute 4883.7 + exposed comm 947.9 + idle 201.1
main rank 2: 2 of 4 steps, window 6032.3 ms = compute 4845.4 + exposed comm 989.8 + idle 197.1
main rank 3: 2 of 4 steps, window 6031.9 ms = compute 4884.2 + exposed comm 961.6 + idle 186.1
main rank 4: 2 of 4 steps, window 6031.2 ms = compute 4969.0 + exposed comm 876.0 + idle 186.2
main rank 5: 2 of 4 steps, window 6031.2 ms = compute 4906.7 + exposed comm 933.1 + idle 191.4
main rank 6: 2 of 4 steps, window 6031.2 ms = compute 4898.5 + exposed comm 949.6 + idle 183.1
main rank 7: 2 of 4 steps, window 6031.3 ms = compute 4890.5 + exposed comm 953.8 + idle 186.9
main: 8 ranks, mean window 6031.8 ms, 41702 GPU activities per step
branch rank 0: 2 of 4 steps, window 4512.1 ms = compute 3384.6 + exposed comm 1027.9 + idle 99.6
branch rank 1: 2 of 4 steps, window 4512.4 ms = compute 3400.2 + exposed comm 1016.4 + idle 95.8
branch rank 2: 2 of 4 steps, window 4512.4 ms = compute 3376.2 + exposed comm 1035.4 + idle 100.8
branch rank 3: 2 of 4 steps, window 4512.3 ms = compute 3390.5 + exposed comm 1021.8 + idle 100.1
branch rank 4: 2 of 4 steps, window 4512.8 ms = compute 3502.6 + exposed comm 908.8 + idle 101.5
branch rank 5: 2 of 4 steps, window 4512.4 ms = compute 3432.0 + exposed comm 984.1 + idle 96.3
branch rank 6: 2 of 4 steps, window 4512.4 ms = compute 3387.8 + exposed comm 1027.6 + idle 96.9
branch rank 7: 2 of 4 steps, window 4512.4 ms = compute 3415.7 + exposed comm 999.3 + idle 97.5
branch: 8 ranks, mean window 4512.4 ms, 23846 GPU activities per step

| row | main ms | branch ms | delta ms (branch - main) |
|---|---:|---:|---:|
| compute: eager elementwise / copy / index | 2252.61 | 176.64 | -2075.98 |
| compute: inductor | 0.00 | 496.92 | 496.92 |
| compute: flash attention | 1371.36 | 1463.12 | 91.76 |
| compute: other compute | 198.47 | 171.08 | -27.39 |
| compute: dense GEMM | 393.33 | 410.51 | 17.17 |
| compute: grouped GEMM | 678.80 | 692.91 | 14.11 |
| exposed comm: SendRecv/AllToAll | 921.52 | 962.07 | 40.55 |
| exposed comm: AllGather | 9.39 | 17.07 | 7.68 |
| exposed comm: ReduceScatter | 11.86 | 16.68 | 4.82 |
| exposed comm: AllReduce | 3.13 | 6.84 | 3.71 |
| idle, host late: aten::copy_ | 31.81 | 0.00 | -31.81 |
| idle, host late: no cpu_op | 14.28 | 6.22 | -8.05 |
| idle, host late: other sites | 10.61 | 16.16 | 5.55 |
| idle, host late: blocked in cudaStreamSynchronize: aten::_local_scalar_dense | 16.05 | 12.24 | -3.81 |
| idle, host late: aten::div | 5.48 | 1.95 | -3.53 |
| idle, host late: aten::copy_ in forward | 3.08 | 0.00 | -3.08 |
| idle, host late: RMSNormFunction in forward | 1.71 | 0.00 | -1.71 |
| idle, host late: aten::mul_ | 1.83 | 3.47 | 1.64 |
| idle, host late: PythonSubclass | 7.28 | 5.82 | -1.45 |
| idle, host late: aten::copy_ in ## Call CompiledFxGraph fku2csp4mz6etpopc2titr7ybtvt76x3hbfoucnq52dlcpldssgf ## | 0.00 | 1.22 | 1.22 |
| idle, host late: aten::copy_ in ## Call CompiledFxGraph fdxgjk5w7cs4cttknpdf6leqxebbudz32ixae5pscr22u2aejyp4 ## | 0.00 | 1.18 | 1.18 |
| idle, host late: aten::copy_ in ## Call CompiledFxGraph fnv3jsuenzkuqr6plhuxkqrlfxlydh3s7wwhgqqdi44tnpkk26kw ## | 0.00 | 1.16 | 1.16 |
| idle, host late: aten::copy_ in ## Call CompiledFxGraph fgf2pbnon7ovfa26bi5sbtraiocaat7rt7ruq6mosxikx7ada5rz ## | 0.00 | 1.15 | 1.15 |
| idle, host late: PythonSubclass in Optimizer.step#AdamW.step | 0.99 | 0.00 | -0.99 |
| idle, host late: no cpu_op in forward | 3.44 | 3.48 | 0.04 |
| idle, queued (stream wait) | 94.72 | 44.50 | -50.22 |
| **traced step window** | **6031.76** | **4512.39** | **-1519.36** |
| tracing overhead / residual (untraced minus traced window) | -66.76 | -28.39 | 38.36 |
| **untraced median step** | **5965.00** | **4484.00** | **-1481.00** |
