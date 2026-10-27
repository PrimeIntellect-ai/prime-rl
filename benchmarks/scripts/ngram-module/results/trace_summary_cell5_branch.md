/home/garrett/tmp/ngram-module/ab/branch/cell5-trace/r0/trace/trace_0.json.gz
5 steps, each ending at an Optimizer.step annotation. Steady steps: [2, 3, 4, 5].

Host-side calls per step (fewer is better for syncs). Lookups: aten::embedding (forward and AC recompute) and aten::embedding_dense_backward, by weight rows x width. a2a: record_param_comms all-to-alls by dtype and the all-to-all op that issued them. Sync point: one CPU op holding a blocking CUDA call or a D2H copy, named by its outermost aten op. An op nested in an op of the same name (e.g. under AC's dispatch mode) counts once. The n-gram table forward runs from its FSDP pre-forward hook to its post-forward hook.

| event                                                         | step 1 | step 2 | step 3 | step 4 | step 5 |
| ------------------------------------------------------------- | -----: | -----: | -----: | -----: | -----: |
| a2a BFloat16 GeneratedBackwardFor__c10d_functional_all_to_all |      2 |      2 |      2 |      2 |      2 |
| a2a BFloat16 _c10d_functional::all_to_all_single              |      2 |      2 |      2 |      2 |      2 |
| a2a BFloat16 prime_rl_collectives::all_to_all_single          |     48 |     48 |     48 |     48 |     48 |
| a2a Long _c10d_functional::all_to_all_single                  |      4 |      4 |      4 |      4 |      4 |
| a2a Long prime_rl_collectives::all_to_all_single_equal        |     16 |     16 |     16 |     16 |     16 |
| aten::_local_scalar_dense                                     |     93 |    102 |    102 |    102 |    102 |
| aten::item                                                    |     93 |    102 |    102 |    102 |    102 |
| cudaDeviceSynchronize                                         |    472 |      1 |      1 |      1 |      1 |
| cudaMemcpyAsync DtoH (Device -> Pageable)                     |     10 |     10 |     10 |     10 |     10 |
| cudaMemcpyAsync DtoH (Device -> Pinned)                       |     77 |     86 |     86 |     86 |     86 |
| cudaStreamSynchronize                                         |    542 |    104 |    104 |    104 |    104 |
| lookup bwd embed_tokens 248320x1024                           |      1 |      1 |      1 |      1 |      1 |
| lookup bwd n-gram 160256x160                                  |      1 |      1 |      1 |      1 |      1 |
| lookup bwd n-gram 160576x160                                  |      1 |      1 |      1 |      1 |      1 |
| lookup fwd embed_tokens 248320x1024                           |      1 |      1 |      1 |      1 |      1 |
| lookup fwd n-gram 160256x160                                  |      1 |      1 |      1 |      1 |      1 |
| lookup fwd n-gram 160576x160                                  |      1 |      1 |      1 |      1 |      1 |
| sync point in CausalConv1dFunction                            |      1 |      0 |      0 |      0 |      0 |
| sync point in CausalConv1dFunctionBackward                    |      1 |      0 |      0 |      0 |      0 |
| sync point in ChunkGatedDeltaRuleFunction                     |      1 |      0 |      0 |      0 |      0 |
| sync point in ChunkGatedDeltaRuleFunctionBackward             |      1 |      0 |      0 |      0 |      0 |
| sync point in LayerNormGatedFunction                          |      1 |      0 |      0 |      0 |      0 |
| sync point in LayerNormGatedFunctionBackward                  |      1 |      0 |      0 |      0 |      0 |
| sync point in aten::copy_                                     |      1 |      1 |      1 |      1 |      1 |
| sync point in aten::is_nonzero                                |      1 |      1 |      1 |      1 |      1 |
| sync point in aten::item                                      |     12 |     21 |     21 |     21 |     21 |
| sync point in aten::repeat_interleave                         |     64 |     64 |     64 |     64 |     64 |
| sync point in aten::to                                        |     16 |     17 |     17 |     17 |     17 |
| sync point inside an n-gram table forward                     |      2 |      2 |      2 |      2 |      2 |

GPU kernel time launched between each module's FSDP pre and post hooks per step, median over steady steps (ms, lower is better). Spans: hook pairs per step (micro-batches, plus AC recomputes in forward). A layer's backward includes its AC recompute.

| module         | pass     | spans | GPU ms |
| -------------- | -------- | ----: | -----: |
| layer 0        | backward |   1.0 |  12.77 |
| layer 0        | forward  |   1.0 |   8.62 |
| layer 1        | backward |   1.0 |  18.19 |
| layer 1        | forward  |   1.0 |   9.87 |
| layer 2        | backward |   1.0 |  12.46 |
| layer 2        | forward  |   1.0 |   5.44 |
| layer 3        | backward |   1.0 |  11.52 |
| layer 3        | forward  |   1.0 |   3.10 |
| layer 4        | backward |   1.0 |  11.75 |
| layer 4        | forward  |   1.0 |   3.88 |
| layer 5        | backward |   1.0 |  17.35 |
| layer 5        | forward  |   1.0 |   6.56 |
| layer 6        | backward |   1.0 |  11.24 |
| layer 6        | forward  |   1.0 |   7.98 |
| layer 7        | backward |   1.0 |   6.59 |
| layer 7        | forward  |   1.0 |   5.05 |
| n-gram table 1 | backward |   1.0 |  19.51 |
| n-gram table 1 | forward  |   1.0 |   1.41 |
| n-gram table 5 | backward |   1.0 |  83.92 |
| n-gram table 5 | forward  |   1.0 |   1.15 |
