# n-gram module A/B: parity and step time

Runs under `/home/garrett/tmp/ngram-module/ab-validate`, collected 2026-10-26 22:03 UTC. Per-step values come from each run's `monitors/file/metrics.jsonl` (rank 0). Recompiles are `Recompiling function` records in every rank's `TORCH_LOGS=recompiles` output, placed in the step whose end time follows them.

## Runs

Peak memory: max over steps of `perf/peak_memory` (rank 0, reserved, GiB, lower is better). Recompiles: total over ranks, fewer is better. Last: the latest step with a recompile on any rank.

| cell                         | arm  | rep | head      | steps | peak GiB | recompiles | last | status          |
| ---------------------------- | ---- | --: | --------- | ----: | -------: | ---------: | ---: | --------------- |
| cell2                        | main |   0 | 1810d5b9a |     5 |     34.0 |          0 |    - | ok              |
| cell2                        | main |   1 | 1810d5b9a |     5 |     34.0 |          0 |    - | ok              |
| cell3-4gpu-graph-breaks      | main |   0 | 1810d5b9a |     5 |     21.2 |         88 |    4 | ok              |
| cell5-4gpu-graph-breaks      | main |   0 | 1810d5b9a |     5 |     17.8 |          0 |    - | ok              |
| cell5-4gpu-trace             | main |   0 | 1810d5b9a |     5 |     17.8 |          0 |    - | ok              |
| cell6-4gpu                   | main |   0 | 1810d5b9a |     0 |        - |          0 |    - | crashed, exit 1 |
| cell6-4gpu                   | main |   1 | 1810d5b9a |     0 |        - |          0 |    - | crashed, exit 1 |
| cell8-4gpu-graph-breaks      | main |   0 | 1810d5b9a |     5 |     12.8 |          0 |    - | ok              |
| cell8-4gpu-graph-breaks-noac | main |   0 | 1810d5b9a |     8 |     14.3 |         32 |    2 | ok              |

Crashes, with the last exception line on any rank:

- cell6-4gpu main r0: `RuntimeError: Autograd not support dtype: Long`
- cell6-4gpu main r1: `RuntimeError: Autograd not support dtype: Long`

## Step time

Seconds per step (`time/step`), lower is better. All: every logged step, including the compile step 1. Post-compile: steps from the first step after both step 1 and the last recompile on any rank.

| cell                         | run     | all med | all min | all max | post from | post n | post med | post min | post max |
| ---------------------------- | ------- | ------: | ------: | ------: | --------: | -----: | -------: | -------: | -------: |
| cell2                        | main r0 |   2.325 |   2.262 |  86.793 |         2 |      4 |    2.295 |    2.262 |    2.440 |
| cell2                        | main r1 |   2.367 |   2.343 |  86.200 |         2 |      4 |    2.363 |    2.343 |    2.760 |
| cell3-4gpu-graph-breaks      | main r0 |   4.956 |   0.335 | 132.336 |         5 |      1 |    0.335 |    0.335 |    0.335 |
| cell5-4gpu-graph-breaks      | main r0 |   0.720 |   0.716 |  87.817 |         2 |      4 |    0.720 |    0.716 |    0.775 |
| cell5-4gpu-trace             | main r0 |   0.973 |   0.949 |  86.694 |         2 |      4 |    0.963 |    0.949 |    1.013 |
| cell6-4gpu                   | main r0 |       - |       - |       - |         2 |      0 |        - |        - |        - |
| cell6-4gpu                   | main r1 |       - |       - |       - |         2 |      0 |        - |        - |        - |
| cell8-4gpu-graph-breaks      | main r0 |   1.462 |   1.445 | 192.933 |         2 |      4 |    1.458 |    1.445 |    1.491 |
| cell8-4gpu-graph-breaks-noac | main r0 |   0.631 |   0.610 | 218.164 |         3 |      6 |    0.623 |    0.610 |    0.643 |

## Startup

Seconds, lower is better, from loguru timestamps (1 s resolution). Launcher: `run_ab.py` start to the launcher's first log line (launcher imports). Trainer import: to the trainer's first log line (pre-download, torchrun, rank imports). Setup: to the training loop (model build, weight load and conversion, FSDP). Step 1: compile and the first step. Wall: the whole run including shutdown.

| cell                         | run     | local venv | launcher | trainer import | setup | step 1 | wall |
| ---------------------------- | ------- | ---------- | -------: | -------------: | ----: | -----: | ---: |
| cell2                        | main r0 | no         |       46 |             55 |    13 |     87 |  216 |
| cell2                        | main r1 | yes        |       26 |             18 |     7 |     87 |  153 |
| cell3-4gpu-graph-breaks      | main r0 | yes        |       14 |             19 |     6 |    133 |  191 |
| cell5-4gpu-graph-breaks      | main r0 | yes        |       15 |             19 |     7 |     88 |  138 |
| cell5-4gpu-trace             | main r0 | yes        |       15 |             19 |     6 |     87 |  221 |
| cell6-4gpu                   | main r0 | yes        |       13 |             20 |     6 |      - |  127 |
| cell6-4gpu                   | main r1 | yes        |       14 |             19 |     7 |      - |  128 |
| cell8-4gpu-graph-breaks      | main r0 | yes        |       15 |             20 |     6 |    194 |  248 |
| cell8-4gpu-graph-breaks-noac | main r0 | yes        |       14 |             19 |     7 |    219 |  271 |

## Parity

Loss gap: max over steps and run pairs of abs(loss x - loss y), in nats. Grad gap: max over steps and run pairs of abs(grad norm x - grad norm y) / grad norm y, y the second-named arm. Both: lower is closer. Halves split the common steps in two. A real bug shows as an arm gap above the floor that grows from the first half to the second. Comparisons without a pair of finished runs are left out.

| cell  | comparison           | pairs | steps | loss all | loss 1st | loss 2nd | grad all | grad 1st | grad 2nd |
| ----- | -------------------- | ----: | ----: | -------: | -------: | -------: | -------: | -------: | -------: |
| cell2 | main vs main (floor) |     1 |     5 |  0.0e+00 |  0.0e+00 |  0.0e+00 |  0.0e+00 |  0.0e+00 |  0.0e+00 |

## Per-step loss and grad norm

### cell2: loss (nats)

| step |   main r0 |   main r1 |
| ---- | --------: | --------: |
| 1    | 12.474824 | 12.474824 |
| 2    | 12.467292 | 12.467292 |
| 3    | 12.472014 | 12.472014 |
| 4    | 12.468057 | 12.468057 |
| 5    | 12.468276 | 12.468276 |

### cell2: grad norm

| step |  main r0 |  main r1 |
| ---- | -------: | -------: |
| 1    | 0.703503 | 0.703503 |
| 2    | 0.641978 | 0.641978 |
| 3    | 0.664328 | 0.664328 |
| 4    | 0.660858 | 0.660858 |
| 5    | 0.656854 | 0.656854 |

### cell3-4gpu-graph-breaks: loss (nats)

| step |   main r0 |
| ---- | --------: |
| 1    | 12.473786 |
| 2    | 12.467033 |
| 3    | 12.471243 |
| 4    | 12.470989 |
| 5    | 12.467202 |

### cell3-4gpu-graph-breaks: grad norm

| step |  main r0 |
| ---- | -------: |
| 1    | 0.752187 |
| 2    | 0.689484 |
| 3    | 0.650804 |
| 4    | 0.672543 |
| 5    | 0.648752 |

### cell5-4gpu-graph-breaks: loss (nats)

| step |   main r0 |
| ---- | --------: |
| 1    | 12.473391 |
| 2    | 12.467816 |
| 3    | 12.470559 |
| 4    | 12.469975 |
| 5    | 12.464944 |

### cell5-4gpu-graph-breaks: grad norm

| step |  main r0 |
| ---- | -------: |
| 1    | 0.751846 |
| 2    | 0.690523 |
| 3    | 0.648359 |
| 4    | 0.675682 |
| 5    | 0.648042 |

### cell5-4gpu-trace: loss (nats)

| step |   main r0 |
| ---- | --------: |
| 1    | 12.473391 |
| 2    | 12.467816 |
| 3    | 12.470559 |
| 4    | 12.469975 |
| 5    | 12.464944 |

### cell5-4gpu-trace: grad norm

| step |  main r0 |
| ---- | -------: |
| 1    | 0.751846 |
| 2    | 0.690523 |
| 3    | 0.648359 |
| 4    | 0.675682 |
| 5    | 0.648042 |

### cell8-4gpu-graph-breaks: loss (nats)

| step |   main r0 |
| ---- | --------: |
| 1    | 12.472073 |
| 2    | 12.472437 |
| 3    | 12.466190 |
| 4    | 12.468506 |
| 5    | 12.469532 |

### cell8-4gpu-graph-breaks: grad norm

| step |  main r0 |
| ---- | -------: |
| 1    | 0.720375 |
| 2    | 0.680420 |
| 3    | 0.690451 |
| 4    | 0.636514 |
| 5    | 0.620347 |

### cell8-4gpu-graph-breaks-noac: loss (nats)

| step |   main r0 |
| ---- | --------: |
| 1    | 12.472607 |
| 2    | 12.475106 |
| 3    | 12.467497 |
| 4    | 12.468192 |
| 5    | 12.466856 |
| 6    | 12.466836 |
| 7    | 12.467986 |
| 8    | 12.466085 |

### cell8-4gpu-graph-breaks-noac: grad norm

| step |  main r0 |
| ---- | -------: |
| 1    | 0.720597 |
| 2    | 0.674990 |
| 3    | 0.694895 |
| 4    | 0.637983 |
| 5    | 0.620432 |
| 6    | 0.652965 |
| 7    | 0.605641 |
| 8    | 0.689576 |
