---
license: mit
pretty_name: Memory Writing Training Task Streams
task_categories:
- reinforcement-learning
language:
- en
tags:
- continual-learning
- memory
- agents
- swe-smith
- alchemy
configs:
- config_name: symbolic_alchemy
  data_files: symbolic_alchemy/train_streams.jsonl
- config_name: codebase_adaptation
  data_files: codebase_adaptation/train_streams.jsonl
- config_name: codebase_adaptation_tasks
  data_files: codebase_adaptation/tasks.jsonl
---

# Training Task Streams for Memory Writing

This repository contains the training task streams used in the blog post *How to Train Memory Writing for Continual Learning*. Each row of a `train_streams.jsonl` file is one stream: an ordered sequence of tasks that the model solves one after another, carrying a memory from each task to the next.

| Domain | File | Streams | Tasks per stream |
|---|---|---|---|
| Symbolic Alchemy | `symbolic_alchemy/train_streams.jsonl` | 950 | 10 |
| Codebase Adaptation | `codebase_adaptation/train_streams.jsonl` | 382 | 12 |

## Symbolic Alchemy

Each stream is one prebuilt episode of the level `perceptual_mapping_randomized_with_random_bottleneck` shipped with [dm_alchemy](https://github.com/google-deepmind/dm_alchemy), which contains 1000 episodes with fixed chemistries. The 10 trials of an episode share the same chemistry and form the tasks of the stream, with at most 20 steps per trial.

- `metadata.episode_index` indexes the prebuilt episode set. The `prompt` field is a placeholder, since the environment state is rebuilt from the episode index.
- `metadata.max_steps_per_trial` is the step budget per task.
- `metadata.rewrite_granularity` is `trial`, meaning memory is written after every task.
- The 20 episodes used for evaluation are excluded. Streams are ordered from easy to hard by the number of edges in the chemistry graph (12, 10, 9, 8, 7), and shuffled within each level.

## Codebase Adaptation

Each stream contains 12 bug-fixing tasks: 6 issues from one Python repository (`repoA`) followed by 6 issues from another (`repoB`). The issues are synthetic bugs from [SWE-smith](https://github.com/SWE-bench/SWE-smith), drawn from 53 repositories. The streams use 4584 distinct issues, and each issue appears in exactly one stream.

`codebase_adaptation/train_streams.jsonl` fields (under `metadata`):

- `instance_ids`: the 12 issues in order. `stage_labels` gives the repository of each issue and `strategies` the SWE-smith bug-generation strategy.
- `difficulty_tier`: the composition of each 6-issue segment, with difficulty bucketed by the measured pass rate of the bug-generation strategy. `pure_easy` (63 streams) has 6 easy issues, `easy_mid` (138) has 3 easy and 3 medium, and `mixed` (181) has 2 easy, 2 medium and 2 hard. Within a segment, issues are ordered from easiest to hardest.

`codebase_adaptation/tasks.jsonl` has one record for each of the 4584 issues:

- `problem_statement`: the issue text shown to the model.
- `repo`, `base_commit`: the SWE-smith mirror repository and the branch that holds the bug.
- `image_name`: the SWE-smith Docker image of the repository.
- `FAIL_TO_PASS`, `PASS_TO_PASS`: the tests used for grading. `PASS_TO_PASS` is restricted to tests in the same files as the `FAIL_TO_PASS` tests.
- `test_patch`: restores the test files that SWE-smith removes from the bug branch, applied before grading.
- `test_command`, `workdir`, `test_timeout`: how the tests are run.
- `strategy`, `tier`, `cluster`, `n_f2p`, `n_p2p_local`: features used to build the streams.

## Sources and licenses

The Codebase Adaptation tasks are derived from [SWE-smith](https://huggingface.co/datasets/SWE-bench/SWE-smith) (MIT license). The Symbolic Alchemy streams index the prebuilt episodes of [dm_alchemy](https://github.com/google-deepmind/dm_alchemy), which are not redistributed here. Please also cite these works if you use the corresponding data.
