# Session Handoff: Rollout Cleanup & ArtifactManager v2

## 任务目的

1. 实现 BlenderGym ArtifactManager v2（PR2），将 artifact 路径解析、I/O、retention 策略集中到 `ArtifactManager` 类中。
2. 发现并修复 prime-rl filesystem transport 的 rollout `.bin` 文件无限累积问题（训练 487 步累积 2.4TB 无用数据）。

## 执行内容

- 按 plan 实现 ArtifactManager v2 PR2：新建 `artifact_manager.py`，重构 `env.py`、`rubric.py`、`render.py`、`schema.py`、`trajectory_writer.py`、`__init__.py`，更新测试。
- 端到端训练验证 ArtifactManager 集成正确，`blender_user/` 不再重复放入每个 `turn_N/`。
- 调查 `outputs/` 目录结构：发现 trainer 写 `output_dir/rollouts/`（micro batch），orchestrator 写 `output_dir/run_default/rollouts/`（training batch），两侧 `.bin` 消费后永不删除。
- 在 `FileSystemTrainingBatchReceiver.receive()` 和 `FileSystemMicroBatchReceiver.receive()` 中各加一行 `unlink`，读完 `.bin` 后立即删除。
- 训练 3 步验证：rollout 目录从 ~5GB/step 降到 ~70MB/step（仅保留 `.jsonl` 日志），节省 98.5% 磁盘。

## 调试经验

- NCCL weight broadcast 在 H20 GPU 上初始化会 hang（>20 分钟无进展），切换到 `--weight-broadcast.type filesystem` 可正常运行。原因未深究，可能是 NCCL 版本或网络配置问题。
- prime-rl CLI 无法覆盖 list-of-tables 中的嵌套字段（如 `orchestrator.train.env.args.work_root`），因为 TOML parser 把 `[orchestrator.train.env.args]` 当 dict 解析，与 list 结构冲突。
- `packer.__init__` 启动时 `shutil.rmtree` 整个 rollout 目录，说明框架已视 rollout `.bin` 为临时文件，只是运行中没有逐步清理。

## 参考代码

| 文件 | 关键位置 | 说明 |
|------|---------|------|
| `src/prime_rl/transport/filesystem.py` | `FileSystemTrainingBatchReceiver.receive()` L95-101, `FileSystemMicroBatchReceiver.receive()` L162-169 | rollout cleanup 的两处改动点 |
| `src/prime_rl/configs/rl.py` | `auto_setup_output_dir()` | trainer `output_dir` vs orchestrator `output_dir/run_default` 的分离逻辑 |
| `src/prime_rl/utils/pathing.py` | `get_rollout_dir()`, `clean_future_steps()` | rollout 目录约定和启动时清理逻辑 |
| `src/prime_rl/trainer/rl/packer.py` | `BasePacker.__init__` | 启动时 `rmtree` rollout 目录 |
| `environments/blendergym/blendergym/artifact_manager.py` | 整个文件 | ArtifactManager v2 核心：`ArtifactPolicy`, `TurnPaths`, `RolloutPaths`, `ArtifactManager` |
| `environments/blendergym/blendergym/env.py` | `BlenderGymEnv` | 集成 ArtifactManager，删除了 `_make_work_dir` / `_populate_inputs_symlinks` |
| `environments/blendergym/blendergym/rubric.py` | `BlenderGymRubric` | 接收 `artifact_manager`，委托 save/cleanup/prune |
| `configs/multimodal/rl_blendergym.toml` | 整个文件 | 当前训练配置：9B 模型，dp=6 推理 + 2 卡 FSDP 训练 |

## 最终方案

Rollout cleanup：在 `filesystem.py` 的两个 receiver 的 `receive()` 中，读完 `.bin` 后直接 `path.unlink(missing_ok=True)`。不加配置开关（无需回滚能力，因为 packer 启动时已 `rmtree`），不改其他文件。总共 1 个文件 +3 行代码。

选择此方案而非"配置开关 + 多文件传参"是因为：`.bin` 文件在整个框架中已被视为临时 IPC 管道，没有合理的保留场景，resume 也不需要它们。

## 下一步任务

尝试研究结合 API（用户描述）。

## 初步方案

- 需要用户进一步明确"结合 API"的具体含义：是指将 prime-rl 的推理/训练能力通过 REST API 暴露？还是指集成外部 API（如 OpenAI API）作为 rollout 后端？还是 BlenderGym 相关的 API？
- 如果是暴露 prime-rl 能力：入口点在 `src/prime_rl/entrypoints/inference.py`（已有 OpenAI-compatible API），可在此基础上扩展。
- 如果是集成外部 API 作为 rollout 后端：看 `src/prime_rl/orchestrator/orchestrator.py` 中的 `setup_inference_pool` 和 client 配置。
- 待用户补充更多细节后再细化方案。
