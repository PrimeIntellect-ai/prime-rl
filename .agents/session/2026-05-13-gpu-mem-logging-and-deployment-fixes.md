# Session Handoff: GPU OOM 诊断与 KAOLA 部署修复

## 任务目的

诊断 BlenderGym RL 训练中的 100% rollout 失败问题，添加 GPU 内存监控日志，修复 S3 同步和 KAOLA 部署流程中的多个问题。

## 当前状态

### 训练正在运行

- **Job**: `ericzyma-job-normal-20260513-195633`（8 GPU，默认 ECR 镜像）
- **进展**: Step 0 约 15 分钟，后续 step 约 2-4 分钟
- **成功率**: 大幅改善，17 个 step 只保留了 5 个失败 rollout（GPU OOM 3 个，xml_parse_failed 2 个）

### gpu_mem 日志仍未验证

`gpu_mem.py` 已改为直接创建 `gpu_mem.log` 文件（绕开 Python logging 框架），当前 job 是第一次运行这个版本。**需要在 Step 0 完成后检查 S3 上是否出现 `gpu_mem.log`**：

```bash
rclone rc --rc-addr=127.0.0.1:5572 vfs/refresh dir="ericzyma/experiments/blendergym-9b-dp6" recursive=true
ls ~/Desktop/codes/s3/experiments/blendergym-9b-dp6/output/logs/envs/train/blendergym/
```

如果 `gpu_mem.log` 存在且有内容，说明修复成功。

## 已修复的问题

### 1. S3 rsync 覆盖失败
- **根因**: S3 FUSE 不支持 `rename()` 和 `chown`，`rsync -a --inplace` 无法覆盖已有文件
- **修复**: `setup_kaola.sh` 中每次 sync 前 `rm -rf "${OUTPUT_S3}"` 全量重建

### 2. 自定义镜像导致 PodInitializing 卡住
- **根因**: Docker Hub 上的 `ericzyma/prime-rl-blendergym:v0.0.5` 需要跨区域拉取
- **修复**: 不指定 `--image`，使用 KAOLA 默认 ECR 镜像（节点已缓存，秒级启动）

### 3. 代码改动未进入 pod
- **根因**: 提交时漏了 `--sync-code .:/data/work/prime-rl`
- **修复**: 必须加 `--sync-code`，并在命令中 `cd /data/work/prime-rl`

### 4. HF_TOKEN 未传入 pod
- **根因**: 单引号 `'...'` 不展开 `$HF_TOKEN`
- **修复**: 用双引号，token 在本地 shell 展开

### 5. S3 guard check
- **功能**: `setup_kaola.sh` 检测到 S3 上已有实验数据时报错退出，防止误覆盖
- **清理**: 用 `rclone purge` 清理 S3（Mac FUSE 的 `rm -rf` 不可靠）

## 未解决的问题

### GPU OOM（Blender 渲染）
- 3/5 个失败 rollout 是 turn 2 的 GPU OOM
- Blender denoiser (OIDN) 加载时从 76 MiB 跳到 1495 MiB
- 与 vLLM KV cache 竞争 GPU 显存
- **需要 gpu_mem 日志确认**：`empty_cache` 在渲染前释放了多少 PyTorch 缓存
- **可能的修复方向**：
  - 降低 `gpu_memory_utilization`（当前 0.80）给 Blender 更多空间
  - 渲染前强制释放更多 PyTorch 缓存
  - 用 CPU denoiser 代替 GPU denoiser（需改 `cycles_denoiser` 参数）

### xml_parse_failed
- 2/5 个失败是模型未输出 `<code>...</code>` 标签
- 可通过优化 prompt 或增加 `max_completion_tokens` 改善

### gpu_mem 日志历程
多次尝试让 GPU 内存监控日志出现在 S3 可见的日志文件中：

| 尝试 | 方法 | 结果 | 原因 |
|------|------|------|------|
| 1 | `StreamHandler(sys.stderr)` + `logging.warning()` | 不出现 | 代码没进 pod（漏了 `--sync-code`） |
| 2 | logger 改名为 `verifiers.blendergym.gpu_mem` | 不出现 | 同上 + env worker 是 spawn 子进程，verifiers handler 不传播 |
| 3 | `StreamHandler(sys.stderr)` + `propagate=False` | 不出现 | env worker 的 stderr 不重定向到文件（`mp.spawn`） |
| 4 | 复制 verifiers 的 FileHandler | 不出现 | handler 可能不在预期的 logger 上 |
| 5 | **直接创建 `gpu_mem.log` 文件** | **待验证** | 从 verifiers FileHandler 拿路径，自己 `open()` 写文件 |

## 关键文件

| 文件 | 说明 |
|------|------|
| `environments/blendergym/blendergym/gpu_mem.py` | GPU 内存监控（当前版本：直接写 `gpu_mem.log`） |
| `environments/blendergym/blendergym/env.py` L342-369 | `add_model_response` 中调用 `log_gpu_mem` |
| `environments/blendergym/blendergym/rubric.py` L159-168 | CLIP 推理前后调用 `log_gpu_mem` |
| `scripts/setup_kaola.sh` | 环境 setup + S3 sync（含 guard check、`--resume` 支持） |
| `.agents/kaola/` | 部署知识库（已更新 README/workflow/troubleshooting） |
| `configs/multimodal/rl_blendergym_kaola.toml` | 训练配置 |

## 正确的提交命令

```bash
cd ~/Desktop/codes/prime-rl
koala submit -m normal -g 8 --sync-code .:/data/work/prime-rl \
    -c "export HF_TOKEN=$HF_TOKEN && export WANDB_API_KEY=$WANDB_API_KEY && export EXP_NAME=blendergym-9b-dp6 && cd /data/work/prime-rl && . scripts/setup_kaola.sh --env blendergym && uv run rl @ configs/multimodal/rl_blendergym_kaola.toml --ckpt.output_dir /local-ssd/checkpoints/\${EXP_NAME}"
```

**关键**：不要指定 `--image`、必须加 `--sync-code`、token 用双引号、用 `.`（source）执行 setup。

## 下一步

1. **检查 gpu_mem.log 是否出现** — 当前 job Step 0 完成后
2. **如果出现**：分析 GPU 内存数据，确定 OOM 根因，调整参数
3. **如果不出现**：verifiers 的 FileHandler 路径可能不对，需要在 debug pod 里手动验证
4. **解决 GPU OOM**：根据 gpu_mem 数据选择方案（降 vLLM 显存 / CPU denoiser / 更积极的 empty_cache）
