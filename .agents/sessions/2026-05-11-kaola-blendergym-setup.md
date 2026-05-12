# Session Handoff: KAOLA BlenderGym 环境搭建与首次训练

## 任务目的

在 KAOLA 集群上搭建 prime-rl BlenderGym RL 训练环境，从零验证全流程（渲染、依赖、配置），并提交 8 GPU 正式训练。

## 执行内容

- 提交 1 GPU debug pod，验证 Blender 4.2 + Cycles OPTIX 渲染管线正常（H200，渲染 1.88s/frame）
- 下载并处理 BlenderGym 数据集（1.9GB zip → 27GB 解压），上传 S3 持久化
- 打包 Blender 和数据集为 tar 文件存 S3（tar 管道恢复 39s vs cp -r 20min，快 30 倍）
- 创建 `scripts/setup_kaola.sh`（`--fast` debug 模式 ~1min / 默认训练模式 ~7min）
- 创建 `configs/multimodal/rl_blendergym_kaola.toml`（路径适配 + render_timeout_s=600）
- 创建 `.agents/kaola/` 知识库（README、paths、workflow、troubleshooting、api）
- 精简 `kaola/.agent/workflow/README.md` 避免重复维护
- 实测 OPTIX cache 行为：driver 级缓存在 `/root/.nv/`，per-rollout blender_user 不影响
- 提交 8 GPU 训练，修复 HF_HOME 只读问题（重定向到 /local-ssd/hf_cache）和 WANDB_API_KEY 缺失
- 训练成功启动（setup 完成 + 模型加载 + wandb 登录通过）

## 调试经验

- `bash scripts/setup_kaola.sh` 在子 shell 运行，export 不传递给后续命令 → 改用 `. scripts/setup_kaola.sh`（source）
- `/threed-code/public_models/` 是只读挂载，HF 下载器无法写入 → `export HF_HOME=/local-ssd/hf_cache`
- `hf_xet` 下载后端在 S3 FUSE 上 panic（创建日志文件失败）→ 设 HF_TOKEN + HF_HOME 到可写路径
- `koala logs -f` 长时间无输出会超时（OPTIX 编译 6min）→ 不用 -f，定期查看
- `koala list` 报 UnicodeEncodeError → `PYTHONIOENCODING=utf-8`
- 训练入口不是 `prime --config` 而是 `uv run rl @ config.toml`（pydantic_config `@` 语法）

## 参考代码

| 文件 | 关键位置 | 说明 |
|------|---------|------|
| `scripts/setup_kaola.sh` | 整个文件 | 环境恢复脚本（source 模式使用） |
| `configs/multimodal/rl_blendergym_kaola.toml` | 整个文件 | KAOLA 平台训练配置 |
| `.agents/kaola/` | 整个目录 | KAOLA 部署知识库（paths/workflow/troubleshooting/api） |
| `environments/blendergym/blendergym/artifact_manager.py` | L165-173 | input symlinks（populate_input_symlinks） |
| `environments/blendergym/blendergym/trajectory_writer.py` | L7-11 | HTML 用相对路径引用 inputs/ |
| `src/prime_rl/trainer/model.py` | L60-67 | pre_download_model（HF snapshot_download） |
| `src/prime_rl/entrypoints/rl.py` | L578-580 | 入口：`rl(cli(RLConfig))` |

## 最终方案

- Setup 脚本用 `source`（`.`）执行，HF_HOME/HF_TOKEN/WANDB_API_KEY 在提交命令中 export
- 数据集用 tar 管道恢复（39s），Blender 也用 tar 管道（8s）
- Checkpoint 通过 `ckpt.output_dir` 实时写 S3，其他产物暂留 /local-ssd/

## 第二次训练（2026-05-12）

首次 8 GPU 训练 `ericzyma-job-normal-20260511-205323` 在 Step 25 保存 checkpoint 时崩溃：
- 原因：`ckpt.output_dir` 指向 S3 FUSE (`/threed-code/`)，PyTorch DCP 调用 `os.rename()` 不被 S3 FUSE 支持（`ENOSYS`）
- 额外发现：训练过程中吞吐量逐步退化（13K→6.5K tok/s），Step 11 有 484s spike（疑似 OPTIX 重编译或渲染瓶颈）

修复：
- `ckpt.output_dir` 改为 `/local-ssd/checkpoints/blendergym-9b-dp6`（本地 NVMe，支持 rename）
- `setup_kaola.sh` 新增 [6/6] 后台 rsync（每 5 min 同步 checkpoint 到 S3，EXIT trap 做最终同步）

## 第三次训练（2026-05-12 13:21）

Job: `ericzyma-job-normal-20260512-132137`（8 GPU，72h TTL）

训练正常运行到 Step 25+，但发现 **S3 上没有任何同步数据**。

### 问题：rsync 后台同步静默失败

**根因**：`setup_kaola.sh` 中的 rsync 使用默认写入模式（创建临时文件 `.filename.XXXX` → `rename()` 为正式文件名），而 S3 FUSE 不支持 `rename()` 系统调用（返回 `ENOSYS`，同 DCP checkpoint 崩溃的原因）。`2>/dev/null || true` 吞掉了所有错误，导致每 5 分钟的同步全部失败但无任何报错。

**已修复**：`setup_kaola.sh` 两条 rsync 命令都加了 `--inplace`（直接写入目标文件，跳过临时文件 + rename）：
```bash
# 修复前
rsync -a --delete "${CKPT_LOCAL}/" "${CKPT_S3}/" 2>/dev/null || true
rsync -a --copy-links --exclude broadcasts/ --exclude '*.bin' "${OUTPUT_LOCAL}/" "${OUTPUT_S3}/" 2>/dev/null || true

# 修复后
rsync -a --inplace --delete "${CKPT_LOCAL}/" "${CKPT_S3}/" 2>/dev/null || true
rsync -a --inplace --copy-links --exclude broadcasts/ --exclude '*.bin' "${OUTPUT_LOCAL}/" "${OUTPUT_S3}/" 2>/dev/null || true
```

**诊断过程**：
- `rclone lsd` 确认 S3 上 `ericzyma/experiments/` 路径完全不存在（无任何文件）
- `rclone ls` 确认无残留临时文件
- `-m normal` 模式不支持 SSH，无法远程修复运行中的 sync 进程
- `koala get` 确认 pod 名称 `ericzyma-job-normal-20260512-132137-master-0`，TTL=72h

### 已解决

已删除旧 job 并用修复后的脚本重新提交。新 job `ericzyma-job-normal-20260512-153459` 已确认 S3 同步正常工作（`experiments/blendergym-9b-dp6/output/` 下有 configs、日志、wandb、eval 渲染图片等 60+ 个文件）。

### 附带修复

- **rclone dir-cache-time**：从 `24h` 改为 `5m`（修改了 `~/Library/LaunchAgents/com.rclone.threed-code.plist` 和 `kaola/.agent/s3-mount/templates/com.rclone.REMOTE.plist`），已重启挂载生效
- **S3 清理**：删除了 12 个旧的 `--sync-code` 时间戳目录 + 空的 `checkpoints/`、`logs/`、`outputs/`、`_scratch/` 目录

## 第四次训练（2026-05-12 15:34）— setup_kaola.sh 模块化重构

Job: `ericzyma-job-normal-20260512-153459`（8 GPU）

### 模块化重构

将 `setup_kaola.sh`（204 行，BlenderGym 专属逻辑混在 base 中）拆分为：
- `scripts/setup_kaola.sh`（150 行）— 通用编排器（参数解析、通用函数、env 调度）
- `scripts/envs/blendergym.sh`（113 行）— BlenderGym 环境插件（`env_setup()` 实现）

#### 核心改动

| 改动 | 说明 |
|------|------|
| `--env` 参数 | 支持 `--env blendergym`，加载 `scripts/envs/<name>.sh`（默认 blendergym） |
| `EXP_NAME` 必须显式设置 | 无默认值，未设置时报错退出（breaking change，防止误写旧目录） |
| `HF_MODEL` 环境变量 | 默认 `Qwen/Qwen3.5-9B`，自动派生短名用于 HF cache tar 文件名 |
| 步骤函数化 | `setup_hf_cache()`、`setup_python_deps()`、`setup_s3_sync()` 封装为通用函数 |
| env 专属逻辑移到插件 | libegl1、Blender、数据集、pip install、OPTIX 全部在 `blendergym.sh` |
| OPTIX cache 持久化 | 首次编译后自动 tar 上传 S3，后续 pod 秒级恢复（跳过 ~6 min 编译） |
| `--ckpt.output_dir` CLI 覆盖 | 训练命令追加 CLI 参数，checkpoint 路径自动跟随 `EXP_NAME` |

#### 步骤顺序优化

```
改动前：[1] libegl1 → [2] Blender → [3] Dataset → [4] HF cache → [5] Python deps → [6] OPTIX warmup（阻塞 ~6 min）→ [7] S3 sync
改动后：[1] Python deps → [2~5] env_setup（OPTIX 后台启动）→ [6] HF cache → [7] S3 sync → wait OPTIX
```

OPTIX warmup 从阻塞 ~6 min 改为后台执行，与 [6] HF cache + [7] S3 sync 并行。
但因 `prime-rl` 的 `check_gpus_available()` 在训练启动前检查 GPU 进程，必须在训练前 `wait` OPTIX 完成。
OPTIX cache 持久化后（后续 pod），warmup 从 ~6 min 降到 ~1s（tar 恢复），彻底消除等待。

#### 调试过程

1. **GPU 冲突**：首次提交时 OPTIX warmup 后台运行，训练启动立即报 `Existing processes found on GPUs: GPU 0`。原因：`check_gpus_available()` 在 model load 之前执行（~10s），非预期的 3-5 min。修复：setup 末尾 `wait ${OPTIX_PID}`。
2. **WANDB_API_KEY 未传入**：`\$WANDB_API_KEY` 在 koala submit 中被转义，pod 内 `$WANDB_API_KEY` 为空（KAOLA 未预注入）。修复：提交时用实际值替代 `\$` 引用。
3. **HF_TOKEN 同理**：与 WANDB_API_KEY 相同原因，改为直接传值。

### 当前 S3 目录结构

```
ericzyma/
├── .koala/                    # koala 系统
├── 20260512_153459/           # 当前 job 代码快照
├── data/                      # BlenderGym 数据集 (blendergym.tar 27GB)
├── experiments/
│   └── blendergym-9b-dp6/
│       ├── checkpoints/       # ← rsync 同步 DCP checkpoint（每 25 step）
│       └── output/            # ← rsync 同步训练输出（configs/logs/wandb/renders）
└── tools/                     # Blender 二进制 + optix_cache.tar（首次编译后自动上传）
```

## 第五次训练调优（2026-05-12 16:14 — 17:30）

### 问题诊断：rollout 失败率高（31/64）

分析 `step_1/train_rollouts.jsonl`（原始配置 `max_completion_tokens=1024`）：

- 64 条 rollout 中 **31 条失败**（reward=0, xml_parse_success=0, render_success=0）
- 失败全部发生在 XMLParser 阶段（还没进入 Blender 渲染）
- **根因：模型输出被 `max_completion_tokens=1024` 截断**
  - 20 条：最后输出有 `<code>` 但没有闭合 `</code>`，代码写到一半被截断
  - 11 条：整轮都在长篇分析，还没输出到 `<code>` 就用完 token
  - 14/31 条的 `output_tokens` 恰好是 3072（= 3 turns × 1024）
- 模型在 system prompt 允许 "Optionally write a few sentences of reasoning first" 下倾向大量 reasoning，浪费 token

### 配置迭代过程

| 轮次 | max_completion_tokens | gpu_memory_utilization | max_model_len | seq_len | 结果 |
|------|----------------------|----------------------|--------------|---------|------|
| 原始 | 1024 | 0.30 | 16384 | 8192 | 33/64 成功，31 失败（截断） |
| 第1次 | **4096** | **0.80** | **32768** | 8192 | **54-59/64 成功**，大幅改善 |
| 第2次 | **8192** | 0.80 | 32768 | 8192 | 崩溃：image token 截断 |
| 第3次 | 8192 | 0.80 | 32768 | **16384** | 未提交（分析发现仍不够） |

### 4096 运行的详细数据

Job: `ericzyma-job-normal-20260512-162731`（已删除）

```
step_0: success 54/64, fail 10, avg_output_tokens=5019, max=11440
step_1: success 59/64, fail 5,  avg_output_tokens=5360, max=12288
step_2: success 58/64, fail 6,  avg_output_tokens=4870, max=11183
```

Step 0 耗时 891s（vs 原始配置的 ~569s），因为平均输出 token 从 ~2600 增加到 ~5000。

### 8192 崩溃分析

```
ValueError: Image features and image tokens do not match, tokens: 512, features: 576
```

**根因**：`src/prime_rl/trainer/batch.py` 的 `prepare_sample()` 在序列超过 `seq_len` 时截断 `input_ids` 和 `mm_token_type_ids`，但 **不截断** `pixel_values` 和 `image_grid_thw`。当 prompt + completion 超过 `seq_len=8192`，图片占位 token 被部分截掉（576→512），但 vision encoder 仍输出完整 576 个 feature，导致不匹配。

**即使提高 `seq_len=16384` 也不够安全**：3 turn 多模态序列的最坏情况（每 turn 8192 completion + 图片 token）可达 ~28K token，仍会超过 16384 触发同样的截断错误。

### OPTIX cache 持久化验证

- ~~确认 OPTIX cache tar 自动上传~~ — **已验证**，首次训练后 `tools/optix_cache.tar` 已存在
- 后续 pod 启动日志显示 `Restored from S3 tar`，warmup 从 ~336s 降到秒级

### 配置文件当前状态（尚未回退）

`configs/multimodal/rl_blendergym_kaola.toml` 当前值：
```toml
seq_len = 16384              # ← 待回退或调整
max_completion_tokens = 8192  # ← 待回退或调整
gpu_memory_utilization = 0.80 # ← 保留（H200 充裕）
max_model_len = 32768         # ← 保留
```

**无运行中的 job**，S3 `experiments/blendergym-9b-dp6` 已清空。

## 下一步任务（待讨论）

### 核心问题：max_completion_tokens vs seq_len vs 多轮图片 token 截断

需要在下一个 session 详细讨论解决方案。可选方向：

1. **回退到 4096 + 8192**（已验证 59/64 成功率）+ **优化 prompt 减少 reasoning 浪费**
   - 修改 `prompts.py` 禁止长 reasoning，要求直接输出 `<code>...</code>`
   - 预期能进一步减少剩余 5-10 个失败
   - 无显存/速度风险

2. **提高 seq_len 到 32768**（覆盖 8192 × 3 turns 最坏情况）
   - H200 单卡 141GB，2 卡训练可能能扛住（有 activation checkpointing）
   - 但 trainer step 会更慢（activation memory 与 seq_len 成正比）
   - 且 rollout 生成也更慢（更多 decode token）

3. **修复 `batch.py` 的截断逻辑**（正确处理多模态截断）
   - 在截断 `input_ids` 时同步截断 `pixel_values` 和 `image_grid_thw`
   - 需要找到被截断的图片并移除对应的 pixel 数据
   - 改动较复杂，但是最正确的方案

4. **组合方案**：适中的 `max_completion_tokens`（如 6144）+ `seq_len=16384` + prompt 优化
   - 在速度和成功率间取折中

### 其他遗留

- ~~确认 OPTIX cache tar 自动上传~~ — 已完成
- **监控吞吐量退化** — 4096 配置下 MFU ~30%，暂无明显退化
- **KAOLA secret 注入** — 未调查
