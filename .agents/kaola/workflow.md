# 操作流程

## Debug 迭代（秒级循环）

```
Mac (代码修改) ──rsync──> Pod (运行/测试) ──查看输出──> 重复
```

### 1. 提交 debug pod（一次）

```bash
cd ~/Desktop/codes/prime-rl
koala submit --sync-code .:/data/work/prime-rl
ssh <pod名>
```

### 2. 容器内首次 setup（一次，~1 min）

```bash
cd /data/work/prime-rl
export EXP_NAME=blendergym-9b-dp6
. scripts/setup_kaola.sh --fast
```

`--fast` 模式：数据集用 S3 symlink、跳过 OPTIX warm-up。首次渲染会触发 ~6 min 编译，后续不再重复。

### 3. 代码迭代（每次修改后）

```bash
# Mac 上推送变更
rsync -avz --exclude '.git' --exclude '.venv' --exclude '__pycache__' \
    ./ <pod>:/data/work/prime-rl/

# Pod 上直接运行
cd /data/work/prime-rl
uv run rl @ configs/multimodal/rl_blendergym_kaola.toml
```

如果修改了 `pyproject.toml` 或 `uv.lock`，需要重跑：

```bash
uv sync --locked --extra flash-attn
```

---

## 正式训练（8 GPU）

一条命令提交，无需人工干预：

```bash
cd ~/Desktop/codes/prime-rl
koala submit -m normal -g 8 --sync-code .:/data/work/prime-rl \
    -c "export HF_TOKEN=$HF_TOKEN && export WANDB_API_KEY=$WANDB_API_KEY && export EXP_NAME=blendergym-9b-dp6 && cd /data/work/prime-rl && . scripts/setup_kaola.sh --env blendergym && uv run rl @ configs/multimodal/rl_blendergym_kaola.toml --ckpt.output_dir /local-ssd/checkpoints/\${EXP_NAME}"
```

Resume 训练（从已有 checkpoint 继续）：

```bash
cd ~/Desktop/codes/prime-rl
koala submit -m normal -g 8 --sync-code .:/data/work/prime-rl \
    -c "export HF_TOKEN=$HF_TOKEN && export WANDB_API_KEY=$WANDB_API_KEY && export EXP_NAME=blendergym-9b-dp6 && cd /data/work/prime-rl && . scripts/setup_kaola.sh --env blendergym --resume && uv run rl @ configs/multimodal/rl_blendergym_kaola.toml --ckpt.output_dir /local-ssd/checkpoints/\${EXP_NAME}"
```

### 提交命令 Checklist

| 必须 | 说明 | 漏了会怎样 |
|------|------|-----------|
| `--sync-code .:/data/work/prime-rl` | 同步本地代码到 pod | pod 里跑镜像内置旧代码，本地改动不生效 |
| **不要**指定 `--image` | 使用 KAOLA 默认 ECR 镜像 | 自定义 Docker Hub 镜像会导致 PodInitializing 卡住 |
| `$HF_TOKEN` 用**双引号** | 在本地 shell 展开为实际值 | 单引号不展开，pod 里 token 为空，报 `HF_TOKEN not set` |
| `. scripts/setup_kaola.sh` | source 执行 | `bash` 子 shell 中 export 不传递给后续命令 |
| `export EXP_NAME=...` | 显式设置实验名 | 无默认值，脚本会报错退出 |
| S3 上**无**同名实验数据 | 或加 `--resume` | guard check 报错退出，防止误覆盖 |

其他注意：
- `--ckpt.output_dir` 让 checkpoint 路径自动跟随 `EXP_NAME`，与 S3 sync 路径一致
- `HF_MODEL` 有默认值（Qwen/Qwen3.5-9B），换模型时追加 `export HF_MODEL=...`
- OPTIX warmup 在 setup 末尾后台启动，与训练初始化（model load ~3-5 min）并行

---

## setup_kaola.sh 模式对比

| 步骤 | `--fast`（debug） | 默认（训练） | 在哪 |
|------|-------------------|-------------|------|
| [1/7] Python deps | uv sync | 同左 | base |
| [2/7~5/7] 系统库/Blender/数据集/env 包 | 数据集 S3 symlink，跳过 OPTIX | 数据集 tar 恢复到 /local-ssd，后台启动 OPTIX | env |
| [6/7] HF 缓存 | tar 恢复 | 同左 | base |
| OPTIX warm-up | 跳过 | **后台启动**，与 HF 缓存和 S3 sync 并行 | env (BG) |
| [7/7] S3 sync | 跳过 | 后台 rsync | base |
| **setup 耗时** | **~1 min** | **~6 min（首次编译）/ ~1-2 min（有缓存）** |
| **到首次渲染** | N/A | **训练启动后无需等待 OPTIX 编译** |

OPTIX warmup（~6 min）在后台运行，并与 HF cache 恢复和 S3 sync 启动并行。`setup_kaola.sh` 末尾会等待 warmup 完成，避免 prime-rl 启动前的 GPU 空闲检查被 warmup 进程挡住。

所有步骤幂等：Blender/数据集已存在则跳过，重跑不会浪费时间。

---

## Pod 恢复（后续 pod）

S3 上的资产持久化，新 pod 只需重跑 setup：

```bash
export EXP_NAME=blendergym-9b-dp6
. scripts/setup_kaola.sh              # 训练模式
. scripts/setup_kaola.sh --fast       # debug 模式
```

OPTIX kernel cache (`/root/.nv/`) 在 pod 重启后丢失，首次渲染需重新编译。但 setup 的 warm-up 步骤会在后台处理。

---

## 查看结果（Mac 上）

```bash
# 刷新 S3 缓存
rclone rc --rc-addr=127.0.0.1:5572 vfs/refresh dir="ericzyma/checkpoints" recursive=true

# 查看 checkpoint
ls ~/Desktop/codes/s3/experiments/${EXP_NAME}/checkpoints/

# Tensorboard
tensorboard --logdir ~/Desktop/codes/s3/experiments/${EXP_NAME}/output/

# 实时日志（仅主进程 trainer 输出，不含 env worker 子进程日志）
koala logs <任务名> -f
```

> **注意**：env worker 日志（GPU 内存监控等）仅通过 S3 rsync 可见，不会出现在 `koala logs` 中。
> 若 S3 上的日志文件没有更新，参见 [troubleshooting.md](troubleshooting.md) 中 "rsync 无法覆盖 S3 上已有文件" 条目。

---

## 换模型 Checklist

换模型时需要同步修改以下 3 处（它们的值必须对应同一个模型）：

| 改什么 | 在哪 | 例子 |
|--------|------|------|
| `HF_MODEL` | 提交命令的 env var | `Qwen/Qwen3.5-9B` → `meta-llama/Llama-3-8B` |
| `[model] name` | TOML 配置文件 | `"Qwen/Qwen3.5-9B"` → `"meta-llama/Llama-3-8B"` |
| `EXP_NAME` | 提交命令的 env var | `blendergym-9b-dp6` → `blendergym-8b-dp6` |

另外需要在 S3 准备新模型的 HF cache tar（可选，不存在则回退到在线下载）。

快速测试时可用 CLI 覆盖代替改 TOML：
```bash
uv run rl @ configs/multimodal/rl_blendergym_kaola.toml --model.name "meta-llama/Llama-3-8B"
```
但正式训练建议保持 TOML 与 `HF_MODEL` 一致，确保配置可追溯。
