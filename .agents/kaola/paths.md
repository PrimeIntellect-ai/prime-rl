# 路径映射与存储决策

## 三端路径对照

| 内容 | Mac 本地 | S3 (持久) | 容器内 |
|------|---------|-----------|--------|
| 代码 | `~/Desktop/codes/prime-rl/` | 临时中转 | `/data/work/prime-rl/` |
| S3 用户空间 | `~/Desktop/codes/s3/` | `s3://arcwm-code-us-west-2/ericzyma/` | `/threed-code/ericzyma/` |
| Blender 4.2 | — | `tools/blender-4.2.0-linux-x64.tar` | `/local-ssd/blender-4.2.0-linux-x64/` |
| 数据集 | — | `data/blendergym/` | `/local-ssd/blendergym`(训练) / symlink(debug) |
| Checkpoint | `~/Desktop/codes/s3/experiments/${EXP_NAME}/checkpoints/` | `experiments/${EXP_NAME}/checkpoints/` | `/local-ssd/checkpoints/${EXP_NAME}` → `aws s3 sync` → S3 |
| 训练输出+渲染 | `~/Desktop/codes/s3/experiments/${EXP_NAME}/output/` | `experiments/${EXP_NAME}/output/` | `/local-ssd/prime-rl-output/` → `aws s3 sync` → S3（含 blendergym-work/） |

KOALA 用户名 = `ericzyma`，Mac 用户名 = `zhiyuanma`。

## 存储决策及原因

| 资产 | 大小 | 存放位置 | 原因 |
|------|------|---------|------|
| HF 模型缓存 | ~18GB tar | S3 `tools/hf_cache_${HF_MODEL_SHORT}.tar` → `/local-ssd/hf_cache/` | `HF_MODEL` 环境变量派生短名（如 `Qwen/Qwen3.5-9B` → `qwen3.5-9b`），避免每次 pod 在线下载 |
| Blender 二进制 | 1.3GB tar | S3 `tools/` → `/local-ssd/` | 启动时加载数百小文件，FUSE 直接运行极慢 |
| BlenderGym 数据集 | 27GB | S3 `data/blendergym.tar` → tar 管道 → `/local-ssd/`(训练) | 6 worker 并发读，tar 管道 39s vs cp -r 20min |
| Checkpoint (DCP) | ~18GB/step | `/local-ssd/checkpoints/` → `aws s3 sync` → S3 | S3 FUSE 不支持 rename，DCP 需要原子提交；本地写后 `aws s3 sync` 走 S3 API |
| 渲染产物 (work_root) | 高频小文件 | `/local-ssd/prime-rl-output/blendergym-work` | 合并到 output_dir 下，`aws s3 sync` 自动覆盖；6 worker 并发写 |
| 训练输出 (output_dir) | 数 GB | `/local-ssd/prime-rl-output/` → `aws s3 sync` → S3 | rollout JSONL、wandb、渲染产物；排除 broadcasts/ 和 *.bin |
| OPTIX kernel cache | 51MB | `/root/.nv/`（自动） | NVIDIA driver 管理，pod 重启后需重新编译 |

## TOML 配置差异

`rl_blendergym_kaola.toml` 相对于 `rl_blendergym.toml` 的变更：

| 字段 | 原值 | kaola 值 | 原因 |
|------|------|----------|------|
| `output_dir` | `outputs/blendergym_v3_9b_dp6_long` | `/local-ssd/prime-rl-output` | 快盘，高频写 |
| `ckpt.output_dir` | (无) | `/local-ssd/checkpoints/${EXP_NAME}` | 现通过 CLI 覆盖 `--ckpt.output_dir /local-ssd/checkpoints/${EXP_NAME}`，TOML 值作为 fallback。本地 NVMe 写入 + 后台 `aws s3 sync` 到 S3（S3 FUSE 不支持 rename） |
| `wandb.name` | `9b-dp6-bs64-cohost` | `9b-dp6-bs64-kaola` | 区分平台 |
| `*.data_root` | `data/blendergym` | `/local-ssd/blendergym` | 本地盘高速读 |
| `*.blender_bin` | `_reference_codes/.../blender` | `/local-ssd/blender-4.2.0-linux-x64/blender` | S3 恢复后路径 |
| `*.work_root` | `outputs/.../blendergym_work` | `/local-ssd/prime-rl-output/blendergym-work` | 合并到 output_dir 下，rsync 自动覆盖 |
| `train.env.args.keep_failed_only` | `false` | `true` | 节省本地盘空间 |
| `*.render_timeout_s` | (无，默认 120) | `600` | 兼容首次 OPTIX 编译 |
