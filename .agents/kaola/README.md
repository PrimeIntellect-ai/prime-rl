# KAOLA 部署指南 — prime-rl BlenderGym

> 平台通用知识（提交命令格式、S3 目录规范、容器存储速查、通用 Gotchas）见 `/Users/zhiyuanma/Desktop/codes/.agents/KOALA.md`。
> 本目录只记录 prime-rl 项目在 KAOLA 上的特有配置和经验。

## 核心文件

| 文件 | 说明 |
|------|------|
| `scripts/setup_kaola.sh` | 通用编排器（`--fast` debug / `--env` 选择环境） |
| `scripts/envs/blendergym.sh` | BlenderGym 环境插件（实现 `env_setup()`） |
| `configs/multimodal/rl_blendergym_kaola.toml` | KAOLA 平台训练配置 |
| `.agents/kaola/paths.md` | 路径映射和存储决策 |
| `.agents/kaola/workflow.md` | 操作流程 |
| `.agents/kaola/api.md` | API keys 和环境变量（HF_TOKEN、EXP_NAME 等） |
| `.agents/kaola/troubleshooting.md` | 踩坑记录（具体的坑和修复） |
| `.agents/kaola/debugging.md` | Debug 方法论（系统化排查方法） |

## 快速命令

```bash
# Debug pod（1 GPU，SSH 交互）
cd ~/Desktop/codes/prime-rl
koala submit --sync-code .:/data/work/prime-rl
ssh <pod名>
cd /data/work/prime-rl
export EXP_NAME=blendergym-9b-dp6
. scripts/setup_kaola.sh --fast    # ~1 min

# 代码迭代（Mac 上，秒级推送）
rsync -avz --exclude '.git' --exclude '.venv' --exclude '__pycache__' \
    ./ <pod>:/data/work/prime-rl/

# 正式训练（8 GPU）
koala submit -m normal -g 8 --sync-code .:/data/work/prime-rl \
    -c "export HF_TOKEN=$HF_TOKEN && export WANDB_API_KEY=$WANDB_API_KEY && export EXP_NAME=blendergym-9b-dp6 && cd /data/work/prime-rl && . scripts/setup_kaola.sh --env blendergym && uv run rl @ configs/multimodal/rl_blendergym_kaola.toml --ckpt.output_dir /local-ssd/checkpoints/\${EXP_NAME}"
```

**关键注意事项**：
- **不要指定 `--image`**：使用 KAOLA 默认 ECR 镜像（集群节点有缓存，秒级拉取）。自定义 Docker Hub 镜像会导致 `PodInitializing` 卡住数分钟甚至失败。
- **必须加 `--sync-code`**：本地代码改动只有通过 `--sync-code` 才能同步到 pod。不加的话 pod 里跑的是镜像内置的旧代码。
- **`$HF_TOKEN` 和 `$WANDB_API_KEY` 用双引号**：确保在本地 shell 展开为实际值。单引号会导致 pod 里变量为空。
- **用 `.`（source）执行 setup 脚本**：`bash` 会在子 shell 运行，export 的变量不传递给后续命令。

## 环境要求

- GPU: NVIDIA H200（已验证）/ H100 / A100（预期兼容）
- 镜像: KAOLA 默认 ECR 镜像（`cuda12.8-efa1.44-ubuntu24.04-zsh-uvcache`），**不要用自定义镜像**
- Python: 3.12（镜像内置）
- 外部依赖: Blender 4.2.0（S3 `tools/` 持久化）、BlenderGym 数据集（S3 `data/blendergym/`）
