# API Keys 与环境变量

训练/推理需要的 API key 和 token。**不要硬编码在代码或脚本中**。

## 获取方式

如果你不知道当前值，从用户的本地终端环境变量中查找：

```bash
# 在 Mac 终端中检查
echo $HF_TOKEN
echo $WANDB_API_KEY
```

或查看 `~/.zshrc` 中的 export 语句。

## 必需的环境变量

| 变量 | 用途 | 默认值 | 在哪设置 |
|------|------|--------|---------|
| `EXP_NAME` | 实验名称，决定 S3 目录和 checkpoint 路径 | **无**（必须显式设置） | 提交命令 `export EXP_NAME=...` |
| `HF_MODEL` | HuggingFace 模型全称，用于 HF cache tar 文件名 | `Qwen/Qwen3.5-9B` | 提交命令 `export HF_MODEL=...`（换模型时） |
| `HF_TOKEN` | HuggingFace 模型下载（避免限速、访问 gated 模型） | — | Mac `~/.zshrc` + pod 内 `export` |
| `WANDB_API_KEY` | Weights & Biases 实验追踪（缺失会导致训练启动失败） | — | Mac `~/.zshrc` + pod 内 `export` |

## Pod 内设置方法

`setup_kaola.sh` 不包含 token 默认值。在 pod 内需要手动设置或通过提交命令传入：

```bash
# 方式 1：SSH 进 pod 后手动 export
export HF_TOKEN="你的token"
export WANDB_API_KEY="你的key"

# 方式 2：在提交命令中传入
koala submit -m normal -g 8 --sync-code .:/data/work/prime-rl \
    -c "export HF_TOKEN=xxx && export WANDB_API_KEY=yyy && export EXP_NAME=blendergym-9b-dp6 && cd /data/work/prime-rl && . scripts/setup_kaola.sh --env blendergym && uv run rl @ configs/multimodal/rl_blendergym_kaola.toml --ckpt.output_dir /local-ssd/checkpoints/\${EXP_NAME}"
```

## 注意事项

- S3 FUSE (`/threed-code/`) 不支持文件权限操作，HF 的 xet 下载器在写入 S3 路径时可能 panic。设置 HF_TOKEN 后通常能避免此问题（authenticated 下载走不同路径）。
- 如果仍然遇到 xet panic，可设 `export HF_HUB_DISABLE_XET=1` 回退到标准 HTTP 下载。
