# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Cross-Project Context

For an overview of sibling projects and shared conventions, see `/Users/zhiyuanma/Desktop/codes/CLAUDE.md`.

@.agents/CODE_INDEX.md

@AGENTS.md

## Key References

> `.agents/notes/` 存放调试经验和架构分析，遇到相关问题时应先查阅已有笔记。

| 文档 | 路径 | 说明 |
|------|------|------|
| Config 系统 | `docs/configs.md` | TOML 配置组合、CLI 覆盖 |
| Entrypoints | `docs/entrypoints.md` | 所有启动入口 |
| Environments | `docs/environments.md` | 自定义 env/reward |
| Troubleshooting | `docs/troubleshooting.md` | 常见问题（OOM、API 超时、TOML 解析） |
| Async 架构 | `docs/async.md` | 异步 rollout 设计 |
| Multimodal | `docs/multimodal.md` | 多模态 RL 配置 |
| Changelog | `CHANGELOG.md` | 破坏性配置变更 |

## Gotchas

- **TOML `@` 后必须有空格**：`uv run rl @ config.toml`，不是 `@config.toml`
- **vLLM 推理后端**：模型通过 vLLM 加载，不是直接 HuggingFace — 修改模型加载逻辑要看 `inference/` 不是 `trainer/`
- **Blender 内嵌 Python 忽略 PYTHONPATH**：worker 脚本中必须手动 `sys.path.insert`，且被导入的包不能有 Blender Python 没有的依赖
- **OPTIX shader 首次编译 ~6min**：pod 重启后缓存丢失，setup 脚本会后台 warmup
- **Koala 踩坑记录**：`.agents/kaola/troubleshooting.md`（具体的坑）、`.agents/kaola/debugging.md`（排查方法论）
