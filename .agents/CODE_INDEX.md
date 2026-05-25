# CODE_INDEX.md — prime-rl 模块索引

| 模块 | 路径 | 说明 |
|------|------|------|
| **entrypoints** | `src/prime_rl/entrypoints/` | CLI 入口：`rl.py`（RL 训练）、`sft.py`（SFT）、`inference.py`（推理服务） |
| **configs** | `src/prime_rl/configs/` | Pydantic 配置类，对应 TOML 配置文件 |
| **trainer** | `src/prime_rl/trainer/` | 训练循环、梯度累积、checkpoint 管理 |
| **orchestrator** | `src/prime_rl/orchestrator/` | 多节点编排、rollout 调度 |
| **inference** | `src/prime_rl/inference/` | vLLM 推理后端封装 |
| **transport** | `src/prime_rl/transport/` | 节点间通信（权重同步、数据传输） |
| **templates** | `src/prime_rl/templates/` | Prompt 模板和 reward 函数 |
| **utils** | `src/prime_rl/utils/` | 通用工具函数 |

## 关键配置文件

- `configs/` — TOML 训练配置（按场景组织）
- `scripts/` — 集群脚本（setup、环境初始化）
- `skills/` — Claude Code 技能文件
