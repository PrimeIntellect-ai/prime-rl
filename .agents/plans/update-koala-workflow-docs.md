# Plan: 更新 koala workflow 文档适配 v1.0.1/v1.1.0

## 目标

将 `prime-rl/.agents/kaola/workflow.md` 和 `.agents/koala/workflow.md` 更新到 koala v1.1.0 语法，消除废弃的 `--sync-code` 用法。

## 关键发现

1. **`prime-rl/.agents/kaola/workflow.md`**（最需要改）：
   - 4 处 `--sync-code .:/data/work/prime-rl` → 改为 `--code "$S3:/data/work/prime-rl"`
   - `ssh <pod名>` → 改为 `ssh koala`（v1.1.0 自动配置）
   - Checklist 表格内容过时

2. **`.agents/koala/workflow.md`**（小改）：
   - SSH 部分加一行提及 `ssh koala` 简化（v1.1.0）
   - BlenderGym 快速开始里 `bash scripts/setup_kaola.sh --fast` 改为 `. scripts/setup_kaola.sh --fast`（source 执行）

3. **不需要改的**：
   - `KOALA.md`（顶层）— 已经是新语法
   - `.agents/koala/workflow.md` 主体 — 已是 `--code` 语法

## 实现步骤

- [x] Step 1: 更新 `prime-rl/.agents/kaola/workflow.md` — 全面替换 `--sync-code` 为 `--code` + aws s3 sync 流程 + ssh koala
- [x] Step 2: 更新 `.agents/koala/workflow.md` — SSH 部分加 v1.1.0 + 修复 source 执行 + rsync 用 koala host

## 状态

**当前阶段**: Done
