# Plan: 更新 troubleshooting

## 目标

将本次 KAOLA 验证 session 中发现的调试经验追加到 `.agents/kaola/troubleshooting.md`。

## 关键发现（Explore 阶段）

对比现有 troubleshooting 内容和本次 session 遇到的问题：

| 本次发现 | 现有文档是否覆盖 | 是否值得加入 |
|---------|----------------|-------------|
| multiprocessing spawn 报错（测试脚本无 `__main__` guard） | ❌ 未覆盖 | ✅ 容易踩到 |
| cadquery 隐式依赖（SDK 内部用了 cadquery 但 filter 没捕获） | ❌ 未覆盖 | ✅ 影响 reward 分布理解 |
| bm25s 未安装导致 import 链失败 | ❌ 未覆盖 | ✅ 安装顺序敏感 |
| zsh heredoc 转义问题 | ❌ 未覆盖 | ⚠️ 边缘 — 只影响调试脚本 |
| aws s3 sync 在 pod 内可用 | ✅ 已在 KOALA.md | ❌ 不需要重复 |
| S3 articraft/ 不完整 | ✅ 已被新 articraft.sh 解决 | ❌ 不需要 |

最终选择加入前 3 条。

## 实现步骤

- [ ] Step 1: 在 `.agents/kaola/troubleshooting.md` 顶部追加 3 条新记录（2026-05-25）

## 状态

**当前阶段**: Done
