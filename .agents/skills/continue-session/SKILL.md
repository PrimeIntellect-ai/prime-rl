---
name: continue-session
description: Summarize the current conversation into a handoff prompt for the next session. Use when the user asks to continue in a new session, wrap up the current session, or create a session summary / handoff.
---

# Continue Session

Produce a single markdown file that gives the next session enough context to pick up immediately, with no re-explanation needed.

## Steps

1. **Read the current conversation** — scan messages for task purpose, actions taken, debugging insights, referenced files, and the final solution.
2. **Ask the user for the next task** (if not already stated): "下一步大概要做什么？"
3. **Draft a rough plan** for that next task based on what you now know of the codebase.
4. **Write the handoff file** using the template below.
5. Save to `.agents/sessions/<YYYY-MM-DD>-<slug>.md` where slug is 2–4 words describing the session.

## Template

```markdown
# Session Handoff: <title>

## 任务目的
<!-- 一两句话说清楚这次 session 要解决什么问题 -->

## 执行内容
<!-- 按时间顺序列出做了什么，每条一句话 -->
- 

## 调试经验
<!-- 遇到的坑、反直觉的行为、被证伪的假设。没有则省略此节 -->
- 

## 参考代码
<!-- 列出本次涉及的关键文件和函数/类，附一句说明 -->
| 文件 | 关键位置 | 说明 |
|------|---------|------|
|      |         |      |

## 最终方案
<!-- 最终采用的设计或修复，说清楚为什么选这个而不是别的 -->

## 下一步任务
<!-- 用户提供的下一步目标 -->

## 初步方案
<!-- 针对下一步任务，基于当前代码库给出大致思路：入口文件、关键改动点、潜在风险 -->
```

## Notes

- Keep each section short — the goal is orientation, not documentation.
- If a section has nothing meaningful, omit it entirely.
- "参考代码" should only list files that are actually needed to understand the next task.
- "初步方案" is a rough sketch, not a detailed spec; 3–5 bullet points is enough.
