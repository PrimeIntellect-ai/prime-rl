# BlenderGym 4 天训练 Crash 分析

**日期**: 2026-05-25
**任务**: `ericzyma-job-normal-20260521-141806` (8 GPU, 5/21 14:18 提交, 5/23 23:46 crash)
**状态**: 根因已定位，S3 sync 问题已修复，thinking-only response 问题待修复

---

## 训练概况

- 模型: Qwen3.5-9B, dp6 + train2
- 环境: BlenderGym (multimodal, tool-use)
- 跑到 trainer step 907 / orchestrator step ~89 后 crash
- 总运行时长: ~4 天

---

## 根因：Thinking-only response 导致 vLLM 400 error

### 现象

训练过程中 throughput 从 ~14k 逐步退化到 ~5.6k tokens/s，多个 step 耗时暴涨（1000-2250s vs 正常 100-200s）。最终 orchestrator exit code 1 终止。

### 原因

模型产生 **只有 reasoning_content 没有实际 content/tool_calls** 的 assistant response：

```json
{"role": "assistant", "content": null, "tool_calls": null, "reasoning_content": "The soccer ball is currently at (0,0,0) which is outside the basket..."}
```

当这个 message 被放回对话历史发给 vLLM 做下一轮推理时，vLLM 返回 400（OpenAI chat format 不允许 `content=null` 且 `tool_calls=null` 的 assistant message）。

Orchestrator 日志:
```
Rollout error in group 722 (blendergym), re-scheduling (0/8 complete):
ModelError() -> BadRequestError('Error code: 400 - 20 validation errors:
  messages[4] = {'role': 'assistant', 'content': None, 'tool_calls': None, 'reasoning_content': "..."}
```

### 退化模式

1. 模型偶尔产生 thinking-only response（只 think 不 act）
2. 整组 rollout re-schedule，造成该 step 耗时暴涨
3. 随着训练进行，这种情况越来越频繁（模型 entropy 下降 → 更容易 hedge）
4. 最终 orchestrator 累积失败崩溃

### 修复方向

在将 assistant response 放回对话历史前，需要处理 `content=None and tool_calls=None` 的情况：

**选项 A**：将 `reasoning_content` 设为 `content`（让 vLLM 认为这是普通文本回复）
**选项 B**：视为无效回复，终止该 rollout 并给低 reward
**选项 C**：在 vLLM 推理前 strip 掉 `reasoning_content` 字段，设 `content = ""`（空字符串合法）

需要确认：
- 这个处理逻辑在哪个模块？应该在 orchestrator 的 message formatting 层
- 是否已有类似的 message sanitization 逻辑？

### 相关代码路径（待确认）

- `src/prime_rl/orchestrator/` — rollout 消息组装
- `src/prime_rl/inference/` — vLLM 请求构建
- verifiers 框架的 env_response / multi-turn 消息拼接

---

## 附带发现：S3 sync 长期失效

详见 `.agents/kaola/troubleshooting.md` 2026-05-25 条目。

**已修复**：`setup_kaola.sh` 的 `sync_all` 改用 `aws s3 sync`（直接走 S3 API）替代 `rsync → S3 FUSE`。

---

## 下一步

1. [x] S3 sync 修复（已改 setup_kaola.sh）
2. [ ] 定位 thinking-only response 的处理代码，添加 sanitization
3. [ ] 删除旧任务，提交 articraft 训练
