# Plan: Fix Thinking-Only Response Crash (Env-Level)

## 目标

在 env 代码中解决 thinking-only response 导致 vLLM 400 error 的问题，不修改 prime-rl 框架代码。

## 关键发现（Explore 阶段）

### verifiers MultiTurnEnv 循环流程

```
while not is_completed(state):
    messages = env.get_prompt_messages(state)    # ← 包含上一轮 completion
    response = await vllm.chat(messages)         # ← 400 error 在这里！
    await env.add_model_response(state, messages, response)  # ← 存入 trajectory
    env_msgs = await env.env_response(messages, state)       # ← env 回应
```

### 400 error 触发路径

1. Turn N: vLLM 返回 `{content: null, tool_calls: null, reasoning_content: "...截断..."}`
2. `add_model_response` → `super()` 存入 `state["trajectory"][-1]["completion"]`
3. Turn N+1: `get_prompt_messages` 从 trajectory 重建 history，包含这个无效 message
4. vLLM 校验 messages → 拒绝 → 400

### 终止机制

- `state["final_env_response"] = []` → 下一轮 `is_completed` 返回 True → 循环退出
- 设置后 `get_prompt_messages` 不再被调用（loop 直接退出）

### 约束

- **不改 prime-rl 代码**（`strip_message_content`, `trajectories.py`, `server.py` 等都不动）
- 只改 env 代码（`environments/blendergym/`, `environments/articraft/`）
- Articraft 的 `env_response` 已经处理 "no tool_calls" → 终止（line 264）
- BlenderGym 的 `env_response` 返回空（line 293），不做 tool_call 检测

## 相关代码

| 文件 | 函数 | 作用 |
|------|------|------|
| `environments/blendergym/blendergym/env.py:295` | `add_model_response()` | 存 completion + 解析 code + render |
| `environments/blendergym/blendergym/env.py:239` | `get_prompt_messages()` | 重建 history（含上轮 completion）|
| `environments/articraft/articraft_env/env.py:252` | `env_response()` | 已有 "no tool_calls" 终止逻辑 |
| `environments/articraft/articraft_env/env.py` | `add_model_response()` | **不存在** — 用 verifiers 默认 |

## 实现步骤

- [ ] Step 1: 提取公共辅助函数 `_sanitize_completion()` 处理 thinking-only messages
- [ ] Step 2: BlenderGym `add_model_response` 中加检测 + sanitize + 终止
- [ ] Step 3: Articraft 新增 `add_model_response`，同样逻辑

## 代码变更预览

### Step 1: 公共辅助函数

两个 env 都需要，各自内部定义（避免跨包依赖）：

```python
def _sanitize_completion(completion: list[dict]) -> bool:
    """Fix thinking-only assistant messages (content=null + tool_calls=null).

    When a thinking model exhausts max_completion_tokens during <think>,
    vLLM returns {content: null, tool_calls: null, reasoning_content: "..."}.
    This is invalid in OpenAI chat format and causes 400 on the next turn.

    Returns True if any message was sanitized (indicates thinking-only response).
    """
    sanitized = False
    for msg in completion:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        tool_calls = msg.get("tool_calls")
        has_no_content = content is None or (isinstance(content, str) and not content.strip())
        has_no_tool_calls = not tool_calls
        if has_no_content and has_no_tool_calls:
            msg["content"] = ""
            sanitized = True
    return sanitized
```

### Step 2: BlenderGym `add_model_response`

```diff
 # environments/blendergym/blendergym/env.py

     async def add_model_response(
         self,
         state: vf.State,
         prompt_messages: vf.Messages,
         response: Any,
     ) -> None:
         # 1. Let the parent push the step so trajectory[-1] is the just-finished step.
         await super().add_model_response(state, prompt_messages, response)

         rollout = require_rollout(state)
         if not state.get("trajectory"):
             return
         completion = state["trajectory"][-1]["completion"]

+        # Detect thinking-only response (model exhausted tokens during <think>).
+        # Sanitize the stored completion and terminate the rollout.
+        if _sanitize_completion(completion):
+            logger.warning(
+                "Thinking-only response detected (no content/tool_calls). "
+                "Terminating rollout %s early.",
+                state.get("trajectory_id", "?"),
+            )
+            state["final_env_response"] = []
+            return
+
         mgr = self.artifact_manager
         turn_idx = rollout.render_count
         ...
```

### Step 3: Articraft `add_model_response`（新增）

Articraft 没有自定义 `add_model_response`，需要新增：

```diff
 # environments/articraft/articraft_env/env.py

+    async def add_model_response(
+        self,
+        state: vf.State,
+        prompt_messages: vf.Messages,
+        response: Any,
+    ) -> None:
+        await super().add_model_response(state, prompt_messages, response)
+        if not state.get("trajectory"):
+            return
+        completion = state["trajectory"][-1]["completion"]
+        if _sanitize_completion(completion):
+            logger.warning(
+                "Thinking-only response detected. Terminating rollout %s.",
+                state.get("trajectory_id", "?"),
+            )
+            state["final_env_response"] = []
+
     async def env_response(
         self,
         messages: vf.Messages,
         state: vf.State,
         **kwargs: Any,
     ) -> vf.Messages:
         ...
```

## 为什么这样做

| 决策 | 原因 |
|------|------|
| 在 `add_model_response` 而非 `env_response` 中修复 | `add_model_response` 在 `env_response` 之前执行，是最早能拦截的点 |
| `msg["content"] = ""` 而非删除 message | 空字符串是合法的 assistant content，不破坏 OpenAI format |
| 设 `state["final_env_response"] = []` 终止 | thinking-only = 模型没有产生任何行动，继续 rollout 毫无意义 |
| `get_prompt_messages` 也加防御 | 双保险：万一 `add_model_response` 的终止信号被绕过（如 verifiers 版本变化）|
| 不改 reward 逻辑 | thinking-only rollout 自然得到低 reward（无 render/compile → 0 分），无需特殊处理 |

## Reward 影响分析

- **BlenderGym**: thinking-only → `add_model_response` 终止前 turns 为空 → rubric 的 `render_quality=0`, `not_truncated=1`（提前终止不算 truncate） → reward ≈ 0
- **Articraft**: thinking-only → no tool_calls → `check_fraction=0`, `build_success=0`, `compile_attempted=0` → reward = 0

模型通过 RL 自然学到 "别把 token 全花在 thinking 上"。

## 状态
**当前阶段**: Planning — 等待确认
