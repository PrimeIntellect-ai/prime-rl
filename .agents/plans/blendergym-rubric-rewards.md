# Plan: 添加 Rule-Based Rubric Rewards

## 目标
在 `BlenderGymRubric` 中添加 2 个 rule-based reward（复用已有 metric 函数 + 新增 truncation 惩罚），
并支持从 TOML 配置文件中调整所有 reward 权重。

## 关键发现（Explore 阶段）

- 当前唯一 reward: CLIP cosine similarity (weight=1.0)
- 已有 3 个 weight=0 的 metric: `xml_parse_success`, `render_success`, `code_non_empty`
- 数据现状 (step 999): format 98% OK, render 97% OK, truncation 11% 且在增长
- TOML `args` dict 直接透传给 `BlenderGymEnv.__init__(**args)` → 再传给 `BlenderGymRubric`
- 因此只需在 `BlenderGymEnv.__init__` 加一个 `reward_weights` 参数，TOML 里就能配置

## 验证结论（Review 阶段）

- **verifiers 评分逻辑**：原始加权求和 `Σ(score × weight)`，**不**归一化
- **`state["is_truncated"]`**：存在，verifiers 在 rollout 结束时设置
- **`state["usage"]`**：存在，类型 `{"input_tokens": float, "output_tokens": float}`（累计所有 turn）
  - ⚠️ **不是** `state["token_usage"]`（该字段仅在 `state_to_output` 后存在于 RolloutOutput 上）
- **`state["sampling_args"]`**：存在，含 `max_completion_tokens`（单 turn 上限）
- **生态内无多目标 rubric 先例**：math-env/code-env 等均为单一 reward。BlenderGym 是第一个
- **已去掉 `syntax_valid`**：97% render 成功 → 语法错误极少，区分度不够
- **已去掉 `format_reward` / `render_success_reward` 冗余函数**：直接复用已有 metric，靠 weight 控制角色

## 设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 复用 vs 新建函数 | 复用已有 `xml_parse_success` / `render_success` | 避免语义重复 |
| brevity reward | 二值（truncated=0, else=1） | 目标是惩罚截断，不惩罚"长但完整" |
| 权重归一化 | 不做，保持 raw sum | 与 verifiers 框架行为一致，advantage 在 group 内已减均值 |
| 总权重 | 1.0 + 0.1 + 0.1 + 0.1 = 1.3 | rule-based 作为 bonus 叠加在主 reward 上 |

## 相关代码
| 文件 | 函数/类 | 作用 |
|------|---------|------|
| `environments/blendergym/blendergym/rubric.py` | `BlenderGymRubric` | 调整 weight + 新增 `not_truncated` |
| `environments/blendergym/blendergym/env.py` | `BlenderGymEnv.__init__` | 透传 reward_weights 给 rubric |
| `configs/multimodal/rl_blendergym_kaola.toml` | `[orchestrator.train.env.args]` | 配置 weights |

## 实现步骤
- [ ] Step 1: `rubric.py` — 添加 `not_truncated` reward func，将已有 metric 提升为 reward，`__init__` 接收 `reward_weights` dict
- [ ] Step 2: `env.py` — 透传 `reward_weights` 参数给 `BlenderGymRubric`
- [ ] Step 3: `rl_blendergym_kaola.toml` — 添加默认 reward_weights 配置
- [ ] Step 4: 验证 — 确认不 break 已有逻辑

## 代码变更预览

### 修改后：`blendergym/rubric.py`
```python
# Default reward weights — overridable from TOML via reward_weights arg
DEFAULT_REWARD_WEIGHTS: dict[str, float] = {
    "clip_similarity": 1.0,
    "xml_parse_success": 0.1,
    "render_success": 0.1,
    "not_truncated": 0.1,
}


class BlenderGymRubric(vf.Rubric):
    """CLIP-similarity reward + rule-based bonus rewards."""

    def __init__(
        self,
        score_service_url: str = "http://localhost:8421",
        parser: vf.Parser | None = None,
        artifact_manager: ArtifactManager | None = None,
        reward_weights: dict[str, float] | None = None,
    ) -> None:
        if artifact_manager is None:
            raise TypeError("artifact_manager is required")
        super().__init__(parser=parser)
        self.score_client = ScoreClient(score_service_url)
        ensure_service_ready(score_service_url, "score")
        self.artifact_manager = artifact_manager

        w = {**DEFAULT_REWARD_WEIGHTS, **(reward_weights or {})}
        self.add_reward_func(self.clip_similarity, weight=w["clip_similarity"])
        self.add_reward_func(self.xml_parse_success, weight=w["xml_parse_success"])
        self.add_reward_func(self.render_success, weight=w["render_success"])
        self.add_reward_func(self.not_truncated, weight=w["not_truncated"])
        # code_non_empty 保留为纯诊断 metric
        self.add_metric(self.code_non_empty)

    # --- existing reward funcs (unchanged) ---

    async def clip_similarity(self, state: vf.State, info: dict[str, Any]) -> float:
        ...  # 不变

    async def xml_parse_success(self, state: vf.State) -> float:
        ...  # 不变，但从 add_metric 提升为 add_reward_func

    async def render_success(self, state: vf.State) -> float:
        ...  # 不变，但从 add_metric 提升为 add_reward_func

    # --- new reward func ---

    async def not_truncated(self, state: vf.State) -> float:
        """0.0 if any turn was truncated, 1.0 otherwise."""
        return 0.0 if state.get("is_truncated") else 1.0

    async def code_non_empty(self, completion, parser) -> float:
        ...  # 不变，保持为纯 metric
```

### 修改后：`blendergym/env.py`
```python
class BlenderGymEnv(vf.MultiTurnEnv):
    def __init__(
        self,
        data_root: str | Path = "data/blendergym",
        ...
        score_service_url: str = "http://localhost:8421",
        render_timeout_s: int = 600,
        # -- reward weights (overridable from TOML) --
        reward_weights: dict[str, float] | None = None,
        # -- artifact policy --
        ...
        **kwargs: Any,
    ) -> None:
        ...
        rubric = BlenderGymRubric(
            score_service_url=score_service_url,
            parser=self.parser,
            artifact_manager=self.artifact_manager,
            reward_weights=reward_weights,
        )
```

### 修改后：`configs/multimodal/rl_blendergym_kaola.toml`
```toml
[orchestrator.train.env.args]
data_root = "/local-ssd/blendergym"
task_types = ["placement"]
max_turns = 3
...
render_timeout_s = 600

[orchestrator.train.env.args.reward_weights]
clip_similarity = 1.0
xml_parse_success = 0.1
render_success = 0.1
not_truncated = 0.1
```

## TOML 配置使用方式

```bash
# 使用默认 weights（不配置 reward_weights 时）
uv run rl @ configs/multimodal/rl_blendergym_kaola.toml

# CLI 覆盖单个 weight
uv run rl @ configs/multimodal/rl_blendergym_kaola.toml \
  --orchestrator.train.env.args.reward_weights.not_truncated=0.2

# 关闭某个 reward
uv run rl @ configs/multimodal/rl_blendergym_kaola.toml \
  --orchestrator.train.env.args.reward_weights.xml_parse_success=0.0
```

## Reward 阶梯效果

```
完全无输出 + 截断    → clip=0  xml=0  render=0  not_trunc=0  total=0.0
格式错（没<code>）   → clip=0  xml=0  render=0  not_trunc=1  total=0.1
格式对但 render 崩   → clip=0  xml=1  render=0  not_trunc=1  total=0.2
全部成功 + 低相似    → clip=X  xml=1  render=1  not_trunc=1  total=X+0.3
全部成功 + 高相似    → clip=1  xml=1  render=1  not_trunc=1  total=1.3
全部成功但被截断     → clip=X  xml=1  render=1  not_trunc=0  total=X+0.2
```

## 风险与缓解

| 风险 | 缓解 |
|------|------|
| xml_parse/render 已 >95%，区分度有限 | weight 仅 0.1，主要防止退化（policy collapse 早期保护） |
| 原始 wandb 中 metric key 对比断裂 | 函数名不变，wandb metrics dict 的 key 保持一致 |
| eval env 未配置 reward_weights | DEFAULT_REWARD_WEIGHTS 兜底，train/eval 行为一致 |

## 状态
**当前阶段**: Planning — 已确认，可执行
