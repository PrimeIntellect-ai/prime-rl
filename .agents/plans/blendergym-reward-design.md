# Plan: BlenderGym 多维 Reward 设计

## 目标
为 BlenderGym placement 训练设计更多 reward signal，替代当前单一的 CLIP cosine similarity，提供更丰富的梯度信号加速收敛。

## 关键发现（Explore 阶段）

### 当前系统
- **唯一 reward**: `CLIP ViT-B/32 cosine similarity(render, goal)` — 权重 1.0
- **诊断 metrics (weight=0)**: `xml_parse_success`, `render_success`, `code_non_empty`
- **框架支持**: `verifiers.Rubric.add_reward_func(func, weight=W)` — 多 reward 加权求和，开箱即用
- **RL loss**: GRPO advantage = reward - per-problem baseline，然后 DPPO+KL loss

### 架构约束
1. **Score Service 是独立 FastAPI**：当前只暴露 `/score` endpoint（CLIP），新 reward 需扩展 API 或新建 service
2. **渲染输出只有 render1.png**：如需 depth/normal 等，需修改 `pipeline_render_script.py`
3. **rubric 是 async 的**：每个 reward func 可以 IO 密集（HTTP 调用）
4. **goal 数据**：每个 task 有 `goal_image`（target render）和 `start.py`（初始代码），可获取 goal 的 3D metadata（物体位置等）
5. **placement task 特点**：只需调整物体的 location（xyz 坐标），不改材质/光照

### 可用信号源
| 信号 | 获取方式 | 成本 |
|------|---------|------|
| CLIP similarity (已有) | Score Service HTTP | ~150ms |
| 像素级 MSE/SSIM/LPIPS | PIL + torchmetrics | ~50ms |
| 物体位置误差 (L2 distance) | Blender scene introspection（需改 render script） | ~0s |
| Depth map similarity | 需 Blender 输出 depth pass | ~3s (额外 render) |
| 语义分割 IoU | 需 object index pass | ~3s |
| 代码语法/结构 reward | 纯文本分析 | ~0ms |
| 多 CLIP 模型 ensemble | 加载多个 CLIP | ~300ms |
| DINOv2 similarity | 类似 CLIP 的 image encoder | ~150ms |

## 相关代码
| 文件 | 函数/类 | 作用 |
|------|---------|------|
| `blendergym/rubric.py` | `BlenderGymRubric` | 唯一改动点：添加新 reward_func |
| `blendergym/services/score/server.py` | `ScoreService` | 需扩展新 endpoint 或新建 service |
| `blendergym/services/score/clip_scorer.py` | `CLIPScorer` | 参考模式：如何加载模型+推理 |
| `blendergym/assets/pipeline_render_script.py` | `main()` | 若需额外 render pass 需修改 |
| `blendergym/render.py` | `RenderResult` | 若输出多文件需扩展 |
| `blendergym/schema.py` | `TurnRecord` | 可能需存额外 metadata |
| `src/prime_rl/orchestrator/advantage.py` | `default_advantage_fn` | reward → advantage，无需改（只看 scalar reward） |

## 方案对比

### 方案 A: 像素级辅助信号 (SSIM + MSE)
| 维度 | 评估 |
|------|------|
| 优点 | 零额外推理成本（PIL 操作）；对位移敏感；无需额外模型 |
| 缺点 | 对光照/噪声敏感（Cycles 随机采样）；低分辨率下可能无意义 |
| 实现难度 | ⭐ 极低 — 在 rubric.py 中读两张 PNG 算 SSIM |
| 与 CLIP 互补性 | 中等 — CLIP 是语义级，SSIM 是像素级，互补 |

### 方案 B: 物体位置误差 (Object Location L2)
| 维度 | 评估 |
|------|------|
| 优点 | 直接度量任务目标（placement = 物体位置）；梯度信号平滑 |
| 缺点 | 需从 Blender scene 提取物体坐标（改 render script 输出 JSON）；goal 需存目标坐标 |
| 实现难度 | ⭐⭐ 中 — 需改 pipeline_render_script 输出 metadata |
| 与 CLIP 互补性 | ⭐⭐⭐ 高 — CLIP 抓视觉整体感，L2 抓精确位置 |

### 方案 C: DINOv2 / 多 CLIP 模型 ensemble
| 维度 | 评估 |
|------|------|
| 优点 | DINOv2 比 CLIP 更关注空间结构；多模型降低 reward hacking 风险 |
| 缺点 | 额外 GPU 显存（DINOv2-L ~1.2GB）；多模型维护复杂度 |
| 实现难度 | ⭐⭐ 中 — 类似 CLIPScorer 模式新建 DINOScorer |
| 与 CLIP 互补性 | ⭐⭐⭐ 高 — DINOv2 空间表征 vs CLIP 语义表征 |

### 方案 D: 渲染进度奖励 (per-turn improvement bonus)
| 维度 | 评估 |
|------|------|
| 优点 | 鼓励多轮 iterative improvement；dense reward 而非 sparse terminal |
| 缺点 | 需跟踪前一轮 score 做 delta；可能导致 reward hacking（小步多次） |
| 实现难度 | ⭐ 低 — 在 rubric 中比较 turns[-1] vs turns[-2] |
| 与 CLIP 互补性 | 正交 — 这是 shaping 而非新信号 |

### 方案 E: 代码质量 / 格式 penalty
| 维度 | 评估 |
|------|------|
| 优点 | 防止 reward hacking（如输出垃圾代码碰巧高 CLIP score） |
| 缺点 | 对 placement 场景价值有限（代码很简单就几行 location） |
| 实现难度 | ⭐ 低 |
| 与 CLIP 互补性 | 低 — 更像 guard rail 而非信号 |

## 推荐组合（优先级排序）

1. **SSIM reward (weight=0.3)** — 立即可做，零成本，与 CLIP 互补
2. **Object Location L2 (weight=0.5)** — 最直接度量 placement 任务目标，但需改 render script
3. **DINOv2 similarity (weight=0.3)** — 空间感知更强的 visual reward

建议**分阶段**：先加 SSIM（验证框架 work），再加 Object L2（验证效果），最后考虑 DINOv2。

## 实现步骤（Phase 1: SSIM）

- [ ] Step 1: 在 `blendergym/services/score/` 新增 `ssim_scorer.py` — 用 `torchmetrics.image.StructuralSimilarityIndexMeasure` 计算 SSIM
- [ ] Step 2: 扩展 Score Service 添加 `/ssim` endpoint
- [ ] Step 3: 扩展 `ScoreClient` 添加 `ssim()` method
- [ ] Step 4: 在 `BlenderGymRubric.__init__` 添加 `self.add_reward_func(self.ssim_similarity, weight=0.3)`
- [ ] Step 5: 测试 — 确认 reward 正确计算、不影响训练性能

## 实现步骤（Phase 2: Object Location L2）

- [ ] Step 6: 修改 `pipeline_render_script.py` — exec 用户代码后，dump 所有物体 location 到 `<output_dir>/locations.json`
- [ ] Step 7: Dataset 扩展 — 为每个 task 预计算 goal locations（从 goal 的 start.py 推断，或渲染 goal 时记录）
- [ ] Step 8: 在 rubric 添加 `object_location_reward` — 读 `locations.json` + goal locations，计算归一化 L2 distance → reward
- [ ] Step 9: 集成测试

## 代码变更预览（Phase 1 - SSIM）

### `blendergym/services/score/server.py`
```diff
+ class SSIMRequest(BaseModel):
+     image_a: str
+     image_b: str
+
+ class SSIMResponse(BaseModel):
+     ssim: float
+     duration_s: float = 0

  class ScoreService(BaseService):
      def __init__(self, ...):
          ...
          self.app.add_api_route("/score", self.score, methods=["POST"])
+         self.app.add_api_route("/ssim", self.ssim_endpoint, methods=["POST"])

+     async def ssim_endpoint(self, req: SSIMRequest) -> SSIMResponse:
+         t0 = time.monotonic()
+         ssim_val = await asyncio.to_thread(compute_ssim, req.image_a, req.image_b)
+         return SSIMResponse(ssim=ssim_val, duration_s=time.monotonic() - t0)
```

### `blendergym/rubric.py`
```diff
  class BlenderGymRubric(vf.Rubric):
      def __init__(self, ...):
          ...
          self.add_reward_func(self.clip_similarity, weight=1.0)
+         self.add_reward_func(self.ssim_similarity, weight=0.3)
          self.add_metric(self.xml_parse_success)
          ...

+     async def ssim_similarity(self, state: vf.State, info: dict[str, Any]) -> float:
+         """SSIM between latest render and goal. Returns 0.0 on failure."""
+         try:
+             rollout = require_rollout(state)
+         except RuntimeError:
+             return 0.0
+         last_render = self.artifact_manager.last_render_path(rollout)
+         goal = rollout.task.goal_image
+         if last_render is None or not goal.is_file():
+             return 0.0
+         return await self.score_client.ssim(str(last_render), str(goal))
```

## 关于 "reward 还是 loss" 的澄清

在 prime-rl 中，这两者的关系是：
```
reward (env/rubric 输出) → advantage (GRPO 标准化) → RL loss (DPPO 梯度)
```

"设计更多 loss" 实际上有两个层面：
1. **Reward signal 层面**（本 plan 的重点）— 在 rubric 中添加多维 reward，框架自动加权求和
2. **Loss function 层面**（`trainer/rl/loss.py`）— 已有 DPPO+KL，如想加 SFT warm-up / entropy bonus 等需改 loss_fn

当前训练配置 GRPO advantage = (weighted_reward_sum) - baseline，已完美支持多 reward 加权。无需改 loss。

## 状态
**当前阶段**: Planning — 等待确认方向后实施
