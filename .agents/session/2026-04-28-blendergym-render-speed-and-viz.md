# Session Handoff: BlenderGym 渲染加速与可视化优化规划

## 前置 session
- [BlenderGym Phase 0–6 端到端集成](2026-04-27-blendergym-phase0-6-integration.md) — Phase 0–6 全部完成，bench 4 步 SUCCESS
- [BlenderGym 环境架构决策](2026-04-25-blendergym-architecture-decisions.md)
- [Verifiers 环境全面精读与知识更新](2026-04-23-verifiers-env-full-survey.md)

## 任务目的
基于 Phase 6 bench 跑出的 164 个 trajectory 数据复盘，规划两个具体优化方向：
1. **加速 Blender 渲染**（目前 ~5s/turn, 占 rollout wall-time 约 17%）
2. **重构落盘文件路径/格式**（让 researcher 能更快从 trajectory 产物可视化定位问题）

## 执行内容（这次 session）
- 复盘 Phase 6 bench 4 步全部数据：`outputs/blendergym_v1/` 下 49 MB env work + 1.3 GB train_rollouts + 120 张 VLM image cache + Rich 表格在 trainer/orchestrator log 里。
- 全局渲染成功率统计：**164 trajectory 中只有 23 个三轮全成功（14%）、110 个三轮全失败（67%）；总 turn 渲染成功率 23.2%**（vs Phase 5 (b) 报告的 50% — 因为 5b 只测了 placement1-5 简单任务）。
- 典型失败模式：`'Object' has no attribute 'move_to'`（模型把 helper function 当 method 调）、`KeyError: bpy_prop_collection key "plant" not found`（hallucinate 物体名）、`module 'bpy' has no attribute 'location'`。
- 深入 1 条高 reward trajectory（`6b715e4a... = placement12`，reward 0.9988）：完整 system prompt + user turn 0（5 块内容：text/image/text/image/text）+ 三轮 assistant 输出全部抽出，确认模型的"copy INITIAL helpers + 改 location"模式。
- 核对输入图片落盘位置：placement12 的 GOAL/INITIAL 图既在 `data/blendergym/placement12/renders/{goal,start}/render1.png`（数据集源头），也在 `outputs/blendergym_v1/run_default/assets/images/<sha256[:16]>.png`（prime-rl 的 VLMImageCache 去重池）。
- 发现 prompt 里 system prompt 硬编码的坐标范围 `x∈(-1,1)/y∈(-0.75,0.75)/z∈(0,1.55)` 与具体 placement 任务的 INITIAL program 注释（如 placement12 是 `x∈(-1.5,1.5)/y∈(-1.8,1.8)/z∈(0,2)`）**互相矛盾** — 是抄 placement1 时引入的 bug。

## 调试经验
- **`turn_N/` 下没有 `render1.png` ≠ env 坏了**：意味着模型 turn N 输出的代码 Blender 执行出错，`run_blender` 会写完整 stderr 到 `blender.log` 但不输出 PNG。要看失败原因永远是 `tail blender.log` 的 STDERR 段。
- **train_rollouts.jsonl 里的 image_url 仍是 base64 data URL**：trajectories.py 里 base64→file:// 替换只在 `vf-eval --save-dataset` 模式触发，RL 训练流程不走那条路；想从 jsonl 里看图必须自己 base64 decode 或者去 `run_default/assets/images/<hash>.png`（命名是 `sha256(b64_string)[:16]`）反查。
- **Phase 6 bench reward 震荡（0.14→0.25→0.18→0.24）不是 RL 在学**：bench 模式 trainer 用 fake data，模型权重不变，reward 波动完全来自 sampling temp 0.0 + example sampling noise。真训练（去掉 `--bench`）才会看 reward 收敛。
- **5b 与 6 的 render_success 差异（0.5 vs 0.23）来自数据集采样**：5b 集中在 placement1-5（简单任务，object 名直观如 "basketball"），6 是 batch_size=32 横扫 placement1-45，包含 hallucination 容易触发的对象（"plant" / "soccer_ball" 等）。

## 参考代码

### Phase 6 产物盘点
| 路径 | 内容 | 大小 |
|------|------|------|
| `outputs/blendergym_v1/blendergym_work/<traj_uuid>/turn_{0,1,2}/` | 每个 rollout 的 `code.py` / `render1.png`（成功才有）/ `blender.log` / `blender_user/` | 49 MB / 164 traj |
| `outputs/blendergym_v1/run_default/rollouts/step_{0,1,2,3}/train_rollouts.jsonl` | 每行一个 rollout：prompt/completion/reward/metrics/advantage/is_filtered，**image 仍是 base64**（未被替换为 file://） | 4 × 35 MB |
| `outputs/blendergym_v1/run_default/assets/images/<sha256[:16]>.png` | VLMImageCache 去重池：goal/init/render 的所有 PNG | 120 张 |
| `outputs/blendergym_v1/logs/{trainer,orchestrator,inference,envs/train/blendergym/env_worker_*}.log` | 4 个 Rich 表格 + 详细日志 | 130 KB |

### 渲染管线相关
| 文件 | 关键位置 | 说明 |
|------|---------|------|
| `environments/blendergym/blendergym/render.py` | `run_blender()` + `RenderResult` + CLI | subprocess 调 Blender 入口；`DEFAULT_TIMEOUT_S=120`，blender_user 隔离 prefs lock |
| `environments/blendergym/blendergym/assets/pipeline_render_script.py` | `_enable_gpu_cycles(samples=512, resolution=512)` + `_render_camera1()` | 在 Blender 进程内执行的脚本；当前固定 Cycles GPU + 512×512 / 512 spp |
| `environments/blendergym/blendergym/env.py` | `add_model_response()` 末尾 | 调 `asyncio.to_thread(run_blender, ...)`；`turn_dir = work_dir / f"turn_{turn_idx}"` |

### 文件落盘相关
| 文件 | 关键位置 | 说明 |
|------|---------|------|
| `environments/blendergym/blendergym/env.py:setup_state` | `state["work_dir"] = self.work_root / state["trajectory_id"]` | trajectory_id 是 verifiers 自动生成的 32-hex uuid，不可读 |
| `environments/blendergym/blendergym/env.py:cleanup_work_dir` | `@vf.cleanup` 钩子 | 当前只在 `keep_failed_only=True` + 全成功时 rmtree；可以在这里写 meta.json + contact_sheet |
| `src/prime_rl/orchestrator/trajectories.py:84-118` + `:411-450` | base64→file:// 替换路径 | 只在 `--save-dataset` 模式触发，RL 训练 jsonl 不走 |

### 现有渲染基线数据
- 单次 Blender 冷启动 + Cycles 512spp / 512×512 / Camera1 only：**~4.5–5.0 s**（Phase 3 smoke）
- 含 GPU 首次初始化：~70 s（已被预热缓存覆盖）
- Phase 6 实测每 rollout 3 turns × ~5 s = 15 s render；同一 rollout LLM gen ~30 s × 3 = 90 s — **render 占 wall-time ≈ 17%，不是瓶颈但有 4–10x 提速空间**

## 现状摘要

Phase 0–6 主线已交付（plan 全 status: completed）。本次 session 是产物复盘 + 下一步技术规划，**没动代码**。代码现状：

```
environments/blendergym/blendergym/
├── env.py       (BlenderGymEnv + setup_state/get_prompt_messages/add_model_response/cleanup)
├── prompts.py   (SYSTEM_PROMPT + TASK_INSTRUCTION + REFINE_INSTRUCTION)
├── dataset.py   (build_dataset)
├── render.py    (run_blender + RenderResult + python -m blendergym.render CLI)
├── rubric.py    (BlenderGymRubric + compute_clip_cosine_similarity)
├── assets/pipeline_render_script.py  (Blender 内部 background 脚本)
└── tests/{test_dataset.py,test_rubric.py}  (9/9 pass)
```

bench 已通：`uv run rl @ configs/multimodal/rl_blendergym.toml --bench` 4 步全 SUCCESS。

## 下一步任务

并行推进两个独立优化（互不依赖，可分头实施）：

### A. 提高 Blender 渲染速度
目标：把单 turn render 从 ~5 s 降到 ~1 s（5x），或更激进降到 ~0.3 s（15x）。

### B. 重构落盘文件路径/方式以方便 researcher 可视化
目标：researcher 拿到一个 `<traj_dir>` 名字就能猜到是哪个 task；打开一张 `contact_sheet.png` 就能看完整个 trajectory；不用查 jsonl 也能拿到 task_id / reward / step。

## 初步方案

### A. 渲染加速

> 28 号实测复盘后，原 A1（降 samples）与 A2（EEVEE 切换）方案被淘汰：
> - **A1 作废**：profile 显示 5s/turn 中只有 1s 是 sampling，其余 4s 是 startup + load + kernel JIT + scene upload；samples 512→64 最多省 0.6s。**真正的瓶颈是 cold start，不是 sampling**。
> - **A2 当前不可行**：实测 Blender background + EEVEE 在本机软件 fallback 到 LLVMpipe，单 turn **27s（比 Cycles GPU 慢 5.9×）**。GPU EEVEE 被两件事锁死：(1) `/dev/dri/card*` 权限是 `root:video`，当前用户不在 video 组；(2) 系统装了 NVIDIA driver 560.35.03 但**没装 NVIDIA EGL ICD**（找不到 `libEGL_nvidia.so.0` / `10_nvidia.json`）。两者都需要 root。
> - ✅ **CLIP cross-engine bias 实测安全**（这条结论会保留到将来）：EEVEE-vs-Cycles 同 state 0.984、跨 state 0.942，差距足够 RL 用；将来如果跑通 GPU EEVEE 不用担心 reward 失真。

**A1' Persistent Blender Cycles Worker**（不依赖管理员权限，4–6 小时实施，**5× 提速**）

每个 env worker 启动一个长驻 Blender 进程，加载完 `.blend` + 编译 kernel 后保持，后续每 turn 通过 IPC 接收 `code.py` 路径 → 在已加载 scene 上 exec → render → 返回 PNG。

```
            cold start  load.blend  kernel JIT  BVH  sampling  total
当前        ~2s         ~2s         ~1s         ~1s  ~1s       ~5s/turn
Persistent  0           0           0           ~0.5 ~1s       ~1.5s/turn  (3.3×)
```

实施要点：
- Blender 进程模板：`blender --background <blend> --python worker_loop.py` ；worker_loop 进 `while True` 等 stdin，每收一行 JSON `{"code_path": ..., "output_dir": ...}` → 调 `bpy.ops.wm.revert_mainfile()` 回初始 scene → exec code → render → write PNG → flush `OK\n` 到 stdout
- **关键 trap**：必须每 turn 调 `bpy.ops.wm.revert_mainfile(use_scripts=False)` 重置 scene，否则上一轮 model 改的 `obj.location` 会污染下一轮基线 reward
- 健康检查：连续 3 次 timeout 或 stdout 无 `OK` → kill + 重启 worker（`run_blender` 当前 fault-tolerant 路径已经有这个语义，迁移到 worker 模式只需再包一层 supervisor）
- 改 `render.py`：`run_blender(blend, code, output_dir, ..., worker=None)` ——传 worker 句柄就走 IPC，不传就 fallback 到当前 subprocess.run 模式（向后兼容 + smoke test 友好）
- 改 `env.py`：`__init__` 创建 `BlenderWorkerPool(gpu_id_pool)`；`add_model_response` 从 pool 借 worker；`@vf.cleanup` 也归还 worker（但 worker 不死，跨 rollout 复用）

复杂度风险：
- **scene 状态泄露**：revert_mainfile 不一定彻底（材质 datablock 可能残留），需 smoke test 验证连续 100 个 rollout 后 reward 分布与冷启动一致
- **Blender 偶发 segfault**：长进程更易触发，需要 watchdog 自动重启
- **placement25 类巨型场景**：21s/turn 中只有 ~6s 是 startup，persistent 也只能省到 15s（1.4×）。placement1 类小场景才能拿到 5×。所以**实际平均加速 ~3×**

**A2' EEVEE 路径（暂时挂起，等管理员加权）**

需要其中之一：
- (a) `sudo usermod -a -G video zhiyuan_ma` + relogin（5 秒命令，需要找管理员）
- (b) `sudo apt install libnvidia-gl-560`（装 NVIDIA EGL ICD，提供 headless GPU OpenGL）

任意一条到位后，把 `pipeline_render_script.py` 的 `scene.render.engine` 从 `"CYCLES"` 改成 `"BLENDER_EEVEE_NEXT"`、`scene.eevee.taa_render_samples = 64`，预期 placement1 → ~0.5s / placement25 → ~1.5s（**总加速 9–14×**）。配合 A1' persistent 后再下一档。Cross-engine reward bias 已实测安全，不需要重渲数据集 goal/init。

**A4 单 worker asyncio 并发**：等 A1' 落地后再看；如果 GPU 6/7 利用率 < 50% 说明可以并发多 Blender 进程在同一 GPU 上压榨吞吐。

**渲染加速验证手段**：
- micro-benchmark：写 `scripts/bench_render.py` 跑 placement1 + placement25 各 20 次取 p50/p99
- 端到端：跑 4-step bench 比对总时长（baseline 13min → A1' 预期 ~5min → A1'+A2' 预期 ~2.5min）

**等管理员的沟通模板**（异步推进）：
> 麻烦在 CUDA 节点上跑 `sudo usermod -a -G video zhiyuan_ma` 然后让我 relogin，或者 `sudo apt install libnvidia-gl-560` 装一下 NVIDIA OpenGL/EGL ICD —— 我需要在 background 模式跑 Blender 4.2 EEVEE 引擎做强化学习数据生成，目前 `/dev/dri/card*` 是 `root:video` 锁死，回退到 LLVMpipe 软件渲染慢 6 倍。

### B. 落盘可视化重构（最终 MVP）

目标：一个 trajectory 目录既适合 researcher 在 Cursor/VSCode 里直接阅读，也适合未来 multi-agent / ATIF-style 数据导出。最终 MVP **不要 HTML、不要 contact_sheet.png**：`trajectory.md` 里直接用 Markdown 表格引用图片即可。

#### 文件结构

```text
outputs/<run>/blendergym_work/
└── placement12__6b715e4a/
    ├── meta.json              # 扁平摘要，方便 jq / 排序 / 批量统计
    ├── trajectory.json        # ATIF-shaped 详细轨迹，预留 multi-agent executor/evaluator
    ├── trajectory.md          # researcher 主入口，含图片 + response + code + error
    ├── inputs/
    │   ├── goal.png  -> data/.../renders/goal/render1.png
    │   ├── init.png  -> data/.../renders/start/render1.png
    │   └── start.py  -> data/.../start.py
    ├── turn_0/
    │   ├── response.txt       # 模型 raw output，parse 成功/失败都保留
    │   ├── code.py            # XMLParser 抽出的 <code> 内容；parse 失败可不存在
    │   ├── render1.png        # render 成功才有
    │   └── blender.log        # subprocess stdout/stderr 全量日志
    ├── turn_1/
    └── turn_2/
```

#### `meta.json`（机器筛选 / jq / index 用）

扁平摘要，避免 researcher 扫 1000+ 个 trajectory 时解析完整嵌套 JSON：

```json
{
  "task_id": "placement12",
  "task_type": "placement",
  "trajectory_id": "6b715e4a7d4049dd8c26093e1b90aa5d",
  "work_dir": "outputs/blendergym_v1/blendergym_work/placement12__6b715e4a",
  "gpu_id": 6,
  "max_turns": 3,
  "num_turns": 3,
  "final_reward": 0.9988,
  "metrics": {
    "clip_similarity": 0.9988,
    "xml_parse_success": 1.0,
    "render_success": 1.0,
    "code_non_empty": 1.0,
    "num_turns": 3.0
  },
  "render_success_per_turn": [true, true, true],
  "xml_parse_success_per_turn": [true, true, true],
  "first_error_hint": null,
  "paths": {
    "goal": "inputs/goal.png",
    "init": "inputs/init.png",
    "start_code": "inputs/start.py",
    "trajectory_md": "trajectory.md",
    "trajectory_json": "trajectory.json"
  }
}
```

> `final_reward` 在 `@vf.cleanup` 时如果还没写进 state，就先写 `null`；更稳的做法是在 `BlenderGymRubric.clip_similarity()` 里顺手写 `state["final_reward"] = reward`。

#### `trajectory.json`（ATIF-shaped，预留 multi-agent）

不引 Harbor 依赖、不做 validator，但字段设计对齐 ATIF 的核心思想：agent step + tool call + observation + metrics。

```json
{
  "schema_version": "blendergym-trajectory-v1",
  "session_id": "6b715e4a7d4049dd8c26093e1b90aa5d",
  "task": {
    "task_id": "placement12",
    "task_type": "placement",
    "blend_file": "data/blendergym/placement12/blender_file.blend",
    "goal_image": "inputs/goal.png",
    "init_image": "inputs/init.png",
    "start_code": "inputs/start.py"
  },
  "agents": [
    {
      "name": "executor",
      "role": "generator",
      "model_name": "Qwen/Qwen3.5-0.8B"
    }
  ],
  "steps": [
    {
      "step_id": 1,
      "turn": 0,
      "source": "agent",
      "agent_name": "executor",
      "message_path": "turn_0/response.txt",
      "tool_calls": [
        {
          "tool_call_id": "render_t0",
          "function_name": "execute_blender_code",
          "arguments": {"code_path": "turn_0/code.py"}
        }
      ],
      "observation": {
        "results": [
          {
            "source_call_id": "render_t0",
            "render_path": "turn_0/render1.png",
            "render_success": true,
            "duration_s": 4.5,
            "error_hint": null,
            "log_path": "turn_0/blender.log"
          }
        ]
      },
      "metrics": {
        "xml_parsed": true,
        "code_non_empty": true,
        "render_success": true
      }
    }
  ],
  "final_metrics": {
    "clip_similarity": 0.9988,
    "xml_parse_success": 1.0,
    "render_success": 1.0,
    "code_non_empty": 1.0,
    "num_turns": 3,
    "final_reward": 0.9988
  }
}
```

未来引入 VLM evaluator / verifier 时无需 schema migration：只要在 `agents[]` 增加 `{name: "evaluator", role: "verifier", model_name: ...}`，并在 `steps[]` 中穿插 `agent_name="evaluator"` 的 step。

#### `trajectory.md`（researcher 主入口；尽量贴近 SWE-agent inspector）

Markdown 是第一版默认可视化格式：Cursor/VSCode 原生 preview、代码块高亮、图片相对路径可用、`<details>` 可折叠长代码/日志。

结构尽量贴近 SWE-agent trajectory inspector：每个 turn 都按 **Query / Response / Thought / Action / Observation / State** 分块。差异是 BlenderGym 的 query 里有 base64 图片，不能原样复制到 md（会把单文件膨胀到几十 MB），所以 **Query 存摘要 + 相对路径引用**，完整 raw prompt 仍然留在 prime-rl 的 `run_default/rollouts/step_N/train_rollouts.jsonl`。

字段映射：

| SWE-agent 字段 | BlenderGym 对应 |
|----------------|-----------------|
| `query` | 本轮 user prompt 摘要 + GOAL/CURRENT 图片路径 + 起始代码路径 |
| `response` | `turn_N/response.txt` 模型原始输出 |
| `thought` | response 中 `<code>` 前的自然语言 reasoning |
| `action` | `execute_blender_code(code_path=turn_N/code.py)` |
| `observation` | Blender render result：`render1.png` / `blender.log` / `error_hint` |
| `state` | turn 后状态：xml_parsed / render_success / last_render_path / gpu_id 等 |

示例结构：

```md
# placement12__6b715e4a

- reward: 0.9988
- render_success: [true, true, true]
- first_error_hint: null

## Images

| GOAL | INIT | TURN 0 | TURN 1 | TURN 2 |
|------|------|--------|--------|--------|
| ![](inputs/goal.png) | ![](inputs/init.png) | ![](turn_0/render1.png) | ![](turn_1/render1.png) | ![](turn_2/render1.png) |

## Timeline

| turn | thought | action | observation | reward-ish |
|---:|---|---|---|---:|
| 0 | Move object locations to match goal | execute code | ✅ render | - |
| 1 | Move soccer ball towards basket | execute code | ✅ render | - |
| 2 | Already matches goal | execute code | ✅ render | 0.9988 |

## Turn 0

### Query

- Goal image: `inputs/goal.png`
- Current image: `inputs/init.png`
- Instruction: rewrite the program so the rendered scene matches GOAL.
- Required output format: `<code>...</code>`
- Initial code: `inputs/start.py`

> Full raw prompt (including base64 image URLs) is stored in `run_default/rollouts/step_N/train_rollouts.jsonl`, not duplicated here.

### Response

<details><summary>response.txt</summary>

```text
I will move the soccer ball ...
```

</details>

### Thought

```text
I will move the soccer ball towards the basket to match the GOAL scene.
```

### Action

```json
{
  "function_name": "execute_blender_code",
  "arguments": {"code_path": "turn_0/code.py"}
}
```

<details><summary>code.py</summary>

```python
...
```

</details>

### Observation

| field | value |
|---|---|
| render_success | true |
| render_path | `turn_0/render1.png` |
| duration_s | 4.52 |
| error_hint | null |

![](turn_0/render1.png)

<details><summary>blender.log tail</summary>

```text
Saved: ...
Blender quit
```

</details>

### State After Turn

```json
{
  "xml_parsed": true,
  "render_success": true,
  "last_render_path": "turn_0/render1.png"
}
```
```

#### 明确不做的项

- **不做 `trajectory.html`**：`trajectory.md` 已覆盖主要阅读需求；HTML 等未来需要浏览器分享/部署时用离线脚本生成。
- **不做 `contact_sheet.png`**：`trajectory.md` 已直接引用 GOAL/INIT/TURN 图，重复生成拼图反而多一份磁盘和维护成本。
- **不写 `turn_N/status.json`**：turn 状态合并进 `meta.json["turns"]` / `trajectory.json["steps"]`，减少小文件数量。
- **不写 `turn_N/prompt.json`**：prompt（含 base64 image）已在 `run_default/rollouts/step_N/train_rollouts.jsonl` 中完整保存，复制会造成磁盘膨胀。
- **不改 `turn_0/turn_1/turn_2` 命名**：保持和当前产物、脚本兼容，不切到 `turns/000`。
- **不在训练循环里生成 index**：index 是整个 run 的离线总览工具，等 100+ 步后再做。
- **不改 cleanup-policy**：第一版继续 `keep_failed_only=false` 全保留，避免误删失败现场；长训练后再做分层清理。

#### index / cleanup-policy 的含义

- **index**：整个 run 的总览页（如 `blendergym_work/index.md` 或 `index.html`），扫所有 `meta.json` 后按 reward / error / task_id 列表排序，链接到每个 `trajectory.md`。它是离线分析工具，不是单 trajectory 产物。
- **cleanup-policy**：长训练时保留/删除哪些文件的策略。例如 `keep_all`（全保留）、`keep_failed_only`（失败全保留，成功只留 md/json/最后一张图）、`minimal`（只留 meta/trajectory）。第一版不动，等实际 50–100 步训练后看磁盘增长再定。

#### 落盘验证手段

- 跑 1 次 vf-eval（5 examples × 1 rollout），检查：
  - 目录名形如 `placement1__abc12345/`
  - `inputs/{goal,init,start.py}` 是有效 symlink
  - `turn_N/response.txt` 保留模型原文
  - `meta.json` 可被 `jq` 读取 reward / error_hint
  - `trajectory.json` 有 `agents[]` + `steps[]`
  - `trajectory.md` 在 Cursor/VSCode preview 里能直接看图
- 再跑 4-step bench，确认新增 md/json 不导致 `blendergym_work` 超过 1GB。

#### 失败场景的落盘语义（必须实现）

当前代码的问题：如果 XML parse 失败，`run_blender()` 不会被调用，旧版几乎不会为该 turn 留下任何现场；如果 Blender 执行失败，则只有 `code.py + blender.log`，没有结构化错误提示。新 MVP 必须把失败也作为一等数据保留下来。

三种失败场景：

1. **XML parse 失败**（模型没输出 `<code>...</code>`）
   ```text
   turn_N/
   └── response.txt        # 模型原始输出，必须保存
   ```
   `meta.json / trajectory.json / trajectory.md` 里记录：
   ```json
   {
     "idx": 0,
     "xml_parsed": false,
     "code_non_empty": false,
     "render_success": false,
     "render_path": null,
     "code_path": null,
     "log_path": null,
     "error_hint": "XMLParser could not find <code>...</code>"
   }
   ```

2. **XML 成功但 Blender 执行失败**
   ```text
   turn_N/
   ├── response.txt
   ├── code.py
   └── blender.log
   ```
   `error_hint` 从 `RenderResult.stderr` 或 `blender.log` 的 stderr 反向扫描提取，例如：
   ```text
   AttributeError: 'Object' object has no attribute 'move_to'
   KeyError: 'bpy_prop_collection[key]: key "plant" not found'
   ```

3. **Blender timeout**
   ```text
   turn_N/
   ├── response.txt
   ├── code.py
   └── blender.log
   ```
   记录：
   ```json
   {
     "render_success": false,
     "timed_out": true,
     "error_hint": "TIMEOUT after 120s"
   }
   ```

成功场景：
```text
turn_N/
├── response.txt
├── code.py
├── render1.png
└── blender.log
```

`trajectory.md` 的 Observation section 对缺图情况要显示：
```md
_No render image produced._

error_hint: AttributeError: ...
```

`extract_error_hint(stderr: str) -> str | None` 建议逻辑：从 stderr 末尾反向扫描，优先返回包含 `Error` / `Exception` / `Traceback` / `TIMEOUT` 的最后一行；不要做复杂 parser，提示性即可。

## 28 号实测复盘（profile + EEVEE 实验）

### Cycles 单 turn 5s 时间分布（placement12 实测）
```
0:00.0 → 0:04.0  Blender 进程冷启动 + 加载 .blend + 解析 scene    4.0s  (80%)
0:04.0 → 0:05.1  同步对象 + 加载 9 张贴图                          1.1s
0:05.1 → 0:06.3  Cycles render kernel JIT compile                 1.2s
0:06.3 → 0:07.5  BVH build + scene/mesh/textures upload to GPU    1.2s
0:07.5 → 0:08.5  Denoise kernel + 512 sample sampling             1.0s
                                                                  ───────
                                                                  ~5s 总
```
**含义**：cold start 是真正瓶颈，sampling 只占 20%。改 cycles samples 治标不治本。

### 数据集复杂度分布（5 任务采样）
| task | 三角形 | 灯光 | 高级材质 | .blend | **实测** |
|------|--------|------|-----------|--------|----------|
| placement1 | 40K | 1 | none | 5.6 MB | **4.6s** |
| placement12 | 305K | 1 | none | 26 MB | **5.0s** |
| placement40 | 122K | 1 | none | 4.1 MB | ~6s 估 |
| placement50 | 557K | 1 | Sheen+Glass | 49 MB | ~10s 估 |
| placement25 | **11.3M** | **12** | SSS+Glass+Coat+Transmission | **640 MB** | **21s** |

placement25 是 infinigen 自动生成的巨型场景（BottleFactory / 12 灯）；复杂度差 250×，渲染时间差 4.6×。退化曲线对 Cycles 很友好。

### EEVEE bias 实验（在 LLVMpipe 软件回退上跑出来）
- EEVEE software 渲 placement1 = **27s**（比 Cycles GPU 慢 5.9×）—— **GPU EEVEE 被锁死**
- EEVEE 输出 vs Cycles 数据集 start: CLIP cos = **0.984**（视觉等价）
- EEVEE 输出 vs Cycles 数据集 goal:  CLIP cos = **0.942**（跨 state 显著低）
- ✅ **CLIP cross-engine reward signal 实测保留**——以后能跑 GPU EEVEE 时直接切，不用先重渲数据集

### Docker / NVIDIA EGL PoC（用户加 docker 组 + docker data-root 后补做）
- 用户已执行 `sudo usermod -aG docker zhiyuan_ma`；当前 shell 未 relogin，但可用 `sg docker -c '...'` 临时进 docker group。
- 用户已把 Docker data-root 挪到 `/data/docker`，避免 10GB Blender image 打满根盘：`Docker Root Dir: /data/docker`。
- `sg docker -c 'docker run --rm --gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi'` 跑通，说明 NVIDIA container runtime 可用。
- `meihaiyi/blender:blender-4.2-cuda12.4.1-ubuntu22.04` 可用，Blender 版本为 **4.2.4 LTS**。
- **直接跑 Docker EEVEE 仍不快**：placement1 render = **50.45s**，日志 `EGL_BAD_MATCH`，本质仍未拿到 NVIDIA EGL 快路径。
- 容器里有 `/usr/share/glvnd/egl_vendor.d/10_nvidia.json`，但没有 `libEGL_nvidia.so.0` / `libnvidia-eglcore` 等库；nvidia-container-toolkit 只能挂 host 已存在的 driver libs，而 host 没装 NVIDIA GL/EGL。
- 无 root fallback 成功一半：下载 NVIDIA 560.35.03 `.run` 到 `/data/zhiyuan_ma/nvidia-egl-560.35.03/`，`--extract-only` 后把 `extracted/` 挂进容器；`eglinfo` 变成 `EGL vendor string: NVIDIA`，NVIDIA EGL 确实可用。
- 但 EEVEE 实测仍有坑（第一轮）：
  - 只挂 vendor json 不设 `LD_LIBRARY_PATH`：Blender 报 `No OpenGL vendor detected` / `epoxy_get_proc_address` assert。
  - 把 extracted 全目录放进 `LD_LIBRARY_PATH` 前，需要补 `libGL.so.1 -> libGL.so.1.7.0`、`libEGL.so.1 -> libEGL.so.1.1.0` 等 symlink，否则 `libGL.so.1: undefined symbol _glapi_tls_Current`。
  - 补 symlink 后 placement1 首次 GPU EEVEE 渲染仍 **47s**，但这是 shader cache 冷编译；挂载 `/data/zhiyuan_ma/blender_docker_cache:/root/.cache` 后第二次降到 **1.89s Blender 内部 / 4.45s 含 Docker cold start**，说明 GPU EEVEE 路径对简单场景已打通。
  - placement25（11.3M tris）在同一 cache 下跑了 >10min 仍未完成（GPU 6 100%），日志：4min 到 `Rendering 1 / 64 samples`、6min 到 `25 / 64`。这说明 **EEVEE 对巨型 Infinigen 场景可能 shader/geometry path 极慢**，不适合直接作为全量 placement 后端。
- 复跑定位实验（第二轮）明确了原因：
  - placement25 **第二次**（同 cache，同 64 samples）到 `Rendering 1 / 64 samples` 只需 **12s**（第一次是 4min），说明首次慢主要是 shader cache 冷编译。
  - 但完整 64 samples 仍需 **345s**（5m45s），说明 shader cache 后 per-sample / EEVEE 高级 pass 仍严重慢于 Cycles。
  - placement25 `taa_render_samples=1` + 尝试关闭 raytracing/shadows/volumetrics 后：**7.93s Blender 内部 / 12.4s 含 Docker cold start**，CLIP vs Cycles start = **0.9397**、vs goal = **0.8609**，signal 保留。
  - placement25 Workbench：**4.50s Blender 内部 / 9.5s 含 Docker cold start**，CLIP vs Cycles start = **0.7590**、vs goal = **0.7058**，signal 较弱但仍有方向。

当前结论：**Docker + extracted NVIDIA EGL 可以让 GPU EEVEE 工作；placement1 缓存后 1.9s，placement25 必须把 `taa_render_samples` 降到 1 才可用（7.9s）。** 直接 64-sample EEVEE 不可用；低样本 EEVEE 可作为研究分支，主线仍应做 Persistent Cycles Worker。

## 风险与决策清单（更新后）

| 项 | 决策 | 理由 |
|---|---|---|
| ~~改 Cycles samples (旧 A1)~~ | **作废** | sampling 只占 20%，最多省 0.6s |
| ~~EEVEE 切换 (旧 A2)~~ | **挂起/研究分支** | Docker + extracted EGL 能让简单场景 1.9s，但 placement25 >10min，复杂场景严重退化 |
| **Persistent Cycles Worker (A1')** | **立即做** | 不依赖管理员，3× 平均提速（小场景 ~5×、大场景 ~1.4×） |
| 目录命名 task__uuid8 (B1) | 立即做 | 一行 setup_state 改动 |
| meta.json (B2) | 立即做 | cleanup 钩子里写 |
| 软链输入图 (B3) | 立即做 | setup_state 末尾 os.symlink |
| contact_sheet.png (B4) | 立即做 | cleanup 钩子里 PIL 拼图 |
| index.html (B5) | 推迟 | 跑 100+ 步后再做离线工具 |

## 推荐执行顺序

1. **B1 + B3**（30 分钟）：目录命名 + 软链输入图。零风险，立刻可看 5 张并排。
2. **B2 + B4**（2 小时）：meta.json + contact_sheet.png。一张图看完一个 rollout。
3. **A1' Persistent Cycles Worker**（4–6 小时）：micro-benchmark 验证 reward 分布与冷启动一致（无 scene 状态泄露）→ 迁移到 env worker → 跑新 bench。
4. **Docker EEVEE 研究分支（可选）**：围绕 placement25 做专项 profiling（`taa_render_samples=1/16/64`、禁 shadows、Workbench engine、scene simplify），只有当 placement25 <3s 且 CLIP signal 保留时才接入。
5. **沟通管理员**（异步）：如果还想走系统级 EEVEE，发模板请求加 `video` 组或装 `libnvidia-gl-560`；但即使系统 EGL OK，也仍需解决 placement25 的复杂场景退化。

## 总加速预期路径

```
现状                                       ~5s/turn   (1×)
+ A1' persistent Cycles                   ~1.5s/turn (3×)   ← 当前能做的上限
+ A2' GPU EEVEE (简单场景已验证)           ~1.9s/turn (placement1)，但 placement25 >10min（暂不可用）
```

## 28 号补充实验：Cycles + low samples + OptiX denoiser（强烈推荐）

在 `outputs/cycles_optix_poc/` 跑了临时矩阵（未改主代码）：`512 no denoise` / `64 + OPTIX` / `32 + OPTIX` / `16 + OPTIX`，placement1 + placement25 各一次。

### 渲染耗时
| task | 512 no denoise | 64 + OptiX | 32 + OptiX | 16 + OptiX |
|------|----------------|------------|------------|------------|
| placement1 | 3.00s | 2.76s | 2.73s | **2.50s** |
| placement25 | **67.72s** | 13.31s | 10.35s | **8.45s** |

> 注意：这个 512 baseline 与最初的 21s placement25 不完全一致，因为临时脚本优先尝试 `OPTIX` compute device，再 fallback CUDA；但同一脚本矩阵内部可比。关键结论是：**16 spp + OptiX denoiser 对 placement25 有 8× 加速，且保留 Cycles 分布**。

### CLIP / PSNR 对比（相对数据集 Cycles start/goal）
placement1:
```
dataset start-vs-goal CLIP = 0.9711
512_none   CLIP(start)=0.9994 CLIP(goal)=0.9697 delta=+0.0297 PSNR=44.66
64_OPTIX   CLIP(start)=0.9996 CLIP(goal)=0.9689 delta=+0.0307 PSNR=44.40
32_OPTIX   CLIP(start)=0.9996 CLIP(goal)=0.9688 delta=+0.0308 PSNR=43.92
16_OPTIX   CLIP(start)=0.9997 CLIP(goal)=0.9705 delta=+0.0292 PSNR=43.28
```

placement25:
```
dataset start-vs-goal CLIP = 0.8792
512_none   CLIP(start)=0.9929 CLIP(goal)=0.8764 delta=+0.1165 PSNR=27.45
64_OPTIX   CLIP(start)=0.9481 CLIP(goal)=0.9166 delta=+0.0315 PSNR=29.18
32_OPTIX   CLIP(start)=0.9452 CLIP(goal)=0.9173 delta=+0.0279 PSNR=29.09
16_OPTIX   CLIP(start)=0.9472 CLIP(goal)=0.9197 delta=+0.0274 PSNR=28.91
```

结论：
- **placement1 近乎无损**：16 spp + OptiX 与 512 spp 的 CLIP/PSNR 几乎一致。
- **placement25 速度从 67.7s → 8.45s（8×）**：比 EEVEE-fast 7.93s 只慢一点，但图像仍是 Cycles/OptiX 分布，训练 reward 风险更低。
- placement25 的 absolute CLIP(start) 从 0.993 降到 ~0.947（denoise/smoothing 引入 domain shift），但 **delta(start-goal) 仍正且稳定 ~0.03**，保留训练信号。

更新后的渲染加速优先级：
1. **Cycles 16/32 spp + OptiX denoiser**（先做，最小代码改动，8× in hard scenes，reward 风险低）
2. **Persistent Cycles Worker**（再做，摊薄 cold start；与 low-sample OptiX 叠加）
3. **EEVEE-fast / Workbench**（降级为研究分支；只有当 OptiX 仍不够快再考虑）

## 28 号结论更新：主线不再需要 Docker

Docker/EEVEE PoC 已完成其探索价值，主线回到**本地 Infinigen Blender + Cycles + OptiX denoiser**：

- 本地 Blender 路径：`_reference_codes/VIGA/utils/third_party/infinigen/blender/blender`
- 不再依赖 Docker / NVIDIA EGL / `/dev/dri` / `meihaiyi/blender` 镜像
- 推荐在主代码里加可配置参数：
  ```toml
  render_backend = "cycles"
  cycles_samples = 16        # 或 32
  cycles_denoiser = "OPTIX"
  cycles_compute_device = "OPTIX"  # fallback CUDA
  ```
- 先实现：`pipeline_render_script.py` 支持 env/argv 传 samples + denoiser；`render.py` 增加参数；`env.py` 透传 TOML args。
- 验证顺序：
  1. placement1 / placement25 smoke
  2. 5×4 vf-eval variance
  3. 4-step `uv run rl @ configs/multimodal/rl_blendergym.toml --bench`

已清理磁盘：
- Docker 镜像 `meihaiyi/blender:blender-4.2-cuda12.4.1-ubuntu22.04`（10GB）已删
- `/data/zhiyuan_ma/nvidia-egl-560.35.03`（1.4GB）已删
- `outputs/eevee_docker_poc` 已删
- 保留 `outputs/cycles_optix_poc`（2.6MB，小且直接记录主线 OptiX 实验图/日志）
- `/data/zhiyuan_ma/blender_docker_cache` 还剩 24KB，由容器 root 创建，当前用户无权限删除；可忽略，或用户 root 手动 `rm -rf /data/zhiyuan_ma/blender_docker_cache`
