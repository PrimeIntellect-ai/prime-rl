# Session Handoff: BlenderGym OptiX + Entity Refactor

## 前置 session

- [BlenderGym 渲染加速与可视化优化规划](2026-04-28-blendergym-render-speed-and-viz.md) — 复盘 Phase 6 bench 产物，提出 A: render 加速、B: trajectory 落盘可视化 MVP 两条路线。
- [BlenderGym Phase 0–6 端到端集成](2026-04-27-blendergym-phase0-6-integration.md) — 初版 BlenderGym env / render / rubric / RL config 跑通。
- [BlenderGym 环境架构决策](2026-04-25-blendergym-architecture-decisions.md)
- [Verifiers 环境全面精读与知识更新](2026-04-23-verifiers-env-full-survey.md)

## 任务目的

本次 session 从前置规划里的两个方向落地实现：

1. **Render 加速 MVP**：把 BlenderGym rollout render 从 512 spp 改成 16 spp + OIDN denoiser + OPTIX compute device，以降低每 turn render wall-time；不做 Persistent Worker（推迟）。
2. **Trajectory 可视化 / schema MVP**：先实现 task-readable work_dir、inputs symlink、`meta.json / trajectory.json / trajectory.md`，随后进一步把 BlenderGym 自己散落在 verifiers `state` dict 里的 runtime 字段收束成 `Task / Rollout / TurnRecord` entity model，并同步把 artifact schema bump 到 v2。

## 执行内容

### A. Render 加速

- 修改 `pipeline_render_script.py::_enable_gpu_cycles`：从 `BLENDERGYM_CYCLES_SAMPLES / BLENDERGYM_CYCLES_DENOISER / BLENDERGYM_CYCLES_COMPUTE_DEVICE` 读取 render 配置，默认 `16 / OPENIMAGEDENOISE / OPTIX`。
- compute device 设置走 `try: prefs.compute_device_type = "OPTIX"`，失败 fallback `"CUDA"`；denoiser 设置走三层 fallback：requested → `OPENIMAGEDENOISE` → `use_denoising=False`。
- `pipeline_render_script.py` 现在会在 Blender stderr 写配置行：`[blendergym] cycles samples=16 denoiser=OPENIMAGEDENOISE compute=OPTIX`，后续排查用 `grep "[blendergym] cycles" blender.log`。
- 修改 `render.py` CLI：新增 `--samples / --denoiser / --compute-device`，main 只把 CLI 参数转成 env var；`run_blender(...)` 签名保持不变。
- 修改 `BlenderGymEnv.__init__`：新增 `cycles_samples / cycles_denoiser / cycles_compute_device` kwargs，并写入 `os.environ`；Blender 子进程通过 `dict(os.environ)` 自动继承。
- 修改 `configs/multimodal/rl_blendergym.toml`：`output_dir = "outputs/blendergym_v2"`；train/eval env args 都加 `cycles_samples=16 / cycles_denoiser="OPENIMAGEDENOISE" / cycles_compute_device="OPTIX"`，`work_root` 同步到 `outputs/blendergym_v2/blendergym_work`。
- 新增 `environments/blendergym/scripts/bench_render.py`，可对单个 task 重复 render N 次，输出 mean/stdev/p50/p99/success/timeouts；`--keep-output` 使用 `shutil.move` 处理跨文件系统移动。

### B. Trajectory 落盘 MVP

- 新增 `trajectory_writer.py`（初版）：`TurnRecord` dataclass、`completion_to_text`、`write_trajectory_artifacts`、Markdown image/timeline/turn section rendering helpers。
- `setup_state` 先改为创建 `<work_root>/<task_id>__<traj_id[:8]>/inputs/`，并把 dataset 的 `goal.png / init.png / start.py` symlink 到 `inputs/`。
- `add_model_response` 改成三步流：写 `turn_N/response.txt`（parse 失败也写）→ `TurnRecord.fill_xml_parse_failure()` 或 `run_blender + fill_from_render()` → append record。
- `BlenderGymRubric.clip_similarity` 成功/兜底两条路径都写 reward，初版写 `state["final_reward"]`。
- `BlenderGymRubric.@vf.cleanup` 接管 artifact 落盘和 `keep_failed_only` rmtree；env-level cleanup 删除，因为 env cleanup 早于 scoring。

### C. Entity model 重构（最终形态）

- 新增 `schema.py`，定义 `Task / Rollout / TurnRecord / ExitStatus / SCHEMA_VERSION / require_rollout`。
- `TurnRecord` 最终字段从旧 15 个精简为 9 个存储字段 + 3 个 property：`xml_parsed / render_success / timed_out` 不再存储，全部由 `exit_status` 派生；删掉 `observation / thought / extras`。
- `Rollout` 成为 BlenderGym 唯一 runtime model：`state["rollout"] = Rollout(...)`；旧的 BlenderGym 私有 state key（`task_id / work_dir / gpu_id / render_count / last_render_path / xml_parsed / render_success / turns / final_reward` 等）不再散落在 state 上。
- `Task` 是 dataset row 的 immutable runtime view，从 `dataset.py` emit 的 `info` dict 构造：`task_id / task_type / blend_file / goal_image / init_image / start_code_path`。
- `env.py` 的 `setup_state / get_prompt_messages / add_model_response` 全部通过 `require_rollout(state)` 读写 rollout。
- `rubric.py` 从 `rollout.last_render_path / rollout.task.goal_image / rollout.gpu_id` 读 CLIP 输入和设备，把 BlenderGym artifact 指标写到 `rollout.final_reward`。
- `trajectory_writer.py` 不再定义 domain model，只从 `schema.py` import `Rollout / TurnRecord / SCHEMA_VERSION`，负责把 Rollout 投影成 v2 artifacts。
- `test_trajectory_writer.py` 改用 `Task/Rollout` 构造 synthetic rollout；新增 `test_env_state.py` 覆盖 `setup_state` 只写 `state["rollout"]`。

## 调试经验

- 本机 Infinigen Blender 4.2 **不支持 OPTIX denoiser**，只支持 `OPENIMAGEDENOISE`；OptiX 加速指的是 Cycles compute device，不是 denoiser。
- 28 号 PoC 里 `denoiser=OPTIX` 被 try/except 吞掉，所以实际速度收益主要来自 `samples=16`，不是 denoiser。
- placement1 smoke：16spp/OIDN/OPTIX，n=3，mean 约 3.60s，`blender.log` 确认 `compute=OPTIX`。
- placement25 smoke：16spp/OIDN/OPTIX，n=3，mean 约 11.09s，stdev/mean 约 5%，因此保留 16 spp 默认值（没有必要切 32 spp）。
- `bench_render.py --keep-output` 最初用 `os.rename`，跨 `/tmp` 和 `/data` 文件系统会报 `Invalid cross-device link`，已改为 `shutil.move`。
- verifiers env-level `@vf.cleanup` 早于 rubric scoring；trajectory artifacts 必须挂在 `BlenderGymRubric.@vf.cleanup`，这样 `rollout.final_reward` 已经可用。这个顺序来自 verifiers `Environment.run_rollout`: `rollout → rubric.score_rollout → rubric.cleanup`。
- e2e smoke 用 fake completion 时不能直接调用父类 `add_model_response`（没有真实 Response object），需要 `_SmokeEnv` 跳过 parent 只跑 BlenderGym render 逻辑。
- `Rollout` 是 runtime object，不是 public JSON state；如果将来 verifiers 序列化完整 state，要排除 `state["rollout"]` 或提供 encoder。当前 durable contract 是 `trajectory.json`。
- `rollout.task` 是 BlenderGym `Task` object，和 verifiers `state["task"]`（通常是 `"default"` 字符串）不是一个概念。
- IDE lints 仍有 import-resolution warning（`verifiers/torch/open_clip/blendergym`），但 pytest 与实际 runtime 正常。

## 实测数据

### Render micro-benchmark

| task | samples | denoiser | compute | n | mean | stdev | p50 | p99 | success |
|------|---------|----------|---------|---|------|-------|-----|-----|---------|
| placement1 | 16 | OPENIMAGEDENOISE | OPTIX | 3 | 3.60s | 0.07s | 3.63s | 3.66s | 3/3 |
| placement25 | 16 | OPENIMAGEDENOISE | OPTIX | 3 | 11.09s | 0.53s | 10.97s | 11.66s | 3/3 |

两次 smoke 的 `blender.log` 都有：

```text
[blendergym] cycles samples=16 denoiser=OPENIMAGEDENOISE compute=OPTIX
```

### Entity model e2e smoke

用 `_SmokeEnv` 跑 placement46，真实 Blender render + CLIP reward + artifact cleanup 全链路通过：

```text
[smoke] task_id=placement46
[smoke] work_dir=.../outputs/blendergym_e2e_smoke/placement46__01234567
[smoke] symlink goal.png -> data/blendergym/placement46/renders/goal/render1.png
[smoke] symlink init.png -> data/blendergym/placement46/renders/start/render1.png
[smoke] symlink start.py -> data/blendergym/placement46/start.py
[smoke] turns=1 exit_status=ok
[smoke] turn_0/response.txt: 129 bytes
[smoke] turn_0/code.py: 91 bytes
[smoke] turn_0/blender.log: ~7 KB
[smoke] turn_0/render1.png: ~316 KB
[smoke] final_reward=0.9817
[smoke] meta.json: 250 bytes
[smoke] trajectory.json: 890 bytes
[smoke] trajectory.md: 8177 bytes
[smoke] schema=blendergym-trajectory-v2
```

## 参考代码

### 目录结构（当前 BlenderGym package）

```text
environments/blendergym/
├── blendergym/
│   ├── assets/pipeline_render_script.py
│   ├── dataset.py
│   ├── env.py
│   ├── prompts.py
│   ├── render.py
│   ├── rubric.py
│   ├── schema.py
│   └── trajectory_writer.py
├── scripts/
│   └── bench_render.py
└── tests/
    ├── test_dataset.py
    ├── test_env_state.py
    ├── test_rubric.py
    └── test_trajectory_writer.py
```

| 文件 | 关键位置 | 说明 |
|------|---------|------|
| `environments/blendergym/blendergym/assets/pipeline_render_script.py` | `_enable_gpu_cycles` | 读取 samples/denoiser/compute env var，配置 Cycles GPU + denoising |
| `environments/blendergym/blendergym/render.py` | `_build_argparser`, `main` | CLI render smoke，转写 cycles args 到 env var |
| `environments/blendergym/scripts/bench_render.py` | `main` | placement render micro-benchmark，输出 mean/stdev/p50/p99 |
| `environments/blendergym/blendergym/schema.py` | `Task`, `Rollout`, `TurnRecord`, `require_rollout` | BlenderGym 内部 runtime contract |
| `environments/blendergym/blendergym/env.py` | `setup_state`, `get_prompt_messages`, `add_model_response` | 构造/消费 `state["rollout"]`，执行每 turn render |
| `environments/blendergym/blendergym/rubric.py` | `clip_similarity`, metrics, `write_artifacts_handler` | CLIP reward、metrics、rubric cleanup artifact 落盘 |
| `environments/blendergym/blendergym/trajectory_writer.py` | `write_trajectory_artifacts` | 输出 `meta.json / trajectory.json / trajectory.md` v2 |
| `configs/multimodal/rl_blendergym.toml` | top-level `output_dir`, train/eval env args | 当前 BlenderGym RL 配置入口 |

## 关键 schema

### `Rollout` runtime model

`state["rollout"]` 是唯一 BlenderGym 私有 state key。关键字段：

```python
Rollout(
    task=Task(...),
    trajectory_id=state["trajectory_id"],
    work_dir=Path("outputs/blendergym_v2/blendergym_work/<task>__<uuid8>"),
    gpu_id=6,
    max_turns=3,
    turns=[TurnRecord(...), ...],
    final_reward=0.9817,
    start_code_text="...",
    goal_image_data_url="data:image/png;base64,...",
    init_image_data_url="data:image/png;base64,...",
)
```

派生 property：

- `render_count = len(turns)`
- `last_turn = turns[-1] if turns else None`
- `last_render_path = work_dir / last_turn.render_path`
- `trajectory_short_id = trajectory_id[:8]`
- `xml_parsed = bool(last_turn and last_turn.xml_parsed)`
- `render_success = bool(last_turn and last_turn.render_success)`

### `TurnRecord` final fields

存储字段只有：

```python
turn: int
exit_status: Literal["ok", "xml_parse_failed", "render_failed", "timeout"] | None
error_hint: str | None
action: str | None
render_path: str | None
code_path: str | None
response_path: str
log_path: str | None
duration_s: float | None
```

`xml_parsed / render_success / timed_out` 都是 property，由 `exit_status` 派生。

### `meta.json` v2

```json
{
  "task_id": "placement46",
  "task_type": "placement",
  "trajectory_id": "0123456789abcdef0123456789abcdef",
  "final_reward": 0.9817,
  "exit_statuses": ["ok"],
  "first_error_hint": null,
  "num_turns": 1,
  "max_turns": 1
}
```

### `trajectory.json` v2

```json
{
  "schema_version": "blendergym-trajectory-v2",
  "trajectory_id": "0123456789abcdef0123456789abcdef",
  "task": {
    "task_id": "placement46",
    "task_type": "placement",
    "blend_file": ".../data/blendergym/placement46/blender_file.blend"
  },
  "final_reward": 0.9817,
  "metrics": {
    "xml_parse_success": 1.0,
    "render_success": 1.0,
    "code_non_empty": 1.0
  },
  "num_turns": 1,
  "max_turns": 1,
  "steps": [
    {
      "turn": 0,
      "exit_status": "ok",
      "error_hint": null,
      "action": "execute_blender_code",
      "render_path": "turn_0/render1.png",
      "code_path": "turn_0/code.py",
      "response_path": "turn_0/response.txt",
      "log_path": "turn_0/blender.log",
      "duration_s": 4.85
    }
  ],
  "runtime": {
    "gpu_id": 6
  }
}
```

## 最终方案

采用“两层收敛”方案：

- Runtime 层：BlenderGym 自己的状态统一进 `state["rollout"]`，`Task` 表示 dataset task，`Rollout` 表示一次 rollout 的上下文/进度/reward/cache，`TurnRecord` 表示每 turn 的执行结果。verifiers/prime-rl 标准 state 字段不动。
- Artifact 层：`meta.json` 只保留快速筛选字段（task/trajectory/final_reward/exit_statuses/num_turns/max_turns），`trajectory.json` v2 保留完整 steps、metrics、runtime debug 信息。删掉 v1 中的 `session_id / agents / paths / render_success_per_turn / final_metrics` 等冗余字段。

这比单纯“删字段”更稳，因为它先明确了 BlenderGym 的内部 entity model，再由 writer 投影成 durable artifact contract。

## 验证结果

### 单测

```bash
uv run pytest environments/blendergym/tests/ -v --no-header
```

结果：31 passed，1 warning（requests dependency warning）。

新增/更新测试：

- `test_env_state.py::test_setup_state_stores_single_rollout_object`
- `test_trajectory_writer.py::test_turn_record_property_derivation`
- `test_trajectory_writer.py::test_rollout_property_derivation`
- `test_trajectory_writer.py::test_require_rollout_missing_or_wrong_type`
- `test_trajectory_writer.py::test_write_trajectory_artifacts_*` 全部更新到 v2 schema

### E2E smoke

临时脚本 `/tmp/blendergym_e2e_smoke.py` 已跑通并清理。它做了：

1. build eval dataset，取 placement46。
2. `env.setup_state` 生成 `state["rollout"]` 和 inputs symlink。
3. 注入 fake completion `<code>...</code>`，真实调用 Blender render。
4. `rubric.score_rollout(state)` 计算 CLIP reward。
5. `rubric.cleanup(state)` 写 v2 artifacts。

结果：

- `final_reward=0.9817`
- `meta.json=250B`
- `trajectory.json=890B`
- `trajectory.md=8177B`
- `schema=blendergym-trajectory-v2`

### Lints

ReadLints 仅剩 import-resolution warnings：

- `verifiers`
- `torch`
- `open_clip`
- `blendergym.*`（测试文件里的 package import）

这些是 IDE/path 解析问题，不影响 `uv run pytest` 或 runtime。

## 下一步任务

运行 BlenderGym 配置，看大模型在多轮 visual feedback 下的推理效果、XML 格式遵循、render 成功率、reward 分布，以及新 artifact 是否足够支持人工复盘。

## 初步方案

### 1. 先跑 bench 链路

```bash
uv run rl @ configs/multimodal/rl_blendergym.toml --bench
```

目的：先确认新 `state["rollout"]` model 没破坏 RL orchestrator/env worker 端到端流程，并生成少量 v2 artifacts。

跑完检查：

```bash
ls outputs/blendergym_v2/blendergym_work | head
```

进入任一 trajectory：

```bash
cd outputs/blendergym_v2/blendergym_work/<task_id>__<uuid8>
ls -la
ls -la inputs/
jq '.final_reward, .exit_statuses, .num_turns' meta.json
jq '.schema_version, .trajectory_id, .runtime.gpu_id, .steps[0].exit_status' trajectory.json
```

期望：

- 目录名形如 `placement46__01234567`
- `inputs/{goal.png,init.png,start.py}` 是 symlink
- `turn_N/response.txt` 始终存在
- `turn_N/code.py / blender.log` 在 XML parse 成功时存在
- `turn_N/render1.png` 仅 render 成功时存在
- `trajectory.md` 在 Cursor preview 可看到 GOAL/INIT/turn render 图和 response/code/log `<details>`

### 2. 统计模型行为

建议先写一两个临时 jq / python 统计，观察：

```bash
jq -r '.exit_statuses[]?' outputs/blendergym_v2/blendergym_work/*/meta.json | sort | uniq -c
jq -r '.final_reward' outputs/blendergym_v2/blendergym_work/*/meta.json | sort -n | tail
jq -r '.first_error_hint // empty' outputs/blendergym_v2/blendergym_work/*/meta.json | head -50
```

重点指标：

- XML parse rate：`xml_parse_failed` 比例
- Render success rate：`ok` 比例
- `render_failed` 的主要 `error_hint`
- `timeout` 是否出现
- 高 reward trajectory 是否真的移动到 goal，而不是 reward 偶然高
- 多 turn 是否基于上轮 render 修正，还是复制/重复初始代码

### 3. 人工 review 样本

优先打开这些：

- reward 最高的 3 条
- reward 最低的 3 条
- 含 `xml_parse_failed` 的 2 条
- 含 `render_failed` 的 2 条

看 `trajectory.md`：

- Turn 0 是否遵守 `<code>...</code>`
- code 是否复用了 `start.py` 中已有 helper，而不是 hallucinate method
- error_hint 是否足够定位 Blender 失败
- 图像反馈是否进入下一 turn prompt 后产生行为变化

### 4. 如果推理效果差，优先排查 prompt 而不是 render

前置 session 已发现一个 priority-0 prompt 问题：

- `prompts.py` system prompt 硬编码坐标范围：`x∈(-1,1) / y∈(-0.75,0.75) / z∈(0,1.55)`
- 但具体 placement 任务（如 placement12）`start.py` 注释中的真实范围可能是 `x∈(-1.5,1.5) / y∈(-1.8,1.8) / z∈(0,2)`

如果模型频繁移动幅度不够或 reward 不上升，下一步应先做 prompt/data task-specific bounds 修复，而不是继续调 render。

### 5. 下一步可能的运行层级

1. `--bench`：确认 pipeline 和 artifacts。
2. 小 eval：选 5–10 个 examples，看真实模型推理（非 fake trainer data）。
3. 小训练：如果 eval trajectory 质量基本 OK，再跑少步数训练观察 reward / render_success 是否稳定。

不要一开始就长训；先用 v2 `trajectory.md` 做人工闭环。
