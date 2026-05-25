# Session Handoff: BlenderGym Smoke30 真训

## 前置 session

- [BlenderGym HTML Viewer + Bench Validation](2026-04-29-blendergym-html-viewer-and-bench.md) — trajectory.md → trajectory.html, completion_to_text Pydantic 修复, 4-step bench 验证。
- [BlenderGym OptiX + Entity Refactor](2026-04-28-blendergym-optix-entity-handoff.md)

## 任务目的

把 BlenderGym 从 4-step bench 切到真训：max_steps=30, rollouts_per_example=8, eval@10/20/30, ckpt@15/30，agent 后台拉起 `uv run rl` 并 5–10 次 check-in 写 STATUS.md，验证多轮 visual feedback + CLIP reward 是否能驱动 Qwen3.5-0.8B 学出 placement 任务。

## 执行内容

### Plan 1: 改 [configs/multimodal/rl_blendergym.toml](configs/multimodal/rl_blendergym.toml)

- `max_steps`: 15 → 30
- `rollouts_per_example`: 4 → 8（抗 zero_advantage）
- 新增 `[ckpt] interval=15 keep_last=3`（拿到 ckpt_step_15 / ckpt_step_30）
- 新增 `[orchestrator.eval] interval=10`（eval@10/20/30）
- env worker / GPU 池 / trainer / inference 全部不动

### 启动 + 巡检

- 删除上一次 bench 残留 `outputs/blendergym_v2/` (5.3 GB)
- 后台启动 `uv run rl @ configs/multimodal/rl_blendergym.toml`，~60s 内 inference / orchestrator / trainer / 2 train env workers + 1 eval env worker 全部就绪
- 每 5–15 min check-in 一次，写入 `outputs/blendergym_v2/STATUS.md` 共 7 条 entry
- 总 wall clock 约 1h32min（07:38 UTC 启动 → 09:11 UTC trainer finished）

### 训练结果

reward / eval 信号噪声很大，30 step 不足以下学习结论：

- Eval `Avg@1`: step 0 `0.1938` → step 10 `0.0000` → step 20 `0.3882` → step 30 partial JSONL 平均约 `0.1965`（orchestrator 没打印 final summary）
- Train reward 范围 `0.0883–0.3773`，最高 step 20 = `0.3773`，最后 step 29 = `0.1408`
- Trainer 全程稳定：mismatch KL 全程 `< 0.014`，grad norm `0.31–0.65`，无发散
- zero_advantage 频繁出现 `8/32` 或 `16/32`，即使 rpe=8 仍占满 1/4 batch
- 全部 ckpt + weights 落盘：`outputs/blendergym_v2/{checkpoints,weights}/step_{15,30}`

### Eval 记录位置 + step 10 = 0% 原因

eval rollout 文件都保存了（注意 eval 文件目录有 off-by-one 命名）：

- `outputs/blendergym_v2/run_default/rollouts/step_0/eval_rollouts.jsonl` → ckpt_step=0
- `outputs/blendergym_v2/run_default/rollouts/step_11/eval_rollouts.jsonl` → ckpt_step=10
- `outputs/blendergym_v2/run_default/rollouts/step_21/eval_rollouts.jsonl` → ckpt_step=20
- `outputs/blendergym_v2/run_default/rollouts/step_30/eval_rollouts.jsonl` → final eval JSONL（存在，但 orchestrator 没打印最终 summary）

step 10 eval 5 条全 0 的具体原因不是 CLIP 低，而是**最后一轮没有成功 render**：

| eval example | work_dir | 结论 |
|-------------|----------|------|
| placement46 | `placement46__528d6321` | turn0/turn1 render ok，turn2 `SyntaxError`，最终 reward=0 |
| placement47 | `placement47__0e6bb652` | 3 turn 都 XML parse failed，找不到 `<code>...</code>` |
| placement48 | `placement48__adeb168a` | 3 turn 都 `KeyError: potted_plant` |
| placement49 | `placement49__6c535170` | turn0 ok，turn1/turn2 `NameError: white_pillow`，最终 reward=0 |
| placement50 | `placement50__b80e7d3e` | `KeyError: blue_lamp` + 最后一轮 `SyntaxError` |

关键机制：`BlenderGymRubric.clip_similarity` 只看 `rollout.last_render_path`，而 `Rollout.last_render_path` 只取最后一个 `TurnRecord.render_path`。所以前面 turn render 成功、最后 turn 失败时，整条 rollout 最终 reward 仍然是 0。step 10 里至少 placement46/placement49 中间成功过，但最后 turn 崩掉，所以平均分是 0。

对比：

- step 0：只有 placement47 成功，`0.9688 / 5 = 0.1938`
- step 20：placement47 + placement49 成功，`(0.9689 + 0.9720) / 5 = 0.3882`
- step 30：JSONL 里 placement49 成功 `0.9823`，平均约 `0.1965`，但 final summary 没写到 log

### Final eval hang (open issue)

Trainer 已经 finished 后，orchestrator 还在跑 final eval：

- `09:10:08 UTC` final eval 启动（在 trainer 写 ckpt_30 + weight broadcast 同时）
- `09:11:16 UTC` 一次 `Pausing/All inference engines resumed`（weight update）
- `09:16:44 UTC` 进度到 `4/5`（example 4 用 `02:14`，正常）
- 之后 `Active tasks: 1 (W0: 1)` 一直挂着 ~7 分钟，orchestrator 没新行
- 取证：`pgrep -af blender` 无 Blender 子进程；GPU 7 显存 3.4 GB / util 0%；`render_timeout_s=120` 没触发；eval env_server 还在打 `Lag stats`
- 09:23 UTC 手动 `kill -TERM` launcher + 子进程，10s 后 SIGKILL 兜底，全部清理完毕，GPU 显存归零

判断：hang **不在 Blender 子进程**（否则会被 timeout 杀掉），更可能是 eval worker 在 vLLM client `await` 上、或 trajectory writer/CLIP forward 上死锁。后续发现 `step_30/eval_rollouts.jsonl` 实际存在且有 5 行，但 orchestrator 没打印 final `Evaluated blendergym-eval...` summary；所以数据可用，log summary 缺失。

### Plan 2: BlenderGym artifact 路径重构 (post-smoke)

跑完 smoke 后讨论目录乱的问题，目录三套系统（prime-rl logs/checkpoints、orchestrator run_default/rollouts、BlenderGym blendergym_work）平铺在 output_dir 下，从 `rollouts/step_N/*.jsonl` 反查 trajectory.html 要靠 `trajectory_id[:8]` 手匹配。决定不动 `src/prime_rl`，只在 BlenderGym + config 内做一阶段重构。改动如下：

- [configs/multimodal/rl_blendergym.toml](configs/multimodal/rl_blendergym.toml)：
  - `work_root` 从 `outputs/blendergym_v2/blendergym_work` 改到 `outputs/blendergym_v2/run_default/blendergym_work`，把 trajectory 收拢到 run_default 下
  - train args 显式 `env_name = "blendergym"`，eval args 显式 `env_name = "blendergym-eval"`（与 wandb sample table 对齐）
  - 顺便把 `args = { ... }` 单行 inline table 拆成 `[orchestrator.train.env.args]` 节式
- [environments/blendergym/blendergym/schema.py](environments/blendergym/blendergym/schema.py) `Rollout`：加固定 schema 的 `metadata: dict | None`（含 schema-lock docstring）
- [environments/blendergym/blendergym/env.py](environments/blendergym/blendergym/env.py)：
  - `__init__` 新增 `env_name: str = "blendergym"` kwarg
  - `setup_state` 构造 metadata（env / split / example_id / task_id / task_type / trajectory_id）
  - `_make_work_dir` 接受 `split / example_id`，新路径 `{work_root}/{split}/example_{id:04d}__{task_id}/{traj8}/`，无 metadata 时退回 `{work_root}/{task_id}__{traj8}` 老布局；example_id 用 `isinstance(_, int)` 判定（防字符串误入）
- [environments/blendergym/blendergym/trajectory_writer.py](environments/blendergym/blendergym/trajectory_writer.py)：`meta.json` / `trajectory.json` 顶层写 `metadata`，HTML header 加 5 行 metadata block (`env / split / example_id / task_id / trajectory_id`)
- [environments/blendergym/tests/test_env_state.py](environments/blendergym/tests/test_env_state.py)：主测试改用 `env_name=` kwarg；新增 4 个测试覆盖结构化路径 + 3 种 fallback 分支（split=None / example_id=None / example_id=str）
- 测试结果：`uv run pytest environments/blendergym/tests/test_env_state.py environments/blendergym/tests/test_trajectory_writer.py` → 30 passed
- metadata schema 与 wandb `log_eval_samples` 表列对齐（多一个 `trajectory_id` 用于反查），为未来双向跳转打基础

待验证：smoke 真训跑一次，确认 `outputs/blendergym_v2/run_default/blendergym_work/{train,eval}/example_*/<traj8>/trajectory.html` 真实落地，且 train/eval 的 `meta.json.metadata.env` 分别是 `"blendergym"` / `"blendergym-eval"`。

## 调试经验

- **`nvidia-smi` 单帧 GPU util 是采样瞬时 SM%，不能直接代表 pipeline 效率**。watch 1s 经常采到 0，但 trainer/env 实际是短 burst + 长等待的多阶段流水线。要看瓶颈应该看 orchestrator step 时间分布、trainer `time/wait_for_batch`、env_server active task / lag、vLLM `gpu_cache_usage`，而不是 `nvidia-smi` util。
- **trainer MFU 低不等于训练算力是瓶颈**。本次 trainer MFU 仅 4.4–6.6%，throughput ~3k tokens/s，但 trainer wall time 大部分在等 batch；orchestrator step 110–240s（总 32 rollouts × 3 turns × 16-spp render 全在这里），env 才是 wall clock 决定者。
- **`Peak FLOPS undefined for NVIDIA H20. Falling back to A100 (312 TFLOPS)`** — MFU 数字是按 A100 折算的，H20 实际 FLOPS 不同，所以 MFU 值不能直接和 H100/A100 比对。
- **vLLM 不是瓶颈**：长期 `Running: 24-30 reqs, Waiting: 0 reqs, GPU KV cache usage: 1.6%-2.2%`，没排队、KV 不紧，inference 侧基本闲置。
- **Pydantic bug 没复发**：抽查 `placement2__c456adac/turn_{0,1,2}/response.txt` 都非空（499/499/386 bytes），上次修复有效。
- **eval step 10 `Avg@1=0.0000` 不一定代表退化**：5 examples × 1 rollout 的 eval 方差太大，step 20 直接回到 `0.3882`。要看趋势必须 num_examples ≥ 30 或 rollouts_per_example > 1。
- **`Detected busy event loop max=5.6s`** 几次 — 短时阻塞，没影响进度，可能是 trajectory_writer 写大文件或 Blender pause。
- **final eval 可能 hang 而不被 render_timeout 救回**：`render_timeout_s=120` 只覆盖 Blender subprocess，覆盖不到 vLLM client / trajectory writer / CLIP forward 的死锁。trainer 已 finished 后 orchestrator 还在等单个 example，要么补 client-side timeout，要么训练结束时无条件给 final eval 一个 wall-clock 上限。
- **本地 artifact 想跟 wandb 对齐就得固定 schema**：`Rollout.metadata` 字段集严格匹配 wandb sample table 列（仅多一个 `trajectory_id` 用于反查），而不是“能塞就塞”。否则 wandb 有的字段本地没，或者本地 metadata 跟 wandb 命名漂移，调试时还得脑内 mapping。
- **example_id 来源不能假设统一**：训练 rollout 走 `Buffer._EnvBuffer` 时 example 已 `example_id=idx`，但 eval rollout 走 `EvalEnv.examples = self.env.get_eval_dataset(...)` 直接从 dataset 出，得看 dataset 列里是否有 `example_id`。`setup_state` 用 `state.get("example_id", info.get("example_id"))` 两层 fallback。
- **env_name 在 setup_state 里拿不到**：prime-rl 是 rollout 完成后才往 `rollout["env_name"]` 注入，而不是 setup 时；并且 eval examples 完全不经过 buffer。所以 BlenderGym 必须自己接 `env_name` kwarg，由 toml 显式声明（train/eval 各一份），不能依赖 `state["env_name"]`。

## 参考代码

| 文件 | 关键位置 | 说明 |
|------|---------|------|
| [configs/multimodal/rl_blendergym.toml](configs/multimodal/rl_blendergym.toml) | 顶层 + `[ckpt]` + `[orchestrator.eval]` | 真训 smoke30 入口 |
| [outputs/blendergym_v2/STATUS.md](outputs/blendergym_v2/STATUS.md) | 7 条 timestamped entry | 完整巡检日志 |
| [environments/blendergym/blendergym/env.py](environments/blendergym/blendergym/env.py) | `add_model_response`, `gpu_id_pool` 轮询, `max_turns`, `_make_work_dir` (结构化 + fallback), `setup_state` (metadata) | rollout / render 调用链 + artifact 路径 |
| [environments/blendergym/blendergym/render.py](environments/blendergym/blendergym/render.py) | `run_blender`, `BLENDERGYM_CYCLES_SAMPLES`, `DEFAULT_TIMEOUT_S=120` | Blender subprocess 入口 |
| [environments/blendergym/blendergym/rubric.py](environments/blendergym/blendergym/rubric.py) | `clip_similarity` 单 reward + 3 metric | CLIP cosine 是唯一梯度信号 |
| [environments/blendergym/blendergym/schema.py](environments/blendergym/blendergym/schema.py) | `Rollout.metadata` (固定 schema, schema-lock docstring) | wandb 对齐用的 metadata 入口 |
| [environments/blendergym/blendergym/trajectory_writer.py](environments/blendergym/blendergym/trajectory_writer.py) | `write_trajectory_artifacts` (写 metadata), `_render_html_header` (展示 metadata) | artifact 内容输出 |
| [src/prime_rl/utils/monitor/wandb.py](src/prime_rl/utils/monitor/wandb.py) | `log_eval_samples` table 列定义 | 本地 metadata 字段集与之对齐 |
| [src/prime_rl/configs/orchestrator.py](src/prime_rl/configs/orchestrator.py) | `[orchestrator]` `max_async_level`, `max_inflight_rollouts` | 控制 async / 在飞 rollout 数 |

## 最终方案

Smoke30 跑通 + step 20 eval 出现正信号（`0.1938 → 0.3882`），证明 BlenderGym 真训 pipeline 端到端可用、Pydantic bug 不复发、checkpoint 正常落盘。但 30 step 不足以判断模型是否真在学习；瓶颈定位为 **rollout/env wall time**（每步 110–240s），trainer 多数时间在等 batch。

后续在不动 `src/prime_rl` 的前提下重构了 BlenderGym artifact 路径，把 trajectory 收拢到 `run_default/blendergym_work/{split}/example_*/<traj8>/`，并在每条 artifact 写入与 wandb sample table 对齐的 `metadata`。30 个 BlenderGym 单测通过；`src/prime_rl/utils/monitor/wandb.py` 加 `trajectory_id` 列做双向跳转留作阶段 2。

## 下一步任务

跑一次新 smoke 同时验证两件事：

1. **bottleneck 缓解**：`num_workers=4`（每 GPU 2 个 Blender + CLIP）能否把 step time 折半，是否撞 OPTIX context / OOM。
2. **artifact 路径重构落地**：确认 `outputs/blendergym_v2/run_default/blendergym_work/{train,eval}/example_*/<traj8>/trajectory.html` 真实生成，且 `meta.json.metadata.env` 在 train/eval 分别为 `"blendergym"` / `"blendergym-eval"`。

次优先：修 final-eval hang。

## 初步方案

候选优化项排序（高收益 → 低收益）：

1. **加 env worker 并发**（成本最低，预期收益最大）
   - 当前 `gpu_id_pool=[6,7]`、`num_workers=2` → 1 worker per GPU，每 GPU 实际只跑 1 个 Blender + 1 个 CLIP forward。
   - H20 有 96 GB 显存，单 Blender + Cycles OPTIX 显存占用 ~1 GB，CLIP ViT-B-32 ~1 GB，单 GPU 跑 4–8 个 Blender 子进程是可能的。
   - 改 `num_workers=4` 或 `=8`，配合 `cycles_compute_device=OPTIX` 共享 GPU；预计 step time 折半。
2. **降低 cycles_samples**：现在 `cycles_samples=16` + OPENIMAGEDENOISE，再降到 `8` 配合 OIDN 通常 CLIP score 损失 < 5%，但 render 时间近似线性减半。
3. **减少 max_turns 或动态截断**：现在 `max_turns=3`，多轮失败的 rollout 会消耗 3× render；可以用 reward 阈值早停（CLIP > 0.85 提前结束）或 xml_parse 失败一次就终止。
4. **打开 async pipelining**：当前 `async_level=1`、`max_inflight_rollouts=32`，所以训练完 step N 必须等 step N+1 全部 rollout 完成；提到 `async_level=2` 可以让 trainer 在等 batch N+2 时用 batch N+1 训练。需要先看 mismatch_kl 是否承受得住。
5. **enforce zero_advantage filter**：现在 `enforce=false` 只是 log；改成 `enforce=true` 直接丢掉 zero-advantage rollout，避免占 trainer batch slot；但会增加每 step 实际 rollout 数（系统会重抽样）。
6. **prompt 修复**：前置 session 已经标记 prompts.py 硬编码坐标范围与实际任务不对齐，可能从源头压低 reward → 间接增加 zero_advantage。

潜在风险：

- 多 worker 共享 GPU：OPTIX context 切换 + Cycles BVH build 可能撞 OOM/竞争，需要先单独压测。
- async_level > 1 在 0.8B 小模型 + 噪声 reward 上可能让 KL 跳变；先小步试。
- 降 cycles_samples 会让 render 噪声变大，需要重新对一遍 CLIP 阈值。

### Final eval hang 修复（次优先级）

- 复现：再跑一次 smoke30，watch orchestrator log；如果 `final eval` 又 hang 在最后 1 个 example，stack trace 一下 eval env_worker 看死锁点（`py-spy dump --pid <env_worker_pid>`）。
- 短期 mitigation：
  - 给 final eval 一个 wall-clock 上限（比如 5 min × num_examples），超时就放弃 missing example 让 orchestrator 退出。
  - 或在 `verifiers` 客户端 wrapper 上加 vLLM HTTP timeout（现在大概是没显式 timeout）。
- 不影响主流程的 workaround：把 `[orchestrator.eval] interval` 设大一点（比如 = max_steps），训练结束时不强制 final eval；想要 step 30 eval 单独跑离线评估即可。
