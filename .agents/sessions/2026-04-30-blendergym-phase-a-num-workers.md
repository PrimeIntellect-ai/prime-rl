# Session Handoff: BlenderGym Phase A — num_workers Sweep

## 前置 session

- [BlenderGym Smoke30 真训](2026-04-29-blendergym-smoke30-real-rl.md) — 30-step 真训跑通、artifact path 重构（Plan 2）落地、final-eval hang 记录。

## 任务目的

按昨天 Plan 2 重构后第一次真实 run，做两件事：

1. **artifact path 落地验证**：确认 `run_default/blendergym_work/{train,eval}/example_*/<traj8>/...` 真实生成、metadata 正确、fallback 不被触发。
2. **Phase A 第一维度：num_workers 1D sweep**：在不动机器、不改 src/prime_rl 的前提下，先用 num_workers 试着缓解 rollout/env 瓶颈。每轮 5-step smoke 比对 step time / GPU peak / failure rate。

## 执行内容

### A0 (num_workers=2 baseline) — `outputs/blendergym_v3_a0_w2/`

- 12:36:49 → 13:03:52 UTC，~26.5 min wall clock
- baseline `step_time* = 160.6s`（step 2-4 mean，去掉 step 0 含 weight update + warmup）
- step 1-4 mean 173.4s
- Eval@0 `Avg@1=0.0000`（5×1 高方差，4 例最后 turn render fail）；Final eval@5 `Avg@1=0.1957`
- GPU 6/7 peak 都是 **2255 MiB**（≈ 2.3% / 96 GB）
- mismatch_kl < 0.014 全程，trainer 稳定
- step 4 zero_advantage=32/32 是 batch 内全部 reward=0（pre-existing 失败模式，不是 regression）
- **artifact path 8 项 check 全过**：`run_default/blendergym_work/train/example_NNNN__placementXX/<traj8>/{trajectory.html,meta.json,...}`，train metadata.env=`blendergym`，eval metadata.env=`blendergym-eval`，inputs/ 三个 symlink 全部解析到真实文件，旧 fallback 路径 0 触发
- **final-eval hang 没有复现**

### A1 (num_workers=4) — `outputs/blendergym_v3_a1_w4/`

- 04:24:54 → 04:49:48 UTC（2026-04-30），~25 min
- step 2-4 mean **169.8s** vs A0 160.6s，**慢 6%**；step 1-4 mean 也慢 6%
- per-step 方差 96s，远大于 A1-A0 的 10s 差异 → 单凭 5 个样本不能下"num_workers=4 真的负收益"的结论
- Eval@0 `Avg@1=0.1963`，Final eval@5 `Avg@1=0.1960` —— 学习信号没变化（5-step 太短）
- GPU 6 peak 4506 MiB，GPU 7 peak 5632 MiB（5.9% / 96 GB）
- 触发了 step time termination 信号，但只 1/4 信号触发（GPU peak / failure / OOM 全 OK）
- **决定继续推 A2**：理由是 GPU 显存还有 90+ GB 空间，且 5-sample 噪声大于 delta，曲线还没扫到拐点

### A2 (num_workers=8) — `outputs/blendergym_v3_a2_w8/`

- 04:52:19 UTC 启动，目前还在跑，无结果
- Watchpoints：GPU peak（4 OPTIX context/card 是否同时启动撞高峰）、step time vs A0/A1、env worker failure rate

### IDE 小调整

- Microsoft Live Preview 设了 `"livePreview.autoRefreshPreview": "Never"`（macOS user settings），避免 trajectory.html 被覆写时 webview 自动跳到新内容、干扰排查旧 rollout

## 调试经验

- **`Active tasks: 1 (W0:1)` 长时间持续 ≠ hang**。A0 eval@0 的 example 5 慢了 ~2:43 我第一时间错读成 hang，重新看 env_server lag stats + 没有 Blender 子进程 + 后续推进，确认是慢不是死。下次别再用单一信号判断。
- **5-step sample 的 wall clock 噪声 ±60s**，远大于 num_workers 翻倍带来的 ~10s 差异。要分辨真实趋势必须 sample > 10 步或者改用 token throughput，而不是单步 wall。
- **GPU memory 比预期低非常多**（2-6 GB / 96 GB H20）。多 worker 共享 GPU 在 OOM 维度完全没风险；真正的天花板是 OPTIX context 切换 + Cycles BVH build 的串行化（GPU compute serial），而不是显存。下次评估 worker concurrency 不要把"显存不够"列为风险。
- **Final-eval hang 在 A0/A1 都没复现**。昨天那次（max_steps=30 + async level + rpe=8）很可能是某种特定组合触发，与 BlenderGym 自身关系不大；先不修，留观察。
- **artifact path 重构 schema 一致**：train metadata.env 是 `blendergym`、eval 是 `blendergym-eval`，全靠 toml 里 `env_name` 显式声明（依赖 `state["env_name"]` fallback 行不通），fallback 路径 0 次触发，结构化分支 100% 命中。

## 参考代码

| 文件 | 关键位置 | 说明 |
|------|---------|------|
| [outputs/blendergym_v3_a0_w2/STATUS.md](outputs/blendergym_v3_a0_w2/STATUS.md) | A0 baseline + 8 项 artifact path 验证 | num_workers=2 baseline `step_time*=160.6s`，GPU peak 2.3 GB |
| [outputs/blendergym_v3_a1_w4/STATUS.md](outputs/blendergym_v3_a1_w4/STATUS.md) | A1 (num_workers=4) 结果 | 169.8s，慢 6%，但 5-sample 噪声内 |
| [outputs/blendergym_v3_a2_w8/STATUS.md](outputs/blendergym_v3_a2_w8/STATUS.md) | A2 (num_workers=8) 启动记录 | 4 OPTIX context/card，正在跑 |
| [configs/multimodal/rl_blendergym.toml](configs/multimodal/rl_blendergym.toml) | `num_workers`, `gpu_id_pool`, `[orchestrator.eval] interval`, `output_dir` | A* 之间唯一改动点；当前 A2 状态 |
| [environments/blendergym/blendergym/env.py](environments/blendergym/blendergym/env.py) | `setup_state`, `_make_work_dir` | artifact path schema 来源；本轮验证它 100% 命中结构化分支 |

## 最终方案

Plan 2 artifact path 重构在第一次真实 run 上 8 项 check 全过，schema 在 train/eval 都正确生成、fallback 不触发；这条线收尾。

num_workers 的 1D sweep 现在的位置：A0 是 baseline，A1 显示轻微负收益但样本不足，A2 在跑。决定**让曲线先扫到拐点（A2 / 可能 A3=16）再下结论**，而不是看一两个点就锁参数。如果 A2/A3 都没改善，承认 OPTIX context serial 是真天花板，把 Phase A 焦点切到 cycles_samples / max_turns / async_level 等其他维度。

## 下一步任务

等 A2 跑完，按 termination 三条信号决定：

- **A2 提升**（step time < A1 且 < A0×0.95）→ 推 A3 (num_workers=16)
- **A2 持平/退化** → 锁 `num_workers* = 2`，Phase A 切到下一维度

## 初步方案

A2 评判：

1. step 1-4 mean 跟 A0 (160.6s) / A1 (169.8s) 直接比对；
2. GPU peak 跟 90 GB 比对（应该还远低于）；
3. failure rate 看 `zero_advantage` / `errored_rollouts` 是否爆炸。

下一维度优先级（如果 num_workers 锁定）：

1. **`cycles_samples=8` + OIDN**：render 时间近似线性折半，CLIP 噪声小，先单独跑离线 CLIP 对齐
2. **`max_turns` 早停**：CLIP > 0.85 提前结束，但要权衡多轮 visual feedback 信号被砍
3. **`async_level=2`**：trainer pipeline overlap，先看 mismatch_kl 是否承受得住（A0 全程 < 0.014 ⇒ 余量很大）

风险：

- OPTIX context 在多 worker 时是否真的 serial，需要 `nsys` / `nvprof` 看 SM 占用才确证；否则只能从 step time 反推
- 5-step smoke 的 wall noise 仍是 ±60s，单条曲线得有 ≥3 个数据点才有判别力
- final-eval hang 暂时不管，A0/A1 没复现；等长 step 数 + max_steps≥20 时再观察
