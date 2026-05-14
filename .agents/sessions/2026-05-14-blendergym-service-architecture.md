# Session Handoff: BlenderGym Render/Score 服务化实现

## 任务目的

将 Blender 渲染和 CLIP 评分从 env worker 进程中解耦为两个独立 FastAPI 服务（Render Service :8420 / Score Service :8421），消除 Blender cold start（~5s → ~1.5s），解决 GPU 争用问题，并释放 vLLM 显存。

## 执行内容

- 按 `blendergym_architecture_rethink` plan 完整实现 Phase 0–2
- 新建 `blendergym/services/` 子包：`base.py`（BaseGPURouter + BaseService）、`semaphore_router.py`（lease 模式路由器）
- 新建 Render Service：`persistent_blender.py`（Unix socket 长驻 Blender + BlenderPool）、`worker_loop.py`（Blender 内渲染循环，复用 `_enable_gpu_cycles`/`_render_camera1`）、`server.py`、`client.py`
- 新建 Score Service：`clip_scorer.py`（per-GPU CLIP 模型）、`server.py`、`client.py`
- 改 `env.py`：移除 `blender_bin`/`gpu_id_pool`/`cycles_*`/`_next_gpu`/`log_gpu_mem`，接入 `RenderClient`
- 改 `rubric.py`：移除本地 CLIP 加载，接入 `ScoreClient`
- 改 `schema.py`：删 `Rollout.gpu_id`，加 `TurnRecord.render_gpu_id`
- 改 `artifact_manager.py`：删 `run_render()`
- 改 `trajectory_writer.py`：`runtime.gpu_id` → `runtime.render_gpus`
- 改两个 TOML config：加 `render_service_url`/`score_service_url`，删旧参数，`gpu_memory_utilization` → 0.75
- 改 `blendergym.sh`：OPTIX warmup 同步化 + 启动双服务 + deep health check + trap 清理
- 改 `pyproject.toml`：加 `fastapi`/`uvicorn[standard]`/`httpx`
- 修所有受影响 test（`test_env_state`/`test_trajectory_writer`/`test_rubric`）
- 新建 `tests/test_persistent_parity.py`（GPU 机器上验证 one-shot vs persistent 一致性）

## 参考代码

| 文件 | 关键位置 | 说明 |
|------|---------|------|
| `blendergym/services/base.py` | `BaseGPURouter.lease()`, `BaseService` | 服务脚手架，lease context manager 保证 GPU 资源归还 |
| `blendergym/services/semaphore_router.py` | `SemaphoreRouter` | Render(max_concurrent=1) / Score(max_concurrent=2) 共用 |
| `blendergym/services/render/persistent_blender.py` | `PersistentBlender`, `BlenderPool` | Unix socket 通信，watchdog 连续 3 次失败重启 |
| `blendergym/services/render/worker_loop.py` | `_handle_request()` | Blender 嵌入式 Python，blend 文件缓存（revert/open），stdout/stderr 捕获 |
| `blendergym/services/render/server.py` | `RenderService.render()` | FastAPI POST /render，router.lease → pool.render |
| `blendergym/services/render/client.py` | `RenderClient.render()` | 同步 httpx，组装 blender.log（=== STDOUT/STDERR ===） |
| `blendergym/services/score/clip_scorer.py` | `CLIPScorer` | `compute_clip_cosine_similarity` 从 rubric.py 迁入 |
| `blendergym/env.py` | `__init__`, `add_model_response`, `close` | 持有 RenderClient，写 blender.log，填 render_gpu_id |
| `blendergym/rubric.py` | `clip_similarity` | 一行 `await self.score_client.score()` 替换整个 CLIP 逻辑 |
| `blendergym/schema.py` | `Rollout`, `TurnRecord` | gpu_id 删除，render_gpu_id 新增 |
| `scripts/envs/blendergym.sh` | `env_setup()` | 启动双服务 + health check + trap 清理 |
| `configs/multimodal/rl_blendergym_kaola.toml` | `[orchestrator.*.env.args]` | render_service_url / score_service_url |

## 最终方案

双 FastAPI 服务架构：
- **Render Service** 管理 PersistentBlender worker pool，通过 Unix socket 与 Blender 子进程通信，`SemaphoreRouter(max_concurrent=1)` 做 GPU 调度
- **Score Service** per-GPU 固定 CLIP 模型，`SemaphoreRouter(max_concurrent=2)` 提供背压
- env worker 只持有 HTTP client，不直接触碰 GPU
- `run_blender()` 保留不动，仍用于 OPTIX warmup 和 parity test

选择此方案而非进程内多线程：Blender 不是线程安全的，必须用子进程隔离；CLIP 读推理虽线程安全但需 semaphore 控制并发防 CUDA 队列堆积。

## 下一步任务

在 KAOLA 上重启训练，验证服务化架构端到端工作正常。

## 初步方案

- 推代码到远端，在 KAOLA 上提交新 job
- `blendergym.sh` 会自动：OPTIX warmup → 启动 Render Service → 启动 Score Service → health check → 进入训练
- 关注点：
  1. 首次启动时 `uv pip install -e environments/blendergym` 需拉取 `fastapi`/`uvicorn`/`httpx`，确认网络通
  2. health check 如果超时，查 `/local-ssd/prime-rl-output/logs/render_service.log`（Blender worker 是否 spawn 成功、socket 是否 ready）
  3. 训练跑起来后观察 wandb 的 `render_success` / `clip_similarity` 指标是否正常
  4. `trajectory.json` 中 `runtime.render_gpus` 应该是 per-turn GPU id 列表（之前是单个 `gpu_id`）
  5. 如果出错需要 fallback，可以在 TOML 里把旧参数加回来并恢复 `env.py` 的 `run_render` 调用（但 `Rollout.gpu_id` 已删，需配合恢复）
