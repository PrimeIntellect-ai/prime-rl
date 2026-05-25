# Session Handoff: 三件套落地 + bpy PoC + 瓶颈翻转到 vLLM

## 前置 session

- [BlenderGym Render Daemon 架构评估](2026-04-30-blendergym-daemon-design-eval.md) — 评估 daemon 4 路径(A/B/C/D)+ 三件套优先级,产出"先并行做三件套 + 30min PyPI bpy PoC"。
- [BlenderGym Phase A — num_workers Sweep](2026-04-30-blendergym-phase-a-num-workers.md) — 实测 165 s/step baseline,加 worker 不加卡 wall 不降。

## 任务目的

落地上一份 handoff 决定的两条并行工作流:

1. **三件套**(resolution 512→256 + cycles_samples 16→8 + max_async_level 1→2 + 6 卡 6 worker)实施 + 真训验证。
2. **PyPI bpy + OPTIX 30min PoC**,产出 daemon 路径 A vs B 决策。

中途因为诊断结果反转,目标变成讨论"加 vLLM dp + Blender 共卡"作为下一份 session 焦点。

## 执行内容

### 1. 三件套落地(plan A 阶段)

代码改动 3 处(都在 `environments/blendergym/`,完全不动 `src/prime_rl/`):

- [`pipeline_render_script.py`](../../environments/blendergym/blendergym/assets/pipeline_render_script.py) `_enable_gpu_cycles` 头部加 `BLENDERGYM_RENDER_RESOLUTION` env hook。
- [`env.py`](../../environments/blendergym/blendergym/env.py) `BlenderGymEnv.__init__` 加 `cycles_resolution` 参数 + 一行 `os.environ[...]` 注入。
- [`render.py`](../../environments/blendergym/blendergym/render.py) CLI 加 `--resolution`(对齐 `--samples / --denoiser / --compute-device` 风格)。

[`configs/multimodal/rl_blendergym.toml`](../../configs/multimodal/rl_blendergym.toml) 改动:

- `output_dir = "outputs/blendergym_v3_real" → "outputs/blendergym_v3_three_kits"`(避免覆盖 baseline)
- train env 块:`num_workers 2→6`,`gpu_id_pool [6,7]→[2,3,4,5,6,7]`,`cycles_samples 16→8`,新增 `cycles_resolution=256`
- eval env 块:同步 `cycles_samples=8 + cycles_resolution=256`,`gpu_id_pool=[7]` 不动
- 新增 `[orchestrator] max_async_level=2` + `[trainer] max_async_level=2`(`validate_shared_max_async_level` 强制两边相等)

`uv run rl --dry-run` 验证通过,启动 50-step 真训。

### 2. PyPI bpy PoC(plan B 阶段)

- 主 venv `uv add bpy` 失败:cp312 没 wheel,只有 cp313。
- 用户提议"独立 venv":`uv venv --python 3.13 /tmp/bpy-poc-venv` + `uv pip install bpy` 装上 `bpy==5.1.1`(380MB)。
- **OPTIX 测试失败**:`OptiX initialization failed with error code 7804`(= `OPTIX_ERROR_FUNCTION_TABLE_SIZE_MISMATCH`,wheel 的 OptiX header 跟系统 driver 560.35.03 OPTIX runtime 函数表不匹配)。fallback 到 CUDA 后,8 张 H20 都枚举到,但 type 全是 `CUDA`。
- **`.blend` 加载成功**:`bpy.ops.wm.open_mainfile` 70ms,Camera1/2/3/4 + 14 对象 + 8 材质 + 7 网格 + 9 图片完整。
- **对照组**:同机器 Infinigen Blender 4.2.0 OPTIX 工作正常(`outputs/blendergym_v3_real/.../turn_0/blender.log` 第 63 行 `compute=OPTIX`,3.35s 完成 16 spp 512²)。
- 报告写到 [`.agents/notes/blendergym_bpy_poc_2026-04-30.md`](../notes/blendergym_bpy_poc_2026-04-30.md)。

**结论**:daemon 路径 B/B' 在当前系统上拿不到 OPTIX,Cycles 只能 CUDA(慢 2-3×),违背 daemon 化目的。**daemon 路径锁定 A**(沿用 Infinigen Blender 二进制 + socket)。

### 3. 50-step 真训诊断(瓶颈翻转)

跑到 step 2 时(过去 ~ 15 分钟)的指标:

| Step | Time | Async Level | Max Off-Policy |
| --- | --- | --- | --- |
| 0 | 394.8 s | 0 | 0(含 5-example 初始 eval 138.9 s) |
| 1 | 199.6 s | 0 | 1 |
| 2 | **182.3 s** | 0 | 1(下降中,稳态 ~150-180 s) |

**`outputs/blendergym_v3_three_kits/run_default/blendergym_work/train`** 124 trajectory / 303 turn 的 Blender duration 实测:

| 状态 | n | p50 | mean | vs baseline p50 |
| --- | --- | --- | --- | --- |
| render_failed | 199 (66%) | 2.40 s | 2.33 s | 2.92 → 2.40 (-18%) |
| ok | 104 (34%) | **4.48 s** | 4.47 s | 6.31 → 4.48 (**-29%**) |

**Avg@1 不退化**:eval at step 0 = `0.3877` vs baseline `0.3932`(降 1.4%,resolution 256 安全)。

**关键反转**:`nvidia-smi` 实测 GPU 2-7 全部 **0% util / 仅 3-5 GB mem**(只是 env_worker + CLIP 常驻),`ps` 整机当下只有 1 个 Blender 子进程在跑。**Blender 不再是瓶颈,vLLM 才是**。vLLM 端实测:

- Running: 25-32 reqs(满 batch),Waiting: 0
- **GPU KV cache usage: 1.9-2.0%**
- Avg generation throughput 总体 ~ 300-400 tokens/s ÷ 32 reqs = **~12 tokens/s/req**
- 单请求 1024 token 生成 ~ 85 s × 3 turn ≈ 255 s,跟 wall 200 s 量级吻合

### 4. deployment / dp / 共卡机制查清

读 [`src/prime_rl/entrypoints/rl.py:117-152`](../../src/prime_rl/entrypoints/rl.py) + [`configs/rl.py:174-191, 764-785`](../../src/prime_rl/configs/rl.py) 后建立完整模型:

- `[deployment] num_infer_gpus = N` → vLLM 占物理 GPU 0..N-1(sequential mapping,不能跳过 GPU 1)。
- `[deployment] num_train_gpus = M` → trainer 占紧接着的 M 张。
- single-node 下 `auto_setup_deployment` 会把 `[inference.parallel] dp = K` **覆盖成** `num_infer_gpus / tp`,且自动设 `api_server_count = dp`。所以**真正决定 dp 的是 `num_infer_gpus`**,不是 toml 里手填的 dp。
- **`check_gpus_available` 只检查 inference + trainer + teacher 占用的卡**,不管 env_worker 用什么卡。**Blender 通过 toml `gpu_id_pool` 跟 vLLM 共卡完全可行**,prime-rl 启动时不会 block。
- `validate_gpu_count` 强制 `num_train + num_infer + num_teacher ≤ gpus_per_node` → 8 卡机器最多 dp=7。

### 5. 决策

用户选择 vLLM/Blender 共卡作为下一步方向。提了三个粒度方案(dp=2 保守 / dp=4 中度 / dp=7 激进),用户表态"下个 session 执行",**没拍最终 dp 值**,把决定带到下一份 session。

## 调试经验

- **PyPI bpy 5.1.1 在 cp313 venv 装得上但 OPTIX 不可用**:error 7804 是 function table size mismatch,wheel 的 OptiX SDK 头版本跟 NVIDIA driver 560.35.03 提供的 OPTIX runtime 不匹配。`.blend` 加载完全 OK。这把 daemon 路径 B/B' 在当前系统下"打到只能跑 CUDA"——比 Infinigen Blender 4.2 OPTIX 慢 2-3×,违背 daemon 化目的。
- **Infinigen 自带 Blender 4.2.0 在同一台机器 OPTIX 工作正常**(从生产 blender.log 推证),所以**问题不是 driver,而是 PyPI bpy 5.1 的 OptiX SDK 版本太新**。要修可能得 downgrade bpy 或 bundle 匹配的 OptiX runtime,不在 daemon plan 范围。
- **三件套真的把 Blender 加速了**(ok p50 6.31→4.48s, fail 2.92→2.40s),但**收益被 vLLM 瓶颈吃掉**——GPU 2-7 整机 0% util,wall step time 反而比 baseline(165s)略高(180s)。这跟前一份 handoff "vLLM 不是瓶颈"的结论**翻转**:那条结论的前提是 Blender 是瓶颈,前提失效后结论失效。
- **KV cache usage 2% 不代表 vLLM 算力空闲**。它只说明"模型小 + batch 32 没把 KV memory 用满",但 vLLM 单请求 latency 主要受两件事决定:(a) prefill compute 跟 prompt 长度成正比、(b) decode 时 batch_size 大 → 单 step 时间长 → 单 req tokens/s 低。这两个都不直接看 KV cache,所以 KV 2% 完全可以同时存在 vLLM 是瓶颈的状态。
- **prime-rl `[inference.parallel] dp = K` 在 single-node 模式被自动覆盖**(rl.py:773-785)。要改 dp 必须改 `[deployment] num_infer_gpus`,不能直接动 toml 里的 dp 字段。
- **prime-rl GPU 物理映射是 sequential**:vLLM 永远占前 `num_infer_gpus` 张,trainer 占紧接着的几张。**不能让 trainer 留在 GPU 1 同时让 vLLM 占 GPU 0,2,3,...**——只能要么调整 num_infer_gpus(trainer 物理位置跟着变),要么不用 prime-rl 的 deployment 系统(自己手动起 inference)。
- **Blender 跟 vLLM 共卡 prime-rl 不会 block**,因为 `check_gpus_available` 不查 env_worker 用的 GPU(它们是 orchestrator 起的子进程,在 prime-rl 的可见性外)。Blender 自己通过 `CUDA_VISIBLE_DEVICES=<gpu_id>` 在 [`render.py:78`](../../environments/blendergym/blendergym/render.py) 指定卡。
- **诊断指标顺序**:`nvidia-smi --query-compute-apps`(看哪个进程在哪张卡)+ `ps | grep blender`(看真正在跑几个 Blender)远比 `nvidia-smi -l` 看 util 直接。当前发现"只有 1 个 Blender 在跑"是用 `ps` 拿到的关键事实。

## 参考代码

| 文件 | 关键位置 | 说明 |
| --- | --- | --- |
| [`environments/blendergym/blendergym/assets/pipeline_render_script.py`](../../environments/blendergym/blendergym/assets/pipeline_render_script.py) | `_enable_gpu_cycles` line 26-90,新增 `BLENDERGYM_RENDER_RESOLUTION` env hook | 三件套之一,resolution env var |
| [`environments/blendergym/blendergym/env.py`](../../environments/blendergym/blendergym/env.py) | `__init__` 第 82-101 行,新增 `cycles_resolution` 参数 | 把 toml `cycles_resolution=256` 透传到子进程 |
| [`environments/blendergym/blendergym/render.py`](../../environments/blendergym/blendergym/render.py) | CLI `_build_argparser` 257-273,`main` 286-291;`_build_subprocess_env:78` `CUDA_VISIBLE_DEVICES` | env_worker 通过这个设定 Blender 单卡 |
| [`configs/multimodal/rl_blendergym.toml`](../../configs/multimodal/rl_blendergym.toml) | 三件套已落地;`[inference] gpu_memory_utilization=0.85`(line 95);`[inference.parallel] dp=1` (line 102-103) | 下一步要加 `[deployment]` block + 改 `gpu_id_pool` 共卡 |
| [`src/prime_rl/entrypoints/rl.py`](../../src/prime_rl/entrypoints/rl.py) | `rl_local` 100-152(GPU mapping);151-152 `check_gpus_available`(只查 train+infer+teacher,**不查 env_worker**) | 关键:**Blender 共卡 vLLM 在 deployment 检查这关不会被拦** |
| [`src/prime_rl/configs/rl.py`](../../src/prime_rl/configs/rl.py) | `SingleNodeDeploymentConfig` 174-191(`num_infer_gpus / num_train_gpus`,total ≤ 8 校验);`auto_setup_deployment` 764-785(**dp = num_infer_gpus / tp 自动覆盖**) | 改 dp 必须改 num_infer_gpus,不能直接改 toml dp |
| [`src/prime_rl/configs/inference.py`](../../src/prime_rl/configs/inference.py) | `ParallelConfig` line 21-50(tp/dp 字段定义);`api_server_count` 自动 = dp | 多 API server 要 cooperate orchestrator 的 client 配置 |
| [`outputs/blendergym_v3_three_kits/`](../../outputs/blendergym_v3_three_kits/) | `logs/orchestrator.log`,`logs/inference.log`,`logs/envs/train/blendergym/env_server.log`,`run_default/blendergym_work/**/trajectory.json` | 三件套真训实证(还在跑,可 kill) |
| [`.agents/notes/blendergym_bpy_poc_2026-04-30.md`](../notes/blendergym_bpy_poc_2026-04-30.md) | 含主 venv 安装失败、独立 cp313 venv 安装 OK 但 OPTIX 7804、.blend 加载 OK 的 stderr 原文 | 路径 A 的实证依据 |

## 最终方案

三件套代码 + toml 改动**已落地**(commit 待用户决定时机),50-step 真训**还在后台跑**(PID 901019,启动 `2026-04-30T10:12:34`)。bpy PoC 完成,daemon 实施路径锁定 A。

但**三件套没拿到 plan §3.5 期望的 wall 砍幅**(预测 30-50s/step 实测 ~180s),原因是瓶颈翻转到 vLLM。下一步**完全脱离原 plan §1/§7 边界**,主动启用 vLLM dp + Blender 共卡(handoff §6 矩阵已预先评估为"可行,推荐")。

`/tmp/bpy-poc-venv` 是隔离 venv,不影响主仓库,可以保留也可以 `rm -rf`。`pyproject.toml` / `uv.lock` 完全没动。

## 下一步任务

**用户原话**:"vLLM/Blender 共卡比较好" + "下个 session 里执行"。

具体焦点:**改 deployment + dp + 共卡**,目标把 wall step time 从 ~180s 砍到 ~ 70-90s/step,不动 max_completion_tokens / max_turns 这种治标变量。

## 初步方案

下个 session 应该按这个顺序展开:

1. **第一件事:停掉当前 50-step 训练**(数据已够,跑完意义不大,占 8 卡 ~ 2.5 h)。

   ```bash
   kill 901019
   # 或者:pkill -f "uv run rl @ configs/multimodal/rl_blendergym.toml"
   ```

   验证 ZMQ + env_worker 子进程都退出干净后再开新 run。

2. **第二件事:先做 dp=2 起步(冒烟验证)**。在没有 known-good dp>1 例子的前提下,直接上 dp=4 风险大。dp=2 改动如下(基于 [`configs/multimodal/rl_blendergym.toml`](../../configs/multimodal/rl_blendergym.toml)):

   ```toml
   output_dir = "outputs/blendergym_v3_dp2_cohost"

   [deployment]
   type = "single_node"
   num_infer_gpus = 2     # vLLM dp=2 自动 -> GPU 0, 1
   num_train_gpus = 1     # trainer -> GPU 2

   [[orchestrator.train.env]]
   id = "blendergym"
   num_workers = 7        # 7 卡共卡 vLLM (0,1) + 独占 (3..7)

   [orchestrator.train.env.args]
   gpu_id_pool = [0, 1, 3, 4, 5, 6, 7]   # 跳过 GPU 2 (trainer)
   work_root = "outputs/blendergym_v3_dp2_cohost/run_default/blendergym_work"
   cycles_resolution = 256
   cycles_samples = 8
   # ...

   [orchestrator.eval.env.args]
   gpu_id_pool = [7]      # eval 5 example,留 1 张独占够
   work_root = "outputs/blendergym_v3_dp2_cohost/run_default/blendergym_work"

   [inference]
   gpu_memory_utilization = 0.30   # 0.85 -> 0.30,腾给 Blender (~53GB / 卡)

   [inference.parallel]
   # dp=1 这行可以删,deployment.num_infer_gpus 会覆盖
   ```

   跑 `max_steps=10` smoke。验证三件事:
   - vLLM 在 prime-rl 下 dp=2 + 多 API server 是否能起来(查 [`logs/inference.log`](../../outputs/blendergym_v3_dp2_cohost/logs/inference.log) 应该看到 `data_parallel_size=2`、`api_server_count=2`)
   - trainer / orchestrator 能不能正常连上多 API server 的端点(看 orchestrator 是否还连 `localhost:8000` 单点,可能需要 router)
   - GPU 0,1 上 vLLM + Blender 共卡 OPTIX 抢占的实际表现(看 [`logs/envs/train/blendergym/env_worker_*.log`](../../outputs/blendergym_v3_dp2_cohost/logs/envs/train/blendergym/) 的 render 时间是否退化,理论 OPTIX context 切换 ~ 100ms 可以接受)

3. **第三件事:dp=2 通过后上 dp=4(中度激进)**。

   ```toml
   [deployment]
   num_infer_gpus = 4     # vLLM 占 GPU 0,1,2,3
   num_train_gpus = 1     # trainer -> GPU 4

   [[orchestrator.train.env]]
   num_workers = 7

   [orchestrator.train.env.args]
   gpu_id_pool = [0, 1, 2, 3, 5, 6, 7]   # 共卡 0-3,独占 5-7
   ```

   预期 wall ~ 70-90s/step,Avg@1 不退化。如果生效,这就是 plan §3.5 目标 30-50s 的最近一次 — 大概率达标。

4. **第四件事:dp=4 OK 后,再决定要不要 dp=7 + 7 卡 vLLM**。dp=7 / 8 是激进版,边际收益越来越小,且 OPTIX 共卡争抢概率上升。建议止步 dp=4 看 wall + Avg@1。

5. **关键风险点**(需要在 PoC 阶段验证):
   - **prime-rl 之前 dp>1 是否真跑过**:扫一下 [`examples/`](../../examples/) 和 [`configs/`](../../configs/) 的 dp>1 toml,如果只有 dp=1 配置在用,意味着 dp=2 是首次,踩坑概率高。
   - **orchestrator 客户端跟 vLLM 多 API server 的连接**:`[orchestrator.client]` 默认是单 base_url,要看 prime-rl 是否自带 routing 把请求散到多个 API server,或者要用 `vllm-router`(`pyproject.toml:134` 已 vendored 但要 `--extra disagg` 安装)。
   - **vLLM gpu_memory_utilization=0.30 是否够 Qwen3.5-0.8B + KV**:0.30 × 96 GB = 29 GB,模型 1.7 GB(实测 [`inference.log:49`](../../outputs/blendergym_v3_three_kits/logs/inference.log) `Model loading took 1.72 GiB`),KV 留 27 GB,远超当前 1.7-2% 实际用量,够。但 multi-modal cache(prefix cache hit ~78%)可能要 ~5 GB,还是够。

6. **跟 plan §1/§7 的边界**:本次方案**主动打破** plan 的"GPU 0 vLLM、GPU 1 trainer"和"不做 vLLM/Blender 共卡"约束。这是用户主动选择,handoff §6 矩阵预先评估过共卡可行。下个 session 第一件事可以是把当前 [`/home/zhiyuan_ma/.cursor/plans/render_three_kits_and_bpy_poc_*.plan.md`](../../../.cursor/plans/) 标记为"已超出范围",新建一份 plan(slug `vllm-dp-cohost`)单独管这次工作,或者直接用 agent mode 跑不开新 plan。

7. **后续(daemon 化暂缓)**:daemon 路径 A 决策已锁定但**优先级再次下调**——如果 dp=4 + 共卡能把 wall 砍到 ~70s/step,daemon 化的边际收益(再砍 ~ 8s/step)可能不再值得 1-2 周工程。下次 session 走完 dp 路径后再判断。
