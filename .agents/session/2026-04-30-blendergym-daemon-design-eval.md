# Session Handoff: BlenderGym Render Daemon 架构评估

## 前置 session

- [BlenderGym Phase A — num_workers Sweep](2026-04-30-blendergym-phase-a-num-workers.md) — Phase A 实测确认 num_workers 加 worker 但不加卡(2→4→8 全在 [6,7] 上)wall time 不降,Phase B 50-step 真训 2h56min、Avg@1 0.1936→0.3932 翻倍,实测 `step_time*` 约 165s。

## 任务目的

不写代码,只做架构评估 + 技术选型预研。承接上次 Phase B 的 165s/step 基线,评估三件事:

1. **GPU 显存 + util 是否真的在浪费**(8 张卡 768GB 实际只用 ~150GB / 20%)。
2. **vLLM/Blender 共卡可行性** — 用户提议把 vLLM 卡也用作 render,因为 H20 96GB 只放 Blender 太浪费。
3. **中心化 render daemon 架构选型**(Blender 二进制 + socket vs PyPI bpy + multiprocessing vs Ray actor)。

产出:plan 文件本身就是评估文档([`/home/zhiyuan_ma/.cursor/plans/render_service_design_doc_c26cf6a9.plan.md`](../../../../.cursor/plans/render_service_design_doc_c26cf6a9.plan.md)),不创建 .agents/notes/ 下的额外文档。

## 执行内容

### 1. 实测 Blender duration 分布(关键事实校准)

扫 [`outputs/blendergym_v3_real/run_default/blendergym_work/`](../../outputs/blendergym_v3_real/run_default/blendergym_work/) 下所有 `trajectory.json` 的 `steps[*].duration_s`,n=4137:

| exit_status | n | 占比 | p50 | mean | 累积 |
| --- | --- | --- | --- | --- | --- |
| **render_failed** | 2936 | **71%** | 2.92 s | 3.22 s | 9465 s |
| **ok** | 1201 | 29% | 6.31 s | 7.55 s | 9065 s |

每 step 96 次 Blender 调用 × 4.48 s/turn = 430 s 累积单线程 Blender 时间,实际并行度 2.6× → wall 165 s。

**最重要的发现**:**71% 是 render_failed**(模型生成代码常 KeyError/NameError),它们 p50 = 2.92 s 几乎全是 cold-start 启动税。daemon 化的真实节约主要落在这 71% 上。

### 2. 提出三轴优化框架

| 轴 | 含义 | 旋钮 |
| --- | --- | --- |
| A. 单次 render 加速 | render_ok 6.31s 怎么压 | A1: resolution 512→256 / A2: cycles_samples 16→8/4 |
| B. 并发数 | 同时几个 render | B1: 扩卡 (gpu_id_pool 2→6) / B2: render daemon |
| C. 显存利用率 | 共卡是否值得 | C1: vLLM+Blender / C2: trainer+Blender |

A 和 C 完全独立;B 内部"扩卡 vs daemon"是替代关系(daemon 是扩卡的超集)。

### 3. 单次 render p50=6.31s 拆解

| 阶段 | 耗时 | 解决轴 |
| --- | --- | --- |
| Blender fork+exec + bpy init | ~ 1 s | B2 (daemon) |
| 加载 .blend | ~ 0.4 s | B2 |
| BVH build | ~ 1.5 s | B2 (同 task 跨 turn 复用 BVH cache) |
| **Cycles raycast (16 spp × 512²)** | **~ 2.5 s** | **A1+A2** |
| OIDN denoise | ~ 0.3 s | 不动 |
| Write PNG | ~ 0.2 s | A1 顺带 |

启动税 ~ 3 s 由 daemon 化解决;raycast 2.5 s 由 A1+A2 解决。**两件事完全独立,可以分开优化**。

### 4. resolution 512→256 的论证(免费午餐)

CLIP `ViT-B-32` 的 `preprocess` 强制 resize 到 224×224(确认在 [`environments/blendergym/blendergym/rubric.py:71-72`](../../environments/blendergym/blendergym/rubric.py)),所以 render 出 512² 还是 256²,CLIP 看到的都是 224² downsample 后的图。**信息损失接近零**,raycast 像素 4×。

代码点 [`environments/blendergym/blendergym/assets/pipeline_render_script.py:26`](../../environments/blendergym/blendergym/assets/pipeline_render_script.py): `resolution` 当前是默认参数 512,**没有 env var hook**,实施时要加一行 `os.environ.get("BLENDERGYM_RENDER_RESOLUTION", "512")`。

### 5. 纠正 vLLM 多卡的误解

用户起初想"每张 vllm 卡都驻 daemon",含混了"vLLM 拓多卡"和"daemon 跟 vLLM 共卡"两件事。澄清:

- vLLM `dp=N` 物理上能做(改一行 toml),但 wall time 不变。当前 KV cache 实测 1.4-2.5%,Waiting=0,**vLLM 不是瓶颈**,加产能没用。
- 真正"8 卡都满"且"训练快"的方案是 daemon 撒到 8 张卡上,vLLM dp=1 即可。
- vLLM 多卡只在显存 KPI 上好看(显存利用率 ~20%→~30%),wall time 等价。

### 6. 共卡可行性矩阵

| 组合 | OOM | SM 抢占 | 错误隔离 | 总体 |
| --- | --- | --- | --- | --- |
| vLLM + Blender (GPU 0) | 中(降 util 0.30) | 低-中(driver 时间分片) | 高 | **可行,推荐** |
| trainer + Blender (GPU 1) | 低 | **高**(trainer fwd/bwd 100% SM) | 高 | 中,需 D7→D8 实验 |
| vLLM + trainer | **高(OOM)** | 高 | 中 | **不推荐** |

vLLM 当前 `gpu_memory_utilization=0.85` 占 81.6 GB pool 但只用 5-7 GB,降到 0.30 释放 53 GB 给 Blender 完全 OK。

### 7. 调研开源 render daemon 参考实现

**没有现成的"Blender daemon for RL"项目**(Blender 主流是离线 batch 不是 online RL),但找到几个核心参考:

| Reference | URL | 借鉴 |
| --- | --- | --- |
| **StackExchange #271096** | [keep-blender-open](https://blender.stackexchange.com/questions/271096/keep-blender-open-after-running-script-from-command-line-for-higher-rendering-th) | 50 行 minimum viable daemon 原型(Blender bg + socket server + Python 主程序 connect) |
| **`bpy` PyPI module** | [bpy on PyPI](https://pypi.org/project/bpy/) + [Blender as Python Module](https://docs.blender.org/api/current/info_advanced_blender_as_bpy.html) | Blender 4.x 官方支持 `pip install bpy` 在普通 Python 进程里 import,daemon 不需要 Blender 二进制!但 OPTIX 支持要 PoC 验证 |
| **DLR-RM/BlenderProc** | [GitHub](https://github.com/DLR-RM/BlenderProc) | 同样是 ML 数据 gen + Blender,用 user code exec 沙箱设计可以借鉴 |
| **princeton-vl/infinigen** | [GitHub](https://github.com/princeton-vl/infinigen) | 我们已经在用其 Blender 二进制(定制版),但其 `manage_jobs` 也是 fork-per-job |
| **Ray actor pool** | [Ray docs](https://docs.ray.io/en/latest/ray-core/actors.html) | Decorator-based actor 模式可以省 ~250 行 dispatcher 代码,但加 Ray 依赖(prime-rl 当前没用) |

### 8. Daemon 实施路径 4 选 1

| 路径 | 用什么 | 估行数 | 风险 |
| --- | --- | --- | --- |
| **A** | Blender 二进制 + socket server(沿用 Infinigen Blender) | 700-900 | 低,基础设施稳定 |
| **B** | PyPI bpy + multiprocessing pool | 400-600 | **PyPI bpy 是否带 OPTIX 支持未知 — 需 PoC** |
| **C** | Ray actor + bpy module | 200-300 | 加 Ray 重依赖 |
| **D** | Ray actor + Blender 二进制 subprocess | 300-400 | 加 Ray 但避开 PyPI bpy 的 OPTIX 风险 |

### 9. 评估内容沉淀

把上面所有讨论以三轴框架整合,沉淀为本 handoff 的依据。**注意**:本次 session 用过一份临时 plan 文件作为评估草稿,但**下次 session 不沿用**——下次会基于本 handoff 新建 plan。本 handoff 是下次 session 唯一需要的输入。

整合的内容(全部已记录在本 handoff 内,具体位置如下):
- 三轴框架(单次 render 加速 / 并发 / 共卡显存)→ §2
- 单次 render 6.31s 拆解 → §3
- 共卡可行性矩阵(vLLM + Blender 推荐;trainer + Blender 慎重)→ §6
- 收益建模摘要:165 → 30 s 拿 80% 由三件套(resolution + samples + async),→ 22 s 由 daemon 化拿剩余 20%(数字来自"调试经验" + "初步方案"测算,未单独成节)
- 开源 reference 调研 → §7
- daemon 实施 4 条路径 A/B/C/D → §8
- 待解决的开放问题 → "初步方案"逐条列出

## 调试经验

- **`nvidia-smi GPU-Util` 在 RL pipeline 里几乎没参考价值**:它是 SM 占用时间比,不是算力利用率。流水线系统大量时间在 CPU/IO/等待,GPU kernel 时间占比天然低,看 nvidia-smi 0% 不代表 GPU 有问题。**正确监控指标是 trainer wait_for_batch / vLLM KV usage / env_server Active tasks / rollout per second**。
- **vLLM 多卡 ≠ 训练快**:vLLM 当前 KV cache 用 1.4-2.5%、Waiting=0,产能 >> 需求。给非瓶颈加资源是 anti-pattern。瓶颈在 Blender,加 vLLM 卡只让大家闲坐。
- **Phase A 没看到加 worker 收益的真实原因不是"加 worker 没用",是"加 worker 但卡数不变"**:OPTIX context 在同一张卡内串行,2 worker / 卡 → 4 worker / 卡 队列深度从 16 变 4 但单 worker 吃同一张 OPTIX,wall time 不变。如果加 worker 同时加卡(独占 1 worker / 卡),理论应该有真并发。这个配置 Phase A 没测过。
- **resolution 512→256 是真免费**:CLIP B-32 反正 resize 到 224,渲染出 512 是浪费 raycast。代码点验证过 [`rubric.py:71-72`](../../environments/blendergym/blendergym/rubric.py)。
- **Blender 调用 71% 是 render_failed**:这个数字之前没意识到,改变了 daemon 化的 ROI 认知。fail 路径 cold-start 占 90%+,daemon 化对 fail 路径节约 ~ 75%(3s→0.5s),比 ok 路径节约多得多。
- **PyPI bpy 把"daemon = Blender 二进制 + stdin loop"简化成"daemon = 普通 Python 进程 + multiprocessing"**,但前提是 PyPI bpy 带 OPTIX 支持。这是 daemon 化最大的未知数,30 分钟 PoC 就能验证。

## 参考代码

| 文件 | 关键位置 | 说明 |
| --- | --- | --- |
| [`environments/blendergym/blendergym/env.py`](../../environments/blendergym/blendergym/env.py) | `add_model_response` 调用 `await asyncio.to_thread(run_blender, ...)` 在 line 372;`_next_gpu()` round-robin 在 line 167 | 需要把 `run_blender` 替换成 `await render_client.render(...)` 的入口 |
| [`environments/blendergym/blendergym/render.py`](../../environments/blendergym/blendergym/render.py) | `run_blender` 主体在 line 86-209;`subprocess.run([blender, --background, blend, --python, script, --, code, out_dir])` 在 line 141-150 | 当前 fork-per-task 实现;daemon 化时改造或保留作 raw-mode fallback |
| [`environments/blendergym/blendergym/assets/pipeline_render_script.py`](../../environments/blendergym/blendergym/assets/pipeline_render_script.py) | `_enable_gpu_cycles` 在 line 26-50,resolution 是默认参数 512;`_exec_user_code` 在 line 94 | Blender 内的 entry point,daemon 化时这部分逻辑要变成 daemon loop body;**resolution 加 env var hook 在这** |
| [`environments/blendergym/blendergym/rubric.py`](../../environments/blendergym/blendergym/rubric.py) | `compute_clip_cosine_similarity` 在 line 55-79;CLIP B-32 preprocess to 224 在 line 71-72 | 证明 resolution 256 不影响 CLIP score 的代码依据 |
| [`environments/blendergym/blendergym/schema.py`](../../environments/blendergym/blendergym/schema.py) | `TurnRecord.duration_s` line 48 + `fill_from_render` line 70-76 | 实测数据来源,duration_s 从 RenderResult 落到 trajectory.json |
| [`configs/multimodal/rl_blendergym.toml`](../../configs/multimodal/rl_blendergym.toml) | `cycles_samples=16` line 58, `gpu_memory_utilization=0.85` line 95, `[inference.parallel] dp=1` line 102-103 | 三个共卡/A 轴优化的旋钮在这里 |
| [`outputs/blendergym_v3_real/run_default/blendergym_work/`](../../outputs/blendergym_v3_real/run_default/blendergym_work/) | `**/trajectory.json` × 4137 turn | duration 实测来源 |
| [`outputs/blendergym_v3_real/logs/inference.log`](../../outputs/blendergym_v3_real/logs/inference.log) | `GPU KV cache usage:` 行 | vLLM 利用率实测,论证"vLLM 不是瓶颈" |

## 最终方案

**没有落地任何代码或配置改动**。本次 session 是 plan mode 设计讨论,产出是本 handoff 文件本身(下次 session 不沿用本次的临时 plan,会基于这份 handoff 重新建 plan)。

评估结论:

1. **三轴框架**:把所有优化映射到 A(单次 render 加速)/ B(并发)/ C(共卡显存)三条独立轴,避免概念混乱。
2. **推荐落地顺序**(按 ROI):resolution 256 + 扩卡 + cycles_samples 8 + max_async_level 2 → 30 分钟工作量能从 165s/step → ~30s/step 拿走 80% 收益。daemon 化是边际优化(再砍 8s/step,1-2 周工程)。共卡是 daemon 化做完后的免费升级(vLLM 共卡几乎无风险,trainer 共卡需实测)。
3. **daemon 化技术选型悬而未决**:4 条路径(A: Blender bin + socket / B: PyPI bpy + mp / C: Ray + bpy / D: Ray + Blender bin)需要先做 30 分钟 PoC 验证 PyPI bpy 是否带 OPTIX,才能确定走哪条。

## 下一步任务

用户明确说:"在下个对话 session 里结合这些 reference 重新讨论一下技术选型"。

具体焦点是 **daemon 化的 4 条实施路径选哪一条**(本 handoff "执行内容 §8" 列出的 A/B/C/D),以及 daemon 化跟"先做炒菜更快(三件套 toml)"的优先级。下次 session 开始时基于本 handoff 直接建新 plan,不要复用本次的临时 plan 文件。

## 初步方案

下次 session 应该按这个顺序展开:

1. **第一件事:30 分钟 PoC 验证 PyPI bpy + OPTIX**(切到 Agent mode 跑):
   ```bash
   uv add bpy
   python -c "import bpy; \
              bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'OPTIX'; \
              bpy.context.preferences.addons['cycles'].preferences.get_devices(); \
              print([d.name for d in bpy.context.preferences.addons['cycles'].preferences.devices])"
   ```
   如果 OPTIX device 列表里有 H20 → **路径 B**(最少代码 ~ 400-600 行);如果只有 CUDA / 没 OPTIX → **路径 A**(沿用 Infinigen Blender 二进制 ~ 700-900 行)。

2. **第二件事:验证 task affinity 命中率上限**(读代码,不跑)。
   - 关键文件:[`src/prime_rl/orchestrator/buffer.py`](../../src/prime_rl/orchestrator/buffer.py) 看 batch 内 rollout 排列。
   - 决定:如果 batch 内同 task 是 chunked(`task1×8 → task2×8 ...`),affinity 命中率 ≈ 7/8;如果 round-robin(`t1, t2, ..., t1, t2, ...`),命中率 ≈ 1/N。
   - 这影响 daemon 化的"BVH cache 复用"实际收益,从而影响是否值得做 affinity 调度。

3. **第三件事:决策"daemon 化先做" vs "先做三件套(resolution + samples + async)"**:
   - 三件套 ROI 极高(30 分钟工作量 → 165→30 s),daemon 化 ROI 中等(1-2 周 → 30→22 s)。
   - 如果用户偏好快速见效:三件套先做,5 步验证后再决定要不要 daemon。
   - 如果用户偏好一次性投入:直接 daemon 化(选定路径 B 或 A),三件套作为后续叠加。

4. **第四件事:如果决定走 daemon,确定最小可行原型规模**:
   - 路径 B 最小原型:1 个 daemon(单 GPU)+ 1 个 worker + raw queue + 100 行代码。先跑通再扩。
   - 路径 A 最小原型:沿用 [StackExchange #271096](https://blender.stackexchange.com/questions/271096/keep-blender-open-after-running-script-from-command-line-for-higher-rendering-th) 50 行 + JSON 协议 + user code exec ≈ 200 行。先跑通再扩。

5. **关键约束**(沿袭前次 session 的边界):
   - 不动 [`src/prime_rl/`](../../src/prime_rl/),只在 [`environments/blendergym/`](../../environments/blendergym/) 内做改造。
   - daemon 化的入口点是替换 [`env.py:add_model_response`](../../environments/blendergym/blendergym/env.py) 里的 `await asyncio.to_thread(run_blender, ...)`,保留 raw fallback 路径。
   - PyPI bpy 替代 Infinigen Blender 之前要确认两件事:(a) OPTIX 支持,(b) Infinigen 有没有定制 patch 影响 .blend 加载。

6. **潜在风险**:
   - PyPI bpy 不带 OPTIX 时,路径 B 直接死,只能回退路径 A 或 D。
   - Infinigen 自带 Blender 是定制版,如果 .blend 文件依赖 Infinigen 特定 features,PyPI bpy 即使带 OPTIX 也读不进 .blend。
   - daemon 化没解决"模型生成代码常错"的本质问题(71% render_failed),只是让失败更快。如果 prompt 修复 / 模型变强,fail rate 下降,daemon 化的累积节约会等比下降。
