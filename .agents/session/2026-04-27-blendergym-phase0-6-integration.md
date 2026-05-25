# Session Handoff: BlenderGym Phase 0–6 端到端集成

## 前置 session
- [BlenderGym 环境架构决策](2026-04-25-blendergym-architecture-decisions.md)
- [Verifiers 环境全面精读与知识更新](2026-04-23-verifiers-env-full-survey.md)
- [VLM RL 探索与 VIGA 集成规划](2026-04-22-vlm-rl-exploration.md)

## 任务目的
按 `~/.cursor/plans/blendergym_env_integration_5f945a23.plan.md` 推进 Phase 0–6，把 BlenderGym 作为独立 verifiers env 包接入 prime-rl，第一版只做 placement，跑通 `uv run rl --bench` 4 步 benchmark 收口。

## 执行内容
- **Phase 0**：从 HF Hub `richard-guyunqi/BG_bench_data` 下 1.96 GB zip，提取 50 个 placement 任务到 `data/blendergym/`；GPU 6 跑 Blender 4.2 smoke 出 render1.png；验证 `Qwen/Qwen3.5-0.8B` 是 VLM (model_type=qwen3_5)；`chmod -R a-w data/blendergym/`；`.gitignore` 加 `data/blendergym/` + `_reference_codes/`。
- **Phase 1**：建独立包 `environments/blendergym/`（pyproject.toml + 包目录），`uv pip install -e environments/blendergym` 装上，`vf.load_environment("blendergym")` 工作。
- **Phase 2**：`dataset.py` 扫 placement 目录构 HF Dataset（45 train + 5 eval，info 7 字段全绝对路径）；`tests/test_dataset.py` 8 个测试；env.py 用 lambda 接入 lazy build。
- **Phase 3**：`pipeline_render_script.py`（VIGA 简化版只 Camera1）+ `render.py`（`RenderResult` dataclass + `run_blender` + `python -m blendergym.render` CLI）；3 张渲染 vs VIGA 数据集自带参考图 PSNR 45 dB（视觉等价）；失败路径包成 `RenderResult(success=False)`，`success` 靠 `image_path.is_file()` 而非 returncode（Blender 在 Python 异常时常常仍返回 0）。
- **Phase 4**：`rubric.py`（`BlenderGymRubric`：lazy CLIP `cuda:{state["gpu_id"]}` 路由、`force_quick_gelu=(pretrained=='openai')` 对齐 OpenAI baseline；reward = clip cosine；3 个 metric）；1 个纯函数 cosine pytest（同色 vs 不同色）。
- **Phase 5**：`prompts.py` + `env.py` 完整重写：parser 从 `XMLParser(["think","code"])` 改成 `XMLParser(["code"])`（去掉 think 避开 Qwen3 chat template 自动剥）；`setup_state`（work_root/trajectory_id 子目录、b64 图、round-robin gpu_id）；`get_prompt_messages`（第 0 轮 5 块 / 第 N 轮 prev_prompt+prev_completion+新渲染）；`add_model_response` 覆盖（`super()` 后 parse `<code>` → `asyncio.to_thread(run_blender)` → 写 state 三件套）；`@vf.cleanup` 4 场景 matrix；`configs/multimodal/inference_blendergym.toml` + 起 vLLM；`vf-eval` 第一次全 reward=0（模型抄 system prompt 里 markdown ` ```python ``` `），改 prompt 用裸 `<code>/</code>` 字面 + INITIAL program 包成 `<initial_code>` 后修复；20 rollout variance check **reward 0.489 std 0.489 / pass@4=1.0 / pass^4=0**。
- **Phase 6**：`rl_blendergym.toml`（seq_len=8192、max_completion_tokens=1024、blendergym train + eval env、gpu_id_pool=[6,7]、work_root 与 output_dir 同步）；`uv run rl @ ... --bench` 4 步全 SUCCESS（reward 0.14→0.25→0.18→0.24，每步 ~190 s，trainer fake-data MFU 18.5%，端到端 15 min，blendergym_work=49 MB ≪ 1 GB，env_worker logs 无 ERROR）。

## 调试经验
- **vf-eval 第一次 reward 全 0 是 prompt bug，不是 env bug**：模型把 system prompt 里 ` ```python ... ``` ` markdown fence 当成"copy this template"指令，每轮吐 fence 而不是 `<code>`。Plan §5b 设计 reward variance sanity check 就是为了捕获这种隐性失败 — 不修复硬上 RL 等于浪费 GPU。修复点：去掉 markdown fence、用裸 `<code>/</code>` + 真物体名 few-shot 示例。
- **`<initial_code>` 字面包含字符串"code"但不会被 `XMLParser(["code"])` 误匹配**：regex 是 `<code>\s*(.*?)\s*</code>` 字面相邻，`<initial_code>` 中夹了 `initial_` 不构成 `<code>` 子串。验证后安全。
- **Blender background 模式 Python 异常仍可能 returncode=0**：`success` 必须靠 `image_path.is_file()` 判断，不能信任 returncode。
- **CLIP device 必须显式 `cuda:{state["gpu_id"]}`**：env worker 主进程没有 `CUDA_VISIBLE_DEVICES` 隔离，裸 `device="cuda"` 会落到 cuda:0（vLLM 占用），与 Blender subprocess 的 GPU 也不是同一个。
- **`python -m blendergym.render` 触发 RuntimeWarning**：`__init__.py` eager `from .render import ...` 让 `blendergym.render` 提前进 `sys.modules`，runpy 再 `__main__` 执行时报 sys.modules 冲突。最干净的办法是把 `RenderResult/run_blender` 从 `__init__.py` 顶层 export 移除，让用户 `from blendergym.render import ...` 直接导子模块。
- **bench mode 下 trainer 用 fake data，不消费 orchestrator rollouts**：所以 4 步 reward 是震荡的（0.14→0.25→0.18→0.24），不是单调上升 — 模型权重根本没更新。真训练（去掉 `--bench`）才会看 reward 收敛。
- **prime-rl 不自动注入 `output_dir` 到 env args**：`work_root` 必须在 TOML 里写两遍（顶层 + `[[orchestrator.train.env]].args.work_root`）。每次新 run 都走一遍 plan 的 new-run checklist。
- **H20 不在 prime-rl peak FLOPS 表里**，trainer fallback 到 A100 的 312 TFLOPS，wandb 上 MFU/throughput 数值偏低、参考意义打折。

## 参考代码

### `environments/blendergym/`（独立 verifiers env 包）
| 文件 | 关键位置 | 说明 |
|------|---------|------|
| `pyproject.toml` | name="blendergym", deps: verifiers + pillow + datasets + open_clip_torch + torch | uv pip install -e 入口 |
| `blendergym/__init__.py` | exports `BlenderGymEnv` / `BlenderGymRubric` / `build_dataset` / `load_environment` | 子模块 `render` 不在顶层 export 避 `python -m` 冲突 |
| `blendergym/dataset.py` | `build_dataset(data_root, task_types, split, eval_holdout)` | 返回 HF Dataset（prompt=[], answer="", info 含 task_id/start_code/init_image_path/goal_image_path/blend_file_path） |
| `blendergym/render.py` | `RenderResult` dataclass + `run_blender(blend_file, code, output_dir, *, blender_bin, render_script, gpu_id, timeout=120)` | subprocess 调 Blender；`CUDA_VISIBLE_DEVICES` + `BLENDER_USER_RESOURCES` 隔离；blender.log 全量落盘；`python -m blendergym.render` CLI |
| `blendergym/assets/pipeline_render_script.py` | Blender 内部 background 脚本，仅渲 Camera1 | VIGA 简化版（去掉 Camera2 分支） |
| `blendergym/rubric.py` | `BlenderGymRubric` + `compute_clip_cosine_similarity(image_a, image_b, *, model, preprocess, device)` | lazy CLIP，`cuda:{state["gpu_id"]}` 路由；3 个 metric：xml_parse_success / render_success / code_non_empty |
| `blendergym/prompts.py` | `SYSTEM_PROMPT` / `TASK_INSTRUCTION` / `REFINE_INSTRUCTION` | 裸 `<code>/</code>` 字面，不要 markdown fence |
| `blendergym/env.py` | `BlenderGymEnv(MultiTurnEnv)` + `load_environment(**kwargs)` | `setup_state` / `get_prompt_messages` / `env_response=[]` / `add_model_response` 覆盖 / `@vf.cleanup cleanup_work_dir` |
| `tests/test_dataset.py` | 8 测试 | placement ≥1 / info 字段完整 / `goal_image_path.suffix == ".png"` / 路径绝对 / split 分区 / 异常 |
| `tests/test_rubric.py` | 1 slow CLIP 测试 | 同色 vs 不同色 cosine 相似度断言（CPU CLIP，~10s） |

### prime-rl 侧
| 文件 | 说明 |
|------|------|
| `configs/multimodal/inference_blendergym.toml` | 独立 inference server 配置（debug 用，phase 6 之后由 `uv run rl` 自动起） |
| `configs/multimodal/rl_blendergym.toml` | RL 训练主配置：output_dir、seq_len=8192、blendergym train+eval env、gpu_id_pool=[6,7]、work_root 与 output_dir 同步 |
| `src/prime_rl/utils/vlm.py:VLM_REGISTRY` | `qwen3_5` 已支持，无需改 |
| `src/prime_rl/configs/orchestrator.py:EnvConfig/TrainEnvConfig/EvalEnvConfig` | env 接受的 args dict 接口；`max_seq_len` 自动注入 `extra_env_kwargs` |
| `src/prime_rl/configs/rl.py:RLConfig.bench` + `auto_setup_bench` | 顶层 `--bench` flag 自动设 trainer fake data + max_steps=4 |
| `src/prime_rl/entrypoints/rl.py:rl_local` | GPU 分配规则：infer=local 0 → physical N0；trainer=local N0+1 → physical N1；env worker GPU 由 env_args.gpu_id_pool 独立指定 |

### 数据 + 二进制（gitignored）
| 路径 | 说明 |
|------|------|
| `data/blendergym/placement{1..50}/` | `blender_file.blend` + `start.py` + `goal.py` + `renders/{start,goal}/render{1..4}.png`，全部只读 |
| `_reference_codes/VIGA/utils/third_party/infinigen/blender/blender` | Blender 4.2.0 二进制 |
| `outputs/blendergym_v1/` | bench 产物：`configs/`（resolved subconfigs）+ `logs/{trainer,orchestrator,inference,envs/train/blendergym/env_worker_*}.log` + `blendergym_work/{trajectory_id}/{turn_N/{code.py,render1.png,blender.log}}` + `run_default/`（trainer wandb 子目录） |

### 知识文档
| 文件 | 说明 |
|------|------|
| `~/.cursor/plans/blendergym_env_integration_5f945a23.plan.md` | 6 个 phase 全 status: completed，每个 phase content 末尾内联记录实际指标 / 偏差 / gotcha |
| `.agents/knowledge/env/verifiers_env_templates.md` | verifiers 环境模板速查表（构建当前 env 时反复参考） |

## 最终方案

### 包结构选择：独立 pip-install 包 vs 内嵌 src/
独立包 `environments/blendergym/`，对外只暴露 `load_environment(**kwargs)`。prime-rl 端只在 TOML 里加 `id = "blendergym"`，不改主框架代码 — 与 color_codeword、prime-rl 整套 envs 一致。后续若要发布到 primeintellect index 也直接 `uv pip publish`。

### Parser：`XMLParser(["code"])` 而非 `["think","code"]`
plan 早期写的 `["think","code"]` 与 Qwen3 chat template 自动剥 `<think>...</think>` 冲突，会触发解析失败。最终用 `["code"]`，让 reasoning 走 chat template，答案部分必须是 `<code>...</code>`。System prompt 显式要求"do NOT use markdown fences"。

### CLIP 设备路由：`cuda:{state["gpu_id"]}` 显式而非裸 `cuda`
env worker 主进程不做 `CUDA_VISIBLE_DEVICES` 隔离（继承 orchestrator），裸 `device="cuda"` 会落到 cuda:0（vLLM 在用）。强制走 state["gpu_id"] = round-robin 从 `gpu_id_pool` 选出。

### Reward：原始 CLIP cosine 不做 rescale
plan §5b 验证显示 reward 在 0/0.97 二元分布，advantage std ≈ 0.49，PPO 梯度方向极清晰，第一版不启用 `(clip_sim − 0.7) / 0.3` fallback。

### Disk 策略：`keep_failed_only=False` 默认
4 步 bench 49 MB；plan 估算 100 步 ≈ 18 GB，500 步 ≈ 90 GB。第一版默认全保留方便 debug，超 100 步真训练再切 `True`（成功 rollout 删、失败保留）。

## 下一步任务

启动**真 RL 训练**（去掉 `--bench`），跑 50–100 步，看 reward 是否能从 baseline 0.49 上升、`render_success` 从 0.5 上升、`zero_advantage_filter` 命中率下降。

## 初步方案

1. **Pre-train cleanup（10 分钟）**：把 `blendergym/env.py` 的 `get_prompt_messages` 返回值从 raw `dict` 改成 `vf.UserMessage`/`vf.AssistantMessage` 等 Pydantic 类型，消除 verifiers 反复 `normalize_messages` 性能 hint。这条不影响正确性，但真训练 100+ 步累积影响明显。

2. **拷一份 long-run TOML**（5 分钟）：`configs/multimodal/rl_blendergym.toml` → `rl_blendergym_v1.toml`：
   - `output_dir` 改成 `outputs/blendergym_v1_train`（**同步改 `args.work_root` + `[[orchestrator.eval.env]].args.work_root`**）
   - `max_steps = 50`（先 50 步看曲线，必要时增量恢复到 100/200）
   - 看跑了第 ~30 步的 disk 占用决定要不要切 `keep_failed_only=true`
   - 加 `[wandb]` 段，让 reward / loss / render_success / zero_advantage_ratio 都进 wandb（默认 shared 模式）

3. **跑真训练**（~3 小时 / 50 步）：
   ```
   uv run rl @ configs/multimodal/rl_blendergym_v1.toml
   ```
   单 step 预算 = 32 rollouts × ~80 s/rollout / async-level（实测 step 1 后 ~190 s/step）+ trainer ~30 s = ~3.5 min/step。50 步 ≈ 3 h。

4. **监控曲线（关键）**：wandb 上盯
   - `train/blendergym/clip_similarity`：mean 应该从 0.489 上升
   - `train/blendergym/render_success`：从 0.5 上升（plan §5b 标定的诊断曲线）
   - `train/blendergym/xml_parse_success`：应稳定在 1.0（如果掉到 < 0.9，说明模型 overfit / collapse）
   - `train/zero_advantage_ratio`：应从 ~50% 下降
   - `trainer/loss` + `trainer/grad_norm`：grad_norm 持续 > 5 = 用 lr=3e-6 太大；< 0.1 = 学不动

5. **风险清单（按概率排序）**：
   - **disk 爆**：100 步 18 GB 还行，但若中途 SIGINT 死掉，下次启动 `clean_output_dir=False` 不会清，要手动 `rm -rf outputs/blendergym_v1_train`。
   - **reward collapse**：模型输出格式漂掉、`<code>` 退化、render_success 跌 — 立即停下回看 wandb 上 reward step 曲线。
   - **GPU 6/7 OOM**：H20 96 GB，CLIP 1.7 GB + Blender Cycles peak 11 GB + parallel rollout 多个 → 单 worker 同时跑 ≥ 4 个 Blender 可能撞内存。如果 env_worker logs 出现 `out of memory`，把 `num_workers` 降到 1 + `gpu_id_pool=[7]` 单 GPU。
   - **vLLM weight broadcast 失败**：plan TOML 默认 `weight_broadcast.type = "filesystem"`，靠 disk IO 传权重；如果 trainer step 完成后 inference 没看到新权重，先看 `outputs/blendergym_v1_train/weights/` 是否在更新。

6. **第一次真训练复盘后才动的可选项**（不要提前优化）：
   - 跑通 100+ 步后切 `keep_failed_only=true`
   - 如果 advantage std 实际偏小再加 `(clip_sim − 0.7) / 0.3` rescale
   - 扩展 `task_types=["placement", "geometry"]` 或更多
   - vf.UserMessage / vf.AssistantMessage 类型替换 raw dict（如果 step 1 没出 normalize_messages 性能问题，可以 defer）
