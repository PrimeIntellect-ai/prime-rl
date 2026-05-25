# Session Handoff: 9B + dp=6 共卡长训启动

## 前置 session

- [vLLM dp + Blender 共卡 pivot](2026-04-30-vllm-dp-cohost-pivot.md) — 把瓶颈从 Blender 切到 vLLM,选了 dp + cohost 路径,产出 dp=2 起步建议。

## 任务目的

继续推进 vLLM dp + Blender 共卡。这一 session 走完 dp=2 → dp=4 → 9B/dp=6 三阶段验证,把 0.8B 模型升级到 Qwen3.5-9B,把 batch_size 从 32 提到 64,启用 NCCL weight broadcast,在 wandb 上启动 long-run 长训。

## 执行内容

### 1. dp=2 + 0.8B cohost smoke (10 step)

[`configs/multimodal/rl_blendergym.toml`](../../configs/multimodal/rl_blendergym.toml) 改成 `num_infer_gpus=2 + num_train_gpus=1 + gpu_id_pool=[0,1,3,4,5,6,7] + gpu_memory_utilization=0.30`。10 step 全过,平均 step 1-9 = **150.79s**(vs three_kits baseline 180s,-16%)。三个验证点全过:

- vLLM dp=2 + 多 API server 起得来(`api_server_count=2`)
- orchestrator 跟 multi-API server 连得上(vLLM 内置 router 单端口分发)
- GPU 0,1 共卡 vLLM + Blender 没崩

### 2. dp=4 + 0.8B cohost smoke (10 step)

`num_infer_gpus=4 + num_train_gpus=1 + gpu_id_pool=[0,1,2,3,5,6,7] + enable_prefix_caching=true`。10 step 跑完,**median 136.85s vs dp=2 的 157.78s**(-13.3%)。但 raw mean 因 step 6 (+90% outlier) 几乎持平 — 用户提醒后修正,trimmed mean -15.4% 才是真实。

**关键反直觉发现**:Blender ok render `max` dp=2=34.78s vs **dp=4=13.50s**(-61%)。共卡比例从 29% 翻到 57% 但**长尾反而更稳** — 因为 dp=4 单卡 vLLM 负载更轻(batch 8 reqs/engine vs dp=2 的 16 reqs/engine),OPTIX 抢占强度小。

### 3. 查清 prime-rl 已默认启用 DP-rank-aware sticky routing

之前担心"prefix cache 不共享"是错的。代码链路:

- [`src/prime_rl/configs/rl.py:890-901`](../../src/prime_rl/configs/rl.py) `auto_setup_dp_rank_count` → `client.dp_rank_count = inference.parallel.dp`
- [`src/prime_rl/utils/client.py:140-163`](../../src/prime_rl/utils/client.py) 把 base_url 扩成 N 个逻辑 client,各加 `X-data-parallel-rank: 0..N-1`
- [`src/prime_rl/orchestrator/scheduler.py:188-194`](../../src/prime_rl/orchestrator/scheduler.py) `group.pinned_client` — 同 example 的 8 rollout 钉到同 dp engine
- vLLM 0.19 `engine/serving.py:866` 识别 `X-data-parallel-rank` header,送到指定 EngineCore_DP{rank}
- `cache_salt = str(self.ckpt_step)` 提供 prefix cache namespace

→ **同 example 8 rollout + multi-turn 全走同 dp engine,prefix cache 完全复用**。

### 4. WebSearch 验证 Qwen3.5-9B 存在

之前推断"Qwen3.5 系列只发了 0.8B"是错的。Qwen3.5-9B 是 9.65B Dense VLM(Gated Delta Networks 混合架构,32 layers, hidden 4096, BF16 ~ 19 GB)。prime-rl 三层全支持:

- [transformers `models/qwen3_5/`](../../../../../data/zhiyuan_ma/code/prime-rl/.venv/lib/python3.12/site-packages/transformers/models/qwen3_5/) ✓
- [vLLM `models/qwen3_5*.py`](../../../../../data/zhiyuan_ma/code/prime-rl/.venv/lib/python3.12/site-packages/vllm/model_executor/models/) ✓
- prime-rl trainer 走 HF `AutoModelForImageTextToText` 路径(无自定义 trainer 模型),`VLM_REGISTRY` 注册 `qwen3_5` ✓

### 5. 9B + dp=6 + train=2 + 100% cohost smoke (5 step)

`num_infer_gpus=6 + num_train_gpus=2 + gpu_id_pool=[0,1,2,3,4,5] + [trainer.model.ac]`。5 step 全过:

| 维度 | 实测 | 之前估值 |
| --- | --- | --- |
| Step 1-3 mean | **236.5s** | 估 800s(慢 5-8x) |
| 9B vs 0.8B 慢比 | **1.5x** | 估 5-8x |
| trainer FSDP shard=2 显存 | **70 GB / 卡** | 估 100 GB(紧) — `activation_ckpt mode=full` 真省 60% |
| GPU 0-5 共卡 | 33-37 GB,稳 | 担心 OOM |
| **trainer MFU** | **27-29%** | H20 上 9B FSDP + ac + bf16 合理 |
| **vLLM KV concurrency** | **9.66x for 16k tokens** | 0.8B 时是 100x — 终于不富余 |
| **Initial Avg@1** | **0.7709** | vs 0.8B 0.0-0.19 → **4-10x** |

100% 共卡风险全部没出现:Blender 长尾未恶化,无 OOM,无 CUDA error。

### 6. 9B + dp=6 + bs=64 + NCCL + wandb 长训启动

切到长训配置:

- `batch_size 32 → 64`(让 trainer 吃满,bs=32 时 trainer 闲 130s/step)
- `max_steps = 20000`(实际 200-300s/step → 几十天,跑到中途看 wandb 决定)
- `[weight_broadcast] type = "nccl"`(省 filesystem pause 1-3s/step)
- `[wandb] project="blendergym-rl" name="9b-dp6-bs64-cohost"`
- 强制 `max_async_level 2 → 1`(NCCL 跟 async=2 互斥,prime-rl `validate` 报错)

在 screen `train_prime-rl_1` 启动,wandb run [`78beab506445458d84957a3b0fe97b53`](https://wandb.ai/zm2354-ma-the-hong-kong-polytechnic-university/blendergym-rl/runs/78beab506445458d84957a3b0fe97b53)。Step 1 实测 **282s**,**trainer GPU 6,7 util 99-100%**(bs=64 真把 trainer 吃满),vLLM 显存涨到 **47 GB**(KV 真接近用满)。

## 调试经验

- **Raw mean 在小样本 + heavy tail 下严重误导**。dp=4 vs dp=2 的 raw mean 差仅 -1.7%,我据此说"dp=4 没拿到收益"。用户挑战后改用 median(-13.3%) / trimmed mean(-15.4%),才看出 dp=4 在 9 step 中 5 step 显著更快(平均 -23%)、3 step 显著更慢(2 个 outlier),median 才是 dp=4 的真实表现。
- **Blender 100% cohost 比 57% cohost 更稳**(反直觉)。dp=2(29% cohost) ok max=34.78s,dp=4(57% cohost) ok max=13.5s。机制:cohost 比例不是关键变量,**单卡 vLLM 负载强度才是** — dp 越多每卡 vLLM batch 越小,SM 抢占强度越低。dp=6 + 9B + 100% cohost 实测稳态进一步验证。
- **prime-rl `dp_rank_count + X-data-parallel-rank` 是 sticky routing 默认行为**,我之前担心要手动配 vllm-router(`--policy consistent_hash`)是错的 — 那是给 multi-node disagg PD 模式用的。single-node multi-API-server 有 vLLM 内置的 round-robin LB + prime-rl 在 client 端把请求按 dp_rank 钉到指定 engine,两层结合实现 group-aware 路由。
- **NCCL weight broadcast 强制 max_async_level=1**(`src/prime_rl/configs/orchestrator.py:1091`),改 NCCL 时要把 trainer + orchestrator 两边的 max_async_level 都从 2 降到 1。filesystem 兼容 async=2,这是两条路径的真正 trade-off,不只是"backend 选择"。
- **9B 显存比 0.8B 推算更友好**。我估 trainer 单卡 100 GB 紧,实测 70 GB(activation_ckpt mode=full 省 60% activation memory)。9B vs 0.8B step time 仅 1.5x 不是 5-8x,因为 dp=6 + batch 32-64 在 vLLM 端 KV 终于真接近用满,decode throughput 利用率显著提升弥补了模型大。
- **Qwen3.5-9B 自身能力强,初始 Avg@1=0.77** vs 0.8B 0.0-0.19。RL 训练的角色从 "教模型学新任务" 变成 "把已经会的任务再调一调",曲线形状会跟 0.8B 完全不同。
- **bs=64 把 trainer GPU util 从 ~ 36% 涨到 ~ 99%**,这是 batch_size 的真正价值 — 不是为了"加速",而是让 trainer 跟 rollout 时间匹配,单样本秒数从 7.5s 降到 4.4s(-44%)。
- **prime-rl 的 RL 是 GRPO 风格 single-pass SGD**,不是 PPO 的 update_epochs。一个 step = 一次完整 rollout + 一次 SGD update(grad-accum 的 micro-batch 是同一次 SGD 内部的拆分,不是多次更新)。`[trainer.optim]` 没 update_epochs 参数,要做 multi-update 得改源码。

## 参考代码

| 文件 | 关键位置 | 说明 |
| --- | --- | --- |
| [`configs/multimodal/rl_blendergym.toml`](../../configs/multimodal/rl_blendergym.toml) | 全文 | 当前 9B/dp=6/bs=64/NCCL/wandb long-run 配置(本次最终落地) |
| [`src/prime_rl/configs/inference.py`](../../src/prime_rl/configs/inference.py) | `WeightBroadcastConfig:105-108` `Literal["nccl", "filesystem"]` | weight broadcast 只支持 2 个 backend |
| [`src/prime_rl/configs/orchestrator.py`](../../src/prime_rl/configs/orchestrator.py) | `max_inflight_rollouts:956-1131` `validate(NCCL+async)` `:1091` | bs=64 → max_inflight=64 全并发,但 num_workers=6 实际限制 6 个并行;NCCL 校验只允许 async=1 |
| [`src/prime_rl/configs/rl.py`](../../src/prime_rl/configs/rl.py) | `auto_setup_dp_rank_count:890-901` `auto_setup_session_headers:743-747` | dp_rank_count 自动设;X-Session-ID=example_id 自动注入 |
| [`src/prime_rl/utils/client.py`](../../src/prime_rl/utils/client.py) | `setup_clients:140-163` | 把 base_url 扩成 N 个逻辑 client,加 `X-data-parallel-rank` header |
| [`src/prime_rl/orchestrator/scheduler.py`](../../src/prime_rl/orchestrator/scheduler.py) | `_select_least_loaded_client:153-164` `group.pinned_client:188-194` | 同 example 8 rollout 钉到同 client |
| [`src/prime_rl/inference/vllm/server.py`](../../src/prime_rl/inference/vllm/server.py) | `WORKER_EXTENSION_CLS:175-178` | `nccl`/`filesystem` 两个 backend dispatch |
| [`src/prime_rl/trainer/rl/train.py`](../../src/prime_rl/trainer/rl/train.py) | `for micro_step:332-475` `optimizer.step:489` | gradient accumulation 实现 |
| [`src/prime_rl/trainer/rl/packer.py`](../../src/prime_rl/trainer/rl/packer.py) | `token_budget = seq_len × dp_world_size:296` | bs=64 切成 14-24 个 micro-batch,token-budget 驱动 |
| [`src/prime_rl/configs/trainer.py`](../../src/prime_rl/configs/trainer.py) | `ActivationCheckpointConfig:46-76` | `mode=full` 默认 全 layer 重算,9B 显存救命 |
| [`outputs/blendergym_v3_9b_dp6_long/`](../../outputs/blendergym_v3_9b_dp6_long/) | `logs/{trainer,orchestrator,inference}.log` | 当前长训 wandb URL: <https://wandb.ai/zm2354-ma-the-hong-kong-polytechnic-university/blendergym-rl/runs/78beab506445458d84957a3b0fe97b53> |

## 最终方案

**长训配置**(已启动,在 screen `train_prime-rl_1` 中跑):

```toml
output_dir = "outputs/blendergym_v3_9b_dp6_long"
seq_len = 8192
max_steps = 20000

[deployment]
type = "single_node"
num_infer_gpus = 6      # GPU 0-5 vLLM dp=6
num_train_gpus = 2      # GPU 6,7 trainer FSDP shard=2

[model]
name = "Qwen/Qwen3.5-9B"

[weight_broadcast]
type = "nccl"           # 强制 max_async_level=1

[wandb]
project = "blendergym-rl"
name = "9b-dp6-bs64-cohost"

[orchestrator]
batch_size = 64
rollouts_per_example = 8
max_async_level = 1

[[orchestrator.train.env]]
num_workers = 6
gpu_id_pool = [0, 1, 2, 3, 4, 5]   # 100% cohost vLLM

[trainer]
max_async_level = 1

[trainer.model.ac]      # mode=full,9B 显存救命

[inference]
gpu_memory_utilization = 0.30
enable_prefix_caching = true
```

**实测稳态**:Step 1 = 282s,trainer GPU 6,7 util 99-100%,vLLM 47 GB / 卡(KV 真满),Avg@1 起点 0.7700。

**为什么不选别的**:

- dp=4 vs dp=6:dp=4 留 GPU 6,7 给 Blender 独占可缓冲长尾,但实测共卡 100% 长尾没恶化 → dp=6 更优
- num_train_gpus=4 vs 2:9B FSDP shard=2 已经 70 GB/卡 不紧,不需要再 shard;给 trainer 多卡反而要从 vLLM 抢卡(dp 减少) → 反而拖慢 rollout
- bs=64 vs bs=32:bs=32 trainer idle 130s/step,bs=64 trainer 99% util,单样本秒数 -44%
- NCCL vs filesystem:NCCL 节省 ~ 1-3s/step pause,代价是 max_async_level 强制 1。9B 已经够慢,1-3s 影响可见

## 下一步任务

用户原话:"接下来我们可以考虑调整 bs 等参数来平衡 training 和 rollout 的吞吐;以及扩展到除了 object placement 等其他 data。"

两条平行路径:

### A. 调 bs 等参数继续平衡 training/rollout 吞吐

观察当前长训 wandb 数据,根据 trainer/rollout 实际比例继续调整。

### B. 扩展数据集到 object placement 之外的其他任务

BlenderGym 还有 `lighting` / `material` / `geometry` / `blendshape` 等 task type,当前只跑 `placement`。扩展任务类型让 reward 信号多样化。

## 初步方案

### Path A:调 bs / 平衡吞吐

1. **观察当前长训 wandb 实际 step time**(可能跟 smoke 不同):
   - 看 `train/Time` 跟 `orchestrator/Step Time` 差距
   - 看 `train/MFU` 是否稳在 25-30%
   - 看 `eval/Avg@1` 是否随 step 上升(reward 收敛信号)

2. **如果 trainer 时间(~ 220s) 还短于 rollout(~ 280s)**,可继续加 bs:
   - bs=64 → bs=128:trainer 时间翻倍到 ~ 440s,rollout 涨到 ~ 350s(因 vLLM dp=6 满了),trainer 可能反过来变 bottleneck
   - 同时加 lr 0.7-1.4×(根号 2 的尺度,GRPO 影响相对小)

3. **如果 rollout 是 bottleneck**(`Generating rollouts` 进度条占大半 step time):
   - 加 `num_workers` 6 → 12(超过 GPU 数也行,Blender 实际靠 worker pool serial 分发,加 worker 让 trajectory 排队更密集)
   - 但每加 worker 内存吃 ~ 1 GB CLIP rubric → 显存可能撞墙
   - 或者降 `max_completion_tokens` 1024 → 512(治标,但当前 9B 输出长度均值 1180 已经接近,可压可不压)

4. **如果想测 dp=8 + train=0**(trainer 用同一卡的不同进程):**不可行**,prime-rl `num_train_gpus >= 1` 强制。

5. **测 `quantize_in_weight_transfer = true`**:NCCL backend 压缩传输,18 GB 权重 → 10 GB(int8)。但当前 NCCL 已经只 ~ 1s/step pause,优化空间小。

### Path B:扩展任务类型

1. **看当前 BlenderGym dataset 全貌**:
   - [`data/blendergym/`](../../data/blendergym/) 看下面有哪些 task 文件夹(`placement01..50`, `lighting01..N`, ...)
   - 数据来源 [Phase 0 session](2026-04-27-blendergym-phase0-6-integration.md) 提到从 HF Hub `richard-guyunqi/BG_bench_data` 下载 1.96 GB,只 extract 了 placement
   - 完整数据集应该有 5 个 task type:placement / lighting / material / geometry / blendshape

2. **如果其他 task type 数据没下来**:
   - 看 [`environments/blendergym/blendergym/data.py`](../../environments/blendergym/blendergym/data.py) 的 dataset loader 怎么过滤 task_types
   - 重跑 phase 0 提取脚本 把其他 task 解压到 `data/blendergym/`

3. **toml 改 task_types**:
   ```toml
   [orchestrator.train.env.args]
   task_types = ["placement", "lighting", "material"]   # 多任务混合
   ```

4. **CLIP rubric 的兼容性**:每个 task type 的 reward 计算可能不同(placement 看物体位置,lighting 看亮度,material 看纹理)。看 [`environments/blendergym/blendergym/rubric.py`](../../environments/blendergym/blendergym/rubric.py) 是否对不同 task 有差异化 reward 函数,如果是统一 CLIP similarity 那就 task-agnostic 直接能跑。

5. **eval 端**:eval split 同步加其他 task 的 holdout examples,否则 Avg@1 只反映 placement 学习曲线。

6. **数据均衡**:不同 task type 难度差异大(geometry 可能比 placement 难),需要看 reward 分布,可能要 stratified sampling。

7. **优先级**:扩展数据是产出 paper 用的实验,**比 path A 更值得做**;path A 是工程优化,长训曲线本身已经在跑,等结果就行。建议下个 session 先做 path B。

## 操作命令

```bash
# 进 screen 看训练
screen -r train_prime-rl_1

# detach 但保持训练:Ctrl+a, d

# 看持久 log
tail -f /tmp/train_prime-rl_1.log

# 看 step time / Avg@1
grep -E 'Step.*Time|Evaluated' outputs/blendergym_v3_9b_dp6_long/logs/orchestrator.log

# 停训练
pkill -9 -f 'rl @ configs/multimodal/rl_blendergym'
pkill -9 -f 'VLLM::|Verifiers::|EngineCore|EnvWorker|prime_rl|infinigen/blender'

# wandb run
# https://wandb.ai/zm2354-ma-the-hong-kong-polytechnic-university/blendergym-rl/runs/78beab506445458d84957a3b0fe97b53
```
