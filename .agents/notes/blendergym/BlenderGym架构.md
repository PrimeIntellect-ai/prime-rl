# BlenderGym 架构

> prime-rl 中 BlenderGym 3D RL 环境的完整技术参考。

## TL;DR

| 维度 | 值 |
|------|---|
| **渲染引擎** | Blender 4.2.0 + Cycles (OPTIX on H200) |
| **服务模型** | 持久 daemon（6 workers）+ FastAPI |
| **Python 栈** | venv 3.12 <-> Blender embedded Python 3.11 |
| **渲染速度** | 3.4s/帧 (512x512, Cycles 16spp) |
| **首次渲染** | ~6 min (OPTIX JIT 编译，后续缓存) |
| **Setup 时间** | 10-12 min (冷启动), 3-4 min (热缓存) |
| **存储策略** | Local-SSD + 后台 rsync → S3 |
| **部署平台** | KAOLA 集群 (`setup_kaola.sh`) |

---

## 1. 服务架构

```
Trainer (GPU 6-7, FSDP sharded Qwen3.5-9B)
  ↓ NCCL broadcast
Orchestrator (GPU 0-5)
  ├─ vLLM inference (生成代码)
  └─ BlenderGym env (6 workers)
       ↓ HTTP localhost:8420/render, localhost:8421/score
  ┌──────────────────────────────────┐
  │ Render Service (FastAPI)          │
  │ Score Service (CLIP)              │
  └──────────┬───────────────────────┘
             ↓ Unix socket /tmp/blendergym_*
  ┌──────────────────────────────┐
  │ 6x Blender Workers           │
  │ (Persistent daemon, OPTIX)   │
  └──────────────────────────────┘
```

### GPU 分配 (单节点 8 GPU)

| GPU | 角色 | 说明 |
|-----|------|------|
| 0-5 | Inference + Rendering | vLLM + Blender Cycles workers 共存 |
| 6-7 | Training | FSDP sharding |

### 数据流

1. Trainer → Orchestrator（发起 rollout）
2. Orchestrator → vLLM (GPU 0-5) 生成 Python 代码
3. Orchestrator → Render Service (HTTP) 执行 Blender 渲染
4. Render Service → Blender workers (Unix socket IPC)
5. Orchestrator → Score Service (HTTP) 获取 CLIP 视觉评分

### RL 环境接口

- **Action**: Python 代码片段（LLM 生成）
- **Observation**: 渲染图像 (512x512 RGB) + trajectory metadata
- **Reward**: CLIP-based 视觉相似度分数（目标 vs 当前渲染）

---

## 2. 包结构

```
environments/blendergym/
├── blendergym/
│   ├── __init__.py                    # PEP 562 lazy imports（关键！）
│   ├── render.py                      # 高层 API: run_blender(), PersistentBlender()
│   ├── schema.py                      # Task/trajectory pydantic models
│   ├── trajectory_writer.py           # Trajectory JSON + HTML
│   ├── dataset.py                     # Dataset 加载 + task 枚举
│   ├── services/
│   │   ├── launcher.py                # 入口：启动 render + score servers
│   │   ├── render/
│   │   │   ├── server.py              # FastAPI /render endpoint
│   │   │   ├── persistent_blender.py  # Daemon: Blender 子进程管理
│   │   │   ├── worker_loop.py         # 在 Blender 内运行（--python 脚本）
│   │   │   └── pool.py               # Multi-GPU Blender pool + socket
│   │   └── score/
│   │       ├── server.py              # FastAPI /score endpoint (CLIP)
│   │       └── clip_scorer.py         # CLIP inference
│   └── assets/
│       └── pipeline_render_script.py  # Blender 场景操作代码
├── tests/
│   ├── test_persistent_parity.py      # 验证 daemon == one-shot
│   ├── test_dataset.py
│   └── test_trajectory_writer.py
└── pyproject.toml
```

---

## 3. Blender Python 隔离（核心坑点）

| 问题 | 原因 | 解决 |
|------|------|------|
| `ModuleNotFoundError` | Blender 4.2 embedded Python 3.11 忽略 `$PYTHONPATH` | `sys.path.insert(0, ...)` in `worker_loop.py` |
| 重 import 失败 | `__init__.py` 引入 torch/datasets | PEP 562 lazy `__getattr__` |
| Worker 变 zombie | Blender 不抑制 SIGPIPE | `signal.signal(SIGPIPE, SIG_IGN)` in `main()` |
| OPTIX 7804 错误 | PyPI bpy headers != driver runtime | 用 Infinigen 的 Blender 4.2 binary |

```python
# ❌ 绝对不能在 __init__.py 顶层 import 重依赖
from blendergym.dataset import build_dataset  # 会拉 torch, datasets, httpx

# ✅ lazy import
def __getattr__(name):
    if name == "dataset":
        from blendergym.dataset import build_dataset
        return build_dataset
```

> 以上所有问题均已修复并记录于 `.agents/kaola/troubleshooting.md`。

---

## 4. OPTIX GPU 渲染

- Blender 首次渲染时 JIT 编译 OPTIX kernels → 缓存于 `/root/.nv/ComputeCache/`
- 缓存是 GPU 架构相关的（H200 vs H100 不兼容）
- **首次**: ~6 min 编译；**后续**: 3.4s（命中缓存）
- 预热策略: `setup_kaola.sh` 在训练开始前后台跑 dummy render
- 持久化: tar 存 S3 (`$S3_PREFIX/tools/optix_cache.tar`)，pod 启动时自动恢复

---

## 5. KAOLA 部署

### 5.1 Setup 脚本

| 脚本 | 作用 |
|------|------|
| `scripts/setup_kaola.sh` | 总控（必须 source 执行！）|
| `scripts/envs/blendergym.sh` | BlenderGym 插件：`env_setup()` |

`blendergym.sh` 执行步骤:
1. 安装系统库 (libegl1)
2. 恢复 Blender binary (tar → `/local-ssd/blender-4.2.0-linux-x64/`)
3. 恢复 dataset (tar → `/local-ssd/blendergym/`) 或 `--fast` 模式下 symlink
4. 安装 blendergym 包 (`uv pip install -e environments/blendergym`)
5. OPTIX warm-up (后台渲染或 tar 恢复)
6. 启动服务 (`blendergym.services.launcher`)

### 5.2 环境变量

```bash
# 用户必须提供（Mac ~/.zshrc）
export HF_TOKEN="hf_..."
export WANDB_API_KEY="wandb..."

# 用户必须设置（setup 前）
export EXP_NAME="blendergym-9b-dp6"  # 无默认值！

# 可选（有默认）
export HF_MODEL="Qwen/Qwen3.5-9B"
export HF_HOME="/local-ssd/hf_cache"  # setup_kaola.sh 自动设

# 脚本自动设置
export GPU_POOL="0,1,2,3,4,5"
export BLENDER_BIN="..."
export BLENDER_USER_RESOURCES="..."   # per-worker 临时目录
export CUDA_VISIBLE_DEVICES="N"       # per-worker GPU 限制
```

### 5.3 存储策略

```
Local-SSD (/local-ssd/)                    — 所有高频 IO
├── blender-4.2.0-linux-x64/               ← S3 tar 恢复
├── blendergym/                            ← Dataset (27GB, tar 恢复)
├── hf_cache/                              ← HF 模型 (18GB)
├── prime-rl-output/                       ← 训练输出 + logs
├── checkpoints/blendergym-9b-dp6/         ← FSDP weights
└── warmup-render/                         ← OPTIX 编译临时

S3 (/threed-code/ericzyma/)                — 持久化（后台 rsync 每 5 min）
└── experiments/blendergym-9b-dp6/
    ├── checkpoints/
    └── output/
```

> S3 FUSE 不支持 `rename()`，所以 checkpoint 必须写本地盘，用 `rsync --inplace` 同步。

### 5.4 训练配置

```toml
# configs/multimodal/rl_blendergym_kaola.toml
output_dir = "/local-ssd/prime-rl-output"
ckpt.output_dir = "/local-ssd/checkpoints/blendergym-9b-dp6"
data_root = "/local-ssd/blendergym"
render_service_url = "http://localhost:8420"
score_service_url = "http://localhost:8421"
render_timeout_s = 600
```

---

## 6. 常用命令

### 正式训练 (KAOLA)

```bash
export EXP_NAME=blendergym-9b-dp6
koala submit -m normal -g 8 --sync-code .:/data/work/prime-rl \
  -c "export HF_TOKEN=$HF_TOKEN && export WANDB_API_KEY=$WANDB_API_KEY && \
      export EXP_NAME=$EXP_NAME && cd /data/work/prime-rl && \
      . scripts/setup_kaola.sh --env blendergym && \
      uv run rl @ configs/multimodal/rl_blendergym_kaola.toml \
        --ckpt.output_dir /local-ssd/checkpoints/\${EXP_NAME}"
```

### Debug Pod (交互)

```bash
koala submit --sync-code .:/data/work/prime-rl
ssh <pod-name>
cd /data/work/prime-rl
export EXP_NAME=blendergym-9b-dp6
. scripts/setup_kaola.sh --fast  # ~1 min
uv run rl @ configs/multimodal/rl_blendergym_kaola.toml
```

---

## 7. 调试方法论（4 层隔离）

| Layer | 测试目标 | 命令 |
|-------|---------|------|
| 0 | Blender binary | `/local-ssd/blender-.../blender --version` |
| 1 | worker_loop.py | `timeout 30 blender --background --python worker_loop.py` |
| 2 | Popen wrapper | `persistent_blender.py` 最小测试 |
| 3 | Render Server | `uv run python -m blendergym.services.render.server --port 8420 ...` |
| 4 | 完整服务链 | via `launcher.py` |

> 详见 `.agents/kaola/debugging.md`。

---

## 8. 性能指标

| 阶段 | 耗时 | 说明 |
|------|------|------|
| Pod 启动 + setup | 10-12 min | 冷启动，OPTIX JIT 主导 |
| 模型加载 (vLLM) | 3-5 min | 6 GPU ramp up |
| 首次渲染 (OPTIX 编译) | 6 min | 1 GPU @ 100% |
| 后续渲染 | 3.4s | 缓存命中 |
| CLIP 评分 | 100-200ms | per-image |
| 训练 step (bs=64) | 30-60s | 2 GPU train + 6 GPU infer |

---

## 9. 架构决策

| 决策 | 选项 | 理由 |
|------|------|------|
| 持久 daemon vs fork-per-render | Daemon | 启动快、OPTIX 缓存复用 |
| Local-SSD + rsync vs 直写 S3 | Local-SSD | S3 FUSE 不支持原子操作 |
| Infinigen Blender vs PyPI bpy | Infinigen | OPTIX 支持 + 验证过的兼容性 |
| 6 worker 共存 inference GPU | 共存 | GPU 利用率最大化，渲染与推理交替 |

---

## 10. 常见错误 Checklist

- [ ] `--sync-code .:/data/work/prime-rl` 必加
- [ ] 不要自定义 `--image`（用默认 ECR）
- [ ] checkpoint 路径指向 `/local-ssd/`，不是 S3
- [ ] `. scripts/setup_kaola.sh`（source 执行！）
- [ ] `EXP_NAME` 必须 export
- [ ] `render_timeout_s = 600`（OPTIX 首次 6 min）
- [ ] `$HF_TOKEN` / `$WANDB_API_KEY` 用双引号（确保 shell 展开）
