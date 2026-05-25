# BlenderGym 索引

> 快速导航 + 问题查找表。遇到具体问题先查这里。

## 文档导航

| 需求 | 去哪 |
|------|------|
| 新手入门 | `.agents/notes/BlenderGym架构.md` (本目录) |
| 技术选型记录 | `.agents/notes/bpy技术选型.md` (本目录) |
| 部署流程 | `.agents/kaola/workflow.md` |
| Token/凭证管理 | `.agents/kaola/api.md` |
| 4 层隔离调试法 | `.agents/kaola/debugging.md` |
| 7 个已知 bug + 修复 | `.agents/kaola/troubleshooting.md` |
| S3 存储策略 | `.agents/kaola/paths.md` |
| 平台概览 | `.agents/kaola/README.md` |

---

## 问题速查

### 执行阻塞

| 症状 | 解决方案 | 详见 |
|------|---------|------|
| Worker 120s 超时崩溃，无日志 | lazy imports + sys.path injection | troubleshooting.md |
| Worker 变 zombie | `signal.SIGPIPE` → `SIG_IGN` | troubleshooting.md |
| Render 卡在 "Loading kernels" | `render_timeout_s = 600` | troubleshooting.md |
| Pod 卡 `PodInitializing` | 不要用自定义 `--image` | troubleshooting.md |

### 数据 & 存储

| 症状 | 解决方案 | 详见 |
|------|---------|------|
| Checkpoint 保存失败 (OSError 38) | 写 `/local-ssd/`，rsync `--inplace` | troubleshooting.md |
| rsync 到 S3 静默失败 | 加 `--inplace` | troubleshooting.md |
| S3 dataset 传输 20 min | tar 管道（快 30 倍） | troubleshooting.md |
| HF 模型下载失败 | `HF_HOME=/local-ssd/hf_cache` | troubleshooting.md |

### 代码部署

| 症状 | 解决方案 | 详见 |
|------|---------|------|
| 本地改动不生效 | `--sync-code .:/data/work/prime-rl` | troubleshooting.md |
| `UnicodeEncodeError` | `PYTHONIOENCODING=utf-8 koala ...` | troubleshooting.md |

### 实验管理

| 任务 | 正确做法 | 详见 |
|------|---------|------|
| 归档旧实验 | `rclone copy checkpoints/` | troubleshooting.md |
| 安全删除实验 | `rclone purge`（不用 `rm -rf`） | troubleshooting.md |

---

## 关键代码文件

| 文件 | 用途 |
|------|------|
| `environments/blendergym/blendergym/__init__.py` | **危险**: 只能 lazy import |
| `blendergym/render.py` | 高层 API |
| `blendergym/services/render/worker_loop.py` | **危险**: 运行在 Blender 内，有 SIGPIPE + sys.path 修复 |
| `blendergym/services/render/server.py` | FastAPI /render |
| `blendergym/services/render/pool.py` | BlenderPool + socket |
| `blendergym/services/launcher.py` | 入口：启动 render + score |
| `blendergym/services/score/server.py` | FastAPI /score (CLIP) |
| `scripts/setup_kaola.sh` | 总控（必须 source!） |
| `scripts/envs/blendergym.sh` | BlenderGym env 插件 |
| `configs/multimodal/rl_blendergym_kaola.toml` | KAOLA 训练配置 |

---

## 调试入口

| 需要什么 | 用什么 |
|---------|--------|
| 系统化找到故障层 | `.agents/kaola/debugging.md` (4 层隔离法) |
| Blender binary 验证 | Layer 0: `blender --version` |
| worker_loop 单独测试 | Layer 1: `blender --background --python worker_loop.py` |
| Render server 测试 | Layer 3: `curl localhost:8420/health` |
| 日志位置 | debugging.md 内有所有服务日志路径 |

---

## 时间线

| 日期 | 事件 |
|------|------|
| 2026-04-30 | PyPI bpy 评估，决定用 Infinigen Blender |
| 2026-05-11 | 首次 pod 部署 + OPTIX warmup |
| 2026-05-12 | rsync + S3 sync 修复 |
| 2026-05-13 | 代码部署 + 实验管理修复 |
| 2026-05-14 | PYTHONPATH + SIGPIPE bug 修复 |
| 2026-05-20 | 文档整理 |
