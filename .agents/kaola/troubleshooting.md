# 踩坑记录

按日期追加，新条目加在最前面。

---

## 2026-05-13 — 自定义镜像导致 PodInitializing 长时间卡住

### 不要用 `--image`，使用 KAOLA 默认 ECR 镜像

**现象**：`koala submit --image "ericzyma/prime-rl-blendergym:v0.0.5"` 提交后，pod 持续停留在 `PodInitializing` 状态（5-10+ 分钟），`koala logs` 报 `container is waiting to start`。集群有空闲 GPU，不是资源问题。

**原因**：自定义镜像托管在 Docker Hub，集群节点需要跨区域拉取大镜像（数 GB）。KAOLA 默认 ECR 镜像（`600627331169.dkr.ecr.us-west-2.amazonaws.com/arcwm/train-aws:cuda12.8-efa1.44-ubuntu24.04-zsh-uvcache`）在同区域 ECR，节点普遍有缓存，秒级拉取。

**解决**：不要指定 `--image`，让 koala 使用默认镜像。所有项目依赖通过 `--sync-code` 同步本地代码 + `setup_kaola.sh` 中 `uv sync` 安装。

---

## 2026-05-13 — 本地代码改动未进入 pod

### 漏了 `--sync-code` 导致 pod 运行镜像内置旧代码

**现象**：本地修改了 `gpu_mem.py` 等文件，但 pod 里的行为没变、新增的日志不出现。

**原因**：`koala submit` 不加 `--sync-code` 时，pod 使用镜像内置的代码。本地改动只存在于 Mac 上。

**解决**：提交时必须加 `--sync-code .:/data/work/prime-rl`，并在 `-c` 命令中 `cd /data/work/prime-rl`（而非镜像内置的 `/code/prime-rl`）。

---

## 2026-05-13 — rsync 无法覆盖 S3 上已有文件

### S3 FUSE mtime 不可靠 + `--inplace` 对已有对象的写入异常

**现象**：新训练 `142027` 启动后，S3 上的 `orchestrator.log`、`env_worker_0.log` 等日志仍保持旧训练 `135718` 的内容（14:10 时间戳）。而新创建的 `trainer/torchrun/` 子目录文件（14:27 时间戳）正常同步。

**原因**：三因素叠加——
1. S3 对象的 mtime 由 FUSE 层模拟，`utimensat()` 不可靠。rsync `-a`（隐含 `-t`）依赖 mtime 判断"未变化"而跳过已有文件。
2. S3 对象不可变，`--inplace` 的 truncate + write 流程在部分 FUSE 实现中对已有文件出错。
3. `2>/dev/null || true` 吞掉所有错误。

新文件不受影响（直接 `creat()` + `write()` + `close()`，FUSE 作为新对象上传）。

**解决**：
1. rsync 改用 `-rlt`（去掉 `-a` 中的 `-o -g`，S3 FUSE 不支持 `chown`）。
2. 每次 sync 前 `rm -rf "${OUTPUT_S3}"` 后全量重建，绕过 `--inplace` 无法覆盖已有文件的问题。
3. `2>/dev/null` 改为写入 `s3_sync.log`，方便排障。
4. 新增 S3 output 存在检查（guard），防止新训练误写旧目录。

**S3 清理注意事项**：Mac 端通过 FUSE `rm -rf` 删除 S3 对象不可靠（可能静默失败）。建议用 `rclone purge` 直接走 S3 API：
```bash
rclone purge "threed-code:arcwm-code-us-west-2/ericzyma/experiments/<EXP_NAME>"
```

**附加发现**：env worker 子进程的 stderr 被 verifiers 框架重定向到日志文件，不出现在 `koala logs` 中。normal pod 不支持 SSH/exec，只能通过 S3 rsync 查看 env worker 日志。

---

## 2026-05-12 — rsync 后台同步到 S3 静默失败

### rsync 默认使用 rename()，S3 FUSE 不支持

**现象**：训练正常运行 25+ step，但 S3 上 `experiments/blendergym-9b-dp6/` 路径完全为空，checkpoint 和 output 均未同步。无任何错误日志。

**原因**：rsync 默认写入流程为「创建临时文件 `.filename.XXXX` → 写入数据 → `rename()` 为正式文件名」。S3 FUSE 不支持 `rename()` 系统调用（返回 `ENOSYS`）。脚本中 `2>/dev/null || true` 吞掉了所有错误，导致每 5 分钟同步全部失败但完全静默。

**解决**：rsync 命令加 `--inplace` 参数，直接写入目标文件，跳过临时文件 + rename 流程。

**教训**：所有写入 S3 FUSE 的工具（rsync、DCP、任何使用 rename/fsync 的程序）都需要绕开 rename。rsync 对应 `--inplace`，DCP 对应写本地盘再同步。关键错误路径不要用 `2>/dev/null || true`，至少保留 stderr 输出到日志文件以便排查。

---

## 2026-05-12 — Step 25 checkpoint 保存崩溃

### S3 FUSE 不支持 os.rename() 导致 DCP 保存失败

**现象**：训练跑到 Step 25（`ckpt.interval=25`），保存 checkpoint 时报 `OSError: [Errno 38] Function not implemented`，traceback 指向 `torch.distributed.checkpoint.filesystem:542` 的 `path.rename()`。

**原因**：`ckpt.output_dir` 指向 `/threed-code/`（S3 FUSE 挂载），PyTorch DCP 在 `finish()` 阶段用 `os.rename(.metadata.tmp, .metadata)` 做原子提交。S3 FUSE 不支持 `rename()` 系统调用（返回 `ENOSYS`）。

**解决**：`ckpt.output_dir` 改为 `/local-ssd/checkpoints/blendergym-9b-dp6`（本地 NVMe 支持全部 POSIX 操作）。`setup_kaola.sh` 新增后台 rsync 进程每 5 min 同步 checkpoint 到 S3 做持久化，EXIT trap 确保退出时做最终同步。

**教训**：S3 FUSE 只适合顺序读/写整个文件，不支持 `rename`、`fsync`、`flock` 等。高频随机 IO 或需要原子操作的目录（checkpoint、wandb）必须放本地盘。

---

## 2026-05-11 — 8 GPU 训练提交

### /threed-code/public_models/ 只读导致模型下载失败

**现象**：`PermissionError: [Errno 1] Operation not permitted: '...models--Qwen--Qwen3.5-9B/blobs/xxx.incomplete'`

**原因**：KAOLA 镜像设 `HF_HOME=/threed-code/public_models/hub`（共享只读挂载），HF 下载器尝试写 `.incomplete` 文件失败。

**解决**：`export HF_HOME=/local-ssd/hf_cache`（写在 setup_kaola.sh 中），模型首次下载到本地盘（~18GB，30s），后续 pod 需重新下载。

### WANDB_API_KEY 缺失导致训练启动失败

**现象**：`wandb.errors.errors.UsageError: No API key configured.`

**原因**：pod 环境没有 WANDB_API_KEY。

**解决**：提交命令中 `export WANDB_API_KEY=xxx`。key 值从 Mac `~/.zshrc` 获取（见 api.md）。

### bash vs source 执行 setup 脚本

**现象**：`setup_kaola.sh` 中的 `export HF_HOME=...` 对后续 `uv run rl` 不生效。

**原因**：`bash scripts/setup_kaola.sh` 在子 shell 运行，export 不回传父 shell。

**解决**：改用 `. scripts/setup_kaola.sh`（source），在当前 shell 执行，所有 export 对后续命令生效。

### koala logs -f 长任务超时

**现象**：OPTIX warm-up 期间 `koala logs -f` 报 `ReadTimeoutError`（30s 超时）。

**原因**：warm-up 约 6 分钟无新日志输出，长连接断开。

**解决**：不用 `-f`，定期 `koala logs <任务名>` 查看。或在 pod 内直接看。

---

## 2026-05-11 — 首次环境搭建验证

### OPTIX 首次编译超时

**现象**：Blender Cycles 渲染卡在 "Loading render kernels (may take a few minutes the first time)"，120s 后被 kill。

**原因**：OPTIX kernel 首次编译在 H200 上需要 ~6 分钟。`blendergym` 默认 `render_timeout_s=120`。

**解决**：`rl_blendergym_kaola.toml` 中设 `render_timeout_s = 600`。`setup_kaola.sh` 默认模式包含 warm-up render 预编译。

### aws CLI 无凭证

**现象**：`aws s3 cp` 报 "Unable to locate credentials"。

**原因**：KAOLA 容器通过 S3 FUSE 挂载提供存储访问，未单独配置 aws credentials。

**解决**：所有 S3 操作改用 FUSE 挂载点 `/threed-code/ericzyma/` 直接读写（`cp`、`cat`、`tar` 管道等）。

### uv 虚拟环境路径

**现象**：`uv pip install -e environments/blendergym` 报 "No virtual environment found"。

**原因**：KAOLA 镜像配置 uv 的 venv 在 `/tmp/uv-venv/`，而非项目内 `.venv`。

**解决**：`uv pip install --python /tmp/uv-venv/bin/python -e environments/blendergym`。运行时用 `uv run` 即可自动使用正确环境。

### OPTIX cache 不受 BLENDER_USER_RESOURCES 影响

**现象**：担心训练时 per-rollout `blender_user/` 目录导致每次重新编译 OPTIX kernel。

**验证**：实测用不同 `BLENDER_USER_RESOURCES` 目录渲染，均 ~2s 完成。

**结论**：OPTIX/CUDA kernel 缓存在 NVIDIA driver 层面（`/root/.nv/`，51MB），与 Blender 用户资源目录无关。无需修改 BlenderGym 代码。

### 数据集实际体积

**现象**：HuggingFace zip 仅 1.9GB，但处理后占 27GB。

**原因**：`blender_files/` 下的 .blend 文件按范围覆盖多个任务（如 `placement_1_5.blend` 覆盖 5 个任务），复制到各任务目录后膨胀约 14 倍。

**影响**：S3 FUSE 上传 27GB 需 ~20 min（后台执行）。`setup_kaola.sh` 训练模式拷贝到本地盘也需数分钟。

### S3 FUSE 大批量文件传输极慢，用 tar 管道替代

**现象**：`cp -r` 27GB（数千文件）到/从 FUSE 挂载点耗时 ~20 min。

**原因**：每个文件一次独立的 S3 API 调用，per-file 延迟 × 文件数。

**解决**：打成单个 tar 文件存 S3，恢复时 `cat xxx.tar | tar xf - -C /local-ssd`。实测 27GB tar 管道恢复仅 **39 秒**（快 30 倍）。

### 训练命令语法

**现象**：`uv run prime --config xxx.toml` 不工作（`prime` 是公司 CLI 工具，不是训练入口）。

**正确用法**：`uv run rl @ configs/multimodal/rl_blendergym_kaola.toml`（`rl` 是训练入口，`@` 是 pydantic_config 的 TOML 加载语法）。

### koala CLI 编码问题

**现象**：`koala list` 报 `UnicodeEncodeError: 'ascii' codec can't encode characters`。

**解决**：`PYTHONIOENCODING=utf-8 koala list`。
