# 踩坑记录

按日期追加，新条目加在最前面。

---

## 2026-05-25 — Orchestrator 启动 2 秒 crash（env_id 模块名冲突）

### verifiers env_id 必须与 importable Python module name 完全一致

**现象**：`uv run rl` 启动后 Orchestrator 2 秒内 exit code 1。`koala logs` 只显示 "Orchestrator failed with exit code 1"，无详细错误。

**原因**：verifiers 的 `load_environment(env_id)` 内部直接用 `importlib.import_module(env_id.replace("-", "_"))` 发现环境包。config 中写 `id = "articraft"`，导致 `import articraft` 导入的是 **articraft SDK 包**（`/data/work/articraft/articraft/__init__.py`），而非 env 包 `articraft_env`。SDK 没有 `load_environment()` 函数 → `AttributeError` → crash。

**根本矛盾**：容器内同时安装了两个包：
- `uv pip install -e /data/work/articraft` → 注册 `articraft` module
- `uv pip install -e environments/articraft` → 注册 `articraft_env` module

**解决**：config 中的 `id` 必须写 **importable module name**，不是 "语义名称"：
```toml
# ❌ 错误：import articraft → 导入 SDK 而非 env
[[orchestrator.train.env]]
id = "articraft"

# ✅ 正确：import articraft_env → 导入 env 包
[[orchestrator.train.env]]
id = "articraft_env"
```

**教训**：
- 新增环境时，env 包的 module name 必须唯一，不能与容器内其他已安装包冲突
- verifiers 不用 entry_points 注册，纯靠 module name 匹配
- 如果看到 "Orchestrator failed with exit code 1" 且无详细错误，从 S3 拉 `orchestrator.log` 查看真实 traceback

### WANDB_MODE=offline 对 shared mode 无效

**现象**：设了 `export WANDB_MODE=offline`，orchestrator 和 trainer 仍然报 `UsageError: No API key configured`。

**原因**：prime-rl 的 `WandbMonitor` 在 shared mode（默认启用）下，代码显式设 `wandb.Settings(mode="shared")`，**完全覆盖**环境变量 `WANDB_MODE`：

```python
# src/prime_rl/utils/monitor/wandb.py L54-56
if shared_mode:
    settings = wandb.Settings(mode="shared", ...)  # 忽略 WANDB_MODE env
```

且 orchestrator 和 trainer 是独立进程，各自调用 `wandb.init()`，所以两个都会 crash。

**解决方案对比**：

| 方案 | 做法 | 适用场景 |
|------|------|---------|
| 禁用 wandb | 注释掉 config 中 `[wandb]` section | Debug 验证 |
| Offline 模式 | `[wandb]` 里加 `offline = true` + `shared = false` | 本地测试 |
| 正式运行 | 传入 `WANDB_API_KEY`（双引号展开） | 正式训练 |

**注意**：单独设 `offline = true` 但保留 `shared = true`（默认）是**无效的**——shared mode 的代码路径根本不检查 `config.offline`。

### 获取 normal pod crash 日志（pod 已清理后）

**现象**：normal pod crash 后被清理，`koala logs` 报 "pods not found"，无法获取错误信息。

**解决**：`setup_kaola.sh` 的 EXIT trap 会在退出时执行 `sync_all`（`aws s3 sync`），将 `/local-ssd/prime-rl-output/` 推到 S3。即使 pod 被清理，S3 上仍有日志：

```bash
# 查看 crash 日志
aws s3 cp "s3://arcwm-code-us-west-2/ericzyma/experiments/<EXP_NAME>/output/logs/orchestrator.log" /tmp/orch.log
cat /tmp/orch.log
```

**进阶**：提交命令中加 `set +e` + 失败后 dump 日志到 stdout（可在 `koala logs` 超时前看到）：
```bash
. scripts/setup_kaola.sh ... && set +e && uv run rl ...; rc=$?; \
if [ $rc -ne 0 ]; then \
  for f in /local-ssd/prime-rl-output/logs/*.log; do \
    [ -f "$f" ] && echo "--- $f ---" && tail -200 "$f"; \
  done; \
fi; exit $rc
```

---

## 2026-05-25 — Articraft 训练提交连环坑

### Guard check 误触发（S3 FUSE 缓存幽灵目录）

**现象**：用 `rclone purge` 在 Mac 删了 S3 上的实验目录，但 normal pod 里 `[ -d /threed-code/.../output/logs ]` 仍然返回 true，setup 脚本 exit 1。

**原因**：容器端 S3 FUSE 有 VFS 缓存，`rclone purge`（走 S3 API）删了对象但 FUSE 缓存没刷新。容器内没有 `rclone rc vfs/refresh` 可用。

**解决**：提交命令加 `--resume` 跳过 guard check。或换一个新的 `EXP_NAME`。

### macOS tar xattr 导致 `set -e` 下 exit code 1

**现象**：Mac 上打的 tar 包在 Linux 容器解压时输出大量 `tar: Ignoring unknown extended header keyword 'LIBARCHIVE.xattr.com.apple.provenance'` 警告，tar 返回 exit code 1（有 warning 时的行为），`set -e` 杀掉整个 setup 脚本。

**原因**：Mac `tar`（基于 libarchive）默认保存 macOS 扩展属性（quarantine、provenance 等）。GNU tar 解压时不认识这些扩展 header，视为 warning 并返回 exit code 1。

**解决**（两端都要做）：
- 打包时：`tar cf ... --no-xattrs --no-mac-metadata`
- 解压时：`tar xf ... --warning=no-unknown-keyword`

### source setup 脚本后 `set -e` 传染主 shell

**现象**：`. scripts/setup_kaola.sh` 成功后，后续 `uv run rl` 报错时 shell 立即退出，训练日志无法上传到 S3 debug。

**原因**：`setup_kaola.sh` 开头 `set -euo pipefail`，source 执行时这些设置**永久作用于调用方 shell**。后续任何命令非零退出都触发 shell 退出。

**影响**：正式训练时无影响（训练报错本来就该退出）。但 debug 时需要 `set +e` 在训练命令前关闭，否则无法捕获错误日志。

**Debug 模板**：
```bash
. scripts/setup_kaola.sh --env articraft --resume && \
set +e && \
uv run rl @ configs/articraft/rl_articraft_kaola.toml > /tmp/train.log 2>&1; \
aws s3 cp /tmp/train.log s3://bucket/user/debug/train.log; \
sleep 60
```

### aws s3 sync 对大量小文件极慢（articraft 92k 文件卡死）

**现象**：`aws s3 sync` 下载 articraft repo（排除 .git/records 后仍有 92k 文件、21MB）到容器，数分钟无输出后 pod 超时被杀。

**原因**：`aws s3 sync` 逐文件 HEAD + GET，92k 文件 = 92k 次 S3 API 调用。加了 `--quiet` 后无输出，koala logs 看起来卡死。

**解决**：打成单个 tar 包存 S3，setup 时 `cat tar | tar xf -` 秒级完成。打包时排除不需要的目录：
```bash
# 打包（Mac 上，一次性）
tar cf articraft-code.tar --no-xattrs --no-mac-metadata \
    --exclude='.git' --exclude='data/records' --exclude='data/cache' \
    --exclude='viewer/web' --exclude='__pycache__' --exclude='.venv' \
    --exclude='tests' -C /path/to/articraft .
aws s3 cp articraft-code.tar s3://bucket/user/data/articraft/articraft-code.tar
```

---

## 2026-05-25 — 后台 S3 sync 4 天内完全失效

### rsync --inplace → S3 FUSE 长期运行静默失败

**现象**：S3 上的日志停在 5/21 19:58（训练开始后 ~5 小时），之后 4 天的日志全部丢失。

**根因**：`setup_kaola.sh` 的 `sync_all` 用 `rsync --inplace → S3 FUSE` 做后台同步。首次 sync 成功后，后续循环中 `rm -rf` S3 FUSE 目录静默失败 → rsync 尝试覆盖已有对象静默失败 → s3_sync.log 累积 15,106 条 "failed to set times: Operation not permitted" 错误但进程照跑不误（`|| true`）。

**修复**：`sync_all` 改用 `aws s3 sync`（直接走 S3 API，不经过 FUSE）：

```bash
aws s3 sync "${OUTPUT_LOCAL}/" "${OUTPUT_S3_BUCKET}/" \
    --delete --exclude 'broadcasts/*' --exclude '*.bin' --quiet
```

S3 API 的 PUT 是原子替换对象，不依赖 rename/mtime/overwrite，长期运行完全可靠。

**教训**：
- `rsync → S3 FUSE` 只适合一次性操作（如 debug 时手动推一次），不适合长期后台 sync
- 后台持久化必须走 S3 API（`aws s3 sync` / `s5cmd sync`）
- 之前 2026-05-12 和 2026-05-13 的 rsync 修复（`--inplace` + `rm -rf`）不够彻底，长期运行仍会失败

---

## 2026-05-25 — Articraft 环境 KAOLA 验证

### multiprocessing spawn 报错（测试脚本无 `__main__` guard）

**现象**：在 debug pod 上用 `uv run python test_script.py` 调用 `compile_urdf_report_maybe_timeout`（`URDF_COMPILE_TIMEOUT_SECONDS=30`），报错：

```
RuntimeError: An attempt has been made to start a new process before the
current process has finished its bootstrapping phase.
```

输出重复 3 遍（fork bomb），compile 结果为 0ms 失败。

**原因**：`compile_urdf_report_maybe_timeout` 内部用 `multiprocessing.Process`（spawn start method）做超时管控。spawn 模式会重新 import 入口模块。如果入口脚本没有 `if __name__ == '__main__'` guard，子进程 import 时再次执行顶层代码 → 递归 spawn。

**解决**：
- 测试脚本加 `if __name__ == '__main__': main()`
- 或设 `URDF_COMPILE_TIMEOUT_SECONDS=0` 禁用 subprocess wrapper（in-process 模式，无 spawn）

**正式训练不受影响**：`rl.py` 入口有 `__main__` guard，且 `env_response` 通过 `asyncio.to_thread` 调用 compile（线程内 spawn subprocess 没有 guard 问题）。

### cadquery 隐式依赖导致部分 record 编译失败

**现象**：dataset filter 已排除 `import cadquery` / `from cadquery` 的 record，但仍有少量 record 编译时报错：

```
RuntimeError: section lofts requires the `cadquery` package, but it is not installed
```

reward = 0.10（RuntimeError 层级）。

**原因**：这些 record 的 `model.py` 不直接 import cadquery，但调用了 SDK 函数（如 `section_lofts()`、`sweep()`），这些函数内部 lazy import cadquery。dataset filter 用正则 `\b(import cadquery|from cadquery)\b` 只能捕获源码级 import，无法检测运行时的间接依赖。

**影响**：约 1-2% 的 record 受影响（6898 条 train set 中大约 50-100 条）。这些 record 编译失败获得低 reward，RL 模型会学到避开这类 SDK 函数，不影响训练稳定性。

**可选改进**（暂不需要）：
- 维护一个 `CADQUERY_DEPENDENT_FUNCTIONS` 黑名单，扫描源码中的函数调用
- 或在编译报错后标记这些 record，下次加载 dataset 时排除

### bm25s 未安装导致 articraft import 链全部失败

**现象**：安装 articraft SDK 后（`uv pip install --no-deps -e articraft`），尝试 `from agent.tools.registry import ToolRegistry` 报：

```
ModuleNotFoundError: No module named 'bm25s'
```

**原因**：articraft 的 import 链较深：
```
agent.tools.__init__.py
  → from agent.tools.find_examples import FindExamplesTool
    → from agent.examples import search_example_documents
      → import bm25s  ← 这里失败
```

`--no-deps` 安装跳过了 articraft 声明的所有依赖，包括 bm25s。

**解决**：`articraft.sh` 的手动依赖列表中必须包含 `bm25s` 和它的依赖 `zstandard`：

```bash
uv pip install --python /tmp/uv-venv/bin/python 'bm25s>=0.3.2.post1' 'zstandard>=0.23.0'
```

**教训**：`--no-deps` 安装后，验证 import 时不能只测 `import sdk`，要测到实际使用的最深层模块（如 `from agent.tools.registry import ToolRegistry`）。`articraft.sh` 的 `setup_ac_verify_imports` 目前只验证 `import sdk`，未来可考虑加深。

---

## 2026-05-14 — Blender worker 120s 超时崩溃（两个独立 bug）

### Bug 1: Blender 内嵌 Python 忽略 PYTHONPATH

**现象**：Render Service 启动后所有 Blender worker 立即崩溃，`wait_ready()` 120s 超时。日志无任何输出（stderr 未重定向）。

**原因**：`worker_loop.py` 通过 `from blendergym.assets.pipeline_render_script import ...` 导入函数，触发 `blendergym/__init__.py`，拉起 `datasets`/`httpx`/`torch` 整条依赖链。Blender 4.2 自带独立 Python 3.11，**忽略 `PYTHONPATH` 环境变量**（`os.environ` 里能看到但 `sys.path` 不包含），且没有这些第三方包。`ModuleNotFoundError` 导致 worker 启动即崩溃。

**解决**：
1. `blendergym/__init__.py` 改为 PEP 562 lazy import（`__getattr__`），不在模块加载时拉起重依赖
2. `worker_loop.py` 顶部 `sys.path.insert(0, ...)` 手动注入包根路径

**关键认知**：Blender `--python <script>` 运行的脚本中，`import` 只能找到 Blender 自带的标准库和 Blender 模块（`bpy` 等）。要导入外部包必须手动操作 `sys.path`，且被导入的包不能有 Blender Python 没有的依赖。

### Bug 2: SIGPIPE 信号杀死 worker

**现象**：Bug 1 修复后，worker 成功初始化 Cycles（日志可见 `cycles samples=8 ...`），但随即变成 zombie 进程。

**原因**：`BlenderPool.wait_ready()` 通过 `connect()` + 立刻 `close()` 探测 socket。Worker `accept()` 后尝试读取已断开连接 → `ConnectionError` → 异常处理器向已关闭 socket 写错误响应 → **SIGPIPE**。Blender 内嵌 Python 不像标准 CPython 那样抑制 SIGPIPE（标准 CPython 在启动时设 `SIG_IGN`），Blender 主进程保持默认 `SIG_DFL`（终止）。

**解决**：`worker_loop.py` 的 `main()` 开头加 `signal.signal(signal.SIGPIPE, signal.SIG_IGN)`。

**关键认知**：在 Blender `--python` 脚本中编写网络服务时，必须手动忽略 SIGPIPE，否则任何对端关闭的 socket 写操作都会杀死 Blender 进程。

---

## 2026-05-14 — S3 实验目录归档（重命名）

### 最小化操作：只需删 `output/logs` 即可绕过 guard，只需 copy `checkpoints` 即可保留进度

**背景**：想把旧实验目录 `blendergym-9b-dp6` 归档为 `blendergym-9b-dp6_2026-05-13`，然后全新开始训练。

**错误做法 1 — 整目录 `rclone copy`**：

```bash
rclone copy threed-code:.../blendergym-9b-dp6 threed-code:.../blendergym-9b-dp6_2026-05-13 --transfers 32
```

卡死。FUSE 挂载会列出 pod 本地盘上的文件（`blendergym-work` 下数万个小文件），但这些文件从未 rsync 到 S3，S3 里不存在。rclone 试图 server-side copy 时大量报 `NoSuchKey`，传输速率跌到 0 B/s，卡在 2%。

**错误做法 2 — 整目录 `rclone purge`**：

```bash
rclone purge threed-code:.../blendergym-9b-dp6
```

运行 8+ 分钟无输出，因为它先要枚举所有对象（含数万个 blendergym-work 小文件），然后批量删除，S3 API 调用次数太多。

**正确做法**：

```bash
# 1. 只归档 checkpoints（实际存在于 S3 的大文件，server-side copy ~9 min）
rclone copy \
    threed-code:arcwm-code-us-west-2/ericzyma/experiments/blendergym-9b-dp6/checkpoints \
    threed-code:arcwm-code-us-west-2/ericzyma/experiments/blendergym-9b-dp6_2026-05-13/checkpoints \
    --progress --transfers 8

# 2. 只删 output/logs（guard 检查的是这一个目录，1 秒完成）
rclone purge threed-code:arcwm-code-us-west-2/ericzyma/experiments/blendergym-9b-dp6/output/logs
```

**关键结论**：
- guard 检查（`setup_kaola.sh` 第 69 行）只判断 `${OUTPUT_S3}/logs` 是否存在，不检查整个实验目录。
- `blendergym-work` 下的文件在 FUSE 挂载上看起来存在，但大多数并未 rsync 到 S3（pod 退出前 rsync 未完成），对 rclone 来说是幽灵文件。
- 实际需要归档的只有 `checkpoints/`（weights + trainer distcp，~136 GB，server-side copy 不走本机带宽）。

---

## 2026-05-14 — `koala submit --sync-code` 也受 Unicode 编码影响

### markdown 文件含中文/箭头字符导致 `UnicodeEncodeError`

**现象**：`koala submit --sync-code .:/data/work/prime-rl ...` 报：

```
UnicodeEncodeError: 'ascii' codec can't encode character '\u2192' in position 48
```

（原因是 `.agents/` 目录下的 markdown 文件含 `→` 等 Unicode 字符。）

**解决**：与 `koala list` 同理，加 `PYTHONIOENCODING=utf-8` 前缀：

```bash
PYTHONIOENCODING=utf-8 koala submit -m normal -g 8 --sync-code .:/data/work/prime-rl ...
```

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

> ⚠️ 此条目的修复方案（`rm -rf` + `rsync --inplace`）后经验证在长期运行下仍会失败。最终方案见 2026-05-25 条目：改用 `aws s3 sync` 直走 S3 API。

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

> ⚠️ 此条目的修复方案（加 `--inplace`）后经验证在长期运行下仍会失败（覆盖已有对象问题）。最终方案见 2026-05-25 条目：改用 `aws s3 sync` 直走 S3 API。

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

**解决**：`PYTHONIOENCODING=utf-8 koala list`。（`koala submit --sync-code` 遇到含 Unicode 字符的文件时同理，见上方 2026-05-14 条目。）
