# Session Handoff: prime-rl 初学者上手与环境运行

## 任务目的
初学者上手 prime-rl，了解项目架构，配置 uv 环境，运行各种 RL 训练示例（VLM、math、math-python）。

## 执行内容
- 介绍了 prime-rl 的整体架构（inference / orchestrator / trainer 三组件异步协作）
- 安装 uv (v0.11.7) 并通过 `uv sync --all-extras` 安装全部依赖（Python 3.12, PyTorch 2.10+cu128, vLLM 0.19.0）
- 成功运行 VLM RL 测试版（`configs/multimodal/rl_color_codeword_test.toml`，3 步）
- 成功运行 VLM RL 正式版（`configs/multimodal/rl_color_codeword.toml`，15 步），Reward 从 0.68 上升到 0.93
- 成功运行 GSM8K math RL（`configs/gsm8k/rl.toml`，5 步，纯文本推理无 Python 工具）
- 尝试运行 math-python RL（`configs/math_python/math_python.toml`），需要 Prime Intellect 云端 sandbox，目前卡在 **PaymentRequiredError**

## 调试经验
- **GPU 占用冲突**：prime-rl 启动时会检查 GPU 上是否有现有进程，需要先 kill 掉；检查命令 `nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader`
- **deep_gemm libcudart.so.13 缺失**：H20 GPU 上 `deep_gemm` 包的 CUDA 13 运行时不兼容，解决方案是 `uv pip uninstall deep-gemm`（H20 不需要 FP8 优化）
- **zero_advantage 过滤器导致崩溃**：当 `rollouts_per_example` 太小时（如 2），模型对简单任务全对或全错，advantage 全为零，被默认 enforce 的 `zero_advantage` 过滤器拦截。解决方案：在 TOML 配置中添加 `[[orchestrator.filters]]` 并设置 `type = "zero_advantage"` + `enforce = false`
- **残留状态导致 "Run evicted by trainer"**：上次失败的 output 目录未清理，新运行会恢复旧的失败状态。解决方案：`rm -rf outputs/<dir>`
- **VLLM_DISABLED_KERNELS 环境变量**：虽然 rl.py 用 `**os.environ` 传递环境变量给子进程，但 vLLM 的 EngineCore 通过 multiprocessing spawn 启动，该环境变量不一定能传递到最内层进程
- **math-python sandbox 只支持云端**：`prime_sandboxes` 库硬编码了 Prime Intellect API（`https://api.primeintellect.ai`），没有本地 Docker 模式
- **疯狂重试导致 429 限速**：sandbox 创建失败后 orchestrator 会无限重试，几万次请求后账号被临时限速

## 参考代码
| 文件 | 关键位置 | 说明 |
|------|---------|------|
| `configs/multimodal/rl_color_codeword.toml` | 全文 | VLM RL 配置，已添加 `enforce=false` 的 zero_advantage 过滤器 |
| `configs/multimodal/rl_color_codeword_test.toml` | 全文 | VLM RL 测试配置，已修改 batch_size/rollouts/tokens |
| `configs/gsm8k/rl.toml` | 全文 | GSM8K 纯文本 math RL 配置 |
| `configs/math_python/math_python.toml` | 全文 | math-python RL 配置，需要 PRIME_API_KEY |
| `src/prime_rl/entrypoints/rl.py` | `rl_local()`, `check_gpus_available()` | RL 入口，启动 inference/orchestrator/trainer 子进程 |
| `.venv/.../prime_sandboxes/sandbox.py` | `AsyncSandboxClient`, `SandboxClient` | sandbox 客户端，通过 Prime API 创建远程容器 |
| `.venv/.../verifiers/envs/python_env.py` | `PythonEnv` | Python REPL 环境，继承自 SandboxEnv |

## 最终方案
- VLM RL 和 GSM8K math RL 均成功运行
- math-python RL 卡在 Prime Intellect sandbox 的 `PaymentRequiredError`，用户已充值但尚未生效

## 下一步任务
排查 Prime Intellect sandbox 的 PaymentRequiredError 问题，使 math-python RL 成功运行。

## 初步方案
- 联网查询 Prime Intellect billing 激活流程，确认是否需要额外操作（如加入 team、验证邮箱、等待到账等）
- 检查 `~/.prime/config.json` 是否需要配置 team_id（sandbox 创建时 `request.team_id` 会从 config 读取）
- 可用单独的测试脚本验证 sandbox 是否可创建：
  ```python
  export PRIME_API_KEY="pit_473d76183e32d4273ec93735ceef18d158472e2cc2785e516f1537e8b4eb9060"
  uv run --no-sync python -c "
  from prime_sandboxes import SandboxClient, CreateSandboxRequest
  from prime_sandboxes.core import APIClient
  client = SandboxClient(APIClient())
  req = CreateSandboxRequest(name='test', docker_image='python:3.11-slim', start_command='echo hello', cpu_cores=1, memory_gb=1, disk_size_gb=1, timeout_minutes=5)
  sb = client.create(req)
  print(f'OK: {sb.id}')
  client.delete(sb.id)
  "
  ```
- 确认 sandbox 可用后，重新运行：
  ```bash
  export PRIME_API_KEY="pit_..."
  rm -rf outputs/math_python_test
  uv run --no-sync rl @ configs/math_python/math_python.toml --max-steps 5 --orchestrator.batch-size 32 --orchestrator.rollouts-per-example 4 --output-dir outputs/math_python_test
  ```
- 注意 `sandbox_client_max_workers` 参数可能需要降低以避免并发过多被限速
