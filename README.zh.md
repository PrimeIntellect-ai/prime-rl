<p align="center">
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/40c36e38-c5bd-4c5a-9cb3-f7b902cd155d#gh-light-mode-only" alt="Prime Intellect" width="312">
  <img src="https://github.com/user-attachments/assets/6414bc9b-126b-41ca-9307-9e982430cde8#gh-dark-mode-only"  alt="Prime Intellect" width="312">
</p>

---

<h3 align="center">
prime-rl：面向大规模集群的异步强化学习训练框架
</h3>

---

</br>
<p align="center">
  <a href="https://github.com/PrimeIntellect-ai/prime-rl/actions/workflows/style.yml">
    <img src="https://github.com/PrimeIntellect-ai/prime-rl/actions/workflows/style.yml/badge.svg" alt="Style" />
  </a>
  <a href="https://github.com/PrimeIntellect-ai/prime-rl/actions/workflows/tests.yml">
    <img src="https://github.com/PrimeIntellect-ai/prime-rl/actions/workflows/tests.yml/badge.svg" alt="Tests" />
  </a>
</p>

<p align="center">
  <a href="README.md">English</a> | <b>简体中文</b>
</p>

## 项目概览 (Overview)

**prime-rl** 是一个专为大规模强化学习（Reinforcement Learning, RL）打造的训练框架。它在设计上兼顾了易用性与极高的代码可塑性（Hackable），同时能够平滑扩展至 **1,000+ 张 GPU** 的大规模集群。以下是使它脱颖而出的核心优势：

1. **全异步强化学习 (Fully Asynchronous RL)**：专为大规模、高吞吐量的智能体（Agentic）强化学习训练而设计。
2. **极致的性能表现**：专为在 1,000+ 张 GPU 上训练万亿参数级（1T+）混合专家（MoE）模型而构建，训练端采用 [FSDP2](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)，推理端采用 [vLLM](https://github.com/vllm-project/vllm)，全面支持 FP8 推理、预填充/解码分离（PD Disaggregation）、专家并行（EP）与上下文并行（CP）等前沿特性。
3. **原生集成 [`verifiers`](https://github.com/PrimeIntellect-ai/verifiers) 环境**：通过 [Environments Hub](https://app.primeintellect.ai/dashboard/environments?ex_sort=most_stars) 无缝接入各类训练与评测环境，内置对软件工程（SWE）及多步智能体环境的开箱即用支持。
4. **端到端后训练全流程**：统一涵盖监督微调（SFT）、强化学习训练（RL）以及基准评测（Evals）。
5. **多节点集群部署**：原生支持基于 Slurm 的多机大规模集群调度部署。
6. **多模态模型支持**：支持 Qwen3.5 等视觉语言模型（VLMs）的多模态训练。
7. **高度模块化与可扩展**：架构设计清晰精炼，极易根据研究与生产需求进行二次定制。
8. **一行命令部署前沿旗舰模型（Slurm）**：例如[结合 P/D 分离、`llm-d` 路由器与 Mooncake KV 卸载的 `GLM-5` FP8 大规模训练方案](examples/advanced/glm-5.3/)。


## 模型支持矩阵 (Models support)


训练器（Trainer）运行位于 `src/prime_rl/trainer/models/` 下的 PrimeRL 原生模型实现，涵盖稠密（Dense）架构的 Llama、Qwen3 与 Qwen3.5，以及下表所列的各大模型家族；支持面向 MoE 层的专家并行（EP）、面向长序列的上下文并行（CP），并集成了 [quack-kernels](https://github.com/quack-kernels/quack-kernels) 等高性能算子库。未列入支持范围的架构会在训练器初始化阶段直接报错提示。

| 模型家族 (Family) | 示例模型 ID (Example IDs) | MoE | 专家并行 (EP) | 上下文并行 (CP) |
|--------|-------------|-----|----|-----|
| GLM-5 (`glm_moe_dsa`) | `zai-org/GLM-5`, `zai-org/GLM-5-FP8` | 是 | ✅ | ✅ |
| GLM-5.3 Flash，纯文本 (`glm5_next`) | `zai-org/GLM-5.3-Flash-BF16` | 是 | ✅ | ✅ |
| Qwen3 MoE (`qwen3_moe`) | `Qwen/Qwen3-30B-A3B`, … | 是 | ✅ | ✅ |
| Qwen3.5 MoE (`qwen3_5_moe`) | `Qwen/Qwen3.5-35B-A3B`, … | 是 | ✅ | ✅ |
| Qwen3.5 VLMs | 参阅 [advanced.md](docs/advanced.md#multimodal-training) (`qwen3_5`, `qwen3_5_moe`) | 仅限 MoE VLM | 仅限 MoE | ✅ |
| Poolside Laguna (`laguna`) | `poolside/Laguna-XS.2` | 是 | ✅ | ✅ |
| MiniMax M2 (`minimax_m2`) | `MiniMax/MiniMax-M2` | 是 | ✅ | ✅ |
| Muse Glimmer，纯文本 (`muse_glimmer`) | `meta-models/Muse-Glimmer-30B` | 否 | — | ✅ |
| Nemotron H (`nemotron_h`) | `nvidia/Nemotron-3-Nano-30B-A3B`, `nvidia/Nemotron-3-Super-120B-A12B`, … | 是 | ✅ | ✅ |
| Trinity (`afmoe`) | `arcee-ai/Trinity-Mini`, … | 是 | ✅ | ✅ |
| GLM-4 · GLM-4.5 MoE · INTELLECT-3 (`glm4_moe`) | `THUDM/GLM-4-9B-0414`, `zai-org/GLM-4.5-Air`, `zai-org/GLM-4.5`, `PrimeIntellect/INTELLECT-3`, … | 是 | ✅ | ✅ |
| GPT-OSS (`gpt_oss`) | `unsloth/gpt-oss-20b-BF16`, … | 是 | ✅ | ✅ |

## 环境安装与配置 (Setup)

> *我们在 NVIDIA RTX 3090/4090/5090、A100、H100、H200 以及 B200 上进行日常开发与测试。如果您在环境搭建过程中遇到任何问题，欢迎提交 [Issue](https://github.com/PrimeIntellect-ai/prime-rl/issues)。*

### 前置条件 (Prerequisites)

目前，**您需要至少一张 NVIDIA GPU 才能运行 prime-rl**。如果您手头暂无可用 GPU，推荐使用我们的[算力云平台 (Compute Platform)](https://app.primeintellect.ai)——无论是按需租用单张 GPU 用于日常开发、调试与小规模消融实验，还是[预订 1,000+ 张 GPU 的大型集群](https://app.primeintellect.ai/dashboard/quotes)开展生产级规模训练，均可一站式满足。

### 一键快速安装 (Quick Setup)

只需一行命令即可完成 prime-rl 的环境安装：

```bash
curl -sSL https://raw.githubusercontent.com/PrimeIntellect-ai/prime-rl/main/scripts/install.sh | bash
```

<details>
<summary>
手动安装步骤 (Manual Setup)
</summary>
<br>

1. 克隆代码仓库：

```bash
git clone https://github.com/PrimeIntellect-ai/prime-rl.git
cd prime-rl
```

2. 初始化 Git 子模块：

```bash
git submodule update --init -- deps/verifiers deps/renderers deps/prime-envs deps/pydantic-config
```

3. 安装 [uv](https://docs.astral.sh/uv/)：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

4. 根据锁文件安装依赖项：

```bash
uv sync --all-extras
```

> *注意*：训练环境（Environments）属于按需选装的 `uv` 工作区成员——执行 `uv sync --all-extras` 默认不会安装它们。如需在内置环境上进行训练，请运行 `uv sync --all-extras --all-packages` 安装全部环境包，或运行 `uv sync --package prime-rl --package <env>` 按需安装指定环境。

</details>

<details>
<summary>
验证您的环境配置 (Validate your environment setup)
</summary>
<br>

1. 检查当前环境是否使用 Python 3.12：

```bash
uv run python -V
```

2. 检查 `flash-attn` 是否已正确安装：

```bash
uv run python -c "import flash_attn"
```

3. 验证能否正常运行 SFT 训练器（*需要 1 张 GPU*）：

```bash
uv run sft @ configs/debug/fake/sft.toml
```

4. 验证能否正常启动推理服务端（*需要 1 张 GPU*）：

```bash
uv run inference --vllm.model Qwen/Qwen3-0.6B
```

5. 验证完整的 RL 技术栈（推理服务 + 编排器 Orchestrator + 训练器 Trainer）能否端到端跑通（*需要 2 张 GPU*）：

```bash
uv run rl @ examples/basic/reverse-text/rl.toml
```

</details>

### 附加配置 (Additional Setup)

1. 如果您希望将训练实验日志记录至 [Weights & Biases (W&B)](https://wandb.ai)，请先登录：

```bash
uv run wandb login
# 或直接设置环境变量：`export WANDB_API_KEY=...`
```

2. 如果您需要访问 [Hugging Face](https://huggingface.co) 上的受限（Gated）或私有模型与数据集，请先登录：

```bash
uv run hf auth login
# 或直接设置环境变量：`export HF_TOKEN=...`
```

## 训练示例 (Training Examples)
我们在 [`examples`](examples) 目录下提供了一系列端到端的训练示例，旨在全方位展示框架的核心特性，并引导您一步步训练自己的专属模型。

### 基础训练示例：1 至 8 张 GPU

循序渐进地掌握 prime-rl 的基础用法。您可以在 1 至 8 张 GPU 上训练自己的模型，非常适合快速入门并探索框架的各项核心能力。这些指南基于轻量级环境与小参数量模型，覆盖了单轮对话、多轮交互、工具调用（Tool Calling）等绝大多数常见应用场景：

1. [**文本反转 (Reverse Text)**](examples/basic/reverse-text/README.md)：训练 `Qwen3-0.6B` 将短文本块按字符反转。展示了极小规模的单轮 SFT 与 RL 完整训练流程，仅需单张消费级显卡即可在数分钟内完成训练，是最佳的入门起点。
2. [**Wordle 猜词游戏**](examples/basic/wordle/README.md)：训练 `Qwen3-1.7B` 玩 Wordle 猜词游戏。这是一个充满趣味的多轮 SFT 与 RL 训练示例，在 2–4 张 H100 GPU 上仅需数小时即可完成训练，非常适合探索框架的多轮交互训练能力。
3. [**字母表排序 (Alphabet Sort)**](examples/basic/alphabet-sort/README.md)：训练 `Qwen3-4B-Instruct-2507` 按字母顺序对人名进行排序。展示了无需 SFT 预热、直接基于 LoRA 进行多轮 RL 训练的流程，在单张 H100 GPU 上仅需一个多小时即可完成，非常适合探索基于 LoRA 的高效训练。
4. [**维基百科检索 (Wiki Search)**](examples/basic/wiki-search/README.md)：训练 `Qwen3-4B-Instruct-2507` 通过检索维基百科来回答百科问答题目，展示了结合搜索工具调用的多轮智能体训练。
5. [**Hendrycks 数学基准验证 (Hendrycks Sanity)**](examples/basic/hendrycks-sanity/README.md)：使用经过筛选的 MATH 数据集子集（筛选出 `DeepSeek-R1-Distill-Qwen-1.5B` 解题准确率在 20%–80% 之间的题目）在 `Qwen3-1.7B` 上运行健全性验证实验，非常适合用于算法消融研究。

### 高阶大规模训练：32 至 2,048 张 GPU

遵循以下指南，在复杂推理与智能体任务上训练大规模参数模型。这些配置方案专为在 Slurm 集群上运行而设计：

1. [**Qwen3-30B-A3B**](examples/advanced/qwen3-30b-a3b/)：在数学推理、软件工程（SWE）及智能体工具调用任务上训练 `Qwen3-30B-A3B`。
2. [**GLM-4.5-Air**](examples/advanced/glm-4.5-air/)：在信息检索、软件工程（SWE）及终端操作（Terminal）任务上训练 `GLM-4.5-Air`。
3. [**GLM-5**](examples/advanced/glm-5.3/)：结合预填充/解码分离（P/D Disaggregation）、`llm-d` 路由器与 FP8 低精度推理的大规模 `GLM-5` / `GLM-5.3` 训练方案。

## 文档导航 (Docs)

欢迎查阅 [`docs`](docs) 目录，获取关于如何使用 prime-rl 的深度指南：

- [**架构概览 (Overview)**](docs/overview.md) —— 系统架构、安装指南以及可直接复制运行的端到端 RL 训练流程
- [**配置系统 (Configuration)**](docs/configuration.md) —— TOML 配置组合、命令行参数覆盖、环境变量与配置校验
- [**训练指南 (Training)**](docs/training.md) —— RL 训练、SFT 微调、检查点（Checkpointing）管理、可观测性与实战经验法则
- [**评测指南 (Eval)**](docs/eval.md) —— 对接任意推理服务端的独立评测、在线评测、断点续评与平台结果上传
- [**规模化扩展 (Scaling)**](docs/scaling.md) —— 从单卡到多机集群、FSDP/EP/CP 并行策略、Slurm 调度与性能基准测试
- [**核心算法 (Algorithms)**](docs/algorithms.md) —— 异步训练与策略陈旧度（Staleness）控制、AIPO 损失函数、优势函数（Advantage）与数据过滤插件、轨迹合并（Trajectory Merging）
- [**高阶专题 (Advanced)**](docs/advanced.md) —— 自定义模型接入、多模态训练与 LoRA 微调
- [**开发者指南 (Development)**](docs/development.md) —— 测试套件、pre-commit 钩子配置以及如何新增模型支持

## 参与贡献 (Contributing)

我们非常欢迎来自开源社区的贡献！我们使用 [GitHub Issues](https://github.com/PrimeIntellect-ai/prime-rl/issues) 来跟踪缺陷反馈、功能请求并分享内部研发路线图。如果您在使用或开发过程中遇到 Bug、痛点，或者有任何新功能想法，欢迎随时提交 Issue。

欢迎通过 Pull Request 贡献代码，请遵循以下规范：
1. 安装 [pre-commit 钩子](#pre-commit-钩子-pre-commit-hooks)，确保您的代码格式符合项目规范。
2. 在 PR 尚未完全就绪前，请将其保持为“草稿（Draft）”状态。
3. 如果您的 PR 解决了某个已有 Issue，请在 PR 描述中关联该 Issue。
4. 条件允许的情况下，请尽量在本地运行[测试套件](#测试-tests)，确保您的改动工作正常。

### Pre-Commit 钩子 (Pre-Commit Hooks)

请安装 [pre-commit](https://pre-commit.com) 钩子以确保代码格式规范：

```bash
uv run pre-commit install
```

### 测试 (Tests)

```bash
uv run pytest -v                    # 运行全部测试
uv run pytest tests/unit -v         # 仅运行单元测试
uv run pytest tests/integration -v  # 仅运行集成测试
uv run pytest -v -m "not gpu"       # 仅运行 CPU 测试（排除带 gpu 标记的测试）
```

## 开源许可证 (License)

本项目基于 Apache 2.0 许可证开源，详情请参阅 [LICENSE](LICENSE) 文件。

## 引用 (Citation)

如果我们的工作对您的研究或项目有所帮助，欢迎按如下格式引用：

```tex
@misc{primeintellect2025prime-rl,
  author = {Prime Intellect},
  title = {prime-rl},
  url = {https://github.com/PrimeIntellect-ai/prime-rl},
  year = {2025}
}
```

---

> _注：本简体中文文档由社区志愿者翻译维护。若与英文原版存在差异，请以 [README.md](README.md) 为准。_
