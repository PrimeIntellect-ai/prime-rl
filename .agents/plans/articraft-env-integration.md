# Plan: Articraft Environment Integration into prime-rl

## 目标

将 articraft 的铰接 3D 物体生成 agent 包装为 `verifiers` 兼容的 RL 训练环境，使其能在 prime-rl 框架下进行多轮 RL 训练（GRPO）。模型通过 tool-calling 与 articraft SDK 交互，获得 compile/QC 信号作为 reward。

## 关键发现（Explore 阶段）

### 架构契合

- **prime-rl 的环境系统**基于 `verifiers` 库的 `vf.MultiTurnEnv` 抽象，BlenderGym 已是一个成功先例
- **articraft agent 的核心循环**：prompt → LLM tool_call → execute_tool → compile_model → feedback → repeat
- **verifiers rollout 模型**：LLM 生成 → env_response(tool 执行) → 再次生成 → ... → rubric 打分
- 两者结构高度一致，articraft 的 compile + QC signals 天然是 reward

### 关键约束

1. **verifiers 仅在 Linux 上运行**（lock 文件有 `platform_machine` marker）— 与集群部署一致
2. **articraft SDK 依赖重**（cadquery, manifold3d, trimesh, fcl）— 通过 `--no-deps` + 手动装核心几何库绕过（cadquery 排除）
3. **编译耗时**：中位数 0.08s（in-process），outlier 可达 12-40s（几何 QC 重的模型）。远低于原估的 5-60s
4. **无 GPU 渲染需求**：articraft 不像 BlenderGym 需要 GPU 渲染，纯 CPU 计算
5. **Tool schema 格式**：verifiers 使用 provider-agnostic 格式（`{"name", "description", "parameters"}`），自动通过 `to_native_tool()` 转为 OpenAI 格式发送给 vLLM
6. **无需服务化/沙盒**：KAOLA 兼容性测试确认 torch+vllm+articraft 无冲突，tool 执行在隔离 work_dir 中（per-rollout 逻辑隔离），不需要容器级沙盒
7. **SFT warm-up 数据极有限**：仅 33 条 records 有 trajectory.jsonl 数据。如需 SFT，必须从 prompt + final model.py 合成 trajectory（或先做纯 RL 试跑）
8. **初期裁减到 4 tools**：只暴露 `write_file`, `replace`, `read_file`, `compile_model`。`probe_model` 和 `find_examples` 作为后续 curriculum 阶段加入
9. **cadquery records 过滤**：31.6% 的 records（3,410/10,797）使用 cadquery，因不装 cadquery 需在 dataset 构建时过滤掉。过滤后仍有 ~7,387 条纯 SDK 几何记录可用

### 已验证风险（Step 0 结论）

- ~~SDK 重依赖可能与 torch/vllm 冲突~~ → **无冲突**（Step 0d 验证通过）
- ~~compile 耗时导致 rollout 慢~~ → **中位数 0.08s**，in-process 模式足够快
- ~~action space 过大~~ → **裁减到 4 tools**，初期可控

---

---

## BlenderGym vs Articraft: 交互模型根本差异

### Tool Set 对比

**BlenderGym**: 无 tool calling。LLM 输出纯文本，用 `<code>...</code>` XML 标签包裹 Python 代码。

**Articraft**: 完整 function calling。LLM 通过 vLLM `tool_call_parser` 输出结构化 tool_calls。

| | BlenderGym | Articraft |
|---|---|---|
| **LLM 输出格式** | 纯文本 + `<code>` XML tag | OpenAI function calling JSON |
| **Tool 数量** | 0（隐式：整个 response 就是 action） | 6 个显式 tools |
| **vLLM 配置** | 无 `tool_call_parser` | `tool_call_parser = "qwen3_coder"` |
| **Parser** | `vf.XMLParser(["code"])` | 不需要 parser（vLLM 原生解析） |
| **Action space** | 每 turn 只能输出一段代码 | 每 turn 可输出多个 tool_calls |

### Articraft Tool Set 详细

| Tool 名 | 参数 | 功能 | RL 中的角色 |
|---------|------|------|------------|
| `read_file` | `path`, `start_line?`, `max_lines?` | 读取 model.py 内容 | 观察/规划 |
| `write_file` | `content` | 整体重写 model.py 的 editable section | 初始写入 |
| `replace` | `old_string`, `new_string`, `description?` | 精确替换代码片段 | 增量修改 |
| `compile_model` | (无参数) | 执行脚本 + QC 检查 → 返回 signals | **核心验证** |
| `probe_model` | `code`, `description?`, `timeout?` | 在已编译模型上执行诊断代码 | 主动诊断 |
| `find_examples` | `query`, `max_results?` | BM25 搜索相似例子 | 参考学习 |

### Function Calling 格式

Articraft 推理时用 OpenAI API 格式。RL 训练时 vLLM 通过 `tool_call_parser = "qwen3_coder"` 将 tool schemas 注入模型 chat template，模型输出结构化 JSON：

```
# vLLM 输入（verifiers to_native_tool() 自动转换为 OpenAI 格式）
# 我们的代码定义 provider-agnostic 格式（见 tools.py），verifiers 负责转换
tools = [
    {"type": "function", "function": {"name": "compile_model", "description": "...", "parameters": {...}}},
    {"type": "function", "function": {"name": "replace", "description": "...", "parameters": {...}}},
    ...
]

# 模型输出（vLLM 解析为 tool_calls）
assistant message: {
    "role": "assistant",
    "tool_calls": [
        {"id": "tc_1", "function": {"name": "replace", "arguments": "{\"old_string\": ..., \"new_string\": ...}"}},
        {"id": "tc_2", "function": {"name": "compile_model", "arguments": "{}"}}
    ]
}

# 环境返回（per tool_call 一条 tool message）
tool messages: [
    {"role": "tool", "tool_call_id": "tc_1", "content": "{\"result\": \"Replaced 5 lines...\"}"},
    {"role": "tool", "tool_call_id": "tc_2", "content": "<compile_signals>...</compile_signals>"}
]
```

### Tool Calling 与 RL 训练的交互

**parser 不影响 RL 训练算法**：训练的是模型生成的原始 token（含 `<tool_call>` XML 标签等格式 token），parser 只在 inference 侧做 post-processing。

| 层 | 作用 | Articraft 注意事项 |
|---|------|-------------------|
| vLLM `tool_call_parser` | 从模型 raw tokens 解析出结构化 `tool_calls` | 必须用 `qwen3_coder`（Qwen3.5） |
| TITO (Token-In-Token-Out) | 记录真实 token + logprob，避免重 tokenize | `env_response` 消息顺序必须兼容 TITO（tool 在前、user 在后） |
| `completion_mask` | 区分 assistant token (True) vs env 注入 token (False) | tool_call XML token 参与 loss；tool result token 被 mask |
| GRPO advantage | 每条 rollout 一个标量，广播到所有 completion token | 与 tool calling 无特殊交互 |

**Loss mask 示例（多轮 tool calling rollout）：**

```
Turn 1: assistant tool_call tokens [T,T,T] + env tool_result tokens [F,F,F]
Turn 2: assistant tool_call tokens [T,T,T] + env tool_result tokens [F,F,F]
Turn 3: assistant tokens [T,T]
→ completion_mask = [T,T,T, F,F,F, T,T,T, F,F,F, T,T]
```

**上线前验证 checklist：**

1. vLLM smoke：`Qwen3.5-9B` + `qwen3_coder` + articraft tools → 返回合法 `tool_calls`
2. 单条 rollout：每 step 有 `tokens`，`completion_logprobs` 非全零（TITO 成功）
3. `completion_mask` 模式正确：assistant `[T...]` + env `[F...]` 交替
4. Decode training sample，确认 tool_call XML token 与 inference 一致
5. 模型输出非法 tool_call 格式时，env 不 crash、scheduler 不 loop

### Env Feedback 对比

**BlenderGym** 的每 turn 反馈：
```
Turn 0:  user=[goal_image + init_image + start_code]
Turn N:  user=[render_of_previous_code + REFINE_INSTRUCTION]
         （或 "parse failed, try again"）
```
反馈全在 `get_prompt_messages()` 中构建，`env_response()` 返回空。

**Articraft** 的每 turn 反馈：
```
Turn 0:  user=[SDK docs + task prompt + runtime_guidance]
         assistant=[tool_calls: write_file, compile_model]
         tool=[write result, compile_signals]
         user=[guidance injections if triggered]

Turn N:  assistant=[tool_calls: replace, compile_model]
         tool=[replace result, compile_signals]
         user=[guidance injections if triggered]
```
反馈在 `env_response()` 中执行 tools 并返回，guidance 作为追加 user message 注入。

### 关键设计差异总结

| 维度 | BlenderGym | Articraft Phase 1 | Articraft Phase 2+ |
|------|-----------|-------------------|-------------------|
| **交互模型** | Single-action per turn（一段 code） | Multi-action per turn（多个 tool_calls） | 同 Phase 1 |
| **env_response** | `return []`（空） | 执行 tools → 返回 tool messages | 同 Phase 1 |
| **反馈载体** | 渲染图片（视觉） | `<compile_signals>` XML（文本） | XML + 渲染截图 |
| **中间引导** | 无（只有图片反馈） | Phase 1: `<compile_required>` + `<edit_retry_guidance>`；Phase 1.5: exact_geometry/baseline_qc guidance | 同 |
| **终止条件** | max_turns | max_turns + freshness 条件终止（text-only/empty response 时检查 code 是否 fresh） | 同 |
| **外部服务** | Render + Score (HTTP) | 无（全 in-process） | Viewer Render + CLIP Score (HTTP) |
| **GPU 需求** | 渲染需要 GPU (OPTIX) | 纯 CPU（几何计算） | CPU + CLIP 推理 GPU |

---

## Articraft 代码复用方案

### 需要复用的 articraft 模块

```
articraft/                    ← 原项目
├── sdk/                      ← 核心 SDK（几何类型、URDF 导出、QC 检查）
│   ├── __init__.py
│   ├── v0/                   ← 主要 API 版本
│   │   ├── types.py          ← Part, Joint, ObjectModel 等核心类型
│   │   ├── articulated_object.py  ← build_object_model 返回的对象
│   │   ├── _urdf_export.py   ← compile_object_to_urdf_xml()
│   │   ├── mesh.py           ← 网格操作
│   │   ├── exact_collisions.py ← 碰撞检测
│   │   └── ...
│   └── _profiles.py          ← SDK profile (scaffold path 等)
├── agent/
│   ├── compiler.py           ← compile_urdf_report_maybe_timeout()
│   ├── feedback.py           ← build_compile_signal_bundle() + render_compile_signals()
│   ├── models.py             ← CompileSignal, CompileSignalBundle, CompileReport
│   ├── tools/
│   │   ├── base.py           ← ToolResult, BaseDeclarativeTool, ToolRegistry
│   │   ├── registry.py       ← ToolRegistry class
│   │   ├── write_code.py     ← WriteFileTool
│   │   ├── edit_code.py      ← ReplaceTool
│   │   ├── compile_model.py  ← CompileModelTool (仅 schema，执行在 harness)
│   │   ├── read_file.py      ← ReadFileTool
│   │   ├── probe_model/      ← ProbeModelTool (诊断代码执行)
│   │   ├── find_examples.py  ← FindExamplesTool (BM25 搜索)
│   │   └── code_region.py    ← editable section 提取
│   ├── harness_compile.py    ← CompileFeedbackLoop
│   └── harness_guidance.py   ← GuidanceInjector
└── scaffold.py               ← 空 model.py 模板
```

### 复用策略

**方案: 将 articraft 作为 git submodule 或 path dependency 安装**

```toml
# environments/articraft/pyproject.toml
[project]
dependencies = [
    "verifiers>=0.1.10",
    "articraft @ file:///${PROJECT_ROOT}/../../../articraft",  # path dep
    # 或: "articraft @ git+https://github.com/mattzh72/articraft.git@main"
]
```

这样 `articraft_env/` 可以直接 import：
```python
from agent.compiler import compile_urdf_report_maybe_timeout
from agent.feedback import render_compile_signals, build_compile_signal_bundle
from agent.models import CompileSignalBundle, CompileReport
from agent.tools.base import ToolResult
from agent.tools.code_region import find_code_region, extract_editable_code, replace_editable_code, map_syntax_error_line_to_editable
from agent.workspace_docs import _DOC_PATH_ALIASES
from sdk._profiles import get_sdk_profile  # 获取 docs 白名单路径
```

### 不复用的部分

| 模块 | 原因 | 替代方案 |
|------|------|---------|
| `agent/harness.py` | 整个 LLM 循环由 verifiers 管理 | `ArticraftEnv.env_response()` |
| `agent/providers/` | LLM 调用由 vLLM 处理 | vLLM `tool_call_parser` |
| `agent/prompts/` | 需要简化（去掉 provider 分化逻辑） | `articraft_env/prompts.py` 手写 |
| `agent/tui/` | 显示层不需要 | W&B + trajectory.json |
| `agent/single_run.py` | 运行编排由 prime-rl orchestrator 负责 | prime-rl trainer |
| `agent/batch_runner.py` | 同上 | prime-rl trainer |
| `agent/cost.py` | 成本追踪不适用于 RL | — |
| `storage.RecordStore` | 只需读 records 目录（简单目录扫描），不需要创建/修改 records 的完整 API | `dataset.py` 简单路径扫描 |
| `storage.DatasetStore` | manifest 索引逻辑偏重，RL dataset 只需 glob 目录 | 同上 |
| `storage.trajectories` | **被 ArtifactManager 替代**——写 BlenderGym 风格的 trajectory.json + meta.json + 保留策略 | `ArticraftArtifactManager` |
| `storage.materialize` | record 物化为完整 artifact 的逻辑，RL 不需要 | — |
| `cli/` | 不需要 CLI | config + orchestrator |
| `viewer/api/`（mutation 路由） | store_mutations / promotion 等编辑功能不直接用于 RL | — |

### 直接复用的部分

| 模块 | 复用方式 | 用途 |
|------|---------|------|
| `storage.layout.StorageLayout` | import | 获取 `records_root` 等路径约定，dataset loader 用于定位 record 目录 |
| `viewer/` (完整) | 独立部署 | (1) RL rollout 可视化：将 rollout 结果写为 viewer 可读格式，用 viewer 浏览 (2) Phase 2 CLIP reward：headless render URDF → 截图 |
| `sdk._profiles.get_sdk_profile()` | import | 获取 `docs_full` 路径列表构建 `read_file` 白名单 |
| Tool 类的参数约定 | 参考（手写 schema，注明对应位置） | `WriteFileTool/ReplaceTool/ReadFileTool/CompileModelTool` 的参数名和语义 |
| `agent.compiler.compile_urdf_report_maybe_timeout` | import | 执行 compile 并获取 `CompileReport` |
| `agent.feedback.render_compile_signals` | import | 将 `CompileSignalBundle` 格式化为 LLM 可读的 XML |
| `agent.tools.code_region` | Phase 1.5 import | `find_code_region()` / `extract_editable_code()` / `replace_editable_code()` / `map_syntax_error_line_to_editable()` — Phase 1 统一用 scaffold（无 marker），全文件可编辑，不需要 code_region 工具。Phase 1.5 如果支持 marker-based records 再引入 |

### 需要适配的部分（直接 import 调用，无需改 articraft）

| 模块 | 适配内容 |
|------|---------|
| `agent/compiler.py` → `compile_urdf_report_maybe_timeout` | 直接调用，传入 `script_path` |
| `agent/feedback.py` → `render_compile_signals`, `build_compile_signal_bundle`, `compile_signal_bundle_from_exception` | 直接调用（成功路径用 `build_compile_signal_bundle`，异常路径用 `compile_signal_bundle_from_exception`） |
| `agent/workspace_docs.py` → `_DOC_PATH_ALIASES` | 构建虚拟路径白名单 |
| `scaffold.py` | 读取内容写入 work_dir/model.py |
| `sdk/_profiles.py` → `get_sdk_profile("sdk").docs_full` | 获取 docs 路径列表 |

### 参考但不直接 import（对齐行为，不 import 代码）

| 模块 | 参考内容 | 阶段 |
|------|---------|------|
| `agent/harness_compile.py` | compile 状态机：freshness 追踪、连续失败计数、repeated hash。Phase 1.5 deferred item #1 的实现参考 | Phase 1.5 |
| `agent/harness_guidance.py` | GuidanceInjector：edit_retry / exact_geometry / baseline_qc。Phase 1.5 deferred item #2 的实现参考 | Phase 1.5 |
| `agent/tools/write_code.py` | WriteFileTool 的验证逻辑（必需函数检查、语法校验、editable region）。我们的 execute_write_file 对齐此行为 | Phase 1 |
| `agent/tools/edit_code.py` | ReplaceTool 的匹配/错误消息。我们的 execute_replace 对齐此行为 | Phase 1 |
| `agent/tools/read_file.py` | ReadFileTool 的 `L{n}: ` 格式、offset/limit 语义。我们的 execute_read_file 对齐此行为 | Phase 1 |
| `agent/tools/compile_model.py` | CompileModelTool 无参 schema、baseline QC 说明、harness 拦截语义。我们的 `COMPILE_MODEL_SCHEMA` 对齐此文件 | Phase 1 |
| `agent/tools/find_examples.py` | FindExamplesTool schema（参数 `limit`、`include_paths`）、返回字段结构（`example_id`/`content`/`match_quality`）。Phase 1.5 实现时需对齐 | Phase 1.5 #4 |
| `agent/tools/base.py` → `make_tool_schema()` | schema 格式约定（`additionalProperties: false`、`strict: true`）。确保手写 schema 格式兼容 | Phase 1 |
| `agent/examples.py` | BM25 `ExampleSearchIndex`。find_examples tool 的核心逻辑在此（tools/find_examples.py 只是薄封装） | Phase 1.5 #4 |
| `agent/tools/probe_model/` | ProbeModelTool：在已编译 model globals 上 exec 诊断代码。依赖 `compiler.load_model_globals()` | Phase 1.5 #3 |
| `agent/compiler.py` → `load_model_globals()` | probe_model runner 依赖的全局变量加载逻辑。Phase 1.5 实现 probe_model 时可能需 **直接 import** | Phase 1.5 #3 |
| `agent/tools/__init__.py` | `build_first_turn_messages()` + `build_first_turn_runtime_guidance()`。验证我们的 Turn 0 构建与之一致 | Phase 1 |
| `agent/workspace_docs.py` → `load_sdk_docs_reference()` | Turn 0 预加载 3 篇 docs 的完整实现。可直接 import 替代手写 | Phase 1（可选） |
| `agent/workspace_docs.py` → `normalize_virtual_workspace_path()` | 虚拟路径规范化（去 `./` 前缀、alias 解析）。我们白名单映射应参考其规则 | Phase 1 |
| `agent/prompts/sections/designer_common.md` | 核心 system prompt 内容源：角色定义、四条硬要求、工具使用原则。手写 prompts.py 的行为对齐依据 | Phase 1 |
| `agent/prompts/sections/sdk_base.md` | SDK 建模与 testing 指南（geometry + TestContext）。system prompt 的技术参考内容来源 | Phase 1 |
| `agent/prompts/sections/link_naming.md` | link 命名规范。system prompt 的命名约定来源 | Phase 1 |
| `agent/harness_codec.py` | `PARALLEL_SAFE_TOOL_NAMES`（read_file, find_examples, probe_model）。确认我们顺序执行即可 | N/A |
| `agent/traces.py` | `TraceWriter` 的 trajectory JSONL 格式。若需导出 viewer 可读 trajectory 可参考 | Phase 2 |
| `agent/record_persistence.py` | record 写入格式。Phase 2 若将 rollout 写为 viewer record 可参考 | Phase 2 |

### 完全不相关（orchestration / 运行环境 / 工具链）

| 模块 | 原因 |
|------|------|
| `agent/runner.py`, `runner_cli.py`, `run_context.py`, `run_config.py` | orchestration 由 prime-rl 负责 |
| `agent/edit.py`, `agent/rerun.py` | record 编辑/重跑工作流 |
| `agent/payload_preview.py` | dry-run preview |
| `agent/defaults.py` | provider 默认值（max_turns per provider） |
| `agent/open_file_limits.py`, `agent/mp_utils.py` | OS 资源管理 |
| `agent/runtime_limits.py` | batch 并发信号量（RL 由 verifiers env server 管） |
| `agent/tools/apply_patch.py` | OpenAI-only patch tool（RL 走 replace+write_file 路径） |
| `agent/tools/registry.py` | ToolRegistry 类（计划不使用 `build_tool_registry()`，手写 schema） |
| `articraft/values.py` | ProviderName/ThinkingLevel 枚举（path dep 安装时随包加载，不主动调用） |
| `sdk/_dependencies.py` | cadquery 可选依赖检查（`--no-deps` 安装时已绕过） |
| `sdk/_extensions/cadquery/` | CadQuery 后端（已排除安装） |
| `agent/models.py` 中非 Compile* 类型 | `TerminateReason`/`AgentResult`/`SessionPaths` 等（RL 不需要） |

> 注：不再需要适配 `build_tool_registry()`——tool schemas 完全手写在 prime-rl 侧。

---

## Articraft 侧修改 vs Prime-rl 侧新写（具体工作清单）

### 核心结论

**articraft 代码不需要任何修改。** 所有适配工作都在 prime-rl 的 `environments/articraft/` 中完成——通过正确使用 articraft 的已有接口 + 少量轻量 wrapper。

### 为什么不需要改 articraft

源码审查确认：

| 组件 | 为什么不改 |
|------|---------|
| `compiler.py` | `compile_urdf_report_maybe_timeout(script_path)` 接受任意路径，无硬编码 |
| `feedback.py` | 纯函数，无外部依赖 |
| `models.py` | 纯 dataclass |
| `CompileFeedbackLoop` | 只需 `file_path` + `sdk_package`，不绑 harness |
| `scaffold.py` | 读文件内容即可复制到 work_dir |
| Tool schemas | `make_tool_schema()` 返回标准 OpenAI function 格式 dict |
| Edit tools | `bind_file_path(path)` 可绑任意路径 |

articraft 唯一的"硬约束"是：**articraft 包必须在 `sys.path`**（`load_model_globals` 会 `chdir` 到 script 目录，script 里 `from sdk import ...` 需要能找到 articraft 的 `sdk/`）。这通过 `uv pip install -e articraft` 就满足了。

---

### Prime-rl 侧需要新写的代码

```
environments/articraft/
├── pyproject.toml                ← 包定义 + articraft path dependency
├── articraft_env/
│   ├── __init__.py              ← PEP 562 lazy imports（避免 import 即拉重依赖）
│   ├── env.py                   ← ArticraftEnv(vf.MultiTurnEnv) + load_environment()
│   ├── rubric.py                ← ArticraftRubric + @vf.cleanup 写 artifacts
│   ├── artifact_manager.py      ← ArticraftArtifactManager（rollout 文件生命周期）
│   ├── dataset.py               ← 从 articraft records 构建 verifiers Dataset
│   ├── prompts.py               ← SYSTEM_PROMPT（概览 + 可用 docs 路径列表）
│   ├── schema.py                ← Task/TurnRecord/Rollout + require_rollout()
│   └── tools.py                 ← 手写 TOOL_SCHEMAS + execute_* 执行逻辑 + 白名单
└── tests/
    └── test_rollout.py          ← MockClient 驱动的端到端测试
```

**总计约 700-900 行新代码**（不含 tests）。下面逐文件说明每个文件的职责和与 articraft 的交互方式：

---

#### `tools.py` — Tool Dispatch（约 150 行，核心 adapter）

这是 articraft 和 prime-rl 之间的**唯一 glue 层**。Schema 从 articraft tool 类 import（保持同步），但执行逻辑自实现（它们需要 VirtualWorkspace 等 harness 绑定，我们不需要）：

```python
"""
为什么不直接用 articraft 的 Tool 类？
- WriteFileTool/ReplaceTool 需要 bind_file_path + VirtualWorkspace 绑定
- CompileModelTool.execute() 是故意返回 error（由 harness 拦截执行）
- ReadFileTool 需要完整 VirtualWorkspace（含虚拟路径映射）

RL 环境中自实现但复刻关键行为：
- write_file: 验证 build_object_model/run_tests 必需函数 → 全文件替换 → 语法校验
  （Phase 1 统一用 scaffold，无 USER_CODE marker，全文件可编辑。Phase 1.5 再支持 marker-based editing）
- replace: 全文件范围内唯一匹配验证 → 替换 → 语法校验（与 articraft 一致）
- read_file: L{n}: 格式输出 + 白名单路径映射（替代 VirtualWorkspace）
- compile_model: compile_urdf_report_maybe_timeout + render_compile_signals

Tool SCHEMAS 手写（不从 articraft import），原因：
- articraft schema description 含 "virtual workspace" 等 RL 中不存在的概念
- RL 中 write_file 无需 path 参数，replace 无需 instruction/allow_multiple
- 每个 schema 旁注明对应 articraft 源文件，方便后续对照检查

ToolResult 序列化：所有 tool 返回 JSON ({"result": ...} 或 {"error": ...})
与 articraft harness 发送给 LLM 的格式一致。
"""

from pathlib import Path
from agent.compiler import compile_urdf_report_maybe_timeout  # compiler.py L618-626
from agent.feedback import render_compile_signals              # feedback.py L1176-1181
from agent.feedback import compile_signal_bundle_from_exception  # feedback.py L1142
from agent.models import CompileReport, CompileSignalBundle    # models.py
from agent.tools.base import ToolResult                        # tools/base.py L12-45

# ── Tool Schemas（手写，每个 tool 一个变量，注明 articraft 对应位置）──

# 对应: articraft/agent/tools/write_code.py WriteFileTool
# 差异: 去掉 path 参数（RL 中只有 model.py）
# 注意: Phase 1 统一用 scaffold（无 marker，全文件可编辑）。write_file 替换整个 model.py 的
#       build_object_model + run_tests 实现部分。
# ⚠️ 格式: verifiers provider-agnostic（不是 OpenAI legacy {"type":"function","function":{...}}）
#    verifiers 通过 to_native_tool() 自动转为 OpenAI 格式发送给 vLLM。
WRITE_FILE_SCHEMA = {
    "name": "write_file",
    "description": "Rewrite the entire model.py implementation. Must include build_object_model() and run_tests() functions.",
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "Full replacement content for model.py. Must define build_object_model() -> ArticulatedObject and run_tests() -> TestReport."},
        },
        "required": ["content"],
    },
}

# 对应: articraft/agent/tools/edit_code.py ReplaceTool
# 差异: 去掉 instruction 和 allow_multiple（简化 action space）
REPLACE_SCHEMA = {
    "name": "replace",
    "description": "Replace text within model.py. old_string must match exactly one occurrence including whitespace.",
    "parameters": {
        "type": "object",
        "properties": {
            "old_string": {"type": "string", "description": "Exact literal text to find. Must match including whitespace and indentation."},
            "new_string": {"type": "string", "description": "Replacement text. May be empty to delete."},
        },
        "required": ["old_string", "new_string"],
    },
}

# 对应: articraft/agent/tools/read_file.py ReadFileTool
# 差异: description 改为列出实际可用路径，不提 "virtual workspace"
# 路径格式: 使用 articraft 虚拟路径（docs/sdk/references/...），不用磁盘路径（sdk/_docs/...）
READ_FILE_SCHEMA = {
    "name": "read_file",
    "description": "Read a file from the workspace with 1-indexed line numbers. Use 'model.py' for your code, or SDK doc paths like 'docs/sdk/references/quickstart.md' for reference.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path to read. Either 'model.py' or an SDK doc path (e.g. 'docs/sdk/references/quickstart.md')."},
            "offset": {"type": "integer", "description": "1-indexed line to start from. Omit for line 1."},
            "limit": {"type": "integer", "description": "Maximum number of lines to return. Omit for full file."},
        },
        "required": ["path"],
    },
}

# 对应: articraft/agent/tools/compile_model.py CompileModelTool
# 差异: 无（schema 完全一致）
COMPILE_MODEL_SCHEMA = {
    "name": "compile_model",
    "description": "Compile and run QC checks on model.py. Returns structured <compile_signals> with status, failures, warnings, and suggested next steps.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}

TOOL_SCHEMAS = [WRITE_FILE_SCHEMA, REPLACE_SCHEMA, READ_FILE_SCHEMA, COMPILE_MODEL_SCHEMA]

def get_tool_schemas() -> list[dict]:
    """返回 4 个 tool 的 verifiers provider-agnostic schemas。
    verifiers 通过 Tool.model_validate(dict) 规范化，再经 to_native_tool() 转为 OpenAI 格式。
    """
    return TOOL_SCHEMAS

# ── 可读文件白名单（model.py + SDK docs）──
# 路径格式使用 articraft 虚拟路径（与 agent 训练/推理一致）

from sdk._profiles import get_sdk_profile
from agent.workspace_docs import _DOC_PATH_ALIASES  # {磁盘相对路径: 虚拟后缀}

def get_readable_paths(articraft_root: Path, work_dir: Path) -> dict[str, Path]:
    """
    返回 read_file 允许的路径映射。
    key = 虚拟路径（如 "docs/sdk/references/quickstart.md"）
    value = 磁盘绝对路径（如 articraft_root / "sdk/_docs/common/00_quickstart.md"）

    源码参考：
    - articraft/agent/workspace_docs.py  build_virtual_workspace() L45-92
    - articraft/agent/workspace_docs.py  _DOC_PATH_ALIASES L12-42（虚拟路径映射）
    - articraft/sdk/_profiles.py  SdkProfile.docs_full（SDK 文档列表）
    """
    profile = get_sdk_profile("sdk")
    paths = {"model.py": work_dir / "model.py"}
    for rel_path in profile.docs_full:
        rel_str = rel_path.as_posix() if hasattr(rel_path, 'as_posix') else str(rel_path)
        virtual_suffix = _DOC_PATH_ALIASES.get(rel_str)
        if virtual_suffix:
            paths[f"docs/sdk/{virtual_suffix}"] = articraft_root / rel_path
    return paths

# ── ToolResult 序列化格式（与 articraft 一致）──
# 成功: {"result": "<output>"}  或  {"result": "<output>", "compilation": {...}}
# 失败: {"error": "<message>"}
#
# ToolMessage.content = json.dumps(tool_result.to_dict())

# ── Harness 行为规格（从 providers/ 和 harness.py 确认）──
#
# == Tool 执行规则 ==
# 1. 执行顺序：tool_calls 数组按顺序串行执行（不并行）
# 2. 畸形 JSON arguments → ToolResult(error="Invalid JSON in tool arguments: ...")
# 3. arguments 解析后不是 object → ToolResult(error="Tool arguments must be a JSON object")
# 4. 未知 tool name → ToolResult(error="Tool {name} not found")
# 5. Pydantic ValidationError → 拼出 missing/invalid 字段说明
# 6. compile_model 传入任何参数 → ToolResult(error="Unexpected parameters: [...]")
# 7. 其它执行异常 → ToolResult(error="Tool execution error: ...")
# 8. 所有错误都写入对话（作为 tool message），不 retry LLM
# 9. tool_call.id 必须存在（vLLM 会生成）
# 10. _mark_code_mutated 只在 write/replace 成功时触发（失败不递增 edit_revision）
#
# == Freshness 追踪（Phase 1 必需）==
# - 每次成功写操作递增 edit_revision
# - 每次成功编译记录 last_successful_compile_revision
# - edit_revision != last_compile_revision → code dirty
# - code dirty 时 compile_model 重新编译
# - code fresh 时 compile_model 返回缓存结果 + "Fresh compile already exists..." 前缀
#
# == 终止与 Turn 计数（Phase 1 必需）==
# - 每次 model response 都消耗 1 turn（含 empty/text-only），max_turns=50
# - Empty/text-only response（无 tool_calls）→ 检查 freshness:
#   - fresh → 终止（设 state["final_env_response"] = []）
#   - dirty → 注入 <compile_required> user message（最多 3 次，超限强制终止）
# - 有 tool_calls 的 response → 执行 tools
#
# == Guidance 注入（Phase 1 简化版）==
# - replace 失败 + "Could not find the old_string" → 注入 <edit_retry_guidance> user message
# - 时机：所有 tool results 追加后、下一次 LLM 调用前
# - 去重：每种 error signature 只注入一次
#
# == replace harness 层拦截 ==
# - old_string="" 且 editable section 非空 → error（建议用 write_file）
# - editable section 空 且 old_string!="" → error（建议用 write_file）
#
# == response_rules（compile 输出的一部分）==
# render_compile_signals 输出末尾包含 <response_rules> 块：
# - failure-type-specific guidance（compile_runtime→"Fix error first", isolated_part→"Fix floating part" 等）
# - repeated=True 时追加 "This failure matches the previous compile attempt"
# - failure_streak>=3 时追加 "You are in a repair loop" + probe_model 建议
# 注：Phase 1 中 repeated/failure_streak 固定为 False/0，response_rules 仍然输出（基于 failure type）
#
# == 编译超时 ==
# - RL 训练用 60s（URDF_COMPILE_TIMEOUT_SECONDS=60）
# - 正常 compile 中位数 0.08s，P90 ~23s；60s 覆盖绝大多数正常 compile
# - 超时后 terminate 子进程 + 返回 TimeoutError ToolResult
#
# Provider 参考：openrouter.py 最接近 vLLM OpenAI-compatible 格式
# 工具集选择：非 OpenAI provider 用 replace+write_file（不是 apply_patch）
# read_file 行为：Phase 1 返回完整 model.py（scaffold 全文件，无 marker）
#   模型看到完整上下文（imports + 空函数体），有助于理解 SDK API

# ── Tool 执行（自实现，复刻 articraft 行为）──

REQUIRED_FUNCTIONS = ("build_object_model", "run_tests")

def execute_write_file(content: str, script_path: Path, rollout: Rollout) -> ToolResult:
    """
    复刻 articraft WriteFileTool 行为。

    源码参考：
    - articraft/agent/tools/write_code.py  WriteCodeInvocation.execute() L56-70
    - articraft/agent/tools/write_code.py  _missing_required_functions() L72-84
    - articraft/agent/tools/write_code.py  _validate_python_syntax() L86-109
    - articraft/agent/harness.py L864-865  is_success() → mark_code_mutated

    流程：
    1. 验证 content 含必需顶层函数 (build_object_model, run_tests)
       - 缺失 → error, 不写入, 不调 mark_code_mutated
       - 源码: _missing_required_functions() L72-84 用 ast.parse 检查 FunctionDef
    2. 写入 model.py（Phase 1: 全文件替换）
       - 源码: L62-63 replace_editable_code（无 marker 时等价全文件替换）
    3. 语法校验 compile(code, "model.py", "exec")
       - 源码: _validate_python_syntax() L86-109
       - 通过 → compilation={"status": "success", "error": None}
       - 失败 → compilation={"status": "error", "error": "Syntax error: {msg} (line {lineno})", "error_line": lineno}
       - 注意：语法错误不阻止写入（写入已在步骤 2 完成）
    4. 写入成功后调 rollout.mark_code_mutated()
       - 源码: harness.py L864-865 is_success() 即 error is None → _mark_code_mutated
       - 即使有语法错误，只要 ToolResult.error 为 None 就 mark

    成功返回: ToolResult(output="Code rewritten successfully", compilation={...})
    失败返回: ToolResult(error="write_code must include required top-level functions...")
             ToolResult(error="File {file_path} not found")
    """
    ...

def execute_replace(old_string: str, new_string: str, script_path: Path, rollout: Rollout) -> ToolResult:
    """
    复刻 articraft ReplaceTool 行为。

    源码参考：
    - articraft/agent/tools/edit_code.py  EditCodeInvocation.execute() L39-110
    - articraft/agent/harness.py L786-840  replace 前置校验（harness 层拦截）
    - articraft/agent/harness.py L864-865  is_success() → mark_code_mutated

    流程：
    1. 读取当前 model.py 全文
       - 源码: L42-44
    2. 前置校验（对齐 harness.py L786-840）：
       - old_string="" 且文件非空 → error（源码 L50-53）:
         "old_string cannot be empty unless the editable code section is empty.
          Please provide the exact string to replace."
       - old_string!="" 且文件为空 → error（源码 L55-63，harness L815-840）:
         "Editable code section is empty. Initialize it with `write_file(content=...)`
          or with replace using old_string=\"\" and new_string containing the initial
          build_object_model() and run_tests() implementation."
    3. 计算 old_string 出现次数（源码 L66）: occurrences = code.count(old_string)
       - 0 次 → error（源码 L66-70）:
         "Could not find the old_string in the code. Make sure the string matches
          exactly, including whitespace and indentation. "
       - >1 次 → error（源码 L73-79）:
         "The old_string appears {occurrences} times in the code. Please provide
          a longer, unique string that appears only once, or use replace_all=true
          to replace all occurrences."
    4. 精确替换 1 次（源码 L82）: code.replace(old_string, new_string, 1)
    5. 写回 + 语法校验（同 write_file 的 _validate_python_syntax）
    6. 替换成功后调 rollout.mark_code_mutated()

    成功返回: ToolResult(output="Code edited successfully", compilation={...})
    """
    ...

def execute_read_file(path: str, readable_paths: dict[str, Path],
                      offset: int | None = None, limit: int | None = None) -> ToolResult:
    """
    复刻 articraft ReadFileTool 行为。

    源码参考：
    - articraft/agent/tools/read_file.py  ReadFileInvocation.execute() L55-103

    流程：
    - 路径解析：readable_paths 白名单映射（对齐 workspace_docs.py _DOC_PATH_ALIASES）
    - 输出格式（源码 L85-89）: 每行 "L{line_number}: {content}"（1-indexed）
    - offset < 1 → error（源码 L72-73）
    - offset > file length → error（源码 L74-77）
    - 空文件 → output=""（源码 L68-69）
    - limit 默认不限（返回全文件）；若指定则截取 [offset, offset+limit)
    """
    resolved = readable_paths.get(path)
    if resolved is None:
        # 源码: workspace_docs.py resolve_virtual_path 失败时
        return ToolResult(error=f"Unable to resolve {path}")
    ...

def execute_compile(script_path: Path, rollout: Rollout, *, sdk_package: str = "sdk") -> ToolResult:
    """
    复刻 articraft compile 执行行为。

    源码参考：
    - articraft/agent/harness_compile.py  CompileFeedbackLoop.execute_compile_model() L177-214
    - articraft/agent/harness_compile.py  latest_code_is_fresh() L85-89
    - articraft/agent/harness_compile.py  _render_reused_compile_tool_output() L142-147
    - articraft/agent/harness_compile.py  _render_compile_tool_output() L125-140
    - articraft/agent/compiler.py         compile_urdf_report_maybe_timeout() L618-626
    - articraft/agent/feedback.py         render_compile_signals() L1176-1181
    - articraft/agent/feedback.py         compile_signal_bundle_from_exception() L1142

    ⚠️ 关键：成功和失败都用 output（不用 error）！
    - 源码: harness_compile.py L193-196 成功用 output; L201-204/L210-213 失败也用 output
    - LLM 看到的都是 {"result": "<compile_signals>..."} 格式
    - failure 信息已在 <compile_signals> XML 内（<failures> 段落），不需要 ToolResult.error
    """
    # ── 1. Freshness cache ──
    # 源码: harness_compile.py L179-184
    # latest_code_is_fresh() = report is not None and revision matches
    if rollout.code_is_fresh() and rollout.last_compile_bundle_dict is not None:
        # 源码: _render_reused_compile_tool_output() L142-147
        # 从 dict 重建 bundle 对象用于 render（或直接从 dict 提取 cached_content）
        cached_bundle = CompileSignalBundle.from_dict(rollout.last_compile_bundle_dict)
        cached_content = render_compile_signals(
            cached_bundle, repeated=False, failure_streak=0
        )
        return ToolResult(
            output=(
                # 前缀文案源码: harness_compile.py L143-146
                "Fresh compile already exists for the current code revision; "
                "`compile_model` was not re-run.\n"
                "Treat that compile result as authoritative unless you are about "
                "to edit code for one specific unresolved defect.\n\n"
                f"{cached_content}"
            ),
            compilation={"status": "success", "error": None},
        )

    # ── 2. 实际编译 ──
    try:
        # 源码: harness_compile.py L188-189 → compiler.py L618-626
        report = compile_urdf_report_maybe_timeout(script_path, sdk_package=sdk_package, ...)
        bundle = report.signal_bundle
        # 源码: _render_compile_tool_output() L125-140 → feedback.py render_compile_signals()
        content = render_compile_signals(bundle, repeated=False, failure_streak=0)
        # 源码: harness_compile.py L191-192
        rollout.mark_compile_attempt(bundle)   # ← 每次都记录（用于 reward）
        rollout.mark_compile_success(bundle)   # ← 仅成功时（用于 freshness）
        return ToolResult(output=content, compilation={"status": "success", "error": None})
    except Exception as exc:
        # 源码: harness_compile.py L206-213 (Exception) 或 L198-205 (TimeoutError)
        # ⚠️ 关键：compile_signal_bundle_from_exception 会从异常中提取
        # compile_signal_bundle 属性（由 compiler.py _attach_compiled_urdf_on_failure 设置）
        # 对于 QC failures（fail_if_isolated_parts 等），bundle 中的 signals 有 group="qc"
        # 这使得 reward 能区分不同严重程度的失败（而非简单二值）
        bundle = compile_signal_bundle_from_exception(exc)  # feedback.py L1142
        rollout.mark_compile_attempt(bundle)   # ← 失败也记录（关键！让 reward 有中间值）
        content = render_compile_signals(bundle, repeated=False, failure_streak=0)
        return ToolResult(output=content, compilation={"status": "error", "error": bundle.summary})
```

#### Phase 1 源码对照：Tool 执行逻辑 ↔ articraft 实现

> 以下逐一列出 RL env 中每个 `execute_*` 函数需要复刻的 articraft 行为，
> 含精确源码位置、错误消息文案、验证逻辑。实现时直接对照此表。

##### `execute_write_file` ↔ `articraft/agent/tools/write_code.py`

| RL env 步骤 | articraft 源码 | 行号 | 具体行为 |
|-------------|---------------|------|---------|
| 1. 必需函数检查 | `WriteCodeInvocation._missing_required_functions()` | L72–84 | `ast.parse` 找顶层 `FunctionDef`/`AsyncFunctionDef`；required = `["build_object_model", "run_tests"]`。`SyntaxError` 时跳过检查（交给步骤 3）|
| 1a. 缺失时报错 | L66–67 | — | `"write_code must include required top-level functions in the editable section: build_object_model, run_tests"` |
| 2. 写入文件 | `WriteCodeInvocation.execute()` | L62–63 | Phase 1 无 marker → 全文件替换（`replace_editable_code` 在无 region 时等价于全文件替换）|
| 3. 语法校验 | `_validate_python_syntax(new_full_code, file_path)` | L86–109 | `compile(code, filename, "exec")`；**不阻止写入**（写入已在步骤 2 完成）|
| 3a. 语法通过 | — | — | `compilation = {"status": "success", "error": None}` |
| 3b. SyntaxError | L96–105 | — | `"Syntax error: {msg} (line {lineno})"` + `{"status": "error", "error": ..., "error_line": lineno}` |
| 4. 成功返回 | L65 | — | `ToolResult(output="Code rewritten successfully", compilation=validation)` |
| — FileNotFoundError | L68 | — | `ToolResult(error="File {file_path} not found")` |
| — 通用异常 | L70 | — | `ToolResult(error="Error writing code: {str(exc)}")` |

**Phase 1 简化**：无 `path` 参数（固定为 `model.py`），无 editable region marker 逻辑。

##### `execute_replace` ↔ `articraft/agent/tools/edit_code.py`

| RL env 步骤 | articraft 源码 | 行号 | 具体行为 |
|-------------|---------------|------|---------|
| 1. 读全文件 | `EditCodeInvocation.execute()` | L42–44 | Phase 1 无 marker → 全文件即为搜索/替换范围 |
| 2a. old_string="" + 文件非空 | L50–53 | — | `"old_string cannot be empty unless the editable section is empty. Please provide the exact string to replace."` |
| 2b. old_string="" + 文件为空 | L55–63 | — | 直接写入 new_string，返回 `"Code edited successfully"` |
| 3. 计算匹配次数 | L66 | — | `occurrences = code.count(old_string)` |
| 3a. 0 次匹配 | L66–70 | — | `"Could not find the old_string in the code. Make sure the string matches exactly, including whitespace and indentation. "` |
| 3b. >1 次匹配 | L73–79 | — | `"The old_string appears {occurrences} times in the code. Please provide a longer, unique string that appears only once, or use replace_all=true to replace all occurrences."` |
| 4. 替换（1 次） | L82 | — | `code.replace(old_string, new_string, 1)`（RL 不暴露 `replace_all`）|
| 5. 写回 + 语法校验 | L89–110 | — | 同 write_file 的 `_validate_python_syntax` |
| 6. 成功返回 | L92 | — | `ToolResult(output="Code edited successfully", compilation=validation)` |

**Phase 1 简化**：去掉 `instruction`、`allow_multiple`/`replace_all` 参数。

##### `execute_read_file` ↔ `articraft/agent/tools/read_file.py`

| RL env 步骤 | articraft 源码 | 行号 | 具体行为 |
|-------------|---------------|------|---------|
| 1. 路径解析 | `ReadFileInvocation.execute()` | L60–62 | RL 用白名单 dict 替代 VirtualWorkspace；不在白名单 → `"Unable to resolve {path}"` |
| 2. offset/limit 验证 | L72–82 | — | `offset < 1` → `"offset must be >= 1"`；`limit < 1` → `"limit must be >= 1"` |
| 3. 空文件 | L84–85 | — | `offset > 1` → `"offset exceeds file length"`；否则 `ToolResult(output="")` |
| 4. offset 越界 | L87 | — | `offset > len(lines)` → `"offset exceeds file length"` |
| 5. 截取行范围 | L89–92 | — | 有 limit → `end = min(len(lines), start + limit)`；无 limit → 读到 EOF |
| 6. 格式化输出 | L96 | — | **`f"L{idx}: {lines[idx - 1]}"`**，`\n` 连接 |

##### `execute_compile` ↔ `articraft/agent/harness_compile.py` + `agent/compiler.py`

| RL env 步骤 | articraft 源码 | 文件:行号 | 具体行为 |
|-------------|---------------|----------|---------|
| 1. Freshness check | `CompileFeedbackLoop.latest_code_is_fresh()` | harness_compile.py:85–89 | `_last_successful_compile_revision == _current_edit_revision` |
| 1a. Fresh → 返回缓存 | `_render_reused_compile_tool_output()` | harness_compile.py:142–147 | 前缀: `"Fresh compile already exists for the current code revision; \`compile_model\` was not re-run.\nTreat that compile result as authoritative unless you are about to edit code for one specific unresolved defect.\n\n"` + rendered signals |
| 2. 实际编译 | `compile_urdf_report_maybe_timeout()` | compiler.py:618–626 | 签名: `(script_path, *, sdk_package="sdk", run_checks=True, ignore_geom_qc=False, target="full", rewrite_visual_glb=None)` → `CompileReport` |
| 2a. 超时 | compiler.py:680–685 | — | 默认 300s（RL 设 60s）。超时消息: `"URDF compile timed out after {timeout_seconds:.0f}s. This can happen if the generated script contains a long-running loop, expensive mesh processing, or very slow overlap checks..."` |
| 3. Bundle 构建 | `build_compile_signal_bundle()` | feedback.py:1100–1106 | `(*, status, warnings=(), test_report=None, exc=None)` → `CompileSignalBundle` |
| 3a. 异常时 | `compile_signal_bundle_from_exception()` | feedback.py:1142 | 优先读 `exc.compile_signal_bundle`；否则从 exc 构建 |
| 4. 渲染 | `render_compile_signals()` | feedback.py:1176–1181 | `(bundle, *, repeated=False, failure_streak=1)` → XML 格式 `<compile_signals>...</compile_signals>` |
| 5. ToolResult | harness_compile.py:199–214 | — | 成功/失败都用 `output`（不用 `error`！）。`compilation={"status": "success"/"error", "error": bundle.summary}` |

**Phase 1 简化**：固定 `repeated=False, failure_streak=0`（不追踪连续失败）。

##### Freshness 追踪 ↔ `articraft/agent/harness_compile.py`

| RL env 概念 | articraft 源码 | 行号 | 行为 |
|------------|---------------|------|------|
| `rollout.edit_revision` | `_current_edit_revision` | L54 | 初始 0 |
| `rollout.mark_code_mutated()` | `mark_code_mutated(tool_name)` | L91–93 | 仅 mutating tools 触发（`MUTATING_TOOL_NAMES = {"apply_patch", "replace", "write_file"}`）|
| `rollout.last_compile_revision` | `_last_successful_compile_revision` | L56 | compile 成功时 = `_current_edit_revision` |
| `rollout.code_is_fresh()` | `latest_code_is_fresh()` | L85–89 | `report is not None and _last_successful_compile_revision == _current_edit_revision` |

##### `<edit_retry_guidance>` 注入 ↔ `articraft/agent/harness_guidance.py`

| RL env 行为 | articraft 源码 | 行号 | 具体行为 |
|------------|---------------|------|---------|
| 触发条件 | `GuidanceInjector.maybe_inject_edit_code_guidance()` | L243–275 | tool 名 = `"replace"` + `result.error` 非空 + 含 `"Could not find the old_string in the code"` |
| One-shot 机制 | L260–261 | — | sig = `"replace_old_string_not_found"`；`_seen_tool_error_sigs` 去重，每 rollout 最多注入 1 次 |
| 注入内容 | L265–273 | — | `<edit_retry_guidance>\n- Your last replace failed because \`old_string\` did not match the file exactly.\n- Do NOT guess. Call \`read_file(path="model.py")\` again, then pick a smaller exact snippet from the current editable code as \`old_string\` and retry.\n- Keep edits surgical.\n</edit_retry_guidance>` |
| 注入方式 | L129–133 | — | 作为 `role: user` 消息追加到 conversation（RL 中即 env_response 返回的 UserMessage） |

##### `<compile_required>` 注入 ↔ `articraft/agent/harness_compile.py`

| RL env 行为 | articraft 源码 | 行号 | 具体行为 |
|------------|---------------|------|---------|
| 触发条件 | `append_compile_required_reminder()` | L96–113 | agent 尝试结束但 `not latest_code_is_fresh()` |
| 内容模板 | L104–108 | — | `<compile_required>\nThe latest code has changed since the last successful compile.\nRun \`compile_model\` before concluding.\n</compile_required>` |
| RL 差异 | — | — | RL 额外加 `compile_required_count`（max 3 次）防无限循环 |

##### `ToolResult.to_dict()` ↔ `articraft/agent/tools/base.py`

| 字段 | 条件 | 格式 |
|------|------|------|
| `"result"` | `error is None` | 值为 `output`（字符串） |
| `"error"` | `error is not None` | 值为 error 消息 |
| `"compilation"` | `compilation is not None` | `{"status": "success"/"error", "error": ...}` |
| `"tool_call_id"` | 若传入 | RL 中不使用此字段 |

##### `scaffold.py` 内容（30 行）↔ `articraft/scaffold.py`

源码参考：`articraft/scaffold.py`（完整文件，模型初始化模板）

```python
# 源码: articraft/scaffold.py（完整内容，~30 行）
# 此文件在 harness.py L614-624 _ensure_code_file() 中被复制到 work_dir/model.py
from sdk import ArticulatedObject, TestContext, TestReport

def build_object_model() -> ArticulatedObject:
    model = ArticulatedObject(name="draft_model")
    return model

def run_tests() -> TestReport:
    ctx = TestContext(object_model)
    return ctx.report()

object_model = build_object_model()
```

---

#### Phase 1.5 源码对照：Deferred Features

##### `repeated` / `failure_streak` 追踪 ↔ `harness_compile.py`

| 步骤 | articraft 源码 | 行号 | 行为 |
|------|---------------|------|------|
| 计算签名 | `_compile_signal_signature(bundle)` | L119–123 | `hashlib.sha1(json.dumps(bundle.to_dict(), sort_keys=True, separators=(",",":")).encode()).hexdigest()` |
| 检测重复 | `_render_compile_tool_output()` | L125–140 | `repeated = (sig == _last_compile_failure_sig)`；更新 `_last_compile_failure_sig` |
| 递增 streak | L133 | — | `_consecutive_compile_failure_count += 1`（成功时重置为 0）|
| 传给 render | L138 | — | `render_compile_signals(bundle, repeated=repeated, failure_streak=_consecutive_compile_failure_count)` |
| 下游效果 | feedback.py:1176+ | — | `repeated=True` → 追加 `"This failure matches the previous compile attempt."`；`streak >= 3` → 追加 `"This is compile failure {N} in a row."` + response_rules 建议 probe_model |

##### `exact_geometry_guidance` ↔ `harness_guidance.py`

| 步骤 | 源码 | 行号 | 行为 |
|------|------|------|------|
| AST 扫描 | `scan_code_contracts(text)` → `CodeContractScan` | L95–108 | 找 `run_tests()` 中 `ctx.expect_*` 引用的 exact element names（`EXACT_ELEMENT_KEYWORDS`）与 `visual(name=...)` 声明名的差集 |
| 触发 | `_maybe_inject_exact_geometry_contract_guidance()` | L143–179 | 成功 mutation 后 + 有 missing names |
| One-shot | — | — | `sig = sha256(json.dumps(missing_names))`；`_seen_exact_geometry_contract_sigs` 去重 |
| 模板 | L165–173 | — | `<exact_geometry_contract>\n- Authored exact checks reference names not present: {names}.\n- Before more geometry edits, restore those visual names or update/remove dependent exact checks.\n</exact_geometry_contract>` |

##### `baseline_qc_guidance` ↔ `harness_guidance.py`

| 步骤 | 源码 | 行号 | 行为 |
|------|------|------|------|
| 检测 | `_maybe_inject_baseline_qc_guidance()` | L181–215 | 扫描 `run_tests()` 中是否有 compiler-owned baseline QC 调用（`BASELINE_QC_CALLS`：`check_model_valid`, `check_mesh_assets_ready`, `fail_if_isolated_parts`, `warn_if_part_contains_disconnected_geometry_islands`, `fail_if_parts_overlap_in_current_pose`） |
| 模板 | L201–209 | — | `<baseline_qc_guidance>\n- run_tests() reintroduces compiler-owned baseline checks: {names}.\n- Leave baseline sanity/QC to compile_model.\n- Keep run_tests() for prompt-specific exact checks only.\n</baseline_qc_guidance>` |

##### `probe_model` 工具 ↔ `articraft/agent/tools/probe_model/`（4 文件）

| 文件 | 行数 | 职责 |
|------|------|------|
| `tool.py` | 203 | `ProbeModelTool` + `ProbeModelInvocation`：参数（`code`, `timeout_ms=600000`, `include_stdout=False`），子进程执行 |
| `runner.py` | 195 | 独立进程入口：`load_model_globals` → `ProbeSession` → `exec(snippet)` + `emit(value)` 契约 |
| `helpers.py` | 1108 | `ProbeSession` 类：只读几何探测 API（`part`, `joint`, `visual`, `aabb`, `dims`, `pair_report`, `gap_report` 等） |
| `description.py` | 36 | schema description 文本 |

**执行流**：
1. 绑定当前 `file_path`（model.py）
2. 子进程 `python -m agent.tools.probe_model.runner`，stdin JSON `{file_path, sdk_package, code}`
3. runner: `load_model_globals(file_path)` → 取 `object_model` → `TestContext` → `ProbeSession.build_namespace(emit=emit)` → `exec(compiled_snippet)`
4. 必须恰好 1 次 `emit(value)` 且 JSON 可序列化

##### `find_examples` 工具 ↔ `articraft/agent/tools/find_examples.py`（116 行）

| 参数 | 类型 | 默认 |
|------|------|------|
| `query` | str | 必填 |
| `limit` | int | 3 |

**执行**：BM25 词法搜索 curated example markdown（`agent.examples.search_example_documents`）。返回 `example_id`, `title`, `description`, `tags`, `content`, `match_quality`, `matched_tokens` 等。

##### Context Window 管理 ↔ `agent/providers/compaction_policy.py`

| 概念 | 源码位置 | 行为 |
|------|---------|------|
| Hard compaction | `HARD_PRESSURE_TRIGGER_RATIO = 0.90` | context 超 90% → 强制压缩 |
| Soft compaction bands | `high(0.85, streak≥3)`, `medium(0.70, streak≥4)`, `early(0.55, streak≥5)` | 需同时满足 pressure ratio + failure streak |
| Cooldown | `SOFT_COMPACTION_COOLDOWN_TURNS = 2` | 上次 soft compaction 后至少 2 turns 才可再压 |
| 实际执行 | `agent/providers/openai.py` / `gemini.py` | 调用 `decide_compaction()`，传入 `consecutive_compile_failure_count` |

RL Phase 2 如需 context 管理，参考此策略在 env_response 中实现 message 压缩。

---

**关键决策**：

1. **Tool schemas 手写**（不从 articraft import）——原因：articraft schema description 含 "virtual workspace"、Gemini parity 等 RL 不需要的概念；RL 版 `write_file` 去掉了 `path` 参数，`replace` 去掉了 `instruction`/`allow_multiple`。每个 schema 旁注明对应 articraft 源文件位置，方便后续对照更新。
2. **执行逻辑复刻 articraft 行为**（不是简化版）：
   - `write_file`：验证必需函数 → 全文件替换 → 语法校验（不阻止写入）→ 返回 `{"result": "Code rewritten successfully", "compilation": {...}}`
   - `replace`：harness 层拦截（empty old_string 检查）→ 全文件内唯一匹配验证 → 替换 → 语法校验 → 返回 `{"result": "Code edited successfully", "compilation": {...}}`
   - `read_file`：白名单路径 + `L{n}: ` 行号格式输出（返回完整文件）
   - `compile_model`：freshness 检查 → 若 fresh 返回缓存 → 否则 `compile_urdf_report_maybe_timeout` + `render_compile_signals`（含 `<response_rules>`）
3. **ToolResult 序列化为 JSON**：`ToolMessage(role="tool", content=json.dumps(tool_result.to_dict()), tool_call_id=tool_call.id)`。注意 `to_dict()` 不含 tool_call_id 字段（在 ToolMessage 层传递）
4. **完全不需要改 articraft 代码**，也不需要复杂的 harness 绑定
5. **只暴露 4 个 tools**（write_file, replace, read_file, compile_model）。`probe_model` 和 `find_examples` 初期不注册
6. **初期不实现 `repeated`/`failure_streak` 追踪**（deferred P1.5）——compile_model 总是以 `repeated=False, failure_streak=0` 调用 render。`response_rules` 仍然基于 failure type 生成
7. **Freshness 追踪 + 条件终止**（Phase 1 必需）：
   - 每次成功写操作递增 `edit_revision`；失败不递增
   - 成功编译记录 `last_compile_revision` + 缓存 `CompileSignalBundle`
   - code fresh 时 `compile_model` 返回缓存 + "Fresh compile already exists" 前缀
   - empty/text-only response 时：fresh → 终止（设 `state["final_env_response"] = []`）；dirty → 注入 `<compile_required>`（最多 3 次）
8. **`<edit_retry_guidance>` 注入**（Phase 1）：replace 失败 + "Could not find the old_string" → 注入一次性 guidance
9. **Provider tool set 选择：Gemini/Anthropic 路径**（replace + write_file，非 apply_patch）
   - Phase 1: 无 marker，read_file 返回完整 scaffold 文件
   - 模型看到完整上下文（imports、scaffold header），有助于理解 SDK API
10. **Phase 1 不支持 multimodal tasks**——只使用 text-only 的 task prompts。含图片的 tasks 如果存在则跳过或忽略 image
11. **编译超时**：使用 `URDF_COMPILE_TIMEOUT_SECONDS=60`（RL 专用，远低于默认 300s）。正常 compile P90 ~23s，60s 足够覆盖同时限制恶性代码影响
12. **In-process compile 进程安全**（长跑训练风险）：
    - `compile_urdf_report_maybe_timeout` 使用子进程模式（`URDF_COMPILE_TIMEOUT_SECONDS > 0`），隔离了 `os.chdir()`、`runpy.run_path()` 和 C++ 库 segfault 风险
    - 子进程模式中位数仅慢 ~0.4s，换来进程级隔离（推荐 RL 长跑时保持子进程模式）
    - 建议在 `scripts/envs/articraft.sh` 中设置 `ulimit -v` 限制单次 compile 内存用量
    - 如果训练中出现内存泄漏（`runpy.run_path` 反复创建模块命名空间），可通过 W&B 监控 worker 内存趋势

---

#### `env.py` — ArticraftEnv + load_environment()（约 280 行）

```python
# 源码参考（articraft 对应物）：
# - articraft/agent/harness.py  ArticraftAgent（整个类 ~1400 行）
# - 详细实现含源码行号见下方 "Step 1 详细计划" 中的完整 env.py 代码

class ArticraftEnv(vf.MultiTurnEnv):
    def __init__(self, articraft_root, max_turns=50, work_root="/tmp/articraft_rl",
                 artifact_policy=None, **kwargs):
        tool_schemas = get_tool_schemas()
        self.artifact_manager = ArticraftArtifactManager(work_root, artifact_policy or ArtifactPolicy())
        rubric = ArticraftRubric(artifact_manager=self.artifact_manager)
        super().__init__(tool_defs=tool_schemas, max_turns=max_turns, rubric=rubric, ...)

    async def setup_state(self, state):
        # 源码: harness.py L1027-1047 run() 初始化 + L614-624 _ensure_code_file
        task = Task.from_info(state["info"])
        work_dir = self.artifact_manager.make_rollout_dir(...)
        rollout = Rollout(task=task, trajectory_id=..., work_dir=work_dir, ...)
        state["rollout"] = rollout
        ...

    async def env_response(self, messages, state, **kwargs):
        # 源码: harness.py L1245-1314（终止 + tool 执行 + guidance）
        rollout = require_rollout(state)
        ...

def load_environment(**kwargs) -> ArticraftEnv:
    """verifiers 约定的工厂入口。"""
    return ArticraftEnv(**kwargs)
```

与 articraft 的交互：
- `setup_state` 读取 scaffold.py 作为初始 model.py（Phase 1 统一用 scaffold，无 marker，全文件可编辑）+ 构建 Turn 0 messages（预加载 docs + runtime guidance + task prompt）
- `env_response` 调 `tools.py` 中的函数（间接调 articraft compiler + feedback），ToolMessage.content = JSON 序列化的 ToolResult
- `load_environment()` 是 verifiers 发现和实例化 env 的约定入口

---

#### `artifact_manager.py` — Rollout 文件生命周期（约 100 行，新增）

```python
# 源码参考（articraft 对应物）：
# - articraft/storage/trajectories.py  TrajectoryStore（保存 trajectory.jsonl）
# - articraft/data/model.py  TurnRecord.to_dict() L58-68（序列化格式）
# - prime-rl/environments/blendergym/blendergym/env.py ArtifactManager（结构参考）
#
# 差异：articraft 用 TrajectoryStore 写 trajectory.jsonl（含完整 messages）；
# RL 版只写轻量 meta.json + trajectory.json（不含完整对话，verifiers 自己追踪 messages）

@dataclass
class ArtifactPolicy:
    save_meta_json: bool = True
    save_trajectory_json: bool = True
    save_per_turn_snapshots: bool = True
    keep_failed_only: bool = False
    max_rollouts_per_example: int = 0  # 0 = 无限

class ArticraftArtifactManager:
    def make_rollout_dir(self, traj_id, record_id, ...) -> Path: ...
    def begin_turn(self, work_dir, turn_idx) -> Path: ...
    def save_trajectory(self, rollout, metrics=None) -> None: ...
    def cleanup_rollout(self, rollout) -> None: ...
    def prune_old_rollouts(self, rollout) -> None: ...
```

仿 BlenderGym `ArtifactManager` 模式，封装所有文件 I/O 操作，统一管理 rollout artifact 的创建、保存和清理策略。

---

#### `rubric.py` — Reward 计算 + Artifact 写入（约 100 行）

```python
class ArticraftRubric(vf.Rubric):
    """
    源码参考（reward 计算逻辑）：
    - articraft/agent/feedback.py L1040-1080  CompileSignalBundle.check_fraction
      （通过/总数 的比值，即 RL reward 的核心信号）
    - articraft/data/model.py L122-135  Rollout.compile_check_fraction()
    - articraft/agent/feedback.py L198-210  QcSignal 分类：blocking_failure / warning / pass
    - prime-rl/environments/blendergym/blendergym/rubric.py BlenderGymRubric（结构参考）
    """
    def __init__(self, artifact_manager, reward_weights=None):
        super().__init__()
        self.artifact_manager = artifact_manager
        # weights 总和 = 1.0，避免 final_reward > 1.0
        w = {"check_fraction": 0.7, "build_success": 0.2, "compile_attempted": 0.1, **(reward_weights or {})}
        self.add_reward_func(self.check_fraction_reward, weight=w["check_fraction"])
        self.add_reward_func(self.build_success_bonus, weight=w["build_success"])
        self.add_reward_func(self.compile_attempted_bonus, weight=w["compile_attempted"])
        self.add_metric(self.blocking_failure_count)
        self.add_metric(self.warning_count)
        self.add_metric(self.turns_used)

    async def build_success_bonus(self, state, info) -> float:
        """代码能执行并构建 ObjectModel（即使 QC 失败）→ 区分 build failure vs QC failure"""
        rollout = require_rollout(state)
        bundle_dict = rollout.last_compile_attempt_dict
        if bundle_dict is None: return 0.0
        bundle = CompileSignalBundle.from_dict(bundle_dict)
        build_fails = [s for s in bundle.signals if s.blocking and s.group == "build"]
        return 0.0 if build_fails else 1.0

    async def check_fraction_reward(self, state, info) -> float:
        # 源码: feedback.py L1040-1080 check_fraction = passed / total
        rollout = require_rollout(state)
        ...

    @vf.cleanup
    async def write_artifacts_handler(self, state) -> None:
        rollout = require_rollout(state)
        self.artifact_manager.save_trajectory(rollout, metrics=state.get("metrics"))
        self.artifact_manager.cleanup_rollout(rollout)
```

与 articraft 的交互：仅读取 `CompileSignalBundle`（纯 dataclass，由 tools.py 生成）。`@vf.cleanup` 在 reward 计算完成后写 artifacts（此时 final_reward 已知）。

---

#### `prompts.py` — System Prompt + Turn 0 Messages（约 80 行）

**Articraft 原始 Turn 0 结构**（需忠实复刻）：

```
[System prompt — API 层面独立传入]
  <role>身份 + 质量要求</role>
  <link_naming>URDF 命名规范</link_naming>
  <tools>4 个 tool 的使用指南</tools>
  <modeling>SDK 用法 + 测试策略</modeling>

[Turn 0 conversation — 2 条 user messages]
  User #1: 预加载 2 篇核心 SDK docs 全文（quickstart + testing；Phase 1.5 加入 probe-tooling）
  User #2: <runtime_task_guidance>工作流指引</runtime_task_guidance> + 任务 prompt
```

**RL 精简版设计**：

```python
# 源码参考：
# - articraft/agent/prompts/system.py  get_system_prompt() → 4 段 XML
# - articraft/agent/prompts/system.py  L35-42  <role> 段
# - articraft/agent/prompts/system.py  L44-85  <tools> 段
# - articraft/agent/prompts/system.py  L87-135 <modeling> 段
# - articraft/agent/tools/__init__.py  build_first_turn_messages() L12-74
# - articraft/agent/prompts/runtime_guidance.py  RUNTIME_TASK_GUIDANCE

# System prompt: 精简 articraft 的 4 段 XML 为 RL 版
SYSTEM_PROMPT = """<role>
You are a 3D articulated object designer. You edit model.py using tools to build
articulated objects with the SDK. Quality bar: realistic geometry, functional joints,
no floating parts, no unintended overlaps.
</role>

<tools>
You have 4 tools: write_file, replace, read_file, compile_model.
- Read model.py before editing.
- Make one coherent change at a time.
- Run compile_model to check your work.
- Available SDK docs (read via read_file):
{available_docs_list}
</tools>

<modeling>
- model.py must define build_object_model() → ArticulatedObject and run_tests() → TestReport
- Import from sdk: ArticulatedObject, Part, Joint, TestContext, TestReport, etc.
- Use compile_model to run QC checks after edits.
</modeling>"""

# Turn 0 预加载文档策略（Token 预算优化）：
#
# ⚠️ 原方案嵌入 2 篇 docs 全文（~7000 tokens），与 seq_len 冲突严重。
# 改为只嵌入精简摘要（~800 tokens），完整版通过 read_file 按需查阅。
# 这也更接近 RL 设计哲学：让模型学会主动获取信息（read_file 是一个 tool）。
#
# articraft 原版嵌入全文是因为：推理时 context window 充裕（128K+）、无训练 seq_len 限制。
# RL 训练时必须在 seq_len 内完成整个对话，因此精简 Turn 0 是必要的。
PRELOAD_DOCS = [
    "docs/sdk/references/quickstart.md",
    "docs/sdk/references/testing.md",
]

# Phase 1 Turn 0 只嵌入摘要，不嵌入全文
DOCS_SUMMARY = """# SDK Quick Reference (use `read_file` for full docs)

## Available SDK Docs (read via read_file tool):
- docs/sdk/references/quickstart.md — SDK basics, Part/Joint/Visual creation
- docs/sdk/references/testing.md — TestContext, expect_* assertions, TestReport

## Key Patterns:
- `model = ArticulatedObject(name="...")`
- `part = Part(name="...", visual=Visual(shape=Box(...)))`
- `joint = Joint(name="...", parent=..., child=..., type="revolute", ...)`
- `model.set_root(root_part)` then add parts/joints
- `ctx = TestContext(object_model)` in run_tests()
- `ctx.report()` returns TestReport
"""

RUNTIME_TASK_GUIDANCE = """<runtime_task_guidance>
- Read the current `model.py` before editing.
- Make one small coherent change at a time.
- Treat visual realism as part of the deliverable.
- Run `compile_model` to check your latest revision.
- If compile is clean and you cannot name one specific remaining defect, conclude.
</runtime_task_guidance>"""

def build_system_prompt(readable_paths: dict[str, Path]) -> str:
    doc_paths = [p for p in readable_paths if p.startswith("docs/sdk/")]
    docs_list = "\n".join(f"  - {p}" for p in sorted(doc_paths))
    return SYSTEM_PROMPT.format(available_docs_list=docs_list)

def build_turn0_messages(readable_paths: dict[str, Path], task_prompt: str) -> vf.Messages:
    """构建完整的 Turn 0 prompt（state["prompt"] 的值）。

    基类 get_prompt_messages 在 Turn 0 直接返回 state["prompt"]，所以这里包含
    system message + user messages，是完整的 initial prompt。

    源码参考：
    - articraft/agent/tools/__init__.py  build_first_turn_messages() L12-74
    - User #1 结构: L21-45（预加载 SDK docs 全文，以 markdown 格式嵌入）
    - User #2 结构: L48-68（runtime_task_guidance XML + task prompt 文本）
    """
    # System message
    system_msg = SystemMessage(role="system", content=build_system_prompt(readable_paths))

    # User #1: SDK 精简摘要（~800 tokens，非全文嵌入）
    # ⚠️ 原 articraft 嵌入 2 篇 docs 全文（~7000 tokens），但 RL 训练
    # 受限于 seq_len，改为精简摘要 + 鼓励模型用 read_file 按需查阅
    docs_content = DOCS_SUMMARY

    # User #2: runtime guidance + 任务 prompt
    task_content = f"{RUNTIME_TASK_GUIDANCE}\n\n{task_prompt}"

    return [
        system_msg,
        UserMessage(role="user", content=docs_content),
        UserMessage(role="user", content=task_content),
    ]
```

**与 articraft 的对照**：

| 维度 | Articraft 原版 | RL 精简版 | 差异说明 |
|------|--------------|----------|---------|
| System prompt 结构 | 4-5 段 XML（provider-specific，~70 行） | 3 段 XML（统一版，~20 行） | 去掉 provider 差异、link_naming 细节 |
| Turn 0 预加载 docs | 3 篇全文（quickstart + probe + testing） | **2 篇**（quickstart + testing） | Phase 1 无 probe_model，去掉 probe-tooling |
| `<runtime_task_guidance>` | 5 条工作流指引 | **同** | 保持一致 |
| 可用 docs 列表 | 隐含在 VirtualWorkspace | 显式列入 system prompt | 替代 VirtualWorkspace 的路径发现 |
| Task prompt 位置 | Turn 0 User #2 | **同** | 保持一致 |

---

#### `dataset.py` — 数据加载（约 100 行）

```python
# 源码参考：
# - articraft/data/model.py  Record.from_directory() L88-115
# - articraft/data/records/rec_*/  目录结构：prompt.txt, model.py, record.json
# - articraft/agent/harness.py L238-242  record 过滤逻辑（cadquery 等）
def build_dataset(data_root, *, split, filter_cadquery=True, ...) -> Dataset:
    # 扫描 articraft/data/records/rec_*/
    # 读 prompt.txt + model.py
    # Phase 1: 过滤 cadquery records（31.6% 的 records 含 `import cadquery`，
    #           因不装 cadquery 这些 model.py 无法编译）
    # 过滤后保留 ~7,387 条纯 SDK 几何记录，数据量充足
    # 构建 HF Dataset
```

与 articraft 的交互：只读 `articraft/data/records/` 目录结构（纯文件 I/O）。

过滤逻辑（regex，覆盖所有 cadquery 导入形式）：
```python
import re
_CADQUERY_IMPORT_RE = re.compile(r'\b(import cadquery|from cadquery)\b')

# 在 build_dataset 循环中：
if filter_cadquery and _CADQUERY_IMPORT_RE.search(model_py_text):
    continue
```
（覆盖 `import cadquery`, `import cadquery as cq`, `from cadquery import ...` 等变体）

---

#### `schema.py` — 数据结构（约 60 行）

```python
SCHEMA_VERSION = "articraft-trajectory-v1"

@dataclass(frozen=True)
class Task:
    """
    源码参考：
    - articraft/data/model.py  Record dataclass（record_id, prompt, category 等）
    - articraft/agent/harness.py L206-210  从 record 提取 task 属性
    """
    record_id: str
    prompt_text: str
    category_slug: str | None = None
    num_test_checks: int = 0  # Phase 1 固定为 0；Phase 2+ 从 dataset 元数据获取
    @classmethod
    def from_info(cls, info: dict) -> "Task": ...

@dataclass
class TurnRecord:
    """
    源码参考：
    - articraft/data/model.py  TurnRecord L42-68（记录每 turn 的 tool_calls 和 compile 状态）
    """
    turn: int
    tool_calls: list[dict]
    compile_attempted: bool = False
    compile_success: bool | None = None
    compile_signals: dict | None = None

@dataclass
class Rollout:
    """
    源码参考（等价 articraft 的分散状态）：
    - edit_revision / last_compile_revision: articraft/agent/harness_compile.py L48-60
      CompileFeedbackLoop._last_compile_report / latest_code_is_fresh()
    - mark_code_mutated(): articraft/agent/harness.py L864-865 _mark_code_mutated()
      → 递增 _code_mutation_counter
    - mark_compile_success(): articraft/agent/harness_compile.py L191-192
      _last_compile_report = report
    - compile_required_count: articraft/agent/harness_compile.py L96-113
      _compile_required_reminder_count
    - edit_retry_injected: articraft/agent/harness_guidance.py L261
      _edit_retry_guidance_injected
    """
    task: Task
    trajectory_id: str
    work_dir: Path
    max_turns: int
    turns: list[TurnRecord] = field(default_factory=list)
    final_reward: float | None = None
    metadata: dict | None = None
    # Freshness 追踪（源码: harness_compile.py L48-60）
    edit_revision: int = 0
    last_compile_revision: int = -1
    # ⚠️ 仅在成功编译时更新（用于 freshness check + 缓存返回）
    last_compile_bundle_dict: dict | None = None
    # ⚠️⚠️ 每次 compile attempt 都更新（成功 OR 失败），用于 reward 计算
    # 这是解决 "reward 二值问题" 的关键：
    # - articraft 的 compile 在 QC failure 时 raise 异常
    # - 异常中的 compile_signal_bundle 包含 group="qc" 的具体 failure signals
    # - 如果只用 last_compile_bundle_dict（仅成功更新），reward 永远只有 0 或 1
    # - 用 last_compile_attempt_dict 让失败的 compile 也参与 reward 计算
    last_compile_attempt_dict: dict | None = None
    # 源码: harness_compile.py L96 _compile_required_reminder_count
    compile_required_count: int = 0
    # Guidance one-shot flags（源码: harness_guidance.py L261）
    edit_retry_injected: bool = False

    def code_is_fresh(self) -> bool:
        # 源码: harness_compile.py L85-89 latest_code_is_fresh()
        return self.last_compile_revision == self.edit_revision and self.last_compile_revision >= 0

    def mark_code_mutated(self):
        # 源码: harness.py L864-865 _mark_code_mutated → _code_mutation_counter += 1
        self.edit_revision += 1

    def mark_compile_attempt(self, bundle):
        """每次 compile 调用都更新（成功和失败），用于 reward 计算。"""
        self.last_compile_attempt_dict = bundle.to_dict()

    def mark_compile_success(self, bundle):
        # 源码: harness_compile.py L191-192 成功后缓存 report
        self.last_compile_revision = self.edit_revision
        self.last_compile_bundle_dict = bundle.to_dict()  # freshness 缓存
        self.compile_required_count = 0

def require_rollout(state: dict) -> Rollout:
    rollout = state.get("rollout")
    if not isinstance(rollout, Rollout):
        raise RuntimeError("Articraft rollout state missing; setup_state likely failed")
    return rollout
```

> 注：`scaffold.py` 不再独立存在。scaffold 读取 + work_dir 初始化在 `env.py` 的 `setup_state` 中内联完成（约 5 行），无需单独模块。

---

### 与 articraft 的调用关系图

```
prime-rl ArticraftEnv
    │
    ├── __init__()
    │       ├── get_tool_schemas()  ← 手写 TOOL_SCHEMAS
    │       ├── ArticraftArtifactManager(work_root, policy)
    │       └── ArticraftRubric(artifact_manager=...)
    │
    ├── setup_state()  ← Turn 0 prompt 也在此构建
    │       ├── artifact_manager.make_rollout_dir(...)
    │       ├── 读 articraft/scaffold.py → 写 work_dir/model.py
    │       ├── get_readable_paths(articraft_root, work_dir)
    │       │       └── SdkProfile.docs_full → 构建白名单 dict
    │       ├── state["rollout"] = Rollout(task=..., work_dir=..., ...)
    │       └── state["prompt"] = [
    │               SystemMessage(build_system_prompt(readable_paths)),
    │               UserMessage(预加载 3 篇核心 docs 全文),
    │               UserMessage(<runtime_task_guidance> + 任务 prompt),
    │           ]
    │
    │   # 不 override get_prompt_messages：
    │   #   Turn 0: 基类直接返回 state["prompt"]（已在 setup_state 构建完毕）
    │   #   Turn N: 基类调用 env_response → 返回 tool results
    │
    ├── env_response()
    │       ├── require_rollout(state)
    │       ├── execute_write_file()    ← 验证必需函数 → 写入 → 语法校验
    │       ├── execute_replace()       ← 唯一匹配验证 → 替换 → 语法校验
    │       ├── execute_read_file()     ← "L{n}: " 格式 + 白名单路径
    │       ├── execute_compile()
    │       │       └── articraft compile_urdf_report_maybe_timeout(script_path)
    │       │               ├── runpy.run_path("model.py")
    │       │               ├── sdk.build_object_model()
    │       │               ├── QC checks
    │       │               └── → CompileReport
    │       │       └── articraft render_compile_signals(bundle)
    │       │               └── → 结构化 XML 文本
    │       ├── ToolMessage.content = json.dumps(tool_result.to_dict())
    │       └── rollout.turns.append(TurnRecord(...))
    │
    └── rubric
            ├── check_fraction_reward → 读 rollout.last_compile_bundle_dict（最后一次 compile 结果）
            ├── compile_attempted_bonus → any(t.compile_attempted)
            └── @vf.cleanup write_artifacts_handler
                    ├── artifact_manager.save_trajectory(rollout)
                    └── artifact_manager.cleanup_rollout(rollout)
```

### 需要从 articraft 获取的"静态资源"

| 资源 | 用途 | 获取方式 |
|------|------|---------|
| `scaffold.py` 内容 | 初始化空 model.py | `articraft_root / "scaffold.py"` 读文件 |
| `sdk/_docs/*.md` (26 个文件) | 模型通过 `read_file` 按需查阅 | 虚拟路径 `docs/sdk/references/...` 映射到磁盘 `sdk/_docs/...`（用 `_DOC_PATH_ALIASES`） |
| 2 篇核心 docs 全文 | Turn 0 预加载（quickstart + testing；Phase 1.5 加入 probe-tooling） | `PRELOAD_DOCS` 用虚拟路径引用 |
| Tool schemas | 传给 vLLM 的 function 定义 | 手写（注明对应 articraft 源文件位置，便于对照） |
| Records 数据 | 训练数据集 | 读 `data/records/rec_*/` |

### 不需要从 articraft 使用的（之前方案列了但实际不用）

| 原方案说的 | 为什么不用 | 备注 |
|-----------|---------|------|
| `build_tool_registry()` | 需要 provider 分化 + harness 绑定，我们只需要 schemas（直接 import tool 类的 `.schema`）| — |
| `CompileFeedbackLoop` | 核心价值是 revision freshness 缓存 + 连续失败计数 + 重复检测。RL 中编译耗时极低（~0.08s），初期不需要缓存 | 若模型刷 compile 浪费 turns，可后加 5 行 freshness check |
| `GuidanceInjector` | 初期不注入 guidance（让 RL 自己学策略，避免依赖提示） | 后续可选加 `edit_retry_guidance` 防 replace 死循环 |
| `ToolRegistry` | 不需要动态注册/查找，4 个 tool 硬编码分发即可 | — |
| `VirtualWorkspace` | 其完整功能含：虚拟文件创建、editable region markers、write tracking、路径映射。Phase 1 中 `read_file` 只需简单白名单映射（`model.py` + `sdk/_docs/*`），全文件可编辑，不需要虚拟文件系统 | 白名单 ~15 行代码 vs VirtualWorkspace ~300 行 |
| `storage.RecordStore` | 不需要创建/修改 records，只需读数据。简单目录扫描即可 | 非 ArtifactManager 替代；我们的使用场景更简单（只读） |
| `storage.DatasetStore` | 其 manifest 索引逻辑偏重，RL 的 dataset loader 只需简单扫描 `records_root` | 同上 |
| `storage.trajectories` | **被 ArtifactManager 替代**——我们写 BlenderGym 风格的 trajectory.json + meta.json，包含保留策略 | ArtifactManager 覆盖此功能 |
| `cli/` | RL 入口是 config + prime-rl orchestrator，不经过 CLI | — |
| `viewer/api/` 的修改功能 | viewer 的 record mutation/编辑功能在 RL 中不需要 | 但浏览/渲染功能会完整复用 |
| `agent/harness.py` (ArticraftAgent) | 见下方详细分析 | — |

#### 为什么不复用 `agent/harness.py`

`harness.py` 是 articraft 的核心 agent runner（~1360 行，`ArticraftAgent` 类），负责 LLM 调用 → tool 执行 → 终止判断的完整循环。

**不复用的核心原因：职责在 RL 中被拆散到不同层**

| harness.py 中的职责 | RL 中由谁负责 | 复用可能性 |
|-------------------|-------------|-----------|
| LLM 调用循环 (`run()` 主循环 ~320行) | verifiers `MultiTurnEnv` rollout 机制 | 不可复用 |
| Provider 初始化 (OpenAI/Anthropic/Gemini) | vLLM（单一推理引擎） | 不可复用 |
| Tool 分发 (`_execute_tool()` ~200行) | 我们的 `tools.py` execute_* 函数 | 参考模式，不直接复用 |
| Compile 执行 (`_execute_compile_model()` ~50行) | 我们的 `execute_compile()` | 简化后重写 |
| Message codec (extract/build ~40行) | verifiers 消息类型 | 不可复用 |
| Cost tracking (~60行) | 不需要（vLLM 自托管） | — |
| OpenAI prompt cache (~80行) | 不需要（vLLM） | — |
| TUI display (~30行) | 不需要 | — |
| Trace writing (~20行) | 我们的 observability | 独立实现 |
| Scaffold 初始化 (`_ensure_code_file` ~10行) | 我们的 `setup_state()` | 等价重写 |
| VirtualWorkspace 初始化 (~10行) | 我们的 `get_readable_paths()` | 简化替代 |
| Guidance injection 调用 (~40行) | 初期不用 | — |
| CompileFeedbackLoop 调用 (~30行) | 初期不用 | — |
| Context window pressure (~30行) | 不需要 | — |
| find_examples 缓存压缩 (~50行) | 不需要此 tool | — |

**决策总结**：harness.py 60%+ 代码与 RL 无关（TUI、cost、prompt cache、provider、context pressure）。剩余 40% 的核心逻辑（LLM 循环 + tool 执行）在 RL 中由 verifiers + vLLM + `ArticraftEnv.env_response()` 分别承担，无法直接 import 任何一个方法。

**可参考的模式**（实现时参考，不 import）：
- `_execute_compile_model()` 的 freshness 缓存逻辑 → 若后续需要防模型刷 compile
- `harness_compile.py` 中 `render_compile_signals` 的调用方式 → 已 import 此函数
- `_minimal_scaffold_text()` → 等价于我们的 `get_scaffold_content()`

---

### 总结：工作量分解

| 工作 | 位置 | 难度 | 行数估算 |
|------|------|------|---------|
| Tool schemas + dispatch + compile + 白名单 | `articraft_env/tools.py` | 中 | ~180 |
| ArticraftEnv + load_environment() | `articraft_env/env.py` | 中 | ~280 |
| Rubric reward + @vf.cleanup | `articraft_env/rubric.py` | 中 | ~100 |
| ArtifactManager（rollout 文件生命周期） | `articraft_env/artifact_manager.py` | 中 | ~100 |
| Dataset loader | `articraft_env/dataset.py` | 低 | ~100 |
| System prompt | `articraft_env/prompts.py` | 低 | ~50 |
| Schema/Rollout/TurnRecord + require_rollout | `articraft_env/schema.py` | 低 | ~60 |
| PEP 562 lazy imports | `articraft_env/__init__.py` | 低 | ~30 |
| TOML 训练配置 | `configs/articraft/rl.toml` | 低 | ~80 |
| **articraft 侧修改** | **无** | — | **0** |

### 部署时的依赖安装

在 KAOLA pod 中：
```bash
# 方案 1: articraft 源码在 --sync-code 中一起上传
uv pip install -e /data/work/articraft          # SDK + agent tools
uv pip install -e /data/work/prime-rl/environments/articraft  # RL env wrapper

# 方案 2: articraft 打包到 S3，pod 启动时安装
cat /threed-code/user/tools/articraft.tar | tar xf - -C /local-ssd/
uv pip install -e /local-ssd/articraft
```

---

## BlenderGym 脚手架参考（逐文件对比）

BlenderGym 包结构：

```
environments/blendergym/
├── pyproject.toml
├── blendergym/
│   ├── __init__.py           ← PEP 562 lazy imports
│   ├── env.py                ← 核心：BlenderGymEnv(vf.MultiTurnEnv)
│   ├── rubric.py             ← reward：BlenderGymRubric(vf.Rubric)
│   ├── dataset.py            ← HF Dataset 构建
│   ├── prompts.py            ← SYSTEM_PROMPT + TASK_INSTRUCTION
│   ├── schema.py             ← Task, Rollout, TurnRecord dataclasses
│   ├── artifact_manager.py   ← 文件管理：work_dir/turn_N/...
│   ├── trajectory_writer.py  ← 持久化 trajectory.json/html
│   ├── render.py             ← Blender subprocess wrapper
│   ├── gpu_mem.py            ← GPU 资源管理
│   └── services/             ← FastAPI render/score 服务
│       ├── launcher.py
│       ├── render/server.py, client.py, persistent_blender.py, worker_loop.py, pool.py
│       └── score/server.py, client.py, clip_scorer.py
└── tests/
```

---

### 1. `pyproject.toml`

**BlenderGym**:
```toml
[project]
name = "blendergym"
version = "0.0.1"
description = "BlenderGym multimodal RL environment for prime-rl (verifiers package)"
requires-python = ">=3.12,<3.13"
dependencies = [
    "verifiers>=0.1.10",
    "pillow>=10.0.0",
    "datasets>=4.0.0",
    "open_clip_torch>=2.24.0",
    "torch>=2.4.0",
    "fastapi>=0.111.0",
    "uvicorn[standard]>=0.30.0",
    "httpx>=0.27.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["blendergym"]
```

**Articraft (proposed)**:
```toml
[project]
name = "articraft-env"
version = "0.0.1"
description = "Articraft articulated-object generation RL environment for prime-rl (verifiers package)"
requires-python = ">=3.12,<3.13"
dependencies = [
    "verifiers>=0.1.10",
    "manifold3d>=3.3.2",
    "trimesh>=4.11.3",
    "python-fcl>=0.7.0.8",
    "networkx>=3.6.1",
    "numpy>=2.0",
    "pydantic>=2.0.0",
    "aiofiles>=24.1.0",
]
# NOTE: cadquery 已排除 — lazy import，不在核心编译路径。
# 安装 articraft 本体时用 --no-deps，这里只声明 RL env 自身直接需要的依赖。

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["articraft_env"]
```

**差异**: BlenderGym 依赖 CLIP/torch/FastAPI（因为有 render+score 服务）。Articraft 依赖几何库（manifold3d/trimesh/fcl），无 GPU、无外部服务。cadquery 已排除（lazy import，非核心路径）。

---

### 2. `__init__.py`

**BlenderGym**:
```python
__all__ = ["ArtifactManager", "ArtifactPolicy", "BlenderGymEnv", "BlenderGymRubric", ...]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "BlenderGymEnv": (".env", "BlenderGymEnv"),
    "load_environment": (".env", "load_environment"),
    "BlenderGymRubric": (".rubric", "BlenderGymRubric"),
    ...
}

def __getattr__(name: str):
    # PEP 562 lazy import pattern
```

**Articraft (proposed)**: 相同模式，只是导出不同的名字。

---

### 3. `env.py` — 核心环境类

**BlenderGym 脚手架**:
```python
class BlenderGymEnv(vf.MultiTurnEnv):
    def __init__(self, data_root, task_types, max_turns, work_root, keep_failed_only,
                 env_name, split, eval_split, eval_holdout,
                 render_service_url, score_service_url, render_timeout_s, **kwargs):
        # 初始化外部服务 client
        self.render_client = RenderClient(render_service_url, render_timeout_s)
        self.artifact_manager = ArtifactManager(self.work_root, policy)
        self.parser = vf.XMLParser(["code"], answer_field="code")
        rubric = BlenderGymRubric(score_service_url=..., parser=self.parser, artifact_manager=...)

        super().__init__(
            dataset=_train_dataset_builder,      # callable → Dataset
            eval_dataset=_eval_dataset_builder,  # callable → Dataset
            system_prompt=SYSTEM_PROMPT,
            max_turns=max_turns,
            rubric=rubric,
            parser=self.parser,
            **kwargs,
        )

    async def setup_state(self, state: vf.State) -> vf.State:
        # 从 state["info"] 构建 Task, 创建 work_dir, 预加载图像
        rollout = Rollout(task=task, trajectory_id=..., work_dir=..., ...)
        state["rollout"] = rollout
        return state

    async def get_prompt_messages(self, state: vf.State) -> vf.Messages:
        # Turn 0: system + goal_image + init_image + start_code
        # Turn N: prev_prompt + prev_completion + latest_render
        ...

    async def env_response(self, messages, state, **kwargs) -> vf.Messages:
        # BlenderGym: 返回空（所有 message 构建在 get_prompt_messages 中）
        return []

    async def add_model_response(self, state, prompt_messages, response) -> None:
        # 1. super() push step
        # 2. parse code from <code>...</code>
        # 3. render via render_client → fill TurnRecord
        ...


def load_environment(**kwargs) -> BlenderGymEnv:
    """Factory for verifiers.load_environment("blendergym")."""
    return BlenderGymEnv(**kwargs)
```

**Articraft (proposed)** — 关键区别：

```python
class ArticraftEnv(vf.MultiTurnEnv):
    """
    源码参考（articraft 对应物）：
    - articraft/agent/harness.py  ArticraftAgent.__init__() L181-268
    - articraft/agent/harness.py  ArticraftAgent.run() L1021-1342（主循环）
    - 对比: prime-rl/environments/blendergym/blendergym/env.py BlenderGymEnv
    """
    def __init__(self, articraft_root, max_turns=50, work_root="/tmp/articraft_rl",
                 sdk_package="sdk", env_name="articraft", split="train",
                 eval_split="eval", eval_holdout=50,
                 artifact_policy=None, **kwargs):
        self.articraft_root = Path(articraft_root)
        self.work_root = Path(work_root)
        self.sdk_package = sdk_package
        tool_schemas = get_tool_schemas()
        policy = artifact_policy or ArtifactPolicy()
        self.artifact_manager = ArticraftArtifactManager(self.work_root, policy)
        rubric = ArticraftRubric(artifact_manager=self.artifact_manager)

        super().__init__(
            dataset=self._train_dataset_builder,
            eval_dataset=self._eval_dataset_builder,
            # 不传 system_prompt：setup_state 完全控制 state["prompt"]
            max_turns=max_turns,
            rubric=rubric,
            tool_defs=tool_schemas,
            **kwargs,
        )

    async def setup_state(self, state: vf.State) -> vf.State:
        """
        源码参考：
        - articraft/agent/harness.py L1027-1047  run() 初始化阶段
        - articraft/agent/harness.py L614-624    _ensure_code_file()（写 scaffold）
        - articraft/agent/tools/__init__.py      build_first_turn_messages()
        - articraft/agent/workspace_docs.py      build_virtual_workspace()
        """
        task = Task.from_info(state["info"])
        traj_id = uuid4().hex[:12]
        work_dir = self.artifact_manager.make_rollout_dir(traj_id, task.record_id, ...)
        # 源码: harness.py L614-624 _ensure_code_file → 写 scaffold
        (work_dir / "model.py").write_text(self._initial_model_code(task))
        # 源码: workspace_docs.py build_virtual_workspace + _DOC_PATH_ALIASES
        readable_paths = get_readable_paths(self.articraft_root, work_dir)
        rollout = Rollout(task=task, trajectory_id=traj_id, work_dir=work_dir,
                          max_turns=self.max_turns)
        state["rollout"] = rollout
        state["readable_paths"] = readable_paths
        # 源码: tools/__init__.py build_first_turn_messages (sdk_docs + task)
        state["prompt"] = build_turn0_messages(readable_paths, task.prompt_text)
        return state

    async def env_response(self, messages, state, **kwargs) -> vf.Messages:
        """
        源码参考：
        - 终止逻辑: articraft/agent/harness.py L1245-1273 (_handle_finish_attempt)
        - compile_required: articraft/agent/harness_compile.py L96-113
        - tool 执行: articraft/agent/harness.py L1275-1298 (_execute_tool_calls_batch)
        - guidance 注入: articraft/agent/harness.py L1305-1314
        - tool message 构造: articraft/agent/harness.py L888-899
        """
        rollout = require_rollout(state)
        readable_paths = state["readable_paths"]
        last_msg = messages[-1]
        tool_calls = getattr(last_msg, "tool_calls", None) or []
        text_content = getattr(last_msg, "content", "").strip()

        # ── 终止 / compile_required 逻辑 ──
        # 源码: harness.py L1245-1273
        # articraft 中 empty/text-only response 走 _handle_finish_attempt
        # 若 latest_code_is_fresh() → return AgentResult (终止)
        # 否则 → append_compile_required_reminder (继续)
        if not tool_calls:
            if rollout.code_is_fresh():
                state["final_env_response"] = []
                return []
            else:
                # 源码: harness_compile.py L96-113 append_compile_required_reminder
                rollout.compile_required_count += 1
                if rollout.compile_required_count > 3:
                    # RL 特有：articraft 无此上限，RL 中防无限 loop
                    state["final_env_response"] = []
                    return []
                # 消息模板源码: harness_compile.py L104-108
                return [UserMessage(content=(
                    "<compile_required>\n"
                    "The latest code has changed since the last successful compile.\n"
                    "Run `compile_model` before concluding.\n"
                    "</compile_required>"
                ))]

        # ── 正常 tool 执行 ──
        # 源码: harness.py L1275-1298 顺序执行 tool_calls
        result_messages = []
        raw_results = []  # JSON strings，用于 guidance 检查
        turn_record = TurnRecord(turn=len(rollout.turns), tool_calls=[])
        for tool_call in tool_calls:
            name, args_str = tool_call.name, tool_call.arguments
            content = await self._dispatch_tool(name, args_str, rollout, readable_paths)
            raw_results.append(content)
            turn_record.tool_calls.append({"name": name, "args": args_str})
            if name == "compile_model":
                turn_record.compile_attempted = True
            # tool message 构造源码: harness.py L888-896
            # content = json.dumps({k:v for k,v in result.to_dict().items() if k != "tool_call_id"})
            result_messages.append(ToolMessage(
                role="tool", content=content,
                tool_call_id=tool_call.id or f"tc_{len(result_messages)}",
            ))

        # ── Guidance 注入（tool results 之后）──
        # 源码: harness.py L1305-1314 依次调
        # maybe_inject_edit_code_guidance → maybe_inject_code_contract_guidance
        guidance_msgs = self._maybe_inject_guidance(tool_calls, raw_results, rollout)

        rollout.turns.append(turn_record)
        # TITO: ToolMessages 先，UserMessages 后（如果有 guidance）
        return result_messages + guidance_msgs

    async def _dispatch_tool(self, name: str, args_str: str, rollout: Rollout,
                             readable_paths: dict[str, "Path"]) -> str:
        """统一 tool 分发入口。返回 JSON 序列化的 ToolResult content string。

        源码参考：
        - articraft/agent/harness.py L682-899 _execute_tool()
        - JSON 解析: harness.py L717-732
        - args 非 dict: harness.py L734-749
        - compile_model 硬编码拦截: harness.py L751-784
        - replace 前置校验: harness.py L786-840
        - 通用 tool 执行: harness.py L842-899
        - 错误处理: harness.py L866-886 (ValidationError / generic Exception)

        ⚠️ compile_model 通过 asyncio.to_thread 包装，避免阻塞事件循环。
        compile_urdf_report_maybe_timeout 是同步函数（持有 _MODEL_EXECUTION_LOCK），
        最高耗时 60s。如果在 async 中直接调用会冻结所有并发 rollout。
        read/write/replace 是微秒级操作，同步调用即可。
        """
        # 1. JSON 解析
        # 源码: harness.py L717-732
        try:
            args = json.loads(args_str) if args_str.strip() else {}
        except json.JSONDecodeError as e:
            # 源码: harness.py L720-721
            return json.dumps({"error": f"Invalid JSON in tool arguments: {e}"})
        # 源码: harness.py L734-749
        if not isinstance(args, dict):
            return json.dumps({"error": "Tool arguments must be a JSON object"})

        # 2. Tool 分发
        script_path = rollout.work_dir / "model.py"
        try:
            if name == "write_file":
                content = args.get("content")
                if content is None:
                    return json.dumps({"error": "Missing required parameter: content"})
                result = execute_write_file(content, script_path, rollout)
            elif name == "replace":
                old = args.get("old_string")
                new = args.get("new_string")
                if old is None or new is None:
                    return json.dumps({"error": "Missing required parameter: old_string and new_string are both required"})
                result = execute_replace(old, new, script_path, rollout)
            elif name == "read_file":
                path = args.get("path", "model.py")
                result = execute_read_file(path, readable_paths, args.get("offset"), args.get("limit"))
            elif name == "compile_model":
                # 源码: harness.py L751-771 unexpected params 检查
                if args:
                    return json.dumps({"error": f"Invalid parameters for compile_model. Unexpected parameters: {sorted(args.keys())}"})
                # ⚠️ compile 是耗时操作（中位数 80ms，P90 23s，max 60s），
                # 必须用 asyncio.to_thread 避免阻塞事件循环
                result = await asyncio.to_thread(execute_compile, script_path, rollout)
            else:
                # 源码: harness.py L848-851 "Tool {func_name} not found"
                return json.dumps({"error": f"Unknown tool: {name}. Available tools: write_file, replace, read_file, compile_model"})
        except Exception as e:
            # 源码: harness.py L883-884 generic exception handling
            return json.dumps({"error": f"Tool execution error ({name}): {type(e).__name__}: {e}"})

        # 3. 序列化
        # 源码: harness.py L893-894
        # json.dumps({k:v for k,v in result.to_dict().items() if k != "tool_call_id"})
        # RL 中 ToolResult 不含 tool_call_id 字段，to_dict() 输出等价
        return json.dumps(result.to_dict())

    async def add_model_response(self, state, prompt_messages, response) -> None:
        await super().add_model_response(state, prompt_messages, response)
        # 可选：per-turn snapshot via artifact_manager


def load_environment(**kwargs) -> ArticraftEnv:
    """verifiers 约定的工厂入口。"""
    return ArticraftEnv(**kwargs)
```

**核心架构差异**:

| 维度 | BlenderGym | Articraft |
|------|-----------|-----------|
| 基类 | `vf.MultiTurnEnv` | `vf.MultiTurnEnv`（不用 ToolEnv） |
| 交互模式 | `<code>` XML 解析 → 环境 render | vLLM tool_calling → `env_response` 手动分发执行 |
| `env_response` | 返回空 `[]` | **返回 tool results**（ToolMessage 列表） |
| `get_prompt_messages` | 完全 override（手动拼图像 + code） | 不 override（Turn 0 在 setup_state 重建 state["prompt"]） |
| `add_model_response` | parse XML + call render service | 记录 compile 状态到 TurnRecord |
| 外部依赖 | Render Service (HTTP) + Score Service (HTTP) | Phase 1: 无（全 in-process）；Phase 2+: Viewer Render + CLIP Score (HTTP) |
| Parser | `vf.XMLParser(["code"])` | 无需 parser（tool_calling 原生支持） |

**为什么不用 `ToolEnv`/`StatefulToolEnv`**：verifiers `ToolEnv` 假设 tools 是简单 Python callable（`convert_func_to_tool_def` 从函数签名生成 schema），`env_response` 只能返回 `ToolMessage`，且 `assert tool_calls is not None`（无 tool_calls 直接崩溃）。Articraft 的 `compile_model` 执行逻辑远超"调函数"（子进程 compile + render_compile_signals），且需要对无 tool_calls 情况优雅处理。`MultiTurnEnv` 的 `env_response` 可返回任意 Messages，手写 tool_call 分发约 ~40 行，比 force-fit ToolEnv 更清晰。

---

### 4. `rubric.py`

**BlenderGym**:
```python
class BlenderGymRubric(vf.Rubric):
    def __init__(self, score_service_url, parser, artifact_manager):
        super().__init__(parser=parser)
        self.score_client = ScoreClient(score_service_url)
        self.artifact_manager = artifact_manager

        self.add_reward_func(self.clip_similarity, weight=1.0)  # 主 reward
        self.add_metric(self.xml_parse_success)                 # 诊断 metric
        self.add_metric(self.render_success)
        self.add_metric(self.code_non_empty)

    async def clip_similarity(self, state, info) -> float:
        # 调 score service 计算 CLIP cosine similarity
        # 返回 0.0-1.0
        ...

    @vf.cleanup
    async def write_artifacts_handler(self, state) -> None:
        # 写 trajectory.json/html，应用 retention policy
        ...
```

**Articraft (proposed)**:
```python
class ArticraftRubric(vf.Rubric):
    """
    源码参考（reward 信号来源）：
    - articraft/agent/feedback.py L1040-1080  CompileSignalBundle 的 check 统计
    - articraft/data/model.py L122-135  Rollout.compile_check_fraction()
    - prime-rl/environments/blendergym/blendergym/rubric.py（结构参考）
    """
    def __init__(self, artifact_manager, reward_weights=None):
        super().__init__()
        self.artifact_manager = artifact_manager
        # ⚠️ weights 总和 ≤ 1.0，避免 final_reward 超过 1.0
        # verifiers Rubric 是纯加权求和（无 clamp），超过 1.0 可能导致 W&B 图不直观
        w = {
            "check_fraction": 0.7,
            "build_success": 0.2,
            "compile_attempted": 0.1,
            **(reward_weights or {}),
        }

        self.add_reward_func(self.check_fraction_reward, weight=w["check_fraction"])
        self.add_reward_func(self.build_success_bonus, weight=w["build_success"])
        self.add_reward_func(self.compile_attempted_bonus, weight=w["compile_attempted"])
        self.add_metric(self.blocking_failure_count)
        self.add_metric(self.warning_count)
        self.add_metric(self.turns_used)

    async def check_fraction_reward(self, state, info) -> float:
        # ⚠️ 使用 last_compile_attempt_dict（每次 compile 都更新，含失败）
        # 而非 last_compile_bundle_dict（仅成功更新）
        # 原因：articraft 的 compile 在 QC failure 时 raise 异常，
        # 如果只取成功的 bundle，reward 永远是 0 或 1（二值，GRPO 无法学习）
        # 失败 bundle 中 QC signals 有 group="qc"，能给出真正的中间值
        rollout = require_rollout(state)
        bundle_dict = rollout.last_compile_attempt_dict
        bundle = CompileSignalBundle.from_dict(bundle_dict) if bundle_dict else None
        return compute_reward(
            bundle, num_task_test_checks=rollout.task.num_test_checks,
            turns_used=len(rollout.turns), max_turns=rollout.max_turns,
        )

    async def build_success_bonus(self, state, info) -> float:
        """代码至少能执行并构建出 ObjectModel（即使 QC 失败）。
        区分 "代码完全不能跑"(build failure) vs "代码能跑但 QC 失败"，
        增加 reward 粒度让 GRPO 有更多梯度信号。
        """
        rollout = require_rollout(state)
        bundle_dict = rollout.last_compile_attempt_dict
        if bundle_dict is None:
            return 0.0
        bundle = CompileSignalBundle.from_dict(bundle_dict)
        build_fails = [s for s in bundle.signals if s.blocking and s.group == "build"]
        return 0.0 if build_fails else 1.0

    async def compile_attempted_bonus(self, state, info) -> float:
        rollout = require_rollout(state)
        return 1.0 if any(t.compile_attempted for t in rollout.turns) else 0.0

    @vf.cleanup
    async def write_artifacts_handler(self, state) -> None:
        rollout = require_rollout(state)
        self.artifact_manager.save_trajectory(rollout, metrics=state.get("metrics"))
        self.artifact_manager.cleanup_rollout(rollout)
```

**差异**: BlenderGym reward 是连续值 CLIP similarity (0-1)。Articraft reward 是连续值 check fraction (0-1) + compile_attempted bonus。BlenderGym 通过 HTTP 调 score service，Articraft 全 in-process。两者共同使用 `@vf.cleanup` 写 artifacts。

---

### 5. `dataset.py`

**BlenderGym**:
```python
def build_dataset(data_root, task_types, *, split, eval_holdout) -> Dataset:
    # 扫描 data_root/<task_type><N>/ 目录
    # 验证 REQUIRED_FILES 存在
    # 返回 HF Dataset with columns: prompt=[], answer="", info={task_id, paths...}
```

**Articraft (proposed)**:
```python
# 源码参考：
# - articraft/data/model.py  Record.from_directory() L88-115
# - articraft/data/records/rec_*/  目录结构: prompt.txt, model.py, record.json
# - articraft/storage/record_store.py  RecordStore.list_records()（列出所有 record）
def build_dataset(data_root, *, split, eval_holdout, max_examples=None) -> Dataset:
    # 扫描 data_root/records/rec_*/record.json
    # 提取 prompt.txt 内容
    # 返回 HF Dataset with columns: prompt=[], answer="", info={record_id, prompt_text, ...}
    # split: 按 record_id hash 分 train/eval
```

**差异**: BlenderGym 数据是目录结构 (blend_file + renders)。Articraft 数据是 `records/rec_*/revisions/rev_*/prompt.txt`。

---

### 6. `schema.py`

**BlenderGym**:
```python
@dataclass(frozen=True)
class Task:
    task_id: str
    task_type: str
    blend_file: Path
    goal_image: Path
    init_image: Path
    start_code_path: Path

@dataclass
class TurnRecord:
    turn: int
    exit_status: ExitStatus | None
    render_path: str | None
    ...

@dataclass
class Rollout:
    task: Task
    trajectory_id: str
    work_dir: Path
    max_turns: int
    turns: list[TurnRecord]
    final_reward: float | None
```

**Articraft (proposed)**:
```python
@dataclass(frozen=True)
class Task:
    # 源码参考: articraft/data/model.py Record
    record_id: str
    prompt_text: str
    category_slug: str | None

@dataclass
class TurnRecord:
    # 源码参考: articraft/data/model.py TurnRecord L42-68
    turn: int
    tool_calls: list[dict]        # {name, args, result_status}
    compile_attempted: bool
    compile_success: bool | None
    compile_signals: list[dict]   # CompileSignal dicts

@dataclass
class Rollout:
    # 源码参考（等价 articraft 的分散状态）：
    # - edit_revision / last_compile_revision: harness_compile.py L48-60
    # - mark_code_mutated(): harness.py L864-865
    # - compile_required_count: harness_compile.py L96
    # - edit_retry_injected: harness_guidance.py L261
    task: Task
    trajectory_id: str
    work_dir: Path
    script_path: Path             # model.py 路径
    max_turns: int
    turns: list[TurnRecord]
    final_compile_success: bool
    final_compile_signals: list[dict]
    edit_revision: int = 0            # 源码: harness.py L864 _code_mutation_counter
    last_compile_revision: int = -1   # 源码: harness_compile.py L48-60
    last_compile_bundle_dict: dict | None = None  # 仅成功时更新（freshness cache）
    last_compile_attempt_dict: dict | None = None  # 每次 compile 都更新（reward 用）
    compile_required_count: int = 0   # 源码: harness_compile.py L96
    edit_retry_injected: bool = False  # 源码: harness_guidance.py L261

    def code_is_fresh(self) -> bool:
        # 源码: harness_compile.py L85-89 latest_code_is_fresh()
        return self.last_compile_revision == self.edit_revision and self.last_compile_revision >= 0

    def mark_code_mutated(self):
        # 源码: harness.py L864-865
        self.edit_revision += 1

    def mark_compile_attempt(self, bundle):
        """每次 compile 都更新（成功和失败），用于 reward。"""
        self.last_compile_attempt_dict = bundle.to_dict()

    def mark_compile_success(self, bundle):
        # 源码: harness_compile.py L191-192
        self.last_compile_revision = self.edit_revision
        self.last_compile_bundle_dict = bundle.to_dict()
        self.compile_required_count = 0
```

---

### 7. `prompts.py`

**BlenderGym**:
```python
SYSTEM_PROMPT = """You are a Blender scripting assistant...
Rules: edit object placements only...
Output format: <code>...</code> tag pair..."""

TASK_INSTRUCTION = "Rewrite the program below so the rendered scene matches the GOAL image..."
REFINE_INSTRUCTION = "Below is the render produced by your previous program..."
```

**Articraft (proposed)**:
```python
SYSTEM_PROMPT = """You are a 3D articulated object designer using the Articraft SDK.
You have 4 tools: write_file, replace, read_file, compile_model.
Your goal: implement build_object_model() and run_tests() in model.py so that compile_model succeeds with no blocking failures.
..."""

# Tool schemas 由 vLLM tool_call_parser 自动注入 chat template
# SDK docs 通过 read_file 按需查阅（不在 prompt 中全文嵌入）
```

**差异**: BlenderGym 需要教模型用 `<code>` XML 格式。Articraft 用 vLLM 原生 tool_calling，格式由 `tool_call_parser=qwen3_coder`（Qwen3.5 系列）自动处理。

---

### 8. 不需要的 BlenderGym 组件

| BlenderGym 组件 | Articraft 是否需要 | 原因 |
|----------------|-------------------|------|
| `services/render/` | ❌ | 无 GPU 渲染 |
| `services/score/` | ❌ | reward 从 compile signals 计算，不需要 CLIP |
| `render.py` (Blender subprocess) | ❌ | 无 Blender |
| `gpu_mem.py` | ❌ | 纯 CPU 计算 |
| `artifact_manager.py` | ⚠️ 简化版 | 只需管理 work_dir + model.py |
| `trajectory_writer.py` | ⚠️ 可选 | 调试用，非必需 |

---

---

## Harness 反馈系统详解

### 一个 Turn 的完整反馈流程

```
LLM 输出 (assistant message with tool_calls)
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ Step 1: 执行 Tool Calls                              │
│                                                      │
│ 对每个 tool_call 按序执行:                            │
│   write_code / replace → 修改 model.py               │
│   read_file → 返回文件内容                            │
│   find_examples → 返回相似例子                        │
│   probe_model → 执行诊断代码片段，返回几何数据          │
│   compile_model → 执行脚本 + QC 检查（见 Step 2）     │
│                                                      │
│ 每个 tool 返回一条 role="tool" 消息                   │
│ 格式: {"output": ..., "error": ..., "compilation": ..}│
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ Step 2: compile_model 执行细节                        │
│                                                      │
│ 1. 检查 revision freshness:                          │
│    - 如果代码自上次成功 compile 后没改过 → 返回缓存    │
│    - 否则执行真正的 compile                           │
│                                                      │
│ 2. compile_urdf_report_maybe_timeout():              │
│    a. 执行 model.py (runpy)                          │
│    b. 调 build_object_model() → 得到 ObjectModel     │
│    c. 导出 URDF XML                                  │
│    d. 运行 run_tests() → TestReport                  │
│    e. 运行 compiler-owned baseline QC checks:         │
│       - check_model_valid                            │
│       - check_single_root_part                       │
│       - check_mesh_assets_ready                      │
│       - fail_if_isolated_parts()                     │
│       - fail_if_parts_overlap_in_current_pose()      │
│       - warn_if_part_contains_disconnected_geometry   │
│    f. 收集 warnings + test failures → 构建           │
│       CompileSignalBundle                            │
│                                                      │
│ 3. 返回给 LLM 的 tool result 内容:                   │
│    render_compile_signals(bundle) → 结构化 XML 文本   │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ Step 3: compile_model 返回给 LLM 的具体文本格式       │
│                                                      │
│ 成功且干净:                                          │
│ ┌──────────────────────────────────────────────┐     │
│ │ <compile_signals>                            │     │
│ │ <summary>                                    │     │
│ │ status=success failures=0 warnings=0 notes=0 │     │
│ │ Compile passed cleanly.                      │     │
│ │ </summary>                                   │     │
│ │ </compile_signals>                           │     │
│ └──────────────────────────────────────────────┘     │
│                                                      │
│ 成功但有 warnings:                                   │
│ ┌──────────────────────────────────────────────┐     │
│ │ <compile_signals>                            │     │
│ │ <summary>                                    │     │
│ │ status=success failures=0 warnings=2 notes=1 │     │
│ │ Primary issue: compile passed with warnings. │     │
│ │ </summary>                                   │     │
│ │                                              │     │
│ │ <warnings>                                   │     │
│ │ Warnings (non-blocking):                     │     │
│ │ - WARNING [geometry_overlap] Geometry overlap │     │
│ │   check reported overlaps.                   │     │
│ │ - WARNING [disconnected_geometry] Exact visual│     │
│ │   connectivity check found disconnected...   │     │
│ │ </warnings>                                  │     │
│ │                                              │     │
│ │ <response_rules>                             │     │
│ │ Suggested next steps:                        │     │
│ │ - Warnings are not blocking, but they are    │     │
│ │   design evidence and should not be ignored. │     │
│ │ </response_rules>                            │     │
│ │ </compile_signals>                           │     │
│ └──────────────────────────────────────────────┘     │
│                                                      │
│ 失败（有 blocking failures）:                        │
│ ┌──────────────────────────────────────────────┐     │
│ │ <compile_signals>                            │     │
│ │ <summary>                                    │     │
│ │ status=failure failures=1 warnings=0 notes=0 │     │
│ │ Primary issue: compiler-owned global QC found│     │
│ │ floating disconnected parts.                 │     │
│ │ </summary>                                   │     │
│ │                                              │     │
│ │ <failures>                                   │     │
│ │ Failures (blocking):                         │     │
│ │ - FAILURE [isolated_part] Floating           │     │
│ │   disconnected component(s) detected.        │     │
│ │   (visual 'knob_body' on link 'knob'...)     │     │
│ │ </failures>                                  │     │
│ │                                              │     │
│ │ <response_rules>                             │     │
│ │ Suggested next steps:                        │     │
│ │ - Treat the compiler-owned floating/         │     │
│ │   disconnected part finding as primary...    │     │
│ │ - If the support path is not obvious,        │     │
│ │   consider using `probe_model` with          │     │
│ │   `find_floating_parts(...)`...              │     │
│ │ </response_rules>                            │     │
│ │ </compile_signals>                           │     │
│ └──────────────────────────────────────────────┘     │
│                                                      │
│ 连续失败时追加:                                      │
│   "This failure matches the previous compile attempt"│
│   "This is compile failure 3 in a row."             │
│   "You are in a repair loop..."                     │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ Step 4: Guidance Injections (tool 执行后自动注入)     │
│                                                      │
│ 触发条件: write_code/replace 成功修改了代码            │
│                                                      │
│ 4a. <exact_geometry_contract>                        │
│     触发: run_tests() 里的 ctx.expect_* 引用了       │
│           build_object_model() 中不存在的 visual 名  │
│     作用: 提醒恢复 visual 或更新 test                 │
│                                                      │
│ 4b. <baseline_qc_guidance>                           │
│     触发: run_tests() 手动调了编译器自带的 check      │
│           (check_model_valid, fail_if_isolated_parts │
│            等 5 个 baseline)                          │
│     作用: 提醒删除冗余检查，由 compile_model 自动处理  │
│                                                      │
│ 4c. <edit_retry_guidance>                            │
│     触发: replace 工具报错 "Could not find the       │
│           old_string in the code"                    │
│     作用: 提醒先 read_file 再重试，保持编辑精确       │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ Step 5: Termination 检查                             │
│                                                      │
│ Agent 想结束（返回纯文本无 tool_calls）时:            │
│                                                      │
│ 5a. 检查 latest_code_is_fresh():                     │
│     - 如果代码自上次成功 compile 后有改动             │
│       → 注入 <compile_required> 提醒                 │
│       → 不允许结束，继续循环                          │
│     - 如果 fresh（最新代码已成功 compile）            │
│       → 允许结束，返回 AgentResult(success=True)     │
│                                                      │
│ 5b. 最终 AgentResult 包含:                           │
│     - success: bool                                  │
│     - reason: CODE_VALID / MAX_TURNS / ERROR / ...   │
│     - final_code: str (model.py 内容)                │
│     - urdf_xml: str (成功时)                         │
│     - compile_warnings: list[str]                    │
│     - turn_count, tool_call_count, compile_attempts  │
└─────────────────────────────────────────────────────┘
```

### 所有可能注入到对话中的反馈消息总表

| # | 消息类型 | role | 触发时机 | 内容 |
|---|---------|------|---------|------|
| 1 | Tool result | `tool` | 每次 tool_call 执行后 | `{"output": ..., "error": ...}` |
| 2 | Compile signals | `tool` (compile_model result) | 调用 compile_model 后 | `<compile_signals>...</compile_signals>` 含 summary + failures + warnings + notes + response_rules |
| 3 | Exact geometry contract | `user` | write/replace 成功后检测到名称不匹配 | `<exact_geometry_contract>...</exact_geometry_contract>` |
| 4 | Baseline QC guidance | `user` | write/replace 成功后检测到冗余 QC 调用 | `<baseline_qc_guidance>...</baseline_qc_guidance>` |
| 5 | Edit retry guidance | `user` | replace 报错 old_string 找不到 | `<edit_retry_guidance>...</edit_retry_guidance>` |
| 6 | Compile required | `user` | agent 想结束但代码未 compile | `<compile_required>...</compile_required>` |

### CompileSignalBundle 结构化数据（用于 Reward 计算）

```python
CompileSignalBundle:
    status: "success" | "failure"
    summary: str  # 人类可读摘要
    signals: tuple[CompileSignal, ...]

CompileSignal:
    severity: "failure" | "warning" | "note"
    kind: str          # 见下表
    code: str          # 机器可读 code (如 "QC_REAL_OVERLAP")
    summary: str       # 一句话描述
    details: str       # 详细信息
    blocking: bool     # True = 必须修复才能算成功
    source: "compiler" | "tests" | "harness"
    group: "build" | "qc" | "design" | "hygiene"
    check_name: str | None  # 触发此 signal 的具体 check 函数名
```

### 完整 Signal Kind 枚举

**Blocking Failures (severity="failure", blocking=True)**:

| kind | group | source | 含义 |
|------|-------|--------|------|
| `compile_runtime` | build | compiler | SyntaxError / ImportError / RuntimeError — 代码无法执行 |
| `single_root_policy` | build | compiler | 不是单一根部件，结构树非法 |
| `model_validity` | build | compiler | 模型结构验证失败 |
| `mesh_assets` | build | compiler | 网格资产缺失或未解析 |
| `isolated_part` | qc | compiler/tests | 浮空/断连部件（无连接路径到根） |
| `real_overlap` | qc | compiler/tests | 部件间真实 3D 体积碰撞 |
| `missing_exact_geometry` | qc | tests | `ctx.expect_*` 引用了不存在的 visual 名称 |
| `exact_contact_gap` | qc | tests | 期望接触处检测到间隙 |
| `test_failure` | qc | tests | 其他用户定义的 `run_tests()` 失败 |

**Non-blocking Warnings (severity="warning", blocking=False)**:

| kind | group | 含义 |
|------|-------|------|
| `geometry_scale` | design | 几何尺寸异常/荒谬 |
| `visual_connectivity` | design | 视觉连通性检查失败 |
| `isolated_part` (warn) | design | 孤立部件（warning 级别，非 blocking） |
| `geometry_overlap` | qc | 重叠检测报告（warning 级别） |
| `overlap_warning` | qc | 重叠传感器报告 |
| `disconnected_geometry` | qc | 单部件内有断连几何岛 |
| `coplanar_surface` | qc | 共面表面启发式告警 |
| `articulation_origin` | qc | 铰接原点远离几何体 |
| `path_hygiene` | hygiene | 使用了 cwd 相对路径 |
| `deprecated_test_api` | hygiene | 使用了废弃 test API |
| `compiler_warning` | qc | 未分类的编译器警告 |
| `test_warning` | qc | 未分类的 test 警告 |

**Notes (severity="note", informational)**:

| kind | 含义 |
|------|------|
| `allowance` | 用户声明的豁免 (`allow_isolated_part`, `allow_overlap`) |
| `allowed_overlap` | 被 justification 豁免的重叠 |
| `allowed_isolated_part` | 被 justification 豁免的孤立件 |
| `coplanar_surface_hint` | 低置信度共面提示 |

---

## Reward 设计（Check Fraction）

### 设计原则

- **无手调阈值** — reward 直接由 compiler 的 check 通过率决定
- **Per-task 动态分母** — 每个 task 的总 check 数不同（test assertions 数量不同），分母动态确定
- **两阶段** — build（代码能否执行）是前提，QC（质量检查通过率）是得分
- **Warning 作为次要 bonus** — 不参与主 reward，只在全部 QC 通过后给小加分

### 信号来源

Articraft `compile_model` 返回 `CompileSignalBundle`，包含：
- `status`: `"success"` | `"failure"`
- `signals`: `list[CompileSignal]`，每个 signal 有 `severity`/`group`/`kind`/`blocking`/`source`

### Reward 公式

```python
# 源码参考：
# - articraft/agent/feedback.py L198-240  CompileSignal 定义（severity, group, kind, blocking）
# - articraft/agent/feedback.py L1040-1080  build_compile_signal_bundle()
# - articraft/agent/compiler.py L580-615  compile_urdf_report（实际跑 QC checks）
# - articraft/agent/compiler.py L298-310  fail_if_isolated_parts
# - articraft/agent/compiler.py L312-340  fail_if_parts_overlap_in_current_pose
# - articraft/data/model.py L122-135  Rollout.compile_check_fraction()（线上评估用的类似公式）

# Compiler-owned QC checks（固定集合，每次 compile 都会跑）
COMPILER_QC_CHECKS = {
    "isolated_part",       # 源码: compiler.py L298-310 fail_if_isolated_parts
    "real_overlap",        # 源码: compiler.py L312-340 fail_if_parts_overlap_in_current_pose
}
# 注：model_valid, single_root, mesh_assets 属于 group="build"，失败直接 → 0
# 源码: compiler.py L254-296 这些检查在 group="build" 中

def compute_reward(
    bundle: CompileSignalBundle | None,
    num_task_test_checks: int = 0,
    turns_used: int = 0,
    max_turns: int = 50,
) -> float:
    """
    多层连续 reward，确保 GRPO 在同一 example 的多 rollout 间有足够方差。

    源码参考：
    - articraft/data/model.py L122-135 Rollout.compile_check_fraction()
    - 本函数在 articraft 的 check_fraction 基础上增加了 RL 所需的细粒度信号

    ⚠️⚠️ 致命语义点（已修复）：
    articraft 的 compile 在 ANY blocking failure 时 raise 异常：
    - compile_urdf_report() 只在零 blocking failure 时正常返回 CompileReport
    - QC checks (fail_if_isolated_parts 等) 失败 → TestReport.failures → raise ValueError
    - compiler.py 捕获后构建 CompileSignalBundle（含 group="qc" 的 signals）再 raise

    因此 mark_compile_success 永远只在零 failure 时被调用。
    如果 reward 只使用 last_compile_bundle_dict（仅成功更新），则 reward 永远是二值（0 或 1）。

    修复：引入 last_compile_attempt_dict（每次 compile 都更新，含失败路径的 bundle）。
    失败路径的 bundle 通过 compile_signal_bundle_from_exception(exc) 获取，
    其中 QC failures 的 signals 有 group="qc"（非 "build"），能给出真正的中间值。

    ⚠️ Phase 1 粒度增强：
    即使有了 QC 中间值，num_task_test_checks=0 时只有 2 个 QC checks。
    额外通过以下方式增加连续性：
    1. 将 warnings 从 bonus 改为连续区分信号
    2. 加入 turns 效率因子（同等 QC 分数下，更快完成 = 更高 reward）
    3. 区分 build 失败的严重程度（SyntaxError vs ImportError vs Runtime vs 结构非法）

    Reward 层级（从低到高）：
    - 从未 compile: 0.0
    - Build failure - SyntaxError: 0.05（代码至少写了，只是有语法错误）
    - Build failure - RuntimeError/其它: 0.10（代码能解析但执行出错）
    - Build failure - 结构非法（non-single-root, mesh_missing）: 0.15（代码跑通了但模型结构不对）
    - QC failures（但 build 成功）: 0.3 + 0.5 * (passed/total)
    - 全 QC 通过 + 有 warnings: 0.8 + 0.1 * warning_clean_fraction
    - 全 QC 通过 + 无 warnings: 0.9
    - 全 QC 通过 + 无 warnings + 快速完成: 0.9 + 0.1 * efficiency
    """
    # 从未 compile
    if bundle is None:
        return 0.0

    blocking = [s for s in bundle.signals if s.blocking]
    warnings = [s for s in bundle.signals if s.severity == "warning"]

    # ── 阶段 1: build failure（连续化）──
    build_failures = [s for s in blocking if s.group == "build"]
    if build_failures:
        kinds = {s.kind for s in build_failures}
        if "syntax_error" in kinds:
            return 0.05
        elif "model_valid" not in kinds and "single_root" not in kinds:
            return 0.10
        else:
            return 0.15

    # ── 阶段 2: QC check fraction ──
    total_qc = len(COMPILER_QC_CHECKS) + num_task_test_checks
    if total_qc == 0:
        qc_score = 1.0
    else:
        qc_failures = [s for s in blocking if s.group == "qc"]
        passed = total_qc - len(qc_failures)
        qc_score = passed / total_qc

    if qc_score < 1.0:
        # Build 成功但有 QC failures: 0.3 ~ 0.8
        return 0.3 + 0.5 * qc_score

    # ── 阶段 3: 全 QC 通过，区分 warnings + 效率 ──
    # Warnings 作为连续信号（越少越好）
    max_expected_warnings = 5
    warning_clean_fraction = max(0.0, 1.0 - len(warnings) / max_expected_warnings)

    if warnings:
        # 0.8 ~ 0.9（按 warning 数量连续变化）
        return 0.8 + 0.1 * warning_clean_fraction

    # 无 warnings → 用 turns 效率区分 rollout 间差异
    # 越快完成越好（让 GRPO 对同一 example 的 8 个 rollout 产生方差）
    efficiency = max(0.0, 1.0 - turns_used / max_turns) if max_turns > 0 else 1.0
    return 0.9 + 0.1 * efficiency
```

### Reward 值域分析

**三维 reward 结构**：`final_reward = 0.7 × compute_reward + 0.2 × build_success + 0.1 × compile_attempted`

- `compute_reward` ∈ [0.0, 1.0]：连续函数，基于 QC pass fraction / warnings / efficiency
- `build_success` ∈ {0, 1}：代码能否执行并构建 ObjectModel（即使 QC 失败也得分）
- `compile_attempted` ∈ {0, 1}：是否至少尝试了 compile

**Phase 1**（num_task_test_checks=0，total_qc=2）典型值域：

| 状态 | compute_reward | build_success | compile_attempted | final_reward |
|------|---------------|---------------|-------------------|-------------|
| 从未 compile | 0.0 | 0 | 0 | **0.00** |
| 至少 compile 但 SyntaxError | 0.05 | 0 | 1 | **0.14** |
| RuntimeError（代码能解析但执行出错） | 0.10 | 0 | 1 | **0.17** |
| 结构非法（non-single-root 等） | 0.15 | 0 | 1 | **0.21** |
| Build 成功 + 2/2 QC 失败 | 0.30 | 1 | 1 | **0.51** |
| Build 成功 + 1/2 QC 失败 | 0.55 | 1 | 1 | **0.69** |
| 全 QC 通过 + 3 warnings | 0.86 | 1 | 1 | **0.90** |
| 全 QC 通过 + 1 warning | 0.88 | 1 | 1 | **0.92** |
| 全 QC 通过 + 0 warnings + 30/50 turns | 0.94 | 1 | 1 | **0.96** |
| 全 QC 通过 + 0 warnings + 10/50 turns | 0.98 | 1 | 1 | **0.99** |

**10+ 档连续值域，max=1.0**。同一 example 的 8 个 rollout 中产生充分方差：
- `build_success_bonus` 提供了关键跳跃（从 ~0.2 到 ~0.5），鼓励模型先写出能执行的代码
- QC fraction 在成功执行后提供连续梯度
- efficiency 在同等 QC 结果的 rollouts 间创造差异

**Phase 2+**（num_task_test_checks>0，total_qc=2+N）：分母更大，QC 阶段本身就有更多档位。

### `num_task_test_checks` 如何确定

每个 record 的 `run_tests()` 函数含不同数量的 `ctx.expect_*` 调用。方案：

1. **Phase 1：固定为 0** — 由于用 scaffold 起步、模型自己写 `run_tests()`，无法预知 assertion 数量。Phase 1 统一 `num_task_test_checks=0`，reward 仅评估 compiler-owned 的 2 个 QC checks（`collision_free` + `joint_limits`）。这使得 reward 信号简洁且一致。
2. **Phase 2+：预计算存入 dataset 元数据** — 对每个 `record.json`，AST 分析最终成功 `model.py` 的 `run_tests()` 统计 `ctx.expect_*` 调用次数，作为 `info["num_test_checks"]` 字段。`Task.from_info()` 中读取：`num_test_checks=info.get("num_test_checks", 0)`
3. **回退默认值** — 若元数据不可用，默认 `num_task_test_checks=0`
4. **后续优化** — 可以用实际 compile 一次 ground-truth solution 来精确计数

### 诊断 Metrics（不参与训练，只记日志）

```python
async def compile_attempted(state) -> float:
    """是否调用过 compile_model"""
    return float(state["rollout"].compile_attempt_count > 0)

async def compile_success_rate(state) -> float:
    """compile 成功率"""
    r = state["rollout"]
    if r.compile_attempt_count == 0:
        return 0.0
    return r.successful_compile_count / r.compile_attempt_count

async def qc_pass_fraction(state) -> float:
    """最终 compile 的 QC 通过率（与 reward 对齐）"""
    # 直接复用 reward 逻辑（不含 bonus）
    ...

async def blocking_failure_count(state) -> float:
    """最终 compile 的 blocking failure 数量"""
    return float(len(state["rollout"].final_blocking_failures))

async def warning_count(state) -> float:
    """最终 compile 的 warning 数量"""
    return float(state["rollout"].final_warning_count)
```

### 设计决策说明

| 决策 | 理由 |
|------|------|
| **Check fraction（非手调阈值）** | reward 直接反映 compiler 的判定，无主观参数 |
| **Per-task 动态分母** | 不同 task 的 test 复杂度不同，固定分母会扭曲 |
| **Build failure = 0** | 前提条件未满足，后续 QC 无意义（cascade failure） |
| **Warning 只做 bonus 不做惩罚** | warnings 是设计提示非正确性指标，不应主导 reward |
| **Outcome-only（episode 结束时）** | GRPO 组内排名，step-level 信号通过 advantage 传递 |
| **0-1 范围** | 与 BlenderGym CLIP reward 对齐 |

### 与旧方案对比

| | 旧方案（6 档 heuristic） | 新方案（check fraction） |
|---|---|---|
| 手调参数 | 6 个 | 1 个（bonus=0.1） |
| 值域 | {0, 0.1, 0.3, 0.6, 0.6~0.85, 1.0} 离散 | [0, 1] 连续（步长 = 1/total_qc） |
| 不同 task 的 reward scale | 相同（ignoring task 复杂度） | 按 task 的 check 数自适应 |
| Reward hacking | "compile 成功就 0.6" | 需要实际通过 QC checks |

### 与 BlenderGym Reward 对比

| | BlenderGym | Articraft (Phase 1) | Articraft (Phase 2+) |
|---|---|---|---|
| 信号来源 | CLIP cosine similarity | CompileSignalBundle | Check fraction + CLIP |
| 值域 | [0.0, 1.0] 连续 | [0.0, 1.0] 连续 | [0.0, 1.0] 连续 |
| 参考标准 | goal image | 无参考，execution-based | render vs prompt + reference render |
| 计算成本 | CLIP ~200ms | compile 0.08s | compile + render + CLIP |

---

### Phase 2+ Reward：CLIP Visual Similarity（加权和）

**设计理念**：check fraction 只验证"模型是否结构正确"，CLIP sim 验证"模型是否语义正确（看起来像描述的东西）"。两者互补。

#### 公式

```python
reward = w_qc * check_fraction + w_clip * clip_similarity

# 初始权重（可调）
w_qc = 0.4    # 结构正确性
w_clip = 0.6  # 语义正确性
```

其中 `clip_similarity` 融合 text sim 和 visual sim：

```python
def compute_clip_similarity(render_image, prompt_text, reference_render=None):
    text_sim = clip_score(render_image, prompt_text)  # render vs 文字描述
    if reference_render is not None:
        visual_sim = clip_score(render_image, reference_render)  # render vs 参考图
        return 0.5 * text_sim + 0.5 * visual_sim
    return text_sim
```

**前提条件**：compile 成功（有 URDF XML）才能渲染。compile 失败时 `clip_similarity = 0`。

#### 渲染 Pipeline

```
compile_model 成功 → URDF XML
    → Three.js headless render service（articraft viewer）
    → 多角度截图（正面、侧面、45°）
    → CLIP 评分
```

**技术方案**：
- **完整复用 articraft viewer**（`viewer/api/` + `viewer/web/`）：
  - viewer 本身就有 URDF → Three.js 渲染能力，已实现 scene setup、URDF loader、材质渲染
  - 直接启动 viewer 服务，通过 Playwright 做 headless 截图
  - 同时 viewer 还可用于 RL rollout 可视化（浏览生成的 records、对比 trajectory）
- Headless 渲染：Playwright（headless Chromium）访问 viewer 的 record 渲染页面
- 部署为 HTTP 微服务（viewer 本身已是 FastAPI，只需加截图 endpoint）
- CLIP 推理：`open_clip_torch`，作为独立 Python service 或集成到 viewer API

#### Reference Render 准备

对每个 training record 的 ground-truth solution：
- 预先 compile → URDF
- 预先渲染多角度图 → 存储为 `data/records/rec_*/reference_renders/`
- Dataset 加载时作为 `info["reference_render_paths"]` 传入

#### 引入时机

| 阶段 | Reward | 备注 |
|------|--------|------|
| Step 1 (mock eval) | check_fraction only | 验证 env 逻辑 |
| Step 2 (真模型 eval) | check_fraction only | 验证 RL pipeline |
| Step 3+ (CLIP 增强) | w_qc * check_fraction + w_clip * clip_sim | 需要 render service |

#### 与 BlenderGym Render Service 的差异

| | BlenderGym | Articraft |
|---|---|---|
| 渲染器 | Blender (Python subprocess, GPU OptiX) | Three.js (headless browser via articraft viewer, CPU/WebGL) |
| 渲染耗时 | ~3-5s (GPU) | ~0.5-1s (headless browser) |
| CLIP 模型 | ViT-B/32 | 同 |
| 评分方式 | CLIP(render, goal_image) | CLIP(render, prompt_text) + CLIP(render, reference) |
| 服务架构 | FastAPI + worker pool | 复用 articraft viewer (FastAPI) + Playwright headless |
| 代码复用 | 从零搭建 render pipeline | 直接用 articraft viewer 已有的 URDF → Three.js 渲染 |

---

## Observability 设计（仿 BlenderGym 四层架构）

### 数据流总览

```
每 Turn（env_response）
  → turn_N/: model.py snapshot, compile_signals.json
Episode 结束（rubric @vf.cleanup）
  → write_artifacts_handler: meta.json + trajectory.json + trajectory.html
  → cleanup: keep_failed_only / prune
verifiers → RolloutOutput
  → state_columns: trajectory, sampling_args
prime-rl orchestrator
  → rollouts/step_N/eval_rollouts.jsonl + W&B metrics/samples
```

### 层 1: 每 Turn 本地产物（`env_response` / `add_model_response` 中写入）

| 文件 | 内容 | 触发时机 |
|------|------|---------|
| `turn_N/model.py` | 当前 model.py 的快照 | 每次 write_file/replace 执行后 |
| `turn_N/compile_signals.json` | `CompileSignalBundle.to_dict()` | 每次 compile_model 执行后 |
| `turn_N/tool_calls.json` | 本 turn 所有 tool_calls + results 摘要 | 每 turn 结束时 |

work_dir 布局：

```
{work_root}/{split}/example_{id:04d}__{record_id}/{trajectory_id[:8]}/
├── model.py              ← 当前版本（最新）
├── turn_0/
│   ├── model.py          ← 该 turn 结束时的快照
│   ├── compile_signals.json
│   └── tool_calls.json
├── turn_1/
│   └── ...
├── meta.json
├── trajectory.json
└── trajectory.html
```

### 层 2: Episode 结束产物（Rubric `@vf.cleanup`）

**`meta.json`**（扁平摘要，便于 grep）：
```python
{
    "record_id": "rec_00042",
    "trajectory_id": "a1b2c3d4",
    "final_reward": 0.83,
    "compile_attempted": True,
    "compile_success": True,
    "qc_pass_fraction": 0.83,
    "blocking_failures": 1,
    "warning_count": 2,
    "num_turns": 5,
    "max_turns": 50,
    "total_tool_calls": 12,
    "total_compile_attempts": 3,
}
```

**`trajectory.json`**（完整结构化轨迹）：
```python
{
    "schema_version": "articraft-trajectory-v1",
    "task": {"record_id": ..., "prompt_text": ..., "category_slug": ...},
    "final_reward": 0.83,
    "metrics": {"qc_pass_fraction": ..., "compile_success_rate": ..., ...},
    "steps": [
        {
            "turn": 0,
            "tool_calls": [
                {"name": "write_file", "arguments": {...}, "result_status": "success"},
                {"name": "compile_model", "arguments": {}, "result_status": "failure"},
            ],
            "compile_signals": {...},  # CompileSignalBundle.to_dict() or None
            "model_py_hash": "abc123",  # 快速比较代码是否变化
        },
        ...
    ],
    "final_compile_bundle": {...},
    "final_model_py": "...",  # 最终代码全文
}
```

**`trajectory.html`**（人类可读时间线）：

与 BlenderGym 不同，Articraft 没有图片，但有代码演化。HTML 应展示：
- Header：record_id、reward、compile status pills（绿/红/黄）
- 代码 diff timeline：每 turn 的 model.py 变更（syntax highlighted diff）
- Compile signals timeline：每次 compile 的 signals 以 badge 方式展示（failure=红, warning=黄, clean=绿）
- Tool call 统计：饼图或条形（write_file/replace/read_file/compile_model 分布）
- 可折叠区域：每 turn 的 tool_calls 详情 + compile_signals 全文

### 层 3: Retention Policy

```python
@dataclass
class ArticraftArtifactPolicy:
    save_meta_json: bool = True
    save_trajectory_json: bool = True
    save_trajectory_html: bool = True
    save_per_turn_snapshots: bool = True    # turn_N/ 目录
    keep_failed_only: bool = False          # train=True, eval=False
    max_rollouts_per_example: int = 0       # 0 = unlimited
```

**`keep_failed_only`** 逻辑（与 BlenderGym 对齐）：
- 最终 compile 成功（无 blocking failure） → 删除整个 work_dir（训练时节省磁盘）
- 最终 compile 失败或从未 compile → 保留（便于调试为什么模型失败）

### 层 4: W&B Metrics + Samples

**标量 metrics（通过 `rubric.add_metric()` → prime-rl orchestrator → W&B）：**

| Metric | 含义 |
|--------|------|
| `qc_pass_fraction` | 最终 compile 的 QC 通过率（= reward 主成分） |
| `compile_success` | 最终 compile 是否无 blocking failure (0/1) |
| `compile_attempted` | 是否调用过 compile_model (0/1) |
| `compile_attempt_count` | compile_model 总调用次数 |
| `blocking_failure_count` | 最终 compile 的 blocking failure 数 |
| `warning_count` | 最终 compile 的 warning 数 |
| `tool_call_count` | 总 tool_call 次数 |
| `turns_used` | 实际使用的 turn 数 |

**W&B Samples Table（自动由 orchestrator 处理）：**
- `messages`：decode 后的完整多轮对话（含 tool_calls XML + tool results）
- `reward`：最终 reward

**反查本地轨迹**：W&B 行中的 `trajectory_id` + `record_id` → 本地 `work_root/{split}/example_{id}__{record_id}/{traj8}/trajectory.html`

### 与 BlenderGym Observability 的对比

| 维度 | BlenderGym | Articraft |
|------|-----------|-----------|
| 每 turn 产物 | code.py + render.png + blender.log | model.py + compile_signals.json |
| HTML 可视化 | 图片条（goal vs renders） | 代码 diff + compile badge timeline |
| 核心调试信息 | 渲染图对比 | compile signal 详情 + 代码演化 |
| Retention 条件 | xml_parsed && render_success | compile 无 blocking failure |
| 磁盘占用 | 大（PNG 渲染图） | 小（纯文本/JSON） |

### Viewer 集成（RL Rollout 可视化 — 与 trajectory.html 并存）

`trajectory.html` 和 Viewer 集成**不互斥**，服务不同场景：

| | `trajectory.html` | Viewer 集成 |
|---|---|---|
| 需要什么 | 只需浏览器打开文件 | 需要启动 viewer 服务 |
| 适用场景 | CI/CD 产物、快速查看、分享 | 深度调试、3D 渲染、交互式浏览 |
| 能力 | 代码 diff + compile badge timeline | 3D URDF 渲染 + trajectory inspector |
| 开发成本 | 需要写 HTML 模板 | 需要适配 viewer 的 record 格式 |

**Viewer 集成方案**：
- RL env 在 episode 结束时，将 rollout 结果写为 articraft viewer 可识别的 record 格式
- 启动 articraft viewer 即可浏览所有 RL 生成的 records：3D 渲染、compile 状态、代码演化
- Phase 2 CLIP reward 和可视化共用一套 viewer 部署

**写入格式**：rollout 完成后，将最终 `model.py` 和 compile 结果写入 viewer 可读的目录结构：
```
work_root/{split}/example_{id}/
├── model.py                    ← 最终模型代码
├── compile_report.json         ← CompileReport
├── metadata.json               ← record_id, prompt, reward, turns 等
└── trajectory.jsonl            ← 完整对话轨迹（viewer trajectory inspector 可读）
```

### 实现优先级

| 组件 | 优先级 | 备注 |
|------|--------|------|
| `trajectory.json` | P0（mock eval 就做） | 调试 env 逻辑的核心 |
| `meta.json` | P0 | grep/dashboard |
| W&B metrics | P0 | 通过 `rubric.add_metric()` 自动流入 |
| `trajectory.html` | P1（接真模型后做） | 自包含 HTML，代码 diff + compile timeline |
| Viewer-compatible output | P1（接真模型后做） | 写为 viewer 可读 record 格式，提供 3D 可视化 |
| Per-turn snapshots | P1 | 占磁盘但调试有用 |
| Retention policy | P2（上集群训练时做） | 本地 dev 不需要 |

---

## 前期验证（Step 0）

在正式实现前，先用测试脚本验证 compile 性能和并发可行性：

- [x] Step 0a: **compile 耗时基准测试** — 20 个随机 records，in-process vs subprocess
- [x] Step 0b: **多 worker 并发压力测试** — 1/4/8/16 workers 并发 compile
- [x] Step 0c: **scaffold compile 基线** — 空 scaffold model.py 编译耗时
- [x] Step 0d: **KAOLA 集群兼容性测试** — H200 pod 上验证 torch+vllm+articraft 共存

测试脚本: `environments/articraft/benchmarks/compile_bench.py`

### Step 0 测试结果（2026-05-21，MacBook 本地，20 条采样）

#### 0c: Scaffold 基线

空 scaffold 编译 **<0.01s**，但会 **fail**（`check_model_valid` 失败，因为空模型没有 parts）。这是正常行为——RL 中模型第一次 compile 必然失败，耗时忽略不计。

#### 0a: 单次 compile 耗时分布

| 模式 | 成功率 | min | median | mean | P90 | max | stdev |
|------|--------|-----|--------|------|-----|-----|-------|
| **in-process** | 19/20 | 0.01s | **0.08s** | 3.72s | 23.2s | 36.4s | 9.60 |
| **subprocess** | 19/20 | 0.12s | **0.55s** | 4.31s | 24.7s | 39.7s | 10.19 |

**关键发现:**
- **双峰分布**: 大部分 compile **<1s**（中位数 0.08s），但有少数 outlier **12-40s**（通常是几何 QC 重的模型，如 barrier_gate、desktop_pc_tower、wheelie_bin）
- **subprocess 开销**: median 从 0.08→0.55s（spawn 子进程固定 ~0.4s 开销）；对快 compile 影响大（6-160x），对慢 compile 影响小（~1.05x）
- **中位数 0.08s/0.55s 远低于预期的 5-60s** — 方案原估计过于保守

**subprocess 开销分析（19 对成功 paired）:**

| 指标 | min | median | max |
|------|-----|--------|-----|
| 绝对开销 | -0.58s | 0.44s | 3.23s |
| 倍率 | 0.95x | 6.52x | 160.65x |

快 compile（<0.1s）的 subprocess overhead 占比极大。对于这些 trivial compile，子进程 spawn 比 compile 本身还慢。

#### 0b: 多 worker 并发

| Workers | Wall time | Sum(compute) | 实际并行度 | Throughput |
|---------|-----------|-------------|-----------|-----------|
| 1 | 85.5s | 85.3s | 1.00x | 0.23/s |
| 4 | 46.9s | 97.4s | **2.07x** | 0.43/s |
| 8 | 43.5s | 100.6s | **2.31x** | 0.46/s |
| 16 | 45.2s | 117.8s | **2.61x** | 0.44/s |

**关键发现:**
- **并发扩展极差**: 4 workers 只有 2x 并行度，8/16 workers 几乎无提升
- **原因: `_MODEL_EXECUTION_LOCK`** — `compiler.py` 中 `load_model_globals()` 使用了 `threading.Lock()` 全局互斥锁（因为 `runpy.run_path` + `os.chdir` 不是线程安全的），subprocess 模式下每个子进程独立持有锁不冲突，但**真正的瓶颈是 outlier 长尾**：barrier_gate（~40s）独占时间片，其他 19 个 compile 都在等它完成
- **4→8→16 workers 无明显提升**说明瓶颈不在并行度，而在那几个 **outlier compile**（一个 40s 的 barrier_gate 决定了 wall time 下界）
- Per-task median 随 workers 增加而增大（0.55→0.79→0.97→1.77s），说明 CPU 争抢导致轻量 compile 变慢

#### RL 训练 Throughput 估算（修正版）

基于 **median=0.55s**（subprocess 模式）：

| 场景 | 串行时间 | 8 workers | 备注 |
|------|---------|-----------|------|
| 3 compiles × 4 rollouts | 7s | 1.6s | 乐观场景 |
| 5 compiles × 8 rollouts | 22s | 2.7s | 标准场景 |
| 8 compiles × 8 rollouts | 35s | 4.4s | 悲观场景 |

**但 outlier 问题严重**：如果一个 group 内有一个 rollout 触发了 barrier_gate 级别的 compile（~40s），整个 group 都要等。

#### 性能优化建议

1. **考虑对 RL 训练禁用 subprocess wrapper**（`URDF_COMPILE_TIMEOUT_SECONDS=0`）：中位数从 0.55→0.08s（7x 加速）。用 `max_turns` 和 `max_total_completion_tokens` 替代 compile 层超时
2. **单独设置 RL 用 QC 开关**：outlier 的主因是几何 QC（overlap/connectivity），可以通过 `URDF_DISABLE_GEOMETRY_OVERLAP_CHECK=1` 关闭最重的 overlap 检查，作为 curriculum 初期简化
3. **`_MODEL_EXECUTION_LOCK` 不影响 subprocess 模式**：子进程之间独立，锁在主进程内。但如果改用 in-process 模式（建议 1），需要用 `asyncio.to_thread` + 确保每个 env worker 是独立进程（verifiers env server 已经是多进程）

#### 0d: KAOLA 集群兼容性测试（2026-05-21，H200 debug pod，CPU-only）

测试脚本: `environments/articraft/benchmarks/compat_test.py`
Setup 脚本: `scripts/envs/articraft.sh`

安装方式：`uv pip install --no-deps -e articraft` + 手动装核心几何库（排除 cadquery）。安装耗时 ~10s。

#### `articraft.sh` 待补充步骤

当前脚本只安装 articraft SDK，**缺少以下步骤**（参考 `blendergym.sh` 模式）：

1. **安装 `articraft_env` 包**：`uv pip install -e environments/articraft`
2. **恢复训练数据集（records）**：从 S3 恢复 articraft records 到 `/local-ssd/articraft-records/`
   - fast mode 可 symlink 到 S3 FUSE
   - 正常模式解压 tar 到本地 SSD
3. **验证 articraft_env 可 import**：`uv run python -c "from articraft_env import ArticraftEnv; print('OK')"`
4. **设置环境变量**：
   - `ARTICRAFT_ROOT` — articraft 代码根目录
   - `ARTICRAFT_RECORDS_ROOT` — records 数据目录
   - `URDF_COMPILE_TIMEOUT_SECONDS=60` — RL 用 60s（正常 P90 ~23s）

完整 `env_setup()` 应为：
```bash
env_setup() {
    setup_ac_install_system_libs
    setup_ac_install_python_pkg
    setup_ac_restore_dataset        # 新增
    setup_ac_install_env_pkg        # 新增: uv pip install -e environments/articraft
    setup_ac_verify_imports         # 更新: 增加 articraft_env 验证
    export ARTICRAFT_ROOT="${ARTICRAFT_DIR}"
    export ARTICRAFT_RECORDS_ROOT="/local-ssd/articraft-records"
    export URDF_COMPILE_TIMEOUT_SECONDS=60
    echo "  [env] Articraft environment ready."
}
```

| 测试项 | 版本 | 结果 |
|--------|------|------|
| torch | 2.11.0+cu128 | PASS |
| vllm | 0.20.2 | PASS |
| numpy | 2.2.6 | PASS |
| manifold3d | 3.4.1 | PASS |
| trimesh | 4.12.2 | PASS |
| python-fcl | 0.7.0.11 | PASS |
| scipy | 1.17.1 | PASS |
| networkx | 3.5 | PASS |
| pydantic | 2.12.5 | PASS |
| articraft sdk | — | PASS |
| agent.compiler | — | PASS |
| agent.feedback | — | PASS |
| CUDA H200 matmul | — | PASS |
| CompileSignalBundle 序列化 | — | PASS |
| render_compile_signals | — | PASS |
| Scaffold compile | — | EXPECTED FAIL（空模型） |

**结论: 方案 A'（直接安装，排除 cadquery）在 KAOLA 集群上完全可行。14/14 import 通过，无依赖冲突。不需要服务化，不需要沙盒。**

numpy 注意事项：articraft 声明 `>=2.4.1`，安装时先升到 2.4.6，但 prime-rl 锁文件又降回 2.2.6。实际运行无报错，说明 articraft SDK 不依赖 numpy 2.4+ 新特性。

---

## 实现阶段总览

### Phase 1：全 In-Process RL 训练（当前目标）

**特征**：无外部服务依赖，reward 纯基于 compiler QC signals，本地或 KAOLA 均可运行。

| 维度 | 说明 |
|------|------|
| Reward | `check_fraction`（QC 通过率）+ `compile_attempted` bonus |
| 计算方式 | in-process 调 `compile_urdf_report_maybe_timeout` |
| 外部服务 | 无 |
| GPU 需求 | 仅 vLLM 推理需要 GPU，env 本身纯 CPU |
| 验证范围 | 模型是否"结构正确"（几何合法、连通、无碰撞） |
| 不验证 | 模型是否"语义正确"（看起来像描述的东西） |

**Phase 1 Steps**：

- [x] Step 0: 前期验证（性能基准 + 架构决策 + 兼容性验证 + API 验证）— **全部完成**
- [ ] **Step 1: Mock Eval Rollout（当前目标）** — 用 MockClient 跑通 env 全流程
- [ ] Step 2: 接真模型 eval（OpenAI API 或 KAOLA vLLM）
- [ ] Step 3: 实现 dataset loader — 从 articraft records 提取 prompts
- [ ] Step 4: 编写 TOML 训练配置 (`configs/articraft/rl.toml`)
- [ ] Step 5: KAOLA setup 脚本更新 — `scripts/envs/articraft.sh`（补充 dataset 恢复 + articraft_env 安装 + 环境变量）
- [ ] Step 6: 端到端 RL 训练

---

### Phase 1.5：Deferred Enhancements（Phase 1 完成后按需启用）

Phase 1 为求简快速跑通，有意 defer 了部分 articraft 特性。以下是每项的**触发条件**、**实现方案**和**预估工作量**：

#### 1. `repeated`/`failure_streak` 编译反馈追踪

**articraft 源码**：`agent/harness_compile.py` `CompileFeedbackLoop._render_compile_tool_output()` (L125–140)

**articraft 行为**：
- 签名计算：`_compile_signal_signature(bundle)` (L119–123) = `hashlib.sha1(json.dumps(bundle.to_dict(), sort_keys=True, separators=(",",":")).encode()).hexdigest()`
- 仅当有 failure signals 时追踪：`repeated = (sig == _last_compile_failure_sig)`
- 成功 compile 重置：`_last_compile_failure_sig = None`, `_consecutive_compile_failure_count = 0`
- 下游效果 (`feedback.py:1176+`)：
  - `repeated=True` → summary 追加 `"This failure matches the previous compile attempt."`
  - `failure_streak >= 3` → 追加 `"This is compile failure {N} in a row."` + response_rules 建议 probe_model

**触发条件**：观察到模型在 RL 训练中反复发出相同的无效 compile（浪费 turns）。

**实现方案**：
```python
# 在 Rollout 中新增字段（对应 harness_compile.py L54–58）
last_failure_sig: str | None = None        # _last_compile_failure_sig
consecutive_failure_count: int = 0         # _consecutive_compile_failure_count

# execute_compile 中（对应 _render_compile_tool_output L125–140）：
sig = hashlib.sha1(json.dumps(bundle.to_dict(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
has_failures = any(s.blocking for s in bundle.signals)
if has_failures:
    repeated = (sig == rollout.last_failure_sig)
    rollout.last_failure_sig = sig
    rollout.consecutive_failure_count += 1
else:
    repeated = False
    rollout.last_failure_sig = None
    rollout.consecutive_failure_count = 0

content = render_compile_signals(bundle, repeated=repeated, failure_streak=rollout.consecutive_failure_count)
```

**工作量**：~15 行，改 `execute_compile` + `Rollout` 加 2 个字段。

---

#### 2. Guidance Injection（`edit_retry_guidance`）

**articraft 源码**：`agent/harness_guidance.py` `GuidanceInjector` (L110–275)

**articraft 行为**（3 种 guidance，均为 one-shot + user message 注入）：

| Guidance | 方法 | 行号 | 触发条件 | One-shot sig |
|----------|------|------|---------|-------------|
| edit_retry | `maybe_inject_edit_code_guidance()` | L243–275 | tool="replace" + error 含 "Could not find the old_string in the code" | `"replace_old_string_not_found"` |
| exact_geometry | `_maybe_inject_exact_geometry_contract_guidance()` | L143–179 | 成功 mutation 后 + AST 扫描发现 `ctx.expect_*` 引用了不存在的 visual name | `sha256(json.dumps(missing_names))` |
| baseline_qc | `_maybe_inject_baseline_qc_guidance()` | L181–215 | 成功 mutation 后 + `run_tests()` 含 compiler-owned baseline QC 调用 | `sha256(json.dumps(baseline_qc_calls))` |

**edit_retry 精确模板** (L265–273)：
```xml
<edit_retry_guidance>
- Your last replace failed because `old_string` did not match the file exactly.
- Do NOT guess. Call `read_file(path="model.py")` again, then pick a smaller exact snippet from the current editable code as `old_string` and retry.
- Keep edits surgical.
</edit_retry_guidance>
```

**注入机制** (L129–133)：通过 `_append_guidance_message()` 作为 `role: user` 消息追加。  
**注入时序** (harness.py:1305–1314)：每轮 tool 执行后依次调 `maybe_inject_edit_code_guidance` → `maybe_inject_code_contract_guidance`。

**触发条件**：观察到模型在 RL 中频繁 replace 失败后不知道 read_file（死循环）。

**实现方案**：
```python
def _maybe_inject_guidance(self, tool_calls, raw_results: list[str], rollout) -> list:
    """检查 replace 失败并注入 edit_retry_guidance（one-shot per rollout）。

    raw_results 是 _dispatch_tool 返回的 JSON 字符串列表。

    源码参考：
    - articraft/agent/harness_guidance.py L243-275  maybe_inject_edit_code_guidance()
    - 触发条件: L249-258 检查 replace 结果是否有 "Could not find" error
    - one-shot 机制: L261 self._edit_retry_guidance_injected flag
    - 文案内容: L263-274  EDIT_RETRY_GUIDANCE 模板
    """
    guidance_msgs = []
    for tool_call, content_json in zip(tool_calls, raw_results):
        if tool_call.name != "replace":
            continue
        parsed = json.loads(content_json)
        error = parsed.get("error", "")
        if ("Could not find the old_string in the code" in error
                and not rollout.edit_retry_injected):
            rollout.edit_retry_injected = True
            guidance_msgs.append(UserMessage(role="user", content=(
                "<edit_retry_guidance>\n"
                '- Your last replace failed because `old_string` did not match the file exactly.\n'
                '- Do NOT guess. Call `read_file(path="model.py")` again, then pick a smaller '
                'exact snippet from the current editable code as `old_string` and retry.\n'
                "- Keep edits surgical.\n"
                "</edit_retry_guidance>"
            )))
            break  # 同轮只注入一次
    return guidance_msgs
```

**TITO 注意**：guidance 作为 UserMessage 必须放在所有 ToolMessage **之后**（TITO 要求 tool 在前 user 在后）。

**工作量**：~20 行，改 `env_response`。

**设计考量**：RL 应让模型自己学会"失败后 read_file"策略。guidance injection 是"给提示"而非"教策略"。如果模型能自主学会，不加 guidance 训练效果更好（策略更鲁棒）。建议**先不加**，仅在模型确实学不会时作为 shaped reward 的替代方案。

---

#### 3. `probe_model` 工具（诊断 tool）

**articraft 源码**：`agent/tools/probe_model/` 包（4 文件，共 ~1540 行）

| 文件 | 行数 | 职责 |
|------|------|------|
| `tool.py` | 203 | `ProbeModelTool` + `ProbeModelInvocation`：注册 schema、参数验证、子进程调度 |
| `runner.py` | 195 | 独立进程入口：`load_model_globals` → `ProbeSession` → `exec(snippet)` + `emit(value)` 契约 |
| `helpers.py` | 1108 | `ProbeSession` 类：只读几何探测 API（`part`, `joint`, `visual`, `aabb`, `dims`, `pair_report` 等） |
| `description.py` | 36 | schema description 文本（`build_probe_model_description()`）|

**articraft 行为**（`tool.py` L47–153）：
1. 参数：`code: str`（必填），`timeout_ms: int = 600000`，`include_stdout: bool = False`
2. 绑定当前 `file_path`（model.py）
3. `local_work_slot(runtime_limits)` 限并发
4. 子进程：`python -m agent.tools.probe_model.runner`，stdin JSON `{file_path, sdk_package, code}`
5. runner (`runner.py` L80+)：
   - `load_model_globals(file_path)` → 取 `object_model`
   - `TestContext(object_model, asset_root=...)` → `ProbeSession(object_model, ctx).build_namespace(emit=emit)`
   - `exec(compiled_snippet)` — 必须恰好 1 次 `emit(value)` 且 JSON 可序列化
6. 错误类型：`lookup_failure`, `emit_contract`, `non_serializable_result`, `snippet_exception`, `load_failure`, `timeout`

**`ProbeSession` 核心 API** (helpers.py)：
- Lookup: `part(name)`, `joint(name)`, `visual(name)`, `parts()`, `joints()`, `visuals()`
- Measurement: `aabb(part)`, `dims(part)`, `center(part)`, `position(joint)`, `projection()`
- Relationship: `pair_report()`, `gap_report()`, `overlap_report()`, `within_report()`, `contact_report()`, `alignment_report()`
- Review: `sample_poses()`, `nearest_neighbors()`, `find_clearance_risks()`, `catalog()`

**触发条件**：Phase 1 训练收敛但 QC 分数卡在 plateau（模型不会主动诊断问题）。

**RL 实现方案**：
```python
PROBE_MODEL_SCHEMA = {
    "name": "probe_model",
    "description": build_probe_model_description(),  # 从 articraft 导入
    "parameters": {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python snippet. Must call emit(value) exactly once."},
            "timeout_ms": {"type": "integer", "description": "Timeout in ms. Default 60000."},
        },
        "required": ["code"],
    },
}

# 执行：复用 articraft runner 子进程模式
# 前提：state 中有最后一次成功 compile 的 script_path
```

**工作量**：~50 行（schema + execute_probe + 子进程管理）。RL 中 timeout 建议 60s（非原版 600s）。

**前提**：需要保存最后成功 compile 的 script_path（已在 Rollout 中保存 `work_dir`）。

---

#### 4. `find_examples` 工具（BM25 检索）

**articraft 源码**：`agent/tools/find_examples.py`（116 行）

**articraft 行为** (L40–68)：
- 参数：`query: str`（必填，非空），`limit: int = 3`（≥1）
- 执行：`agent.examples.search_example_documents(query, limit=limit)`（BM25 词法搜索 curated example markdown）
- 返回字段：`example_id`, `title`, `description`, `tags`, `content`, `match_quality`, `matched_tokens`, `matched_fields`
- OpenAI 外 provider 额外含 `path`（`include_paths=True`）

**触发条件**：Phase 1 训练收敛但泛化能力差（模型不会类比已有范例）。

**RL 实现方案**：
```python
FIND_EXAMPLES_SCHEMA = {
    "name": "find_examples",
    "description": "Search example models by keyword. Returns up to `limit` matching examples with code.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Short search query."},
            "limit": {"type": "integer", "description": "Max results. Default 3."},
        },
        "required": ["query"],
    },
}

# 执行：复用 articraft agent.examples.search_example_documents
# 或从 records 构建独立 BM25 index（避免依赖 articraft 的 example corpus 格式）
```

**工作量**：~80 行 + 预处理脚本。

**部署依赖**：BM25 corpus 文件需上传到 KAOLA pod（~50MB）。articraft 的 example corpus 位于 `agent/examples/` 目录。

---

#### 5. SFT Warm-up 数据准备

**触发条件**：Phase 1 的 RL from scratch 完全失败（模型 0% compile 通过率）。

**⚠️ 数据可用性**：仅 33 条 records 有 `traces/trajectory.jsonl.zst` 数据。原假设的 "10K+ 成功 trajectories" **不成立**。SFT 不能直接依赖已有 trajectory 数据。

**可行方案**（按优先级）：
1. **合成 trajectory**（推荐）：从 prompt + final model.py 逆向合成 (prompt → write_file(final_code) → compile_model → success) 单步 trajectory。简单但有效——教会模型基本的 write_file + compile_model 工作流
2. **从 33 条真实 trajectory 做 SFT**：数据量极少，仅作为 format 对齐验证
3. **跳过 SFT，纯 RL**：如果 base model 已有基本 tool-calling 能力（Qwen3.5-9B 经过 function calling 训练），可能直接 RL 就能冷启动

**实现方案（方案 1）**：
1. 遍历过滤后的 ~7,387 条 records
2. 每条生成: system_prompt + task_prompt → `write_file(content=final_model_py)` → `compile_model()` → success signal
3. 转换为 `qwen3_coder` XML 格式
4. 格式化为 prime-rl SFT 训练数据格式
5. 先 SFT 再 RL

**工作量**：~300 行（合成脚本 + 格式转换）。

**关键约束**：SFT 数据的 tool_call 格式必须与 vLLM `qwen3_coder` parser 输出一致，否则 RL 阶段 format 冲突。

---

#### 6. Context Window 管理

**articraft 源码**：`agent/providers/compaction_policy.py` + `agent/providers/base.py`

**articraft 行为**：
- `ContextWindowPressure` (`base.py`)：`build_context_window_pressure(usage)` 计算 pressure ratio
- `decide_compaction()` (`compaction_policy.py`)：决策 hard/soft compaction

| 策略 | 触发条件 | 行为 |
|------|---------|------|
| Hard compaction | `pressure_ratio >= 0.90` | 强制压缩旧 messages |
| Soft (high_pressure) | ratio ≥ 0.85 + failure_streak ≥ 3 + compactable ≥ 2 | 压缩 |
| Soft (medium) | ratio ≥ 0.70 + streak ≥ 4 + compactable ≥ 2 | 压缩 |
| Soft (early) | ratio ≥ 0.55 + streak ≥ 5 + compactable ≥ 3 | 压缩 |
| Cooldown | `SOFT_COMPACTION_COOLDOWN_TURNS = 2` | 上次 soft 后 2 turns 内不再 |

**触发条件**：观察到 max_turns=50 的 rollout 超出 context window（Qwen3.5-9B 的 32K 或 128K）。

**RL 实现方案**：
- `max_total_completion_tokens` 在 verifiers `MultiTurnEnv` 中已内置
- 若需更精细控制：在 `env_response` 中估算累计 token，接近阈值时：
  1. 简单方案：直接 `state["final_env_response"] = []` 终止
  2. 复杂方案：参考 articraft compaction_policy，压缩早期 tool 结果（保留最近 N turns 完整）

**工作量**：简单方案 ~10 行；复杂方案 ~100 行。

---

#### Deferred Items 优先级路线图

```
Phase 1 完成（Step 6）
    │
    ├── 观察训练曲线
    │
    ├── 模型反复无效 compile? ──→ 启用 #1 (repeated/failure_streak)
    ├── 模型 replace 死循环?  ──→ 启用 #2 (edit_retry_guidance)
    ├── QC 分数 plateau?      ──→ 启用 #3 (probe_model)
    ├── 泛化能力差?           ──→ 启用 #4 (find_examples)
    ├── 从零 RL 完全失败?     ──→ 执行 #5 (SFT warm-up)
    ├── Context OOM?          ──→ 启用 #6 (context management)
    │
    └── 所有结构问题解决 ──→ Phase 2 (CLIP reward)
```

---

### Phase 2+：CLIP Visual Similarity Reward（后续增强）

**特征**：引入外部渲染+评分服务，reward 融合结构正确性和语义正确性。

| 维度 | 说明 |
|------|------|
| Reward | `w_qc * check_fraction + w_clip * clip_similarity` |
| 新增依赖 | articraft viewer（headless render）+ CLIP scorer service |
| 新增部署 | Playwright headless Chromium + `open_clip_torch` FastAPI |
| GPU 需求 | vLLM + CLIP 推理（可共享或分 GPU） |
| 验证范围 | 结构正确性 + 语义正确性（渲染结果 vs 文本/参考图） |
| 前提条件 | Phase 1 完成 + compile 成功才触发渲染 |

**Phase 2 额外 Steps**：

- [ ] Step 7: 部署 articraft viewer 为 headless render service（FastAPI + Playwright）
- [ ] Step 8: 实现 CLIP scorer service（open_clip_torch）
- [ ] Step 9: 预渲染 reference renders（ground-truth solutions）
- [ ] Step 10: 扩展 `ArticraftRubric` 加入 `clip_similarity_reward`
- [ ] Step 11: 联调 render + CLIP + RL pipeline

**Phase 2 服务架构**：

```
┌─────────────────────────────────────────────────┐
│  prime-rl GRPO trainer                          │
│    └── ArticraftEnv                             │
│          ├── tools.py (in-process compile)      │
│          └── rubric.py                          │
│                ├── check_fraction (in-process)  │
│                └── clip_similarity (HTTP call)  │
│                        │                        │
└────────────────────────┼────────────────────────┘
                         │ HTTP
        ┌────────────────┼────────────────┐
        │                ▼                │
        │  Render Service (viewer)        │
        │  ┌──────────────────────────┐   │
        │  │ articraft viewer FastAPI  │   │
        │  │ + Playwright headless     │   │
        │  │ URDF → Three.js → 截图    │   │
        │  └──────────────────────────┘   │
        │                │                │
        │                ▼                │
        │  Score Service (CLIP)           │
        │  ┌──────────────────────────┐   │
        │  │ open_clip_torch FastAPI   │   │
        │  │ render_img vs prompt_text │   │
        │  │ render_img vs ref_render  │   │
        │  └──────────────────────────┘   │
        └─────────────────────────────────┘
```

> 注：Phase 2 的详细设计见下方 "Phase 2+ Reward：CLIP Visual Similarity" 章节。Phase 1 完成并验证 RL 训练效果后再启动 Phase 2。

---

### Step 1 详细计划：Mock Eval Rollout

**目标**：实现最小可用的 ArticraftEnv，用 verifiers MockClient 模拟模型输出 tool_calls，端到端验证 env 逻辑（setup → tool 执行 → compile → 终止 → reward）。不需要 GPU、vLLM、或任何外部服务，本地 MacBook 即可运行。

#### 文件结构（参考 BlenderGym 包结构）

```
environments/articraft/
├── pyproject.toml
├── articraft_env/
│   ├── __init__.py          ← PEP 562 lazy imports（参考 blendergym/__init__.py）
│   ├── env.py               ← ArticraftEnv(vf.MultiTurnEnv) + load_environment()
│   ├── tools.py             ← TOOL_SCHEMAS + execute_* 函数 + 白名单
│   ├── rubric.py            ← ArticraftRubric + @vf.cleanup 写 artifacts
│   ├── prompts.py           ← SYSTEM_PROMPT（精简版，不含完整 docs）
│   ├── schema.py            ← Task/TurnRecord/Rollout + require_rollout()
│   └── artifact_manager.py  ← ArticraftArtifactManager（仿 BlenderGym 模式）
└── tests/
    └── test_rollout.py      ← MockClient 驱动的端到端测试
```

#### 从 BlenderGym 参考的设计模式

| BlenderGym 模式 | Articraft 对应 | 说明 |
|----------------|---------------|------|
| `ArtifactManager` 类 | `artifact_manager.py` | 文件 I/O 封装：make_rollout_dir, begin_turn, save_trajectory, cleanup |
| `require_rollout(state)` | `schema.py` 中同名函数 | 安全取 state["rollout"] + 错误定位 |
| `Rollout.metadata` dict | 同样设计 | 只存 W&B sample table 需要的字段，保持 schema 稳定 |
| `@vf.cleanup` 写 artifacts | `rubric.py` 中 | 在 reward 计算后写 meta.json + trajectory.json（此时 final_reward 已知） |
| `load_environment(**kwargs)` | `env.py` 末尾 | verifiers 约定的工厂入口 |
| `Dataset.from_list([{prompt, answer, info}])` | `dataset.py` | verifiers 期望的 dataset 格式 |
| PEP 562 lazy imports | `__init__.py` | 避免 import 包就拉起 articraft SDK 重依赖 |
| 多 reward 函数加权 | `rubric.py` | 主 reward (check_fraction) + 可选 shaped bonus (compile_attempted) |

#### 核心实现要点

**`tools.py` — 手写 schemas + 复刻 articraft 行为的执行逻辑**

- `WRITE_FILE_SCHEMA` / `REPLACE_SCHEMA` / `READ_FILE_SCHEMA` / `COMPILE_MODEL_SCHEMA` — 每个 tool 一个 schema dict（注明对应 articraft 源文件位置）
- `TOOL_SCHEMAS` — 上述 4 个 schema 的列表
- `get_tool_schemas()` — 返回 `TOOL_SCHEMAS`
- `get_readable_paths(articraft_root, work_dir)` — 白名单映射（model.py + sdk/_docs/*）
- `execute_write_file(content, script_path, rollout)` — 验证必需函数 → 全文件替换 → 语法校验 → `rollout.mark_code_mutated()` → JSON ToolResult
- `execute_replace(old_string, new_string, script_path, rollout)` — harness 层拦截 → 唯一匹配验证 → 替换 → 语法校验 → `rollout.mark_code_mutated()` → JSON ToolResult
- `execute_read_file(path, readable_paths, offset, limit)` — `L{n}: ` 格式输出（完整文件）
- `execute_compile(script_path, rollout)` — freshness check → 若 fresh 返回缓存 → 否则 `compile_urdf_report_maybe_timeout` + `render_compile_signals`（含 response_rules）→ `rollout.mark_compile_success()` → JSON ToolResult
- `_dispatch_tool(name, args_str, rollout, readable_paths)` — 统一入口（详见 env.py 完整实现）：
  1. JSON 解析失败 → `{"error": "Invalid JSON in tool arguments: ..."}`
  2. 未知 tool name → `{"error": "Unknown tool: {name}. Available tools: ..."}`
  3. 必需参数缺失 → `{"error": "Missing required parameter: ..."}`
  4. compile_model 收到参数 → `{"error": "Invalid parameters for compile_model. Unexpected parameters: [...]"}`
  5. 执行时异常 → `{"error": "Tool execution error ({name}): ..."}`
  6. 正常路径 → `json.dumps(result.to_dict())`
- `ToolMessage(role="tool", content=json.dumps(tool_result.to_dict()), tool_call_id=tool_call.id)` — content 格式与 articraft harness 一致，tool_call_id 通过 ToolMessage 层传递

**`env.py` — ArticraftEnv + 终止/guidance 逻辑**

- `env_response()` — 终止判断（empty/text-only + freshness）+ tool 执行 + guidance 注入
- `_maybe_inject_guidance(tool_calls, results, rollout)` — edit_retry_guidance（replace 失败时一次性注入）
- 终止通过 `state["final_env_response"] = []` 触发（内置 `@vf.stop has_final_env_response` 检测）
- `_initial_model_code(task)` — Phase 1 统一返回 scaffold.py 内容（无 marker，全文件可编辑）

**`prompts.py` — System Prompt + Turn 0 Messages**

- `SYSTEM_PROMPT` — 精简版 XML 结构（`<role>` + `<tools>` + `<modeling>`）
- `PRELOAD_DOCS` — Turn 0 预加载的 2 篇核心 docs（quickstart + testing；Phase 1.5 加 probe-tooling）
- `RUNTIME_TASK_GUIDANCE` — 工作流指引（5 条，与 articraft 一致）
- `build_system_prompt(readable_paths)` — 拼入可用 docs 路径列表
- `build_turn0_messages(readable_paths, task_prompt)` — 2 条 user messages（docs 全文 + guidance + task）

**`env.py` — ArticraftEnv（参考 BlenderGymEnv 结构）**

- `__init__`: 构建 `ArtifactManager`、`ArticraftRubric`、dataset builders，传 `tool_defs=get_tool_schemas()`, `max_turns`, `rubric`
- `setup_state`: 创建 work_dir + scaffold model.py + `Rollout`，并重建 `state["prompt"]` = `build_turn0_messages()`（Turn 0 完整 messages）
- 不 override `get_prompt_messages`：Turn 0 基类直接返回 `state["prompt"]`；Turn N 基类自动调 `env_response`
- `env_response`: 按顺序串行执行 tool_calls → 畸形 JSON/未知 tool 返回 error ToolResult → 返回 `[ToolMessage...]`
- `add_model_response`（可选 override）: 记录 TurnRecord 到 `rollout.turns`
- 终止：`max_turns` + freshness 条件终止。`env_response` 检测 empty/text-only response → 若 code fresh 则设 `state["final_env_response"] = []` 终止；否则注入 `<compile_required>` user message 继续（最多 3 次，超限也终止）
- Turn 计数：每次 model response 都消耗 1 turn（含 text-only），因此 max_turns=50 留足裕量

**`rubric.py` — ArticraftRubric（参考 BlenderGymRubric 结构）**

```python
class ArticraftRubric(vf.Rubric):
    # 源码参考（reward 信号来源）：
    # - articraft/agent/feedback.py L1040-1080  check_fraction 计算
    # - articraft/data/model.py L122-135  Rollout.compile_check_fraction()
    def __init__(self, artifact_manager, reward_weights=None):
        super().__init__()
        self.artifact_manager = artifact_manager
        # weights 总和 = 1.0，避免 final_reward 超过 1.0
        w = {
            "check_fraction": 0.7,
            "build_success": 0.2,
            "compile_attempted": 0.1,
            **(reward_weights or {}),
        }
        self.add_reward_func(self.check_fraction_reward, weight=w["check_fraction"])
        self.add_reward_func(self.build_success_bonus, weight=w["build_success"])
        self.add_reward_func(self.compile_attempted_bonus, weight=w["compile_attempted"])
        self.add_metric(self.blocking_failure_count)
        self.add_metric(self.warning_count)
        self.add_metric(self.turns_used)

    @vf.cleanup
    async def write_artifacts_handler(self, state) -> None:
        rollout = require_rollout(state)
        self.artifact_manager.save_trajectory(rollout, metrics=state.get("metrics"))
        self.artifact_manager.cleanup_rollout(rollout)
```

**`schema.py` — 数据结构（参考 BlenderGym schema.py）**

```python
SCHEMA_VERSION = "articraft-trajectory-v1"

@dataclass(frozen=True)
class Task:
    # 源码参考: articraft/data/model.py Record（record_id, prompt, category）
    record_id: str
    prompt_text: str
    category_slug: str | None = None

    @classmethod
    def from_info(cls, info: dict) -> "Task": ...

@dataclass
class TurnRecord:
    # 源码参考: articraft/data/model.py TurnRecord L42-68
    turn: int
    tool_calls: list[dict]
    compile_attempted: bool = False
    compile_success: bool | None = None
    compile_signals: dict | None = None

@dataclass
class Rollout:
    # 源码参考（等价 articraft 的分散状态）：
    # - edit_revision / last_compile_revision: harness_compile.py L48-60
    # - mark_code_mutated(): harness.py L864-865
    # - mark_compile_success(): harness_compile.py L191-192
    # - compile_required_count: harness_compile.py L96
    # - edit_retry_injected: harness_guidance.py L261
    task: Task
    trajectory_id: str
    work_dir: Path
    max_turns: int
    turns: list[TurnRecord] = field(default_factory=list)
    final_reward: float | None = None
    metadata: dict | None = None
    edit_revision: int = 0
    last_compile_revision: int = -1
    last_compile_bundle_dict: dict | None = None   # 仅成功更新（freshness）
    last_compile_attempt_dict: dict | None = None  # 每次 compile 都更新（reward）
    compile_required_count: int = 0
    edit_retry_injected: bool = False

    def code_is_fresh(self) -> bool:
        # 源码: harness_compile.py L85-89 latest_code_is_fresh()
        return self.last_compile_revision == self.edit_revision and self.last_compile_revision >= 0

    def mark_code_mutated(self):
        # 源码: harness.py L864-865
        self.edit_revision += 1

    def mark_compile_attempt(self, bundle):
        self.last_compile_attempt_dict = bundle.to_dict()

    def mark_compile_success(self, bundle):
        # 源码: harness_compile.py L191-192
        self.last_compile_revision = self.edit_revision
        self.last_compile_bundle_dict = bundle.to_dict()
        self.compile_required_count = 0

def require_rollout(state: dict) -> Rollout:
    rollout = state.get("rollout")
    if not isinstance(rollout, Rollout):
        raise RuntimeError("Articraft rollout state missing; setup_state likely failed")
    return rollout
```

**`artifact_manager.py` — 文件管理（仿 BlenderGym ArtifactManager）**

```python
# 源码参考（articraft 的 trajectory 存储）：
# - articraft/storage/trajectories.py  TrajectoryStore.save()
# - articraft/data/model.py  Rollout.to_dict() / TurnRecord.to_dict()
# - prime-rl/environments/blendergym/blendergym/env.py ArtifactManager（结构参考）

@dataclass
class ArtifactPolicy:
    save_meta_json: bool = True
    save_trajectory_json: bool = True
    save_per_turn_snapshots: bool = True
    keep_failed_only: bool = False
    max_rollouts_per_example: int = 0

class ArticraftArtifactManager:
    def __init__(self, work_root: Path, policy: ArtifactPolicy): ...
    def make_rollout_dir(self, traj_id, record_id, split, example_id) -> Path: ...
    def begin_turn(self, work_dir, turn_idx) -> Path: ...  # 返回 turn_N/ 目录
    def save_trajectory(self, rollout, metrics=None) -> None: ...  # 写 meta.json + trajectory.json
    def cleanup_rollout(self, rollout) -> None: ...  # keep_failed_only 逻辑
    def prune_old_rollouts(self, rollout) -> None: ...  # max_rollouts_per_example
```

**`tests/test_rollout.py` — Mock 驱动的验证**

用 verifiers `MockClient` 模拟一个 `max_turns=3` 的轨迹：
1. Turn 1: 模型输出 `write_file` + `compile_model`（scaffold 空模型，compile 会失败）
2. Turn 2: 模型输出 `replace`（写入有效模型代码）+ `compile_model`（应成功）
3. Turn 3: 模型输出 `compile_model`（确认仍成功）→ 达到 max_turns 自然终止

验证：reward > 0，state 中有正确的 compile signals，rollout 在 3 turns 后结束。

#### MockClient 用法（来自 verifiers conftest.py）

```python
from verifiers.testing import MockClient  # 或从 conftest 复制

mock = MockClient()
mock.add_response(
    messages_match=...,
    response=AssistantMessage(
        role="assistant",
        tool_calls=[
            ToolCall(id="tc_1", name="write_file", arguments='{"content": "..."}'),
            ToolCall(id="tc_2", name="compile_model", arguments='{}'),
        ]
    )
)
```

#### 运行方式

```bash
cd prime-rl
uv pip install -e environments/articraft
uv run pytest environments/articraft/tests/test_rollout.py -v
```

#### 验证后的下一步（Step 2）

mock 通过后，可用真模型验证：
- `OPENAI_API_KEY` + `env.evaluate(client=ClientConfig(..., api_base_url="https://api.openai.com/v1"), model="gpt-4.1-mini")`
- 或 KAOLA 上起 vLLM + `uv run vf-eval articraft -m Qwen/Qwen3.5-9B ...`

#### TOML 训练配置示例（Step 4: `configs/articraft/rl_articraft_kaola.toml`）

```toml
# Articraft RL training config — KAOLA platform.
#
# Articraft 不需要外部渲染/评分服务（所有 compile 和评估 in-process），
# 比 BlenderGym 更简单：无 render_service_url / score_service_url。
#
# Setup: . scripts/envs/articraft.sh  (must use source, not bash)
# Train: uv run rl @ configs/articraft/rl_articraft_kaola.toml

output_dir = "/local-ssd/prime-rl-output"
# ⚠️ seq_len 必须容纳整个对话（Turn 0 prompt + 所有 completions + tool results）
# Token 预算分析：
#   Turn 0 prompt（精简 docs 摘要版）: ~3000 tokens
#   System prompt: ~500 tokens
#   每 turn assistant output: ~300 tokens × 15 turns = ~4500 tokens
#   每 turn tool results: ~800 tokens × 15 turns = ~12000 tokens（compile output 较长）
#   合计典型 rollout: ~20000 tokens
# 结论：seq_len=16384 太紧，改用 32768
seq_len = 32768
max_steps = 20000

[deployment]
type = "single_node"
num_infer_gpus = 6
num_train_gpus = 2

[ckpt]
interval = 25
keep_last = 2
output_dir = "/local-ssd/checkpoints/articraft-9b-dp6"

[model]
name = "Qwen/Qwen3.5-9B"

[weight_broadcast]
type = "nccl"

[wandb]
project = "articraft-rl"
name = "9b-dp6-bs64-kaola-phase1"

[orchestrator]
batch_size = 64
rollouts_per_example = 8
oversampling_factor = 1.5
max_async_level = 1

[orchestrator.eval]
interval = 10

[orchestrator.train.sampling]
# max_completion_tokens 是单次 LLM 调用的上限（含 thinking tokens）
# 典型 tool-call turn: ~200-500 tokens assistant output
# Phase 1 不启用 reasoning（去掉 thinking_token_budget），减少变量
max_completion_tokens = 4096

[[orchestrator.filters]]
type = "zero_advantage"
enforce = false

[[orchestrator.train.env]]
id = "articraft"
num_workers = 6
# max_total_completion_tokens = 所有 turns 的 assistant completion 总和上限
# 15 turns × ~300 tokens/turn = ~4500；留 buffer 设 8192
# 注意：这只限制 completion tokens，不含 prompt/tool results
max_total_completion_tokens = 8192

[orchestrator.train.env.args]
articraft_root = "/local-ssd/articraft"
max_turns = 50
work_root = "/local-ssd/prime-rl-output/articraft-work"
env_name = "articraft"
split = "train"
eval_split = "eval"
eval_holdout = 50
filter_cadquery = true

[orchestrator.train.env.args.reward_weights]
# weights 总和 = 1.0，确保 final_reward ∈ [0, 1]
check_fraction = 0.7       # 主信号：QC 通过率
build_success = 0.2        # 代码能否执行（区分 build failure vs QC failure）
compile_attempted = 0.1    # 至少尝试了 compile

[[orchestrator.eval.env]]
id = "articraft"
name = "articraft-eval"
num_workers = 1
num_examples = 10
rollouts_per_example = 1

[orchestrator.eval.env.args]
articraft_root = "/local-ssd/articraft"
max_turns = 50
work_root = "/local-ssd/prime-rl-output/articraft-work"
env_name = "articraft-eval"
split = "eval"
eval_split = "eval"
eval_holdout = 50
filter_cadquery = true

[trainer]
max_async_level = 1

[trainer.model]
optimization_dtype = "bfloat16"
reduce_dtype = "bfloat16"

[trainer.model.ac]

[trainer.optim]
lr = 3e-6

[inference]
gpu_memory_utilization = 0.75
enable_prefix_caching = true

[inference.model]
max_model_len = 32768
enforce_eager = true
trust_remote_code = true
tool_call_parser = "qwen3_coder"
# ⚠️ Phase 1 不启用 reasoning_parser：
# 1. thinking tokens 占用 max_completion_tokens 额度（减少实际 tool_call 输出空间）
# 2. TITO completion_mask 对 thinking tokens 的处理需要验证
# 3. 减少变量，先建立 baseline；Phase 2 作为 ablation 开启
# reasoning_parser = "qwen3"  # Phase 2 ablation 时取消注释
```

**与 BlenderGym TOML 的关键区别**：
- 无 `render_service_url` / `score_service_url`（所有 compile 和评估 in-process，无外部服务依赖）
- `max_turns = 50`（BlenderGym 仅 3 turns，Articraft 是多轮 tool-use 场景）
- `tool_call_parser = "qwen3_coder"`（BlenderGym 用 XML parser，不需要 tool_call_parser）
- `filter_cadquery = true`（过滤掉 `import cadquery` 的 records）
- 无 `[model.vlm]` section（Articraft 是纯文本场景，不需要 vision encoder）
- Phase 1 不启用 `reasoning_parser`（避免 thinking tokens 占用 completion 额度 + 简化 debug）

## 方案对比

| 方案 | 优点 | 缺点 |
|------|------|------|
| **A: 复用 articraft tool 执行代码（推荐）** | 保持与推理时行为一致；tool 逻辑已测试过；reward 信号丰富 | 需要把 articraft 作为依赖安装；SDK 重 |
| **B: 简化为 code-as-action（无 tool calling）** | Action space 更小；无需 tool_call_parser | 与现有 agent 不一致；丢失 probe/find_examples 能力 |
| **C: 外部 articraft 服务（HTTP）** | 解耦依赖；可独立扩展 worker | 增加 infra 复杂度；类似 BlenderGym 的 render service |

## 确认的选择

| 维度 | 决定 | 验证状态 |
|------|------|---------|
| 部署方案 | **A'（直接安装，排除 cadquery）** | ✅ KAOLA 实测通过 |
| 基类 | **`vf.MultiTurnEnv`** — 自实现 env_response 中的 tool 分发（ToolEnv 不匹配） | ✅ 源码分析确认 |
| 服务化 | **不需要** — 无冲突、无资源隔离需求、compile 够快 | ✅ 基准测试 + 兼容性测试 |
| 初始 tools | **4 个**（write_file, replace, read_file, compile_model） | — |
| 模型 | **Qwen3.5-9B** — 与 BlenderGym RL 一致，2 GPU train + 6 GPU infer | — |
| 数据 | **从现有 records 提取 prompt** — 利用 articraft storage 中已有的 ~10K 成功 records | — |

---

## 依赖冲突分析

### 实际冲突面

| 依赖 | articraft 要求 | prime-rl 要求 | 冲突？ |
|------|---------------|--------------|--------|
| numpy | `>=2.4.1` | `>=2.2.6` | ❌ 兼容 |
| scipy | `>=1.17.1` | 不需要 | ❌ 新增即可 |
| manifold3d | `>=3.3.2` | 不需要 | ❌ 纯 C++ wheel |
| trimesh | `>=4.11.3` | 不需要 | ❌ 纯 Python |
| python-fcl | `>=0.7.0.8` | 不需要 | ⚠️ 需要 libfcl 系统库 |
| cadquery | `>=2.5` | 不需要 | ⚠️ 依赖 OCP (OpenCASCADE)，巨型 C++ 库 |
| networkx | `>=3.6.1` | 不需要 | ❌ 纯 Python |

### 关键发现：cadquery 是 lazy import

```python
# sdk/_dependencies.py
def require_cadquery(*, feature: str) -> Any:
    try:
        return importlib.import_module("cadquery")
    except Exception as exc:
        raise RuntimeError(...)
```

cadquery 只在 `sdk._extensions.cadquery` 扩展中使用（cadquery-specific 网格操作）。**核心 SDK 路径不需要 cadquery** — model.py 脚本用的是 `manifold3d` 为基础的内置网格操作。

### 真正的重依赖

| 依赖 | 大小/复杂度 | 编译时是否 model.py 必需 |
|------|-----------|------------------------|
| **manifold3d** | ~5MB wheel, pre-built | ✅ 所有几何操作的核心 |
| **trimesh** | 纯 Python, ~2MB | ✅ QC 检查（碰撞检测输入网格） |
| **python-fcl** | 需要 `libfcl-dev` 系统库 | ✅ 碰撞检测 |
| **cadquery (OCP)** | ~200MB wheel | ❌ 可选扩展，不是核心路径 |

### 结论

- **无 numpy/torch 版本冲突**（articraft numpy≥2.4 兼容 prime-rl numpy≥2.2）
- **主要风险是 python-fcl** 需要系统库 `libfcl-dev`，KAOLA 默认镜像可能没有
- **cadquery 可以不装**（lazy import + 不在核心编译路径）
- **manifold3d + trimesh** 是纯 wheel/Python，与 torch/vllm 无冲突

---

## 服务化方案评估

### 方案 C-Lite: Compile Service（仅重操作服务化）

```
┌──────────────────────────────────┐     ┌──────────────────────────────┐
│ prime-rl process                  │     │ Articraft Compile Service     │
│ (torch + vllm + verifiers)        │     │ (manifold3d + trimesh + fcl) │
│                                   │     │                              │
│ ArticraftEnv:                     │     │ FastAPI:                     │
│   read_file  → local (trivial)   │     │   POST /compile              │
│   write_file → local (trivial)   │     │     body: {script_content}   │
│   replace    → local (trivial)   │     │     resp: CompileSignalBundle│
│   compile_model → HTTP client ───────→ │                              │
│   probe_model   → HTTP client ───────→ │   POST /probe                │
│   find_examples → local (BM25)   │     │     body: {code, script}     │
│                                   │     │     resp: {output}           │
└──────────────────────────────────┘     └──────────────────────────────┘
```

**只有 compile_model 和 probe_model 需要服务化**（它们执行 model.py 脚本，依赖 SDK 几何库）。read_file/write_file/replace 是纯文件操作，find_examples 是 BM25 搜索 — 都不需要 SDK。

### 方案对比（更新版）

| 方案 | 优点 | 缺点 | 部署复杂度 |
|------|------|------|-----------|
| **A: 直接安装 articraft（原方案）** | 最简单、延迟最低、单进程 | 需要在 prime-rl venv 装 manifold3d + fcl；万一冲突难排查 | 低 |
| **A': 直接安装但不装 cadquery** | 去掉最大风险（200MB OCP）；其他都是轻量 wheel | 还是需要 libfcl-dev 系统库 | 低 |
| **C-Lite: 只服务化 compile+probe** | 完全隔离 SDK 依赖；compile service 有独立 venv；轻量 tools 留本地 | 多一个进程；HTTP 延迟 ~5-10ms/call；需要管理 service 生命周期 | 中 |
| **C-Full: 所有 tools 都服务化** | 彻底解耦 | 过度工程；read_file/write_file 走网络无意义 | 高 |

### 方案推荐

**推荐 A'（直接安装，排除 cadquery）— 已实测验证 ✅**

理由：
1. **已验证无冲突**：KAOLA H200 pod 上 torch 2.11 + vllm 0.20 + articraft SDK 全部 import 通过，CUDA 正常工作（Step 0d）
2. manifold3d、trimesh、scipy、networkx 都是纯 wheel/Python 包，与 torch 零冲突
3. python-fcl 需要 `apt install libfcl-dev`，setup 脚本已覆盖（`scripts/envs/articraft.sh`）
4. cadquery 不在核心编译路径（lazy import），直接不装，节省 ~200MB
5. 避免了服务化带来的额外复杂度（进程管理、HTTP 通信、错误传播、生命周期管理）
6. compile 中位数 0.08s（in-process），HTTP round-trip 开销反而是净负担

**C-Lite 服务化方案降级为存档参考，不再作为 fallback。** 实测已排除冲突风险，无需保留 fallback 路径。

### ~~服务化 Fallback 的实现成本~~（已不需要）

> 以下仅作存档参考。Step 0d 兼容性测试已确认方案 A' 可行，无需服务化。

如果走 C-Lite，需要：
1. `articraft_compile_service/server.py` — FastAPI, `POST /compile` + `POST /probe`
2. `articraft_compile_service/client.py` — httpx sync client（类似 `blendergym/services/render/client.py`）
3. 独立 venv/container 安装 articraft SDK
4. 在 `scripts/setup_kaola.sh` 中启动 service

工作量约 2 天，且有 BlenderGym 完整参考可抄。

## 状态
**当前阶段**: Step 0 全部完成（性能基准 + 架构决策 + 兼容性验证 + API 验证），待进入 Step 1 实现

### Step 0 结论摘要

| 验证项 | 结论 | 依据 |
|--------|------|------|
| 依赖兼容性 | ✅ 无冲突 | KAOLA H200 pod 实测 14/14 import 通过 |
| 服务化必要性 | ❌ 不需要 | 无冲突 + compile 中位数 0.08s + 无资源隔离需求 |
| 环境基类 | `vf.MultiTurnEnv` | ToolEnv 假设不匹配（tool_calls assert、只返回 ToolMessage、简单 callable） |
| cadquery | 不装 | lazy import，不在核心编译路径 |
| 部署方案 | A'（直接安装，排除 cadquery） | 实测验证 |
| Setup 脚本 | `scripts/envs/articraft.sh` | apt install libfcl-dev + uv pip install |
| 性能瓶颈 | outlier compile（~40s，几何 QC 重） | compile_bench.py 基准测试 |
| 优化方向 | 禁用 subprocess wrapper + 可选关闭重 QC | 性能优化建议 1-3 |
| verifiers API 兼容性 | ✅ 全部通过 | 源码审查（见下方 Step 0e） |

---

### Step 0e: verifiers API 验证（2026-05-21，源码审查 rev `77a9f28`）

对方案中所有 verifiers/vLLM API 假设做了源码级验证，**全部通过**：

| API 假设 | 结论 | 代码位置 |
|----------|------|---------|
| `MultiTurnEnv` 接受 `tool_defs` | ✅ 通过 `**kwargs` → `Environment.__init__(tool_defs=...)` | `verifiers/envs/environment.py:97-134` |
| `env_response` 返回 `[ToolMsg..., UserMsg?]` | ✅ `concat_messages` 无 role 过滤；TITO 要求 tool 在前、user 在最后 | `verifiers/envs/multiturn_env.py:88-97`, `clients/openai_chat_completions_token_client.py:47-55` |
| `state["final_env_response"]` 触发终止 | ✅ 内置 `@vf.stop has_final_env_response` | `verifiers/envs/multiturn_env.py:79-82` |
| `AssistantMessage.tool_calls` | ✅ `list[ToolCall]`，含 `.id/.name/.arguments` | `verifiers/types.py:118-132` |
| vLLM tool schema 注入 | ✅ `get_model_response` → `client.get_response(tools=state["tool_defs"])` | `verifiers/envs/environment.py:524-551` |
| `tool_call_parser` for Qwen3.5-9B | ✅ 应为 `qwen3_coder`（非 `hermes`） | `prime_rl/inference/vllm/server.py` `MODEL_TOOL_CALL_PARSER` |

#### TITO 兼容性约束（重要）

verifiers 的 token-in-token-out 优化（`OpenAIChatCompletionsTokenClient`）对 `env_response` 尾部消息有格式要求：

```python
# _is_valid_env_tail: 所有前置消息须为 tool，最后一条可为 tool 或 user
[ToolMessage, ToolMessage, ..., UserMessage]  # ✅ 兼容
[ToolMessage, UserMessage, ToolMessage]       # ❌ TITO fallback
```

Articraft 的 `env_response` 返回 `[ToolMessage(tool1), ToolMessage(tool2), ..., UserMessage(guidance)]` **天然符合此约束**。

**终止时 `return []` 安全性**：当 `state["final_env_response"] = []` 被设置后，rollout loop（`while not await self.is_completed(state)`）在下一次迭代顶部检查到 `has_final_env_response` 为 True，直接退出循环，不再调用 `get_prompt_messages`。因此空列表不会进入 TITO 拼接路径，无需担心 `_is_valid_env_tail` 校验失败。

#### vLLM tool schema 注入路径

```
ArticraftEnv.__init__(tool_defs=schemas)
  → Environment.__init__ → self.tool_defs
  → init_state() → state["tool_defs"]
  → get_model_response() → client.get_response(tools=tool_defs)
  → OpenAI chat.completions.create(tools=oai_tools)
```

不需要在 TOML 中配置 tool schema；TOML 只需配 `tool_call_parser`。

#### 修正：`tool_call_parser` 选择

方案中部分地方写了 `hermes`，应统一为 **`qwen3_coder`**（Qwen3.5 系列专用）：

| 模型系列 | parser |
|---------|--------|
| Qwen3 (0.6B–235B) | `hermes` |
| **Qwen3.5 (0.8B–397B，含 Qwen3.5-9B)** | **`qwen3_coder`** |

#### 参考先例

| 环境 | 基类 | tool calling | parser |
|------|------|-------------|--------|
| wiki-search | `vf.ToolEnv` | ✅ vLLM 原生 | `hermes` (Qwen3) |
| math-python | `vf.PythonEnv` → `StatefulToolEnv` | ✅ vLLM 原生 | `hermes` (Qwen3) |
| BlenderGym | `vf.MultiTurnEnv` | ❌ XML `<code>` | 无 |
| **Articraft（计划）** | `vf.MultiTurnEnv` | ✅ vLLM 原生 | `qwen3_coder` (Qwen3.5) |

Articraft 是 **首个在 `MultiTurnEnv` 上使用 vLLM 原生 tool calling 的环境**（wiki-search 用 ToolEnv，BlenderGym 用 XML）。tool 分发逻辑参考 `ToolEnv.env_response` 约 20 行即可。
