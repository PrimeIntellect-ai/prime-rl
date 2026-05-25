# Articraft 环境迁移方案审查

## 目标

将 Articraft agent 的多轮 tool-calling 环境忠实地迁移到 prime-rl 的 `verifiers` 框架中，使得 GRPO 训练出的模型能在 Articraft 任务上获得有效提升。

完整计划：`@prime-rl/.agents/plans/articraft-env-integration.md`

## 当前方案概要

- 基于 `vf.MultiTurnEnv`，实现 `ArticraftEnv`
- 4 个 tools: write_file, replace, read_file, compile_model（tool schemas 手写）
- Reward: compile QC 的 check fraction（通过的 checks / 总 checks）
- Freshness 追踪 + 条件终止（code fresh 时 text-only/empty response 触发终止）
- Turn 0: system prompt(XML) + 2 条 user messages（docs 全文 + guidance + task prompt）
- 训练数据来源: articraft records（model.py 含 USER_CODE_START/END marker）
- 已完成: compile bench、KAOLA 兼容性测试、SDK 安装脚本

---

## 需要继续检查的问题

### 一、verifiers 框架兼容性（最高优先级）

1. **条件终止机制**：env_response 检测 text-only/empty response → 调用 `final_env_response()` 终止。verifiers MultiTurnEnv 是否有此 API？如何终止一个 episode——返回空列表？设置 state 标志？还是必须等 max_turns？需阅读 verifiers rollout 循环源码。

2. **Turn 计数语义**：articraft 中 empty response 不消耗 turn。verifiers 的 turn counter 何时递增？如果 env_response 调用本身就计为 1 turn，"免费重试" 不可能实现。

3. **env_response 只返回 UserMessage**：text-only response 时注入 `<compile_required>`，无 ToolMessage。verifiers/TITO 是否允许？BlenderGym 的 env_response 是否有类似情况？

4. **completion_mask**：env_response 返回的 guidance UserMessages 不应参与 loss。verifiers 如何处理？是否 env_response 的所有返回都自动 mask 为 non-completion？

5. **get_prompt_messages override**：Turn 0 返回 2 条 user messages。verifiers 的调用时机、返回格式约束、与 system_prompt 参数的关系？

6. **Rubric 获取最终 compile 结果的时机**：`check_fraction_reward` 需要读最后一次 compile 的结果。如果模型在最后一个 turn 没有调 compile_model（直接输出 text 终止），rubric 读什么？是读 rollout 中最后一次 compile 的缓存结果，还是强制执行一次 compile？

7. **state dict 的生命周期**：`state["rollout"]` 对象在整个 episode 中持续存在。verifiers 是否保证 state dict 在 setup_state → 多次 env_response → rubric 之间完整传递？有无 pickle/serialize 要求？

### 二、Tool 执行与 dispatch 逻辑

8. **tool_call 解析格式**：vLLM `qwen3_coder` parser 输出的 tool_call 对象结构是什么？`tool_call.name` / `tool_call.arguments` 是字符串还是 dict？arguments 是 JSON 字符串需要 json.loads 吗？

9. **多 tool_calls in one response**：模型可能在一次 response 中输出多个 tool_calls（如 `[write_file, compile_model]`）。我们按顺序执行，每个的结果是独立 ToolMessage。verifiers 和 vLLM 如何处理一次 response 多个 tool_calls？是分成多个 ToolMessage 还是合并？

10. **tool_call_id 的生成和匹配**：每个 ToolMessage 需要一个 tool_call_id 与对应的 tool_call 关联。vLLM 生成的 tool_call 是否带 id？verifiers 是否做 id 匹配校验？

11. **tool_call 失败时的整体 episode 行为**：如果一个 turn 中有 `[replace, compile_model]`，replace 失败（返回 error ToolMessage），compile_model 是否仍然执行？articraft harness 是全部执行的，我们也应该全部执行。但需确认这与 verifiers 的期望一致。

### 三、数据与初始状态

已确认的事实（从实际 records 读取）：
- 总 records: **10,797**，244 个 categories
- 几乎全是 **final_status=success**（10,795/10,797）— model.py 是 agent 最终完成的代码
- model.py 含 `USER_CODE_START/END` marker，header 是 `from __future__ import annotations` + imports，footer 是 `object_model = build_object_model()`
- prompt 在 `revisions/rev_000001/prompt.txt`（简短的物体描述文本）
- 只有 `rev_000001`（无 rev_000000 初始版本）
- 大部分由 openai/gpt-5.4 生成，rating 5 占 70%
- traces 目录有 `trajectory.jsonl.zst`（压缩的完整对话 trajectory）
- `revision.json` 含 `run_summary`（turn_count, tool_call_count, compile_attempt_count, final_status）
- `record.json` 含 `category_slug`, `rating`, `provider`, `model_id`, `display.title`

12. **RL 初始状态的根本问题**：所有 records 的 model.py 都是**成功完成的最终代码**（不是 scaffold）。RL agent 的起点应该是什么？
    - 选项 A: 用 **scaffold.py**（空模板，无 marker）作为 model.py 起点，用 record 的 prompt.txt 作为 task → 从零开始写
    - 选项 B: 用 record 的**最终 model.py** 作为起点 → 模型"改进"已完成代码（意义不大？）
    - 选项 C: 用带 marker 但 editable region 清空的模板 → 保留 imports scaffold + 空 editable region
    - 这个选择直接决定 RL 的难度和目标，**必须在实现前明确**

13. **prompt.txt 的格式**：已确认是简短物体描述。是否所有 records 都有？是否有 multi-turn prompt（`prompt_series_json`）？

14. **Rating 对 dataset 的影响**：rating 分布不均（5:7494, 4:2571, 3:573, 2:134, 1:11）。是否只用高 rating records？low rating 意味着什么？

15. **含图片的 tasks 识别**：`record.json` 有 `artifacts.inputs_dir`。如何判断 task 是否包含参考图片？

16. **Trajectory 数据用于 SFT**：`trajectory.jsonl.zst` 含完整 agent 对话历史（10K+ 成功 trajectories）。这对 SFT warm-up 极有价值，但格式是 OpenAI provider 格式，需要转为 qwen3_coder。

### 四、Prompt 与 System Prompt

16. **System prompt 的精简程度**：我们将 articraft 的完整 system prompt（~2000 tokens）精简为 ~300 tokens 的 RL 版。精简的内容是否包含模型成功完成任务所需的关键信息？是否遗漏了重要的 SDK 用法提示或约束？

17. **预加载 docs 的选择**：当前选 quickstart + probe-tooling + testing 三篇。为什么是这三篇？probe-tooling 在 Phase 1 中 probe_model 工具未启用，预加载其 docs 是否有意义？是否应该换成更相关的 docs（如 geometry API、joint types）？

18. **RUNTIME_TASK_GUIDANCE 的 5 条指引**：这些指引是固定的还是需要根据 task 类型调整？articraft 原版是否有 task-specific 的 guidance？

19. **System prompt 中的 `{available_docs_list}` 占位**：列出所有可读的 docs 路径。这个列表有 26 个文件路径。加上系统 prompt + docs 全文 + guidance，Turn 0 总共消耗多少 tokens？会不会挤占模型的工作空间？

### 五、Reward 与 Rubric

20. **check_fraction 的精确计算**：
    - `total_checks` 包括哪些？failures + warnings + notes？还是只有 failures + passes？
    - `passed_checks` 如何定义？所有不是 failure 的 signal？
    - 如果 compile 完全失败（runtime error），bundle 中可能只有 1 个 failure signal（compile_runtime）且没有其他 checks。此时 check_fraction = 0/1 = 0 还是 0/N（N=所有应运行的 checks）？
    - baseline QC early exit：如果 check_model_valid 失败，后续 heavy checks 被跳过。total 是"实际运行数"还是"全部应运行数"？这影响 reward 的语义——early exit 得到的 0/1 vs 0/7 对梯度信号不同。

21. **compile_attempted_bonus 的设计**：权重多少？如果 bonus 太大，模型可能学会每个 turn 都 compile（即使代码没改）来刷 bonus。需要限制为"episode 中至少 compile 过一次"还是"每次 compile 都给 bonus"？

22. **最终 reward 取哪次 compile**：如果模型在 episode 中多次 compile（turn 5 失败、turn 10 成功、turn 15 又失败），reward 用哪次的结果？
    - 最后一次 compile？
    - 最高 check_fraction 的一次？
    - episode 结束时强制 compile？

23. **Reward 为 0 的情况**：如果模型整个 episode 都没有调用 compile_model（只做了 write_file/replace/read_file），check_fraction 无法计算。返回 0？-1？需要明确。

### 六、Compile 链

24. **`render_compile_signals` 的完整签名**：需要确认是 `render_compile_signals(bundle)` 还是 `render_compile_signals(bundle, repeated=False, failure_streak=0)`。如果 response_rules 逻辑在该函数内部依赖参数，我们需要传参；如果是从 CompileFeedbackLoop 状态推导，我们可能需要单独处理。

25. **`build_compile_signal_bundle` 的调用**：成功 compile 时需要从 `CompileReport` 构建 `CompileSignalBundle`。这个函数的完整签名和所需参数是什么？是否需要 `task_prompt`、`sdk_package` 等上下文？

26. **compile 子进程 + env server 进程模型**：
    - `compile_urdf_report_maybe_timeout` 用 multiprocessing.Process + terminate
    - verifiers env server 是独立进程？还是 vLLM worker 的线程？
    - 如果 env server 有 CUDA context，multiprocessing.fork 可能死锁
    - 需确认 verifiers 的进程架构和 articraft compiler 的 start method

27. **`_MODEL_EXECUTION_LOCK` 并发影响**：threading.Lock 在 compiler.py 中。如果 verifiers 的 env server 用 asyncio + run_in_executor（ThreadPoolExecutor），多个并发 rollout 的 compile 会被这个锁串行化。影响 throughput 的估算？

28. **compile timeout 值**：默认 300s。但如果模型写出死循环代码（`while True: pass`），每次 compile 都会 block 300s。RL 训练中这种情况可能频繁出现。是否需要更短的 timeout（如 30-60s）？

### 七、Freshness 与终止

29. **Freshness 与 rubric 的交互**：如果模型终止时 code dirty（被 max_turns 强制终止），最后一次成功编译的 bundle 可能是若干 turns 前的。rubric 应该用"最后一次成功编译结果"还是"强制在 episode 结束时再编译一次"？

30. **条件终止的 reward 计算**：模型输出 text-only → code fresh → 终止。此时 rubric 用 `last_compile_bundle`（缓存的成功结果）。这是否意味着模型学到"成功 compile 后立即输出 text 终止"是最优策略？这是好的还是需要额外的 turn-efficiency bonus？

31. **compile_required 注入后模型如何响应**：注入 `<compile_required>` 后模型重新生成。但模型可能再次输出 text-only response（无限循环）。是否需要一个 max_compile_required_retries 来防止？

### 八、articraft.sh 与部署

32. **Records 数据位置**：训练数据（articraft records）存在哪？
    - 在 articraft 仓库的 `data/records/` 下？
    - 需要单独打包到 S3？
    - 数据大小估算（如果每个 record 含完整 model.py ~10KB，数百个 records 可能只有几 MB）

33. **articraft_env 包结构**：`environments/articraft/` 需要 `pyproject.toml`（参考 blendergym 的结构）。当前 plan 中的 `pyproject.toml` 列了 `cadquery>=2.5` 作为依赖——但我们决定排除 cadquery！需要从依赖列表中移除。

34. **articraft 仓库在 KAOLA 上的位置**：`ARTICRAFT_DIR="/data/work/articraft"`。这个路径如何初始化——从 git clone？从 S3 恢复？还是预装在镜像中？

35. **环境变量传递**：`ARTICRAFT_ROOT`、`ARTICRAFT_RECORDS_ROOT`、`URDF_COMPILE_TIMEOUT_SECONDS` 如何从 setup 脚本传到 ArticraftEnv 的 constructor？通过 TOML config？通过环境变量 os.environ？

### 九、pyproject.toml 与包依赖

36. **articraft 作为 path dep 的安装**：当前用 `uv pip install --no-deps -e articraft`。但 `articraft_env/` 的 pyproject.toml 不声明 articraft 为依赖（因为是 --no-deps 安装的）。这在 `uv pip install -e environments/articraft` 时会报错吗？

37. **verifiers 版本锁**：`verifiers>=0.1.10`。当前 prime-rl 用的具体 verifiers 版本是多少？MultiTurnEnv 的 API 在不同版本间是否稳定？

38. **SDK 运行时依赖**：model.py 在 compile 时 `import sdk`。sdk 的运行时依赖（manifold3d, trimesh, fcl 等）必须在 env server 进程中可用。如果 env server 是独立 venv 还是共享主 venv？

### 十、观测与调试

39. **Trajectory 格式**：ArtifactManager 保存 trajectory.json。它的 schema 是什么？包含哪些字段？是否与 viewer 可读格式兼容？

40. **W&B 集成**：计划中提到 W&B。具体 log 什么 metrics？每个 episode 还是每个 batch？reward distribution、compile 成功率、turn 数分布、token usage？

41. **MockClient 测试的覆盖范围**：test_rollout.py 用 MockClient 测试。它能覆盖：
    - 正常 tool_call → compile 成功 路径？
    - tool_call 失败（replace 找不到 old_string）路径？
    - empty response → compile_required 注入路径？
    - text-only response + code fresh → 条件终止 路径？
    - max_turns 耗尽路径？
    - compile timeout 路径？

42. **错误诊断**：如果 RL 训练中 reward 始终为 0，如何诊断？需要哪些 logging 来判断是"模型写的代码太烂"还是"环境实现有 bug"？

### 十一、已有代码与 plan 的一致性

`prime-rl/environments/articraft/` 已有两个 benchmark 脚本：

43. **`compat_test.py` 中的调用方式**：它用 `compile_urdf_report(script_path, sdk_package="sdk")` 调用编译。plan 中用的是 `compile_urdf_report_maybe_timeout`。两者的关系是什么？`_maybe_timeout` 版本是否只是加了 subprocess 包装？`sdk_package` 参数在两个版本中都需要传吗？

44. **`compile_bench.py` 的发现与 plan 的对齐**：bench 中发现了 `_MODEL_EXECUTION_LOCK` 的并发问题、subprocess overhead 等。这些发现是否都已反映在 plan 中？是否有 bench 中发现但 plan 遗漏的问题？

45. **`compat_test.py` 设置 `URDF_COMPILE_TIMEOUT_SECONDS=0`**：这禁用了 subprocess wrapper（直接 in-process 编译）。plan 中是否明确了 RL 训练时用哪个模式——subprocess（有 timeout 保护但慢）还是 in-process（快但死循环会 block）？

### 十二、articraft 代码路径中的隐式依赖

46. **`compile_urdf_report` 需要的文件系统状态**：编译 model.py 时，SDK 的 `AssetContext.from_script(__file__)` 需要什么？是否需要 `__file__` 指向实际路径？work_dir 的结构有要求吗？

47. **SDK 运行时的 CWD 依赖**：`runpy.run_path` 执行 model.py 时 CWD 是什么？compiler.py 中是否有 `os.chdir` 到 script 所在目录？如果有，多线程下是否安全？

48. **mesh 文件生成**：某些 model.py 可能调用 `mesh_from_cadquery()` 或 `mesh_from_geometry()`。如果 cadquery 未安装，这些调用会失败吗？还是 manifold3d 路径可以替代？records 中的 model.py 是否使用了 cadquery？

49. **`ASSETS = AssetContext.from_script(__file__)` 和 `HERE`/`MESH_DIR`**：records 的 model.py header 中有这些变量。它们在我们的 work_dir 中能正确解析吗？需要什么目录结构？

50. **SDK version 兼容性**：records 是用不同版本的 articraft SDK 生成的（跨越 2026-03 到 2026-05）。当前 SDK 版本能否编译所有历史 records 的 model.py？是否有 API 变更导致旧 records 无法编译？

---

## 检查方式

请逐条通过阅读源码来确认或否定假设：
- `verifiers` 源码：`MultiTurnEnv` rollout loop、`env_response` 调用语义、终止机制、completion_mask、state 传递
- `prime-rl` 的 GRPO trainer：variable-length episode 处理、reward normalization
- `articraft/agent/feedback.py`：`render_compile_signals` 和 `build_compile_signal_bundle` 完整签名
- `articraft/agent/harness_compile.py`：`_render_reused_compile_tool_output` 精确格式
- `articraft/agent/tools/write_code.py` 和 `edit_code.py`：harness 层拦截的精确错误消息
- `articraft/agent/compiler.py`：`compile_urdf_report` vs `compile_urdf_report_maybe_timeout` 的区别、`sdk_package` 参数、CWD 处理
- `articraft/data/records/`：model.py 中 AssetContext 依赖、cadquery 使用情况
- `prime-rl/environments/articraft/benchmarks/`：已有代码与 plan 的一致性
- `prime-rl/environments/blendergym/`：作为参考（终止、turn 计数、env_response 格式、rubric）

输出格式（每个问题）：
1. **确认** — 假设正确，计划无需调整
2. **问题** — 假设有误，说明影响和修改建议
3. **需要决策** — 列出选项和 trade-off

**重要：所有检查结果和发现的问题必须更新到 `@prime-rl/.agents/plans/articraft-env-integration.md` 的对应位置**。具体规则：
- "确认" 类结果 → 在 plan 中对应假设旁标注 `[已验证]`
- "问题" 类结果 → 修改 plan 中的错误假设，更新为正确方案
- "需要决策" 类结果 → 在 plan 对应位置标注 `[待决策]` + 列出选项
- 如果发现 plan 中完全没有覆盖某个问题 → 在相关 section 新增段落说明

不要只输出检查报告而不更新 plan。plan 文档是 single source of truth。

优先级：verifiers API > 数据/初始状态 > reward/rubric > compile 链 > 终止逻辑 > 部署/依赖 > 观测
