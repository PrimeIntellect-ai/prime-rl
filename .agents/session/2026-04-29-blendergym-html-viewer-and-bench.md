# Session Handoff: BlenderGym HTML Viewer + Bench Validation

## 前置 session

- [BlenderGym OptiX + Entity Refactor](2026-04-28-blendergym-optix-entity-handoff.md) — 16 spp render 加速、Task/Rollout/TurnRecord schema、trajectory.md/json v2 schema 落地。
- [BlenderGym 渲染加速与可视化优化规划](2026-04-28-blendergym-render-speed-and-viz.md)
- [BlenderGym Phase 0–6 端到端集成](2026-04-27-blendergym-phase0-6-integration.md)

## 任务目的

把 BlenderGym 单 trajectory 可视化从 markdown(+base64 inline) 切到 HTML(+ 相对路径引图),解决 Cursor markdown preview 在 symlinked workspace 下无法显示 trajectory 图片的问题;同时验证 4-step bench 链路在新 writer 下不退化。中途发现并修复 `completion_to_text` 不支持 verifiers Pydantic `AssistantMessage` 导致的 0-byte response.txt 长期 bug。

## 执行内容

### Plan 1: trajectory.md → trajectory.html

- `trajectory_writer.py`: 删除 base64 inline 路线,新增 `_render_html(rollout)` 拆成 5 个渲染函数 + 4 个 helper(`_html_escape` / `_html_attr` / `_read_optional_text` / `_rel_img_src`)。
- 加入 `<meta name="generator" content="blendergym-trajectory-html-v1">` 版本标记;暗色模式 `prefers-color-scheme` CSS;`<a><img loading="lazy"></a>` 原图点击放大;Timeline 表 + 每 turn `<details>` 折叠 response/code/blender.log。
- `blender.log` 超过 80 KB 截尾;`html.escape()` 处理所有用户内容;`_html_escape(None) -> "-"` 避免 `None` 字面量泄漏。
- `_atomic_write_text(.tmp + os.replace)` 保护 meta.json/trajectory.json/trajectory.html 三个产物的写入;同步 `unlink(missing_ok=True)` 旧 `trajectory.md`。
- `write_trajectory_artifacts(rollout, *, metrics=None)` 签名不变,调用方零改动。
- 新增 `environments/blendergym/scripts/regenerate_artifacts.py`,扫已有 `blendergym_work` 重生成 trajectory.html(支持 `--limit / --dry-run / --overwrite / --quiet`)。
- 测试集从 19 → 26 个,新增 escape / removes_old_markdown / render_failed_turn / truncates_large_log / xml_parse_failure_renders_dash_not_none / handles_pydantic_assistant_message。

### `completion_to_text` Pydantic bug 修复

- 问题:扫描 138 trajectory × ~3.2 turn = 448 个 `response.txt` **全部 0 字节**。
- 根因:`completion_to_text` 第一行 `if not isinstance(msg, dict): continue` 把所有 message 跳过,因为 verifiers `parse_response_message` 返回 `[AssistantMessage(...)]`(Pydantic CustomBaseModel,**不是 dict**)。XMLParser 内部有自己的 normalization 所以 code.py 没受影响,但我们的 helper 没有。
- 修复:新增 `_msg_content` / `_content_block_text` 两个 helper,同时支持 dict 和 Pydantic;新增 `test_completion_to_text_handles_pydantic_assistant_message` 用真实 `AssistantMessage` + `TextContentPart` 锁住行为。

### Bench 验证

- 删除 `outputs/blendergym_v2/` 重跑 `uv run rl @ configs/multimodal/rl_blendergym.toml --bench`,exit code 0,4 step trainer + orchestrator 全成功。
- 新跑 448 个 `response.txt` **全部非空**(3.3–4.4 KB),抽样内容含模型 reasoning + `<code>` 块。
- 139 个 trajectory.html 写出,`trajectory.md` 0 个残留;orchestrator reward 0.12 → 0.21 → 0.27 → 0.17(fake-data trainer 不学,reward 是真实 rollout 但 weight 没更新)。

## 调试经验

- **Cursor markdown preview 在 symlinked workspace(`~/code -> /data/...`)下被 webview CSP 拦截本地图片**,改成 HTML 相对路径走浏览器 file:// 才能避开。`markdown.preview.localResourceRoots` 设置 / Markdown Security Settings 都无效。`Cmd+P` 直接打开 PNG 是正常的,所以是 preview 渲染的限制,不是文件权限。
- **vf-tui (`verifiers.scripts.tui`)** 是终端 Textual TUI,只识别 `outputs/evals/<env>--<model>/<run>/{metadata.json,results.jsonl}` 目录结构,且**图片只显示 `[image]` 占位符**;它适合多 run 概览,不适合单 trajectory 看图。`VF_OUTPUTS_DIR` 环境变量可指自定义路径(默认 `./outputs`)。
- **prime-rl orchestrator 写 `step_<N>/{train,eval}_rollouts.jsonl` 时 `exclude_keys={"trajectory"}`**,所以这两份 jsonl 不带完整 trajectory;trainer 不依赖它们(走 `train_rollouts.bin` filesystem transport)。
- **base64 inline markdown 在表格里会变成 ~3MB 单行**,Cursor preview 渲染会 stall;HTML 相对路径每条 ~10–30 KB,体积小 100×。
- **regenerate_artifacts.py 不能补 response.txt**,因为 raw assistant text 没存到 `meta.json/trajectory.json`(schema 不带 prompt/completion)。Pydantic bug 导致的旧 0-byte response 没法回填,只能下次 rollout 时正确生成。

## 参考代码

| 文件 | 关键位置 | 说明 |
|------|---------|------|
| [environments/blendergym/blendergym/trajectory_writer.py](environments/blendergym/blendergym/trajectory_writer.py) | `_render_html` / `_atomic_write_text` / `completion_to_text` | HTML 渲染 + Pydantic message 兼容 |
| [environments/blendergym/blendergym/rubric.py](environments/blendergym/blendergym/rubric.py) | `BlenderGymRubric.write_artifacts_handler` | `@vf.cleanup` 后调 writer,跑在 score_rollout 之后 |
| [environments/blendergym/blendergym/env.py](environments/blendergym/blendergym/env.py) | `add_model_response` | rollout cleanup 时 push TurnRecord 到 `state["rollout"]` |
| [environments/blendergym/blendergym/schema.py](environments/blendergym/blendergym/schema.py) | `Rollout` / `TurnRecord` / `Task` | runtime entity model;`SCHEMA_VERSION` |
| [environments/blendergym/scripts/regenerate_artifacts.py](environments/blendergym/scripts/regenerate_artifacts.py) | `--work-root` | 重生成 trajectory.html 的离线脚本 |
| [environments/blendergym/tests/test_trajectory_writer.py](environments/blendergym/tests/test_trajectory_writer.py) | 26 个测试 | HTML escape / Pydantic msg / atomic write 全覆盖 |
| [configs/multimodal/rl_blendergym.toml](configs/multimodal/rl_blendergym.toml) | `output_dir`, train/eval env args | 当前 RL 入口,GPU 0/1 trainer+infer,GPU 6/7 env worker |

## 最终方案

trajectory.md (+base64) → trajectory.html (+ 相对路径) 是行业惯例(SWE-agent / Inspect AI / Cua),不入侵 verifiers / prime-rl,blast radius 限制在 BlenderGym package 内。Pydantic bug 修复不动 schema,纯 helper 加固。

## 下一步任务

跑真正的 BlenderGym RL 训练(去掉 `--bench`),观察 reward / render_success / xml_parse_success 是否随 step 上升,验证多轮 visual feedback + CLIP reward 是否真的能驱动 Qwen3.5-0.8B 学出 placement 任务。

## 初步方案

详见本次会话讨论。要点:

1. **先跑 30 step 小训练** 验证 pipeline,看 reward 曲线趋势 + render_success_rate + zero_advantage 比例。
2. **打开 wandb 监控**:rl_blendergym.toml 加 `[monitor.wandb]`,记录 reward / metrics / sample_table。
3. **监控指标**:render_success_rate(目前 bench 里很多 render_failed)、xml_parse_success(模型能不能稳定输出 `<code>`)、reward 趋势。
4. **潜在风险**:
   - prompts.py 硬编码坐标范围与具体 placement 任务不一致(前置 session 已记录),可能压制 reward 上限。
   - rollouts_per_example=4 时 zero_advantage 比例较高(bench 里 50%),可能需要提到 8 或加 reward shaping。
   - 0.8B 模型容量小,复杂 placement 任务(N=45)未必能学会;可先在 train 子集上跑。
5. **阶段性 checkpoint**:training 中途 save 至少 ckpt_step_15 / ckpt_step_30,失败时能 resume。
