# Renderer Plugin

This example trains [reverse-text](../../basic/reverse-text/README.md) with a renderer that lives outside the [`renderers`](https://github.com/PrimeIntellect-ai/renderers) package. [`renderer.py`](renderer.py) subclasses the PrimeIntellect Qwen3 renderer and adds one template control, `instruction`, which it appends to every system prompt. It needs two GPUs: one trainer and one inference GPU.

A plugin is a Python file (or an importable module) with a renderer class that sets `config_class` to its own `BaseRendererConfig` subclass. The TOML selects it with `name = "plugin"` and a `target`; every other `[renderer]` key is validated by the plugin's config:

```toml
[renderer]
name = "plugin"
target = "examples/extra/renderer-plugin/renderer.py:InstructedQwen3Renderer"
instruction = "Reverse every character, including spaces and punctuation."
```

Relative `target` paths resolve against the directory you launch from, so run these commands from the repository root.

## SFT

```bash
uv run sft @ examples/extra/renderer-plugin/sft.toml
```

The trainer renders every sample through the plugin. Online evals go through the inference server's chat template, so they do not see `instruction`; see [`inference.vllm_plugins`](../../../docs/inference.md#local-vllm-plugins) to teach vLLM a matching tokenizer mode when a plugin changes the prompt format.

## RL

```bash
uv run rl @ examples/extra/renderer-plugin/rl.toml
```

The orchestrator forwards the plugin config to its env servers, which render every rollout through the plugin and send token ids to vLLM. The rendered prompts land in the run's traces (`outputs/run_default/monitors/file/traces`), where each system message ends with the `instruction` text.
