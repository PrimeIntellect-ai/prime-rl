# Custom Renderer

This example trains [reverse-text](../../basic/reverse-text/README.md) with a renderer that lives outside the [`renderers`](https://github.com/PrimeIntellect-ai/renderers) package. [`renderer.py`](renderer.py) subclasses the PrimeIntellect Qwen3 renderer and adds one template control, `instruction`, which it appends to every system prompt. It needs two GPUs: one trainer and one inference GPU.

A custom renderer is a class in a Python file (or an importable module) that sets `config_class` to its own `BaseRendererConfig` subclass. The TOML selects it with `name = "custom"` and an `import_path`; every other `[renderer]` key is validated by the renderer's config:

```toml
[renderer]
name = "custom"
import_path = "examples/extra/custom-renderer/renderer.py:InstructedQwen3Renderer"
instruction = "Reverse every character, including spaces and punctuation."
```

`import_path` is `my_module.Class` for an importable module or `path/to/file.py:Class` for a file. Relative file paths resolve against the directory you launch from, so run these commands from the repository root.

## SFT

```bash
uv run sft @ examples/extra/custom-renderer/sft.toml
```

The trainer renders every sample through the custom renderer. Online evals go through the inference server's chat template, so they do not see `instruction`; see [`inference.vllm_plugins`](../../../docs/inference.md#local-vllm-plugins) to teach vLLM a matching tokenizer mode when a custom renderer changes the prompt format.

## RL

```bash
uv run rl @ examples/extra/custom-renderer/rl.toml
```

The orchestrator forwards the renderer config to its env servers, which render every rollout through the custom renderer and send token ids to vLLM. The rendered prompts land in the run's traces (`outputs/run_default/monitors/file/traces`), where each system message ends with the `instruction` text.
