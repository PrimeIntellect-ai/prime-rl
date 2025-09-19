# Configs

**Sources**

We support the following sources for configuration, in this order of precedence:

1. **Command-line arguments**: You can pass (nested) arguments as `--key.subkey value` to the script. For example, to set the model name you can run `--model.name`

2. **Config files**: You can pass `.toml` config files (defined in the `configs` directory) using the `@` prefix. For example, to use the `debug.toml` config file, you can run `uv run inference @ configs/debug/infer.toml`. (*If you leave a space between the `@` and the config file, you will get shell path auto-completions.*)

3. **Environment variables**: You can set environment variables to override the config values. All environment variables must be prefixed with `PRIME_` and use the `__` delimiter to nest the keys. For example, to set the model name you can run `export PRIME_MODEL__NAME=Qwen/Qwen3-0.6B`.

4. **Defaults**: For almost all config arguments, we have a default value which will be used if no other source is provided.

In general we recommend setting configurations via config files to define reproducible experiments and use command-line arguments to override the config values to run variants of the same experiment. Environment variables are usually only used in production settings to communicate with the [Prime Protocol](https://github.com/PrimeIntellect-ai/protocol) worker. In most cases, you should not need to use environment variables.

The precedence order will be important if multiple sources try to configure the same argument. For example, in the following command, all sources will define a model name

```toml
# qwen8b.toml
[model]
name = "Qwen/Qwen3-8B"
```

```toml
# qwen14b.toml
[model]
name = "Qwen/Qwen-14B"
```

```bash
PRIME_MODEL__NAME=Qwen/Qwen3-4B uv run inference @qwen8b.toml @qwen14b.toml --model.name Qwen/Qwen3-32B
```

In this example, the CLI argument `--model.name Qwen/Qwen3-32B` will take precendence and the script will use `Qwen/Qwen3-32B` as the model name. If the CLI argument wasn't set, then the second config file would take precedence and the script would use `Qwen/Qwen-14B` as the model name. If the second config file wasn't set, then the first config file would take precedence and the script would use `Qwen/Qwen3-8B` as the model name. Finally, if the first config file wasn't set, then the environment variable would take precedence and the script would use `Qwen/Qwen-4B` as the model name. If the environment variable wasn't set, then the default value would be used and the script would use `Qwen/Qwen3-0.6B` as the model name.