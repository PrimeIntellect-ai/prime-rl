# prime-rl-configs

Slim config schema for [`prime-rl`](https://github.com/PrimeIntellect-ai/prime-rl), with no GPU or ML deps.

`pip install prime-rl-configs` gives you `prime_rl.configs.*` (RL/SFT/inference/orchestrator/trainer/env-server schemas) without pulling in `torch`, `vllm`, `transformers`, `wandb`, etc. The full training stack lives in `prime-rl`, which depends on this package.

## Install

```sh
pip install git+https://github.com/PrimeIntellect-ai/prime-rl.git#subdirectory=packages/prime-rl-configs
```

## Usage

The pip *distribution name* (`prime-rl-configs`) and the *import path* (`prime_rl.configs.*`) are different on purpose: this package contributes submodules to the shared `prime_rl` namespace.

```python
from pydantic_config import cli
from prime_rl.configs.rl import RLConfig

config = cli(RLConfig, args=["@", "path/to/rl.toml"])
```

Other config classes live alongside `RLConfig` under `prime_rl.configs.*` (`sft`, `inference`, `orchestrator`, `trainer`, `env_server`).

`import prime_rl` on its own succeeds but is empty — it's a [PEP 420](https://peps.python.org/pep-0420/) namespace package with no top-level attributes. Always import a submodule.
