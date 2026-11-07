# prime-rl-configs

Config schema for [`prime-rl`](https://github.com/PrimeIntellect-ai/prime-rl).

On this CyberStrike run branch, `prime-rl-configs` also installs the pinned CyberStrike environment, Flow relay, and kernel-aware Harbor taskset before schema validation. Their CPU-side dependencies include Harbor, Transformers, and tokenizers; `torch` and `vllm` are not required. Every schema consumer of this source revision receives these plugins. The full GPU training stack remains in `prime-rl`.

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
