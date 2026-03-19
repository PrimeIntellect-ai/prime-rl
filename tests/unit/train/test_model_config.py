from prime_rl.configs.trainer import ModelConfig


def test_moe_optim_defaults_are_valid():
    config = ModelConfig(name="PrimeIntellect/GLM-0.5B")
    assert config.moe_optim.routing == "torch"
    assert config.moe_optim.scatter == "torch"
    assert config.moe_optim.gather == "torch"
    assert config.moe_optim.routed_ffn == "torch"
