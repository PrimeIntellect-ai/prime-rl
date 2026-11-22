def test_importing_glyphbench_registers_dummy_env():
    import glyphbench  # noqa: F401
    from glyphbench.core.registry import all_glyphbench_env_ids
    assert "glyphbench/__dummy-v0" in all_glyphbench_env_ids()


def test_make_env_can_make_dummy_env_via_id():
    import glyphbench  # noqa: F401
    from glyphbench.core import make_env
    env = make_env("glyphbench/__dummy-v0")
    obs, info = env.reset(0)
    assert isinstance(obs, str)
    assert "@" in obs
