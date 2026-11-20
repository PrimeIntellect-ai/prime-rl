from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from prime_rl.configs.inference import VllmRouterConfig

TEMPLATES = Path(__file__).parents[2] / "src" / "prime_rl" / "templates"


def _render(router: VllmRouterConfig) -> str:
    env = Environment(loader=FileSystemLoader(TEMPLATES), keep_trailing_newline=True)
    return env.get_template("_launch_router.sh.j2").render(router=router, pd_bidirectional=None)


def test_pd_router_retries_turns_on_connection_errors():
    script = _render(VllmRouterConfig())
    pd_line = next(line for line in script.splitlines() if "--vllm-pd-disaggregation" in line)
    assert "--pd-turn-retries 2" in pd_line
    assert "--pd-turn-retries 0" in _render(VllmRouterConfig(pd_turn_retries=0))
    # The regular (non-P/D) router does not take the flag.
    regular_line = next(line for line in script.splitlines() if "--worker-urls" in line)
    assert "--pd-turn-retries" not in regular_line
