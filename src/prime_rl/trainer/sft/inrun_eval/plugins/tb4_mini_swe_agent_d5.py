"""Harness id ``tb4-mini-swe-agent-d5``: verifiers 24107a4ea's ``mini_swe_agent`` harness with spec decision D5
applied to the in-sandbox bootstrap of mini-swe-agent's uv-script venv. Not imported by prime-rl: the in-run eval
copies this file next to the harness virtualenv (verifiers imports a harness plugin by its id) and the derived flow
configs name it in ``[flow.solver.harness] id``.

What changes, only in the preparation step (``Runtime.prepare_uv_script``, before the rollout's agent starts):

- ``UV_EXCLUDE_NEWER=<prepare_exclude_newer>`` is added to the environment of the ``uv sync --script`` command, so
  the script's unlocked dependencies (mini-swe-agent 2.4.6 and its ~126 packages) resolve as at collection time.
- With ``prepare_uv_version``: a box whose image has no script-capable uv gets that uv version, installed the way
  verifiers' ``_ENSURE_UV`` would install the latest one (``pip install -q -U --user uv``, else the astral installer
  script into ``$HOME/.local/bin``), before the bootstrap runs. The bootstrap then finds it and uses it. An image's own
  script-capable uv is used as is, as in collection. A box where neither works falls through to verifiers' unchanged
  bootstrap (the host's uv 0.13.1, as in collection).

What does not change: ``program.py`` and so the script's sha256 (``2cc22997...``) and the venv path the agent sees
(``environments-v2/2cc22997...-fd57668a0b15dabf``: uv names a script environment by the script's file stem and a
digest of its path, neither of which depends on the environment); the program's argv and its environment
(``resolved_env`` + ``MSWEA_*``): ``UV_EXCLUDE_NEWER`` never reaches the agent's commands; the harness config
(``command_timeout``, version 2.4.6) and every other verifiers file. Only the harness id in host-side error messages
reads ``tb4-mini-swe-agent-d5``.
"""

from __future__ import annotations

import logging
import shlex

from verifiers.v1.harness import Harness
from verifiers.v1.harnesses.mini_swe_agent import MiniSWEAgentHarness, MiniSWEAgentHarnessConfig
from verifiers.v1.runtimes.base import _INSTALL_CURL, FALLBACK_BIN

logger = logging.getLogger(__name__)

PIN_FAILED = 3


class D5MiniSWEAgentHarnessConfig(MiniSWEAgentHarnessConfig):
    prepare_exclude_newer: str | None = None
    """``UV_EXCLUDE_NEWER`` (RFC 3339) for the ``uv sync --script`` of mini-swe-agent's venv; None: unset."""
    prepare_uv_version: str | None = None
    """uv to install when the image has none that runs scripts; None: verifiers' unpinned install."""

    @property
    def prepare_env(self) -> dict[str, str]:
        return {"UV_EXCLUDE_NEWER": self.prepare_exclude_newer} if self.prepare_exclude_newer else {}


def pin_uv_script(version: str) -> str:
    """Install uv ``version`` the way verifiers' ``_ENSURE_UV`` installs the latest, unless a script-capable uv is
    already on its PATH. Exits ``PIN_FAILED`` when it could not."""
    v = shlex.quote(version)
    installer = shlex.quote(f"https://astral.sh/uv/{version}/install.sh")
    return (
        f'export PATH="$HOME/.local/bin:{FALLBACK_BIN}:$PATH" UV_INSTALL_DIR="$HOME/.local/bin"; '
        "uv_supports_scripts() { "
        "uv sync --help 2>/dev/null | grep -q -- '--script' "
        "&& uv python find --help 2>/dev/null | grep -q -- '--script'; }; "
        "uv_supports_scripts && exit 0; "
        f"{{ pip install -q -U --user uv=={v} 2>/dev/null && hash -r && uv_supports_scripts; }} && exit 0; "
        f"{{ {{ command -v curl >/dev/null 2>&1 && curl -LsSf {installer} | sh; }} "
        f"|| {{ command -v wget >/dev/null 2>&1 && wget -qO- {installer} | sh; }}; }} >/dev/null 2>&1; "
        f"hash -r; uv_supports_scripts && exit 0; exit {PIN_FAILED}"
    )


class _PrepareWith:
    """The runtime, except that its uv-script preparation gets ``extra`` added to the environment it is given (the
    preparation command's only); every other call goes to the runtime unchanged."""

    def __init__(self, runtime, extra: dict[str, str]) -> None:
        self._runtime = runtime
        self._extra = dict(extra)

    def __getattr__(self, name):
        return getattr(self._runtime, name)

    async def prepare_uv_script(self, script, env=None, **kwargs):
        return await self._runtime.prepare_uv_script(script, {**(env or {}), **self._extra}, **kwargs)


class D5MiniSWEAgentHarness(MiniSWEAgentHarness, Harness[D5MiniSWEAgentHarnessConfig]):
    async def setup(self, runtime) -> None:
        if self.config.prepare_uv_version:
            await self._pin_uv(runtime)
        await super().setup(_PrepareWith(runtime, self.config.prepare_env))

    async def launch(self, ctx, trace, runtime, endpoint, secret, mcp_urls, data):
        # launch() prepares the script again: a no-op on the box setup prepared, a full preparation (with the same
        # D5 environment) if the box lost it. The program itself runs with launch()'s own environment.
        return await super().launch(
            ctx, trace, _PrepareWith(runtime, self.config.prepare_env), endpoint, secret, mcp_urls, data
        )

    async def _pin_uv(self, runtime) -> None:
        cache = runtime.backend.cache
        if runtime.user is not None and not cache.curl_ready:
            # As prepare_uv_script does: only root can install a downloader for the user's uv bootstrap.
            curl = await runtime.with_user(None).run(["sh", "-c", _INSTALL_CURL], {})
            cache.curl_ready = curl.exit_code == 0
        result = await runtime.run(
            ["sh", "-c", pin_uv_script(self.config.prepare_uv_version)], self.config.resolved_env
        )
        if result.exit_code != 0:
            logger.warning(
                "tb4-mini-swe-agent-d5: could not install uv %s in the box (exit %s); verifiers' bootstrap installs "
                "its own",
                self.config.prepare_uv_version,
                result.exit_code,
            )


__all__ = ["D5MiniSWEAgentHarness", "D5MiniSWEAgentHarnessConfig"]
