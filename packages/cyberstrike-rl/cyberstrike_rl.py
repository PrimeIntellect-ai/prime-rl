"""Native single-agent CyberStrike RL with host-only Prime Inference grading."""

from prime_flow.qc.relay import GraderModels, GraderRelay, record
from verifiers.v1.env import Env
from verifiers.v1.envs.single_agent import SingleAgentEnv, SingleAgentEnvConfig

__all__ = ["CyberStrikeRL"]


class CyberStrikeRLConfig(SingleAgentEnvConfig):
    grader_models: GraderModels = GraderModels()


class CyberStrikeRL(SingleAgentEnv, Env[CyberStrikeRLConfig]):
    """Keep each grader grant alive through the native agent's scoring phase.

    Select the kernel-aware taskset independently in ``taskset``. Provider credentials
    resolve on this host through the relay's native Prime configuration; only a
    short-lived grant enters the task's verifier environment.
    """

    async def start(self) -> None:
        await super().start()
        self._grader = GraderRelay(self.config.grader_models)
        await self._grader.start()

    async def stop(self) -> None:
        try:
            if relay := getattr(self, "_grader", None):
                await relay.stop()
        finally:
            await super().stop()

    async def run(self, task, agents) -> None:
        relay = self._grader
        if not relay.routed(task):
            await super().run(task, agents)
            return
        async with relay.grant(None) as grant:
            trace = await agents.agent.run(relay.route(task, grant))
            record(trace, "grader", grant.calls)
