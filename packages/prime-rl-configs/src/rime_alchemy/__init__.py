"""Verifiers plugin id: rime-alchemy."""

import verifiers.v1 as vf

from rime_alchemy.configs import (
    RIMEAlchemyEnvConfig,
    RIMEAlchemyTasksetConfig,
    StreamTask,
)


class RIMEAlchemyEnv(vf.Env[RIMEAlchemyEnvConfig]):
    async def setup(self, agents):
        from rime_study.hosted import RIMEAlchemyEnv as RuntimeEnv

        self.runtime = RuntimeEnv(self.config)
        await self.runtime.setup(agents)

    async def run(self, task, agents):
        await self.runtime.run(task, agents)


class RIMEAlchemyTaskset(vf.Taskset[StreamTask, RIMEAlchemyTasksetConfig]):
    def load(self):
        from rime_study.hosted import RIMEAlchemyTaskset as RuntimeTaskset

        return RuntimeTaskset(self.config).load()


__all__ = ["RIMEAlchemyEnv", "RIMEAlchemyEnvConfig", "RIMEAlchemyTaskset", "RIMEAlchemyTasksetConfig"]
