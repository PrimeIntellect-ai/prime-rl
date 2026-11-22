"""Procgen suite - importing registers all envs."""

from glyphbench.core.registry import register_env
from glyphbench.envs.procgen.bigfish import (
    BigFishEnv,
)
from glyphbench.envs.procgen.bossfight import (
    BossFightEnv,
)
from glyphbench.envs.procgen.caveflyer import (
    CaveFlyerEnv,
)
from glyphbench.envs.procgen.chaser import (
    ChaserEnv,
)
from glyphbench.envs.procgen.climber import (
    ClimberEnv,
)
from glyphbench.envs.procgen.coinrun import (
    CoinRunEnv,
)
from glyphbench.envs.procgen.dodgeball import (
    DodgeballEnv,
)
from glyphbench.envs.procgen.fruitbot import (
    FruitBotEnv,
)
from glyphbench.envs.procgen.heist import (
    HeistEnv,
)
from glyphbench.envs.procgen.jumper import (
    JumperEnv,
)
from glyphbench.envs.procgen.leaper import (
    LeaperEnv,
)
from glyphbench.envs.procgen.maze import (
    MazeEnv,
)
from glyphbench.envs.procgen.miner import (
    MinerEnv,
)
from glyphbench.envs.procgen.ninja import (
    NinjaEnv,
)
from glyphbench.envs.procgen.plunder import (
    PlunderEnv,
)
from glyphbench.envs.procgen.starpilot import (
    StarPilotEnv,
)

_REGISTRATIONS = {
    "glyphbench/procgen-coinrun-v0": CoinRunEnv,
    "glyphbench/procgen-maze-v0": MazeEnv,
    "glyphbench/procgen-heist-v0": HeistEnv,
    "glyphbench/procgen-leaper-v0": LeaperEnv,
    "glyphbench/procgen-chaser-v0": ChaserEnv,
    "glyphbench/procgen-bigfish-v0": BigFishEnv,
    "glyphbench/procgen-climber-v0": ClimberEnv,
    "glyphbench/procgen-jumper-v0": JumperEnv,
    "glyphbench/procgen-ninja-v0": NinjaEnv,
    "glyphbench/procgen-fruitbot-v0": FruitBotEnv,
    "glyphbench/procgen-miner-v0": MinerEnv,
    "glyphbench/procgen-dodgeball-v0": DodgeballEnv,
    "glyphbench/procgen-caveflyer-v0": CaveFlyerEnv,
    "glyphbench/procgen-starpilot-v0": StarPilotEnv,
    "glyphbench/procgen-plunder-v0": PlunderEnv,
    "glyphbench/procgen-bossfight-v0": BossFightEnv,
}

_MEMORY_DEFAULT_TRUE = {
    # Maze-like objectives with partially visible layout and off-screen
    # goals/keys/pellets where remembering explored structure matters.
    "glyphbench/procgen-chaser-v0",
    "glyphbench/procgen-heist-v0",
    "glyphbench/procgen-maze-v0",
}
_UNKNOWN_MEMORY_DEFAULT_TRUE = _MEMORY_DEFAULT_TRUE - set(_REGISTRATIONS)
if _UNKNOWN_MEMORY_DEFAULT_TRUE:
    raise RuntimeError(
        "unknown Procgen memory-default ids: "
        f"{sorted(_UNKNOWN_MEMORY_DEFAULT_TRUE)}"
    )

for _id, _cls in _REGISTRATIONS.items():
    register_env(_id, _cls, default_use_memory=_id in _MEMORY_DEFAULT_TRUE)
