"""Minihack suite - importing registers all envs."""

from glyphbench.core.registry import register_env
from glyphbench.envs.minihack.boxoban import (
    MiniHackBoxobanHardEnv,
    MiniHackBoxobanMediumEnv,
    MiniHackBoxobanUnfilteredEnv,
)
from glyphbench.envs.minihack.corridor import (
    MiniHackCorridorR2Env,
    MiniHackCorridorR3Env,
    MiniHackCorridorR5Env,
)
from glyphbench.envs.minihack.corridorbattle import (
    MiniHackCorridorBattleDarkEnv,
    MiniHackCorridorBattleEnv,
)
from glyphbench.envs.minihack.hidenseek import (
    MiniHackHideNSeekBigEnv,
    MiniHackHideNSeekEnv,
    MiniHackHideNSeekLavaEnv,
    MiniHackHideNSeekMappedEnv,
)
from glyphbench.envs.minihack.keyroom import (
    MiniHackKeyRoomDarkS5Env,
    MiniHackKeyRoomDarkS15Env,
    MiniHackKeyRoomS5Env,
    MiniHackKeyRoomS15Env,
)
from glyphbench.envs.minihack.mazewalk import (
    MiniHackMazeWalk9x9Env,
    MiniHackMazeWalk15x15Env,
    MiniHackMazeWalk45x19Env,
    MiniHackMazeWalkMapped9x9Env,
    MiniHackMazeWalkMapped15x15Env,
    MiniHackMazeWalkMapped45x19Env,
)
from glyphbench.envs.minihack.memento import (
    MiniHackMementoF2Env,
    MiniHackMementoF4Env,
    MiniHackMementoHardEnv,
    MiniHackMementoShortEnv,
)
from glyphbench.envs.minihack.river import (
    MiniHackRiverEnv,
    MiniHackRiverLavaEnv,
    MiniHackRiverMonsterEnv,
    MiniHackRiverMonsterLavaEnv,
    MiniHackRiverNarrowEnv,
)
from glyphbench.envs.minihack.room import (
    MiniHackRoom5x5Env,
    MiniHackRoom15x15Env,
    MiniHackRoomDark5x5Env,
    MiniHackRoomDark15x15Env,
    MiniHackRoomMonster5x5Env,
    MiniHackRoomMonster15x15Env,
    MiniHackRoomRandom5x5Env,
    MiniHackRoomRandom15x15Env,
    MiniHackRoomTrap5x5Env,
    MiniHackRoomTrap15x15Env,
    MiniHackRoomUltimate5x5Env,
    MiniHackRoomUltimate15x15Env,
)
from glyphbench.envs.minihack.skill_eat import (
    MiniHackEatDistractEnv,
    MiniHackEatEnv,
)
from glyphbench.envs.minihack.skill_lavacross import (
    MiniHackLavaCrossFullEnv,
    MiniHackLavaCrossLevitateEnv,
    MiniHackLavaCrossPotionInvEnv,
    MiniHackLavaCrossRingInvEnv,
)
from glyphbench.envs.minihack.skill_pray import (
    MiniHackPrayDistractEnv,
    MiniHackPrayEnv,
)
from glyphbench.envs.minihack.skill_quaff import (
    MiniHackQuaffDistractEnv,
    MiniHackQuaffEnv,
)
from glyphbench.envs.minihack.skill_read import (
    MiniHackReadDistractEnv,
    MiniHackReadEnv,
)
from glyphbench.envs.minihack.skill_sink import (
    MiniHackSinkDistractEnv,
    MiniHackSinkEnv,
)
from glyphbench.envs.minihack.skill_wield import (
    MiniHackWieldDistractEnv,
    MiniHackWieldEnv,
)
from glyphbench.envs.minihack.skill_wod import (
    MiniHackWoDEasyEnv,
    MiniHackWoDHardEnv,
    MiniHackWoDMediumEnv,
    MiniHackWoDProEnv,
)

_REGISTRATIONS = {
    "glyphbench/minihack-room-5x5-v0": MiniHackRoom5x5Env,
    "glyphbench/minihack-room-15x15-v0": MiniHackRoom15x15Env,
    "glyphbench/minihack-room-random-5x5-v0": MiniHackRoomRandom5x5Env,
    "glyphbench/minihack-room-random-15x15-v0": MiniHackRoomRandom15x15Env,
    "glyphbench/minihack-room-dark-5x5-v0": MiniHackRoomDark5x5Env,
    "glyphbench/minihack-room-dark-15x15-v0": MiniHackRoomDark15x15Env,
    "glyphbench/minihack-room-monster-5x5-v0": MiniHackRoomMonster5x5Env,
    "glyphbench/minihack-room-monster-15x15-v0": MiniHackRoomMonster15x15Env,
    "glyphbench/minihack-room-trap-5x5-v0": MiniHackRoomTrap5x5Env,
    "glyphbench/minihack-room-trap-15x15-v0": MiniHackRoomTrap15x15Env,
    "glyphbench/minihack-room-ultimate-5x5-v0": MiniHackRoomUltimate5x5Env,
    "glyphbench/minihack-room-ultimate-15x15-v0": MiniHackRoomUltimate15x15Env,
    "glyphbench/minihack-corridor-r2-v0": MiniHackCorridorR2Env,
    "glyphbench/minihack-corridor-r3-v0": MiniHackCorridorR3Env,
    "glyphbench/minihack-corridor-r5-v0": MiniHackCorridorR5Env,
    "glyphbench/minihack-keyroom-s5-v0": MiniHackKeyRoomS5Env,
    "glyphbench/minihack-keyroom-s15-v0": MiniHackKeyRoomS15Env,
    "glyphbench/minihack-keyroom-dark-s5-v0": MiniHackKeyRoomDarkS5Env,
    "glyphbench/minihack-keyroom-dark-s15-v0": MiniHackKeyRoomDarkS15Env,
    "glyphbench/minihack-mazewalk-9x9-v0": MiniHackMazeWalk9x9Env,
    "glyphbench/minihack-mazewalk-15x15-v0": MiniHackMazeWalk15x15Env,
    "glyphbench/minihack-mazewalk-45x19-v0": MiniHackMazeWalk45x19Env,
    "glyphbench/minihack-mazewalk-mapped-9x9-v0": MiniHackMazeWalkMapped9x9Env,
    "glyphbench/minihack-mazewalk-mapped-15x15-v0": MiniHackMazeWalkMapped15x15Env,
    "glyphbench/minihack-mazewalk-mapped-45x19-v0": MiniHackMazeWalkMapped45x19Env,
    "glyphbench/minihack-river-v0": MiniHackRiverEnv,
    "glyphbench/minihack-river-narrow-v0": MiniHackRiverNarrowEnv,
    "glyphbench/minihack-river-monster-v0": MiniHackRiverMonsterEnv,
    "glyphbench/minihack-river-lava-v0": MiniHackRiverLavaEnv,
    "glyphbench/minihack-river-monsterlava-v0": MiniHackRiverMonsterLavaEnv,
    "glyphbench/minihack-hidenseek-v0": MiniHackHideNSeekEnv,
    "glyphbench/minihack-hidenseek-mapped-v0": MiniHackHideNSeekMappedEnv,
    "glyphbench/minihack-hidenseek-lava-v0": MiniHackHideNSeekLavaEnv,
    "glyphbench/minihack-hidenseek-big-v0": MiniHackHideNSeekBigEnv,
    "glyphbench/minihack-corridorbattle-v0": MiniHackCorridorBattleEnv,
    "glyphbench/minihack-corridorbattle-dark-v0": MiniHackCorridorBattleDarkEnv,
    "glyphbench/minihack-eat-v0": MiniHackEatEnv,
    "glyphbench/minihack-eat-distract-v0": MiniHackEatDistractEnv,
    "glyphbench/minihack-pray-v0": MiniHackPrayEnv,
    "glyphbench/minihack-pray-distract-v0": MiniHackPrayDistractEnv,
    "glyphbench/minihack-sink-v0": MiniHackSinkEnv,
    "glyphbench/minihack-sink-distract-v0": MiniHackSinkDistractEnv,
    "glyphbench/minihack-read-v0": MiniHackReadEnv,
    "glyphbench/minihack-read-distract-v0": MiniHackReadDistractEnv,
    "glyphbench/minihack-quaff-v0": MiniHackQuaffEnv,
    "glyphbench/minihack-quaff-distract-v0": MiniHackQuaffDistractEnv,
    "glyphbench/minihack-wield-v0": MiniHackWieldEnv,
    "glyphbench/minihack-wield-distract-v0": MiniHackWieldDistractEnv,
    "glyphbench/minihack-wod-easy-v0": MiniHackWoDEasyEnv,
    "glyphbench/minihack-wod-medium-v0": MiniHackWoDMediumEnv,
    "glyphbench/minihack-wod-hard-v0": MiniHackWoDHardEnv,
    "glyphbench/minihack-wod-pro-v0": MiniHackWoDProEnv,
    "glyphbench/minihack-lavacross-full-v0": MiniHackLavaCrossFullEnv,
    "glyphbench/minihack-lavacross-levitate-v0": MiniHackLavaCrossLevitateEnv,
    "glyphbench/minihack-lavacross-levitate-potion-inv-v0": MiniHackLavaCrossPotionInvEnv,
    "glyphbench/minihack-lavacross-levitate-ring-inv-v0": MiniHackLavaCrossRingInvEnv,
    "glyphbench/minihack-boxoban-medium-v0": MiniHackBoxobanMediumEnv,
    "glyphbench/minihack-boxoban-hard-v0": MiniHackBoxobanHardEnv,
    "glyphbench/minihack-boxoban-unfiltered-v0": MiniHackBoxobanUnfilteredEnv,
    "glyphbench/minihack-memento-short-v0": MiniHackMementoShortEnv,
    "glyphbench/minihack-memento-hard-v0": MiniHackMementoHardEnv,
    "glyphbench/minihack-memento-f2-v0": MiniHackMementoF2Env,
    "glyphbench/minihack-memento-f4-v0": MiniHackMementoF4Env,
}

_MEMORY_DEFAULT_TRUE = {
    # Dark/local-vision tasks where the map must be built over time.
    "glyphbench/minihack-room-dark-5x5-v0",
    "glyphbench/minihack-room-dark-15x15-v0",
    "glyphbench/minihack-room-ultimate-5x5-v0",
    "glyphbench/minihack-room-ultimate-15x15-v0",
    "glyphbench/minihack-keyroom-dark-s5-v0",
    "glyphbench/minihack-keyroom-dark-s15-v0",
    "glyphbench/minihack-mazewalk-9x9-v0",
    "glyphbench/minihack-mazewalk-15x15-v0",
    "glyphbench/minihack-mazewalk-45x19-v0",
    "glyphbench/minihack-corridorbattle-dark-v0",
    # Explicit hidden/erased-cue memory mechanics.
    "glyphbench/minihack-eat-distract-v0",
    "glyphbench/minihack-pray-distract-v0",
    "glyphbench/minihack-read-distract-v0",
    "glyphbench/minihack-wod-hard-v0",
    "glyphbench/minihack-wod-pro-v0",
    "glyphbench/minihack-memento-short-v0",
    "glyphbench/minihack-memento-hard-v0",
    "glyphbench/minihack-memento-f2-v0",
    "glyphbench/minihack-memento-f4-v0",
}

for _id, _cls in _REGISTRATIONS.items():
    register_env(_id, _cls, default_use_memory=_id in _MEMORY_DEFAULT_TRUE)
