"""Classics suite - importing registers all envs."""

from glyphbench.core.registry import register_env
from glyphbench.envs.classics.artillery import (
    ArtilleryEnv,
)
from glyphbench.envs.classics.battleship import (
    BattleshipEnv,
)
from glyphbench.envs.classics.bomberman import (
    BombermanEnv,
)
from glyphbench.envs.classics.connect_four import (
    ConnectFourEnv,
)
from glyphbench.envs.classics.farm_sim import (
    FarmSimEnv,
)
from glyphbench.envs.classics.flappy import (
    FlappyEnv,
)
from glyphbench.envs.classics.flood_fill import (
    FloodFillEasyEnv,
    FloodFillHardEnv,
)
from glyphbench.envs.classics.frogger import (
    FroggerEnv,
)
from glyphbench.envs.classics.gravity_maze import (
    GravityMazeEnv,
)
from glyphbench.envs.classics.guard_evasion import (
    GuardEvasionEasyEnv,
    GuardEvasionHardEnv,
    GuardEvasionMediumEnv,
)
from glyphbench.envs.classics.ice_sliding import (
    IceSlidingEasyEnv,
    IceSlidingHardEnv,
    IceSlidingMediumEnv,
)
from glyphbench.envs.classics.lights_out import (
    LightsOutEasyEnv,
    LightsOutHardEnv,
)
from glyphbench.envs.classics.lunar_lander import (
    LunarLanderEnv,
)
from glyphbench.envs.classics.match3 import (
    Match3Env,
)
from glyphbench.envs.classics.maze_runner import (
    MazeEasyEnv,
    MazeHardEnv,
    MazeMediumEnv,
)
from glyphbench.envs.classics.memory_match import (
    MemoryMatchEasyEnv,
    MemoryMatchHardEnv,
)
from glyphbench.envs.classics.minesweeper import (
    MinesweeperEnv,
)
from glyphbench.envs.classics.mirror_laser import (
    MirrorLaserEnv,
)
from glyphbench.envs.classics.nim import (
    NimEasyEnv,
    NimHardEnv,
)
from glyphbench.envs.classics.nonogram import (
    NonogramEasyEnv,
    NonogramHardEnv,
)
from glyphbench.envs.classics.pipe_connect import (
    PipeConnectEasyEnv,
    PipeConnectHardEnv,
    PipeConnectMediumEnv,
)
from glyphbench.envs.classics.platformer import (
    PlatformerEnv,
)
from glyphbench.envs.classics.rush_hour import (
    RushHourEasyEnv,
    RushHourHardEnv,
)
from glyphbench.envs.classics.ski import (
    SkiEnv,
)
from glyphbench.envs.classics.slide2048 import (
    Slide2048Env,
)
from glyphbench.envs.classics.snake import (
    SnakeEasyEnv,
    SnakeHardEnv,
    SnakeMediumEnv,
)
from glyphbench.envs.classics.sokoban import (
    SokobanEasyEnv,
    SokobanHardEnv,
    SokobanMediumEnv,
)
from glyphbench.envs.classics.tetris import (
    TetrisEnv,
)
from glyphbench.envs.classics.tower_defense import (
    TowerDefenseEnv,
)
from glyphbench.envs.classics.tron import (
    TronEnv,
)
from glyphbench.envs.classics.warehouse import (
    WarehouseEnv,
)
from glyphbench.envs.classics.wave_defense import (
    WaveDefenseEnv,
)

_REGISTRATIONS = {
    "glyphbench/classics-artillery-v0": ArtilleryEnv,
    "glyphbench/classics-battleship-v0": BattleshipEnv,
    "glyphbench/classics-bomberman-v0": BombermanEnv,
    "glyphbench/classics-connectfour-v0": ConnectFourEnv,
    "glyphbench/classics-farm-v0": FarmSimEnv,
    "glyphbench/classics-flappy-v0": FlappyEnv,
    "glyphbench/classics-floodfill-easy-v0": FloodFillEasyEnv,
    "glyphbench/classics-floodfill-hard-v0": FloodFillHardEnv,
    "glyphbench/classics-frogger-v0": FroggerEnv,
    "glyphbench/classics-gravitymaze-v0": GravityMazeEnv,
    "glyphbench/classics-guardevasion-easy-v0": GuardEvasionEasyEnv,
    "glyphbench/classics-guardevasion-medium-v0": GuardEvasionMediumEnv,
    "glyphbench/classics-guardevasion-hard-v0": GuardEvasionHardEnv,
    "glyphbench/classics-icesliding-easy-v0": IceSlidingEasyEnv,
    "glyphbench/classics-icesliding-medium-v0": IceSlidingMediumEnv,
    "glyphbench/classics-icesliding-hard-v0": IceSlidingHardEnv,
    "glyphbench/classics-lightsout-easy-v0": LightsOutEasyEnv,
    "glyphbench/classics-lightsout-hard-v0": LightsOutHardEnv,
    "glyphbench/classics-lunarlander-v0": LunarLanderEnv,
    "glyphbench/classics-match3-v0": Match3Env,
    "glyphbench/classics-maze-easy-v0": MazeEasyEnv,
    "glyphbench/classics-maze-medium-v0": MazeMediumEnv,
    "glyphbench/classics-maze-hard-v0": MazeHardEnv,
    "glyphbench/classics-memorymatch-easy-v0": MemoryMatchEasyEnv,
    "glyphbench/classics-memorymatch-hard-v0": MemoryMatchHardEnv,
    "glyphbench/classics-minesweeper-v0": MinesweeperEnv,
    "glyphbench/classics-mirrorlaser-v0": MirrorLaserEnv,
    "glyphbench/classics-nim-easy-v0": NimEasyEnv,
    "glyphbench/classics-nim-hard-v0": NimHardEnv,
    "glyphbench/classics-nonogram-easy-v0": NonogramEasyEnv,
    "glyphbench/classics-nonogram-hard-v0": NonogramHardEnv,
    "glyphbench/classics-pipeconnect-easy-v0": PipeConnectEasyEnv,
    "glyphbench/classics-pipeconnect-medium-v0": PipeConnectMediumEnv,
    "glyphbench/classics-pipeconnect-hard-v0": PipeConnectHardEnv,
    "glyphbench/classics-platformer-v0": PlatformerEnv,
    "glyphbench/classics-rushhour-easy-v0": RushHourEasyEnv,
    "glyphbench/classics-rushhour-hard-v0": RushHourHardEnv,
    "glyphbench/classics-ski-v0": SkiEnv,
    "glyphbench/classics-2048-v0": Slide2048Env,
    "glyphbench/classics-snake-easy-v0": SnakeEasyEnv,
    "glyphbench/classics-snake-medium-v0": SnakeMediumEnv,
    "glyphbench/classics-snake-hard-v0": SnakeHardEnv,
    "glyphbench/classics-sokoban-easy-v0": SokobanEasyEnv,
    "glyphbench/classics-sokoban-medium-v0": SokobanMediumEnv,
    "glyphbench/classics-sokoban-hard-v0": SokobanHardEnv,
    "glyphbench/classics-tetris-v0": TetrisEnv,
    "glyphbench/classics-towerdefense-v0": TowerDefenseEnv,
    "glyphbench/classics-tron-v0": TronEnv,
    "glyphbench/classics-warehouse-v0": WarehouseEnv,
    "glyphbench/classics-wavedefense-v0": WaveDefenseEnv,
}

_MEMORY_DEFAULT_TRUE = {
    # Face-down cards hide information after observation; efficient play
    # depends on remembering previously revealed card identities.
    "glyphbench/classics-memorymatch-easy-v0",
    "glyphbench/classics-memorymatch-hard-v0",
}

for _id, _cls in _REGISTRATIONS.items():
    register_env(_id, _cls, default_use_memory=_id in _MEMORY_DEFAULT_TRUE)
