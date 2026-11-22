"""Minigrid suite - importing registers all envs."""

from glyphbench.core.registry import register_env
from glyphbench.envs.minigrid.crossing import (
    MiniGridCrossingN1Env,
    MiniGridCrossingN1SafeEnv,
    MiniGridCrossingN2Env,
    MiniGridCrossingN2SafeEnv,
    MiniGridCrossingN3Env,
    MiniGridCrossingN3SafeEnv,
    MiniGridSimpleCrossingEasyN1Env,
    MiniGridSimpleCrossingEasyN2Env,
    MiniGridSimpleCrossingEasyN3Env,
    MiniGridSimpleCrossingN1Env,
    MiniGridSimpleCrossingN2Env,
    MiniGridSimpleCrossingN3Env,
)
from glyphbench.envs.minigrid.distshift import (
    MiniGridDistShift1Env,
    MiniGridDistShift2Env,
)
from glyphbench.envs.minigrid.doorkey import (
    MiniGridDoorKey5x5Env,
    MiniGridDoorKey6x6Env,
    MiniGridDoorKey8x8Env,
    MiniGridDoorKey16x16Env,
)
from glyphbench.envs.minigrid.dynamic_obstacles import (
    MiniGridDynamicObstacles5x5Env,
    MiniGridDynamicObstacles6x6Env,
    MiniGridDynamicObstacles8x8Env,
    MiniGridDynamicObstacles16x16Env,
)
from glyphbench.envs.minigrid.empty import (
    MiniGridEmpty5x5Env,
    MiniGridEmpty6x6Env,
    MiniGridEmpty8x8Env,
    MiniGridEmpty16x16Env,
    MiniGridEmptyRandom5x5Env,
    MiniGridEmptyRandom6x6Env,
)
from glyphbench.envs.minigrid.fetch import (
    MiniGridFetch5x5N2Env,
    MiniGridFetch6x6N2Env,
    MiniGridFetch8x8N3Env,
    MiniGridPutNear6x6N2Env,
    MiniGridPutNear8x8N3Env,
)
from glyphbench.envs.minigrid.fourrooms import (
    MiniGridFourRoomsEnv,
)
from glyphbench.envs.minigrid.goto import (
    MiniGridGoToDoor5x5Env,
    MiniGridGoToDoor6x6Env,
    MiniGridGoToDoor8x8Env,
    MiniGridGoToObject6x6N2Env,
)
from glyphbench.envs.minigrid.keycorridor import (
    MiniGridKeyCorridorS3R1Env,
    MiniGridKeyCorridorS3R2Env,
    MiniGridKeyCorridorS3R3Env,
    MiniGridKeyCorridorS4R3Env,
    MiniGridKeyCorridorS5R3Env,
    MiniGridKeyCorridorS6R3Env,
)
from glyphbench.envs.minigrid.lavagap import (
    MiniGridLavaGapS5Env,
    MiniGridLavaGapS6Env,
    MiniGridLavaGapS7Env,
)
from glyphbench.envs.minigrid.lockedroom import (
    MiniGridBlockedUnlockPickupEnv,
    MiniGridLockedRoomEnv,
)
from glyphbench.envs.minigrid.memory import (
    MiniGridMemoryS7Env,
    MiniGridMemoryS9Env,
    MiniGridMemoryS11Env,
    MiniGridMemoryS13Env,
    MiniGridMemoryS17Env,
)
from glyphbench.envs.minigrid.multiroom import (
    MiniGridMultiRoomN2S4Env,
    MiniGridMultiRoomN4S5Env,
    MiniGridMultiRoomN6Env,
)
from glyphbench.envs.minigrid.obstructedmaze import (
    MiniGridObstructedMaze1DlEnv,
    MiniGridObstructedMaze1DlhbEnv,
    MiniGridObstructedMaze1DlhEnv,
    MiniGridObstructedMaze1QEnv,
    MiniGridObstructedMaze2DlEnv,
    MiniGridObstructedMaze2DlhbEnv,
    MiniGridObstructedMaze2DlhEnv,
    MiniGridObstructedMaze2QEnv,
    MiniGridObstructedMazeFullEnv,
)
from glyphbench.envs.minigrid.playground import (
    MiniGridPlaygroundEnv,
)
from glyphbench.envs.minigrid.redbluedoors import (
    MiniGridRedBlueDoors6x6Env,
    MiniGridRedBlueDoors8x8Env,
)
from glyphbench.envs.minigrid.unlock import (
    MiniGridUnlockEnv,
    MiniGridUnlockPickupEnv,
)

_REGISTRATIONS = {
    "glyphbench/minigrid-empty-5x5-v0": MiniGridEmpty5x5Env,
    "glyphbench/minigrid-empty-6x6-v0": MiniGridEmpty6x6Env,
    "glyphbench/minigrid-empty-8x8-v0": MiniGridEmpty8x8Env,
    "glyphbench/minigrid-empty-16x16-v0": MiniGridEmpty16x16Env,
    "glyphbench/minigrid-empty-random-5x5-v0": MiniGridEmptyRandom5x5Env,
    "glyphbench/minigrid-empty-random-6x6-v0": MiniGridEmptyRandom6x6Env,
    "glyphbench/minigrid-doorkey-5x5-v0": MiniGridDoorKey5x5Env,
    "glyphbench/minigrid-doorkey-6x6-v0": MiniGridDoorKey6x6Env,
    "glyphbench/minigrid-doorkey-8x8-v0": MiniGridDoorKey8x8Env,
    "glyphbench/minigrid-doorkey-16x16-v0": MiniGridDoorKey16x16Env,
    "glyphbench/minigrid-fourrooms-v0": MiniGridFourRoomsEnv,
    "glyphbench/minigrid-multiroom-n2-s4-v0": MiniGridMultiRoomN2S4Env,
    "glyphbench/minigrid-multiroom-n4-s5-v0": MiniGridMultiRoomN4S5Env,
    "glyphbench/minigrid-multiroom-n6-v0": MiniGridMultiRoomN6Env,
    "glyphbench/minigrid-unlock-v0": MiniGridUnlockEnv,
    "glyphbench/minigrid-unlockpickup-v0": MiniGridUnlockPickupEnv,
    "glyphbench/minigrid-keycorridor-s3r1-v0": MiniGridKeyCorridorS3R1Env,
    "glyphbench/minigrid-keycorridor-s3r2-v0": MiniGridKeyCorridorS3R2Env,
    "glyphbench/minigrid-keycorridor-s3r3-v0": MiniGridKeyCorridorS3R3Env,
    "glyphbench/minigrid-keycorridor-s4r3-v0": MiniGridKeyCorridorS4R3Env,
    "glyphbench/minigrid-keycorridor-s5r3-v0": MiniGridKeyCorridorS5R3Env,
    "glyphbench/minigrid-keycorridor-s6r3-v0": MiniGridKeyCorridorS6R3Env,
    "glyphbench/minigrid-lockedroom-v0": MiniGridLockedRoomEnv,
    "glyphbench/minigrid-blockedunlockpickup-v0": MiniGridBlockedUnlockPickupEnv,
    "glyphbench/minigrid-distshift1-v0": MiniGridDistShift1Env,
    "glyphbench/minigrid-distshift2-v0": MiniGridDistShift2Env,
    "glyphbench/minigrid-lavagap-s5-v0": MiniGridLavaGapS5Env,
    "glyphbench/minigrid-lavagap-s6-v0": MiniGridLavaGapS6Env,
    "glyphbench/minigrid-lavagap-s7-v0": MiniGridLavaGapS7Env,
    "glyphbench/minigrid-crossing-n1-v0": MiniGridCrossingN1Env,
    "glyphbench/minigrid-crossing-n2-v0": MiniGridCrossingN2Env,
    "glyphbench/minigrid-crossing-n3-v0": MiniGridCrossingN3Env,
    "glyphbench/minigrid-crossing-n1-safe-v0": MiniGridCrossingN1SafeEnv,
    "glyphbench/minigrid-crossing-n2-safe-v0": MiniGridCrossingN2SafeEnv,
    "glyphbench/minigrid-crossing-n3-safe-v0": MiniGridCrossingN3SafeEnv,
    "glyphbench/minigrid-simplecrossing-n1-v0": MiniGridSimpleCrossingN1Env,
    "glyphbench/minigrid-simplecrossing-n2-v0": MiniGridSimpleCrossingN2Env,
    "glyphbench/minigrid-simplecrossing-n3-v0": MiniGridSimpleCrossingN3Env,
    "glyphbench/minigrid-simplecrossing-easy-n1-v0": MiniGridSimpleCrossingEasyN1Env,
    "glyphbench/minigrid-simplecrossing-easy-n2-v0": MiniGridSimpleCrossingEasyN2Env,
    "glyphbench/minigrid-simplecrossing-easy-n3-v0": MiniGridSimpleCrossingEasyN3Env,
    "glyphbench/minigrid-dynamic-obstacles-5x5-v0": MiniGridDynamicObstacles5x5Env,
    "glyphbench/minigrid-dynamic-obstacles-6x6-v0": MiniGridDynamicObstacles6x6Env,
    "glyphbench/minigrid-dynamic-obstacles-8x8-v0": MiniGridDynamicObstacles8x8Env,
    "glyphbench/minigrid-dynamic-obstacles-16x16-v0": MiniGridDynamicObstacles16x16Env,
    "glyphbench/minigrid-gotodoor-5x5-v0": MiniGridGoToDoor5x5Env,
    "glyphbench/minigrid-gotodoor-6x6-v0": MiniGridGoToDoor6x6Env,
    "glyphbench/minigrid-gotodoor-8x8-v0": MiniGridGoToDoor8x8Env,
    "glyphbench/minigrid-gotoobject-6x6-n2-v0": MiniGridGoToObject6x6N2Env,
    "glyphbench/minigrid-fetch-5x5-n2-v0": MiniGridFetch5x5N2Env,
    "glyphbench/minigrid-fetch-6x6-n2-v0": MiniGridFetch6x6N2Env,
    "glyphbench/minigrid-fetch-8x8-n3-v0": MiniGridFetch8x8N3Env,
    "glyphbench/minigrid-putnear-6x6-n2-v0": MiniGridPutNear6x6N2Env,
    "glyphbench/minigrid-putnear-8x8-n3-v0": MiniGridPutNear8x8N3Env,
    "glyphbench/minigrid-redbluedoors-6x6-v0": MiniGridRedBlueDoors6x6Env,
    "glyphbench/minigrid-redbluedoors-8x8-v0": MiniGridRedBlueDoors8x8Env,
    "glyphbench/minigrid-obstructedmaze-1dl-v0": MiniGridObstructedMaze1DlEnv,
    "glyphbench/minigrid-obstructedmaze-1dlh-v0": MiniGridObstructedMaze1DlhEnv,
    "glyphbench/minigrid-obstructedmaze-1dlhb-v0": MiniGridObstructedMaze1DlhbEnv,
    "glyphbench/minigrid-obstructedmaze-2dl-v0": MiniGridObstructedMaze2DlEnv,
    "glyphbench/minigrid-obstructedmaze-2dlh-v0": MiniGridObstructedMaze2DlhEnv,
    "glyphbench/minigrid-obstructedmaze-2dlhb-v0": MiniGridObstructedMaze2DlhbEnv,
    "glyphbench/minigrid-obstructedmaze-1q-v0": MiniGridObstructedMaze1QEnv,
    "glyphbench/minigrid-obstructedmaze-2q-v0": MiniGridObstructedMaze2QEnv,
    "glyphbench/minigrid-obstructedmaze-full-v0": MiniGridObstructedMazeFullEnv,
    "glyphbench/minigrid-memory-s7-v0": MiniGridMemoryS7Env,
    "glyphbench/minigrid-memory-s9-v0": MiniGridMemoryS9Env,
    "glyphbench/minigrid-memory-s11-v0": MiniGridMemoryS11Env,
    "glyphbench/minigrid-memory-s13-v0": MiniGridMemoryS13Env,
    "glyphbench/minigrid-memory-s17-v0": MiniGridMemoryS17Env,
    "glyphbench/minigrid-playground-v0": MiniGridPlaygroundEnv,
}

_MEMORY_DEFAULT_TRUE = {
    # Explicit memory/cue tasks: the decisive cue disappears or must be
    # retained after it is no longer present in the full-grid observation.
    "glyphbench/minigrid-memory-s7-v0",
    "glyphbench/minigrid-memory-s9-v0",
    "glyphbench/minigrid-memory-s11-v0",
    "glyphbench/minigrid-memory-s13-v0",
    "glyphbench/minigrid-memory-s17-v0",
    "glyphbench/minigrid-playground-v0",
}

for _id, _cls in _REGISTRATIONS.items():
    register_env(_id, _cls, default_use_memory=_id in _MEMORY_DEFAULT_TRUE)
