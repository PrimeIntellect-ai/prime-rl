# glyphbench.envs.dummy

A 3×3 walk-to-goal env used only as a test fixture. Not part of any benchmark suite. Subsequent plans import this to exercise harness, runner, and provider infrastructure end-to-end before real pilot envs are implemented.

## Env id

`glyphbench/__dummy-v0`

## Mechanics

- 3×3 grid. Agent starts at (0, 0). Goal at (2, 2).
- Actions: NORTH, SOUTH, EAST, WEST, NOOP.
- Reward: +1 on reaching goal, 0 otherwise.
- Terminates on reaching goal. Truncates at max_turns (default 20).
