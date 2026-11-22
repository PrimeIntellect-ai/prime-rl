"""nethack suite - full NetHack through the local NLE fork."""

from glyphbench.core.registry import register_env
from glyphbench.envs.nethack.full import NetHackFullEnv

_REGISTRATIONS = {
    "glyphbench/nethack-full-v0": NetHackFullEnv,
}

for _id, _cls in _REGISTRATIONS.items():
    register_env(_id, _cls, default_use_memory=True)
