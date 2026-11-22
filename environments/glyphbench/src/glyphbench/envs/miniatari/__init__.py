"""miniatari suite — short-horizon, [-1, 1]-bounded redesigns of atari games.

Each env uses a smaller play field, a tight terminal win condition, a bounded
horizon, and a structural [-1, 1] reward shape.
"""
from __future__ import annotations

from glyphbench.core.registry import register_env
from glyphbench.envs.miniatari.alien import MiniAlienEnv
from glyphbench.envs.miniatari.amidar import MiniAmidarEnv
from glyphbench.envs.miniatari.assault import MiniAssaultEnv
from glyphbench.envs.miniatari.asterix import MiniAsterixEnv
from glyphbench.envs.miniatari.asteroids import MiniAsteroidsEnv
from glyphbench.envs.miniatari.atlantis import MiniAtlantisEnv
from glyphbench.envs.miniatari.bankheist import MiniBankHeistEnv
from glyphbench.envs.miniatari.battlezone import MiniBattleZoneEnv
from glyphbench.envs.miniatari.beamrider import MiniBeamRiderEnv
from glyphbench.envs.miniatari.berzerk import MiniBerzerkEnv
from glyphbench.envs.miniatari.bowling import MiniBowlingEnv
from glyphbench.envs.miniatari.boxing import MiniBoxingEnv
from glyphbench.envs.miniatari.breakout import MiniBreakoutEnv
from glyphbench.envs.miniatari.centipede import MiniCentipedeEnv
from glyphbench.envs.miniatari.choppercommand import MiniChopperCommandEnv
from glyphbench.envs.miniatari.defender import MiniDefenderEnv
from glyphbench.envs.miniatari.demonattack import MiniDemonAttackEnv
from glyphbench.envs.miniatari.doubledunk import MiniDoubleDunkEnv
from glyphbench.envs.miniatari.enduro import MiniEnduroEnv
from glyphbench.envs.miniatari.fishingderby import MiniFishingDerbyEnv
from glyphbench.envs.miniatari.freeway import MiniFreewayEnv
from glyphbench.envs.miniatari.frostbite import MiniFrostbiteEnv
from glyphbench.envs.miniatari.gopher import MiniGopherEnv
from glyphbench.envs.miniatari.icehockey import MiniIceHockeyEnv
from glyphbench.envs.miniatari.kangaroo import MiniKangarooEnv
from glyphbench.envs.miniatari.kungfumaster import MiniKungFuMasterEnv
from glyphbench.envs.miniatari.mspacman import MiniMsPacmanEnv
from glyphbench.envs.miniatari.phoenix import MiniPhoenixEnv
from glyphbench.envs.miniatari.pong import MiniPongEnv
from glyphbench.envs.miniatari.qbert import MiniQbertEnv
from glyphbench.envs.miniatari.riverraid import MiniRiverRaidEnv
from glyphbench.envs.miniatari.robotank import MiniRobotankEnv
from glyphbench.envs.miniatari.seaquest import MiniSeaquestEnv
from glyphbench.envs.miniatari.skiing import MiniSkiingEnv
from glyphbench.envs.miniatari.spaceinvaders import MiniSpaceInvadersEnv
from glyphbench.envs.miniatari.stargunner import MiniStarGunnerEnv
from glyphbench.envs.miniatari.surround import MiniSurroundEnv
from glyphbench.envs.miniatari.tennis import MiniTennisEnv
from glyphbench.envs.miniatari.timepilot import MiniTimePilotEnv
from glyphbench.envs.miniatari.upndown import MiniUpNDownEnv
from glyphbench.envs.miniatari.wizardofwor import MiniWizardOfWorEnv
from glyphbench.envs.miniatari.yarsrevenge import MiniYarsRevengeEnv
from glyphbench.envs.miniatari.zaxxon import MiniZaxxonEnv

REGISTRY: dict = {
    "glyphbench/miniatari-pong-v0": MiniPongEnv,
    "glyphbench/miniatari-tennis-v0": MiniTennisEnv,
    "glyphbench/miniatari-icehockey-v0": MiniIceHockeyEnv,
    "glyphbench/miniatari-doubledunk-v0": MiniDoubleDunkEnv,
    "glyphbench/miniatari-fishingderby-v0": MiniFishingDerbyEnv,
    "glyphbench/miniatari-boxing-v0": MiniBoxingEnv,
    "glyphbench/miniatari-surround-v0": MiniSurroundEnv,
    "glyphbench/miniatari-breakout-v0": MiniBreakoutEnv,
    "glyphbench/miniatari-spaceinvaders-v0": MiniSpaceInvadersEnv,
    "glyphbench/miniatari-freeway-v0": MiniFreewayEnv,
    "glyphbench/miniatari-frostbite-v0": MiniFrostbiteEnv,
    "glyphbench/miniatari-asteroids-v0": MiniAsteroidsEnv,
    "glyphbench/miniatari-centipede-v0": MiniCentipedeEnv,
    "glyphbench/miniatari-demonattack-v0": MiniDemonAttackEnv,
    "glyphbench/miniatari-enduro-v0": MiniEnduroEnv,
    "glyphbench/miniatari-skiing-v0": MiniSkiingEnv,
    "glyphbench/miniatari-bowling-v0": MiniBowlingEnv,
    "glyphbench/miniatari-atlantis-v0": MiniAtlantisEnv,
    "glyphbench/miniatari-beamrider-v0": MiniBeamRiderEnv,
    "glyphbench/miniatari-choppercommand-v0": MiniChopperCommandEnv,
    "glyphbench/miniatari-defender-v0": MiniDefenderEnv,
    "glyphbench/miniatari-phoenix-v0": MiniPhoenixEnv,
    "glyphbench/miniatari-qbert-v0": MiniQbertEnv,
    "glyphbench/miniatari-robotank-v0": MiniRobotankEnv,
    "glyphbench/miniatari-timepilot-v0": MiniTimePilotEnv,
    "glyphbench/miniatari-upndown-v0": MiniUpNDownEnv,
    "glyphbench/miniatari-stargunner-v0": MiniStarGunnerEnv,
    "glyphbench/miniatari-assault-v0": MiniAssaultEnv,
    "glyphbench/miniatari-battlezone-v0": MiniBattleZoneEnv,
    "glyphbench/miniatari-berzerk-v0": MiniBerzerkEnv,
    "glyphbench/miniatari-alien-v0": MiniAlienEnv,
    "glyphbench/miniatari-amidar-v0": MiniAmidarEnv,
    "glyphbench/miniatari-asterix-v0": MiniAsterixEnv,
    "glyphbench/miniatari-bankheist-v0": MiniBankHeistEnv,
    "glyphbench/miniatari-gopher-v0": MiniGopherEnv,
    "glyphbench/miniatari-kangaroo-v0": MiniKangarooEnv,
    "glyphbench/miniatari-kungfumaster-v0": MiniKungFuMasterEnv,
    "glyphbench/miniatari-seaquest-v0": MiniSeaquestEnv,
    "glyphbench/miniatari-wizardofwor-v0": MiniWizardOfWorEnv,
    "glyphbench/miniatari-yarsrevenge-v0": MiniYarsRevengeEnv,
    "glyphbench/miniatari-zaxxon-v0": MiniZaxxonEnv,
    "glyphbench/miniatari-mspacman-v0": MiniMsPacmanEnv,
    "glyphbench/miniatari-riverraid-v0": MiniRiverRaidEnv,
}

for env_id, cls in REGISTRY.items():
    # Miniatari tasks are compact, current-screen arcade games. Prefer
    # action-only/default prompts; use n_frames for short temporal cues.
    register_env(env_id, cls, default_use_memory=False)
