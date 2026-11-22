"""Atari suite - importing registers all envs."""

from glyphbench.core.registry import register_env
from glyphbench.envs.atari.alien import (
    AlienEnv,
)
from glyphbench.envs.atari.amidar import (
    AmidarEnv,
)
from glyphbench.envs.atari.assault import (
    AssaultEnv,
)
from glyphbench.envs.atari.asterix import (
    AsterixEnv,
)
from glyphbench.envs.atari.asteroids import (
    AsteroidsEnv,
)
from glyphbench.envs.atari.atlantis import (
    AtlantisEnv,
)
from glyphbench.envs.atari.bankheist import (
    BankHeistEnv,
)
from glyphbench.envs.atari.battlezone import (
    BattleZoneEnv,
)
from glyphbench.envs.atari.beamrider import (
    BeamRiderEnv,
)
from glyphbench.envs.atari.berzerk import (
    BerzerkEnv,
)
from glyphbench.envs.atari.bowling import (
    BowlingEnv,
)
from glyphbench.envs.atari.boxing import (
    BoxingEnv,
)
from glyphbench.envs.atari.breakout import (
    BreakoutEnv,
)
from glyphbench.envs.atari.centipede import (
    CentipedeEnv,
)
from glyphbench.envs.atari.choppercommand import (
    ChopperCommandEnv,
)
from glyphbench.envs.atari.crazyclimber import (
    CrazyClimberEnv,
)
from glyphbench.envs.atari.defender import (
    DefenderEnv,
)
from glyphbench.envs.atari.demonattack import (
    DemonAttackEnv,
)
from glyphbench.envs.atari.doubledunk import (
    DoubleDunkEnv,
)
from glyphbench.envs.atari.enduro import (
    EnduroEnv,
)
from glyphbench.envs.atari.fishingderby import (
    FishingDerbyEnv,
)
from glyphbench.envs.atari.freeway import (
    FreewayEnv,
)
from glyphbench.envs.atari.frostbite import (
    FrostbiteEnv,
)
from glyphbench.envs.atari.gopher import (
    GopherEnv,
)
from glyphbench.envs.atari.gravitar import (
    GravitarEnv,
)
from glyphbench.envs.atari.hero import (
    HeroEnv,
)
from glyphbench.envs.atari.icehockey import (
    IceHockeyEnv,
)
from glyphbench.envs.atari.jamesbond import (
    JamesBondEnv,
)
from glyphbench.envs.atari.kangaroo import (
    KangarooEnv,
)
from glyphbench.envs.atari.krull import (
    KrullEnv,
)
from glyphbench.envs.atari.kungfumaster import (
    KungFuMasterEnv,
)
from glyphbench.envs.atari.montezumarevenge import (
    MontezumaRevengeEnv,
)
from glyphbench.envs.atari.mspacman import (
    MsPacManEnv,
)
from glyphbench.envs.atari.namethisgame import (
    NameThisGameEnv,
)
from glyphbench.envs.atari.phoenix import (
    PhoenixEnv,
)
from glyphbench.envs.atari.pitfall import (
    PitfallEnv,
)
from glyphbench.envs.atari.pong import (
    PongEnv,
)
from glyphbench.envs.atari.privateeye import (
    PrivateEyeEnv,
)
from glyphbench.envs.atari.qbert import (
    QbertEnv,
)
from glyphbench.envs.atari.riverraid import (
    RiverRaidEnv,
)
from glyphbench.envs.atari.roadrunner import (
    RoadRunnerEnv,
)
from glyphbench.envs.atari.robotank import (
    RobotankEnv,
)
from glyphbench.envs.atari.seaquest import (
    SeaquestEnv,
)
from glyphbench.envs.atari.skiing import (
    SkiingEnv,
)
from glyphbench.envs.atari.solaris import (
    SolarisEnv,
)
from glyphbench.envs.atari.spaceinvaders import (
    SpaceInvadersEnv,
)
from glyphbench.envs.atari.stargunner import (
    StarGunnerEnv,
)
from glyphbench.envs.atari.surround import (
    SurroundEnv,
)
from glyphbench.envs.atari.tennis import (
    TennisEnv,
)
from glyphbench.envs.atari.timepilot import (
    TimePilotEnv,
)
from glyphbench.envs.atari.tutankham import (
    TutankhamEnv,
)
from glyphbench.envs.atari.upndown import (
    UpNDownEnv,
)
from glyphbench.envs.atari.venture import (
    VentureEnv,
)
from glyphbench.envs.atari.videopinball import (
    VideoPinballEnv,
)
from glyphbench.envs.atari.wizardofwor import (
    WizardOfWorEnv,
)
from glyphbench.envs.atari.yarsrevenge import (
    YarsRevengeEnv,
)
from glyphbench.envs.atari.zaxxon import (
    ZaxxonEnv,
)

_REGISTRATIONS = {
    "glyphbench/atari-pong-v0": PongEnv,
    "glyphbench/atari-montezumarevenge-v0": MontezumaRevengeEnv,
    "glyphbench/atari-venture-v0": VentureEnv,
    "glyphbench/atari-kangaroo-v0": KangarooEnv,
    "glyphbench/atari-pitfall-v0": PitfallEnv,
    "glyphbench/atari-frostbite-v0": FrostbiteEnv,
    "glyphbench/atari-mspacman-v0": MsPacManEnv,
    "glyphbench/atari-berzerk-v0": BerzerkEnv,
    "glyphbench/atari-wizardofwor-v0": WizardOfWorEnv,
    "glyphbench/atari-bankheist-v0": BankHeistEnv,
    "glyphbench/atari-amidar-v0": AmidarEnv,
    "glyphbench/atari-qbert-v0": QbertEnv,
    "glyphbench/atari-breakout-v0": BreakoutEnv,
    "glyphbench/atari-spaceinvaders-v0": SpaceInvadersEnv,
    "glyphbench/atari-hero-v0": HeroEnv,
    "glyphbench/atari-freeway-v0": FreewayEnv,
    "glyphbench/atari-surround-v0": SurroundEnv,
    "glyphbench/atari-phoenix-v0": PhoenixEnv,
    "glyphbench/atari-demonattack-v0": DemonAttackEnv,
    "glyphbench/atari-assault-v0": AssaultEnv,
    "glyphbench/atari-atlantis-v0": AtlantisEnv,
    "glyphbench/atari-gopher-v0": GopherEnv,
    "glyphbench/atari-centipede-v0": CentipedeEnv,
    "glyphbench/atari-defender-v0": DefenderEnv,
    "glyphbench/atari-seaquest-v0": SeaquestEnv,
    "glyphbench/atari-choppercommand-v0": ChopperCommandEnv,
    "glyphbench/atari-stargunner-v0": StarGunnerEnv,
    "glyphbench/atari-timepilot-v0": TimePilotEnv,
    "glyphbench/atari-riverraid-v0": RiverRaidEnv,
    "glyphbench/atari-upndown-v0": UpNDownEnv,
    "glyphbench/atari-beamrider-v0": BeamRiderEnv,
    "glyphbench/atari-namethisgame-v0": NameThisGameEnv,
    "glyphbench/atari-yarsrevenge-v0": YarsRevengeEnv,
    "glyphbench/atari-alien-v0": AlienEnv,
    "glyphbench/atari-crazyclimber-v0": CrazyClimberEnv,
    "glyphbench/atari-tutankham-v0": TutankhamEnv,
    "glyphbench/atari-kungfumaster-v0": KungFuMasterEnv,
    "glyphbench/atari-jamesbond-v0": JamesBondEnv,
    "glyphbench/atari-roadrunner-v0": RoadRunnerEnv,
    "glyphbench/atari-krull-v0": KrullEnv,
    "glyphbench/atari-privateeye-v0": PrivateEyeEnv,
    "glyphbench/atari-bowling-v0": BowlingEnv,
    "glyphbench/atari-boxing-v0": BoxingEnv,
    "glyphbench/atari-tennis-v0": TennisEnv,
    "glyphbench/atari-icehockey-v0": IceHockeyEnv,
    "glyphbench/atari-skiing-v0": SkiingEnv,
    "glyphbench/atari-enduro-v0": EnduroEnv,
    "glyphbench/atari-doubledunk-v0": DoubleDunkEnv,
    "glyphbench/atari-fishingderby-v0": FishingDerbyEnv,
    "glyphbench/atari-asterix-v0": AsterixEnv,
    "glyphbench/atari-asteroids-v0": AsteroidsEnv,
    "glyphbench/atari-battlezone-v0": BattleZoneEnv,
    "glyphbench/atari-robotank-v0": RobotankEnv,
    "glyphbench/atari-zaxxon-v0": ZaxxonEnv,
    "glyphbench/atari-gravitar-v0": GravitarEnv,
    "glyphbench/atari-solaris-v0": SolarisEnv,
    "glyphbench/atari-videopinball-v0": VideoPinballEnv,
}

for _id, _cls in _REGISTRATIONS.items():
    # Atari glyph grids render the current screen/state; use frame stacking,
    # not the costly text-memory scaffold, when temporal velocity helps.
    register_env(_id, _cls, default_use_memory=False)
