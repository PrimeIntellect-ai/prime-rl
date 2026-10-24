"""Whether the platform monitors can authenticate, from the environment and the CLIs'
config files only: no network, no prompt, no heavy imports, so launchers can ask too."""

import netrc
import os
from pathlib import Path
from urllib.parse import urlparse

PRIME_MODE_VAR = "PRIME_RUNS_MODE"


def missing_prime_credentials() -> str | None:
    """Why the prime monitor cannot authenticate, or None when it can."""
    from prime_sandboxes import Config as PrimeConfig

    if os.getenv(PRIME_MODE_VAR) or os.getenv("PRIME_API_KEY") or PrimeConfig().api_key:
        return None
    return (
        "No Prime API key found - the run is not logged to Prime. Authenticate with "
        "`prime login` (or set PRIME_API_KEY), or turn the monitor off with --no-monitors.prime"
    )


def missing_wandb_credentials(offline: bool = False) -> str | None:
    """Why the W&B monitor cannot authenticate, or None when it can: the credentials
    ``wandb.init`` would load, the API key or identity token variables, else the netrc
    entry for the W&B host. An offline run needs none."""
    if offline or os.environ.get("WANDB_MODE") in ("offline", "disabled"):
        return None
    if os.environ.get("WANDB_API_KEY") or os.environ.get("WANDB_IDENTITY_TOKEN_FILE"):
        return None
    host = urlparse(os.environ.get("WANDB_BASE_URL", "https://api.wandb.ai")).hostname
    path = Path(os.environ.get("NETRC", Path.home() / ".netrc")).expanduser()
    try:
        if host and netrc.netrc(path).authenticators(host):
            return None
    except (OSError, netrc.NetrcParseError):
        pass
    return (
        "No W&B API key found - the run is not logged to W&B. Authenticate with `wandb login` "
        "(or set WANDB_API_KEY), or turn the monitor off with --no-monitors.wandb"
    )
