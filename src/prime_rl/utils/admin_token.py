"""The run's admin token, which guards the engines' admin routes and the control APIs.

A process that calls a guarded route holds the token in ``$<var>`` and sends it in the
``X-Prime-Admin-Token`` header. A process that only checks it can be given its SHA-256
digest in ``$<var>_SHA256`` instead, so holding what it checks against does not let it
make the calls itself.
"""

import hashlib
import hmac
import os
import secrets
from collections.abc import Mapping
from dataclasses import dataclass


def digest_var(token_var: str) -> str:
    """The variable a server reads the token's digest from."""
    return f"{token_var}_SHA256"


def admin_token_digest(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def expected_admin_digest(token_var: str) -> str | None:
    """The digest a server checks requests against: ``$<var>_SHA256``, else the digest of
    ``$<var>``; None when neither is set."""
    if digest := os.environ.get(digest_var(token_var)):
        return digest
    if token := os.environ.get(token_var):
        return admin_token_digest(token)
    return None


def admin_token_matches(provided: str | None, expected_digest: str) -> bool:
    return provided is not None and hmac.compare_digest(admin_token_digest(provided), expected_digest)


@dataclass(frozen=True)
class TokenUse:
    """How one process a launcher starts uses the admin token."""

    sends: str | None = None
    """Variable it reads the token from, to call guarded routes."""

    checks: str | None = None
    """Variable naming the token it checks requests against. It gets the digest, in ``<var>_SHA256``."""


@dataclass(frozen=True)
class AdminTokenUses:
    """How each process a launcher starts uses the run's admin token."""

    by_process: dict[str, TokenUse]

    @property
    def names(self) -> list[str]:
        """Every variable a process reads the token, or its digest, through."""
        names = (name for use in self.by_process.values() for name in (use.sends, use.checks))
        return list(dict.fromkeys(name for name in names if name is not None))

    @property
    def guards(self) -> bool:
        """Whether a process checks the token, so the launcher needs one."""
        return any(use.checks is not None for use in self.by_process.values())

    def strip(self, environ: Mapping[str, str]) -> dict[str, str]:
        """``environ`` without the token or its digest, for the processes that use neither."""
        hidden = {*self.names, *map(digest_var, self.names)}
        return {key: value for key, value in environ.items() if key not in hidden}

    def split(self, environ: Mapping[str, str]) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
        """``environ`` without the token or its digest, and what each process adds to it.

        The token is the operator's when ``environ`` sets it, a fresh one when a process
        checks it, and absent otherwise."""
        token = next((environ[name] for name in self.names if environ.get(name)), None)
        if token is None and self.guards:
            token = secrets.token_urlsafe(32)
        base = self.strip(environ)
        if token is None:
            return base, {}
        digest = admin_token_digest(token)
        envs = {}
        for process, use in self.by_process.items():
            env = {}
            if use.sends is not None:
                env[use.sends] = token
            if use.checks is not None:
                env[digest_var(use.checks)] = digest
            envs[process] = env
        return base, envs
