"""ActionSpec: per-env action vocabulary."""

from __future__ import annotations

from dataclasses import dataclass, field

# Cross-suite name aliases. Some suites use "WAIT"/"DONE" as their no-op
# action, but models — especially Qwen — heavily reach for "NOOP" as a
# universal fallback. We accept either when the env owns the corresponding
# semantic action; collision (env defines BOTH NOOP and WAIT) skips the alias.
# All five are mutually-aliased: any of {NOOP, WAIT, DONE, PASS, SKIP}
# emitted by the model resolves to the env's actual no-op action — whichever
# of those names appears in `names`.
_NO_OP_FAMILY: tuple[str, ...] = ("NOOP", "WAIT", "DONE", "PASS", "SKIP")
_DEFAULT_ALIASES: dict[str, tuple[str, ...]] = {
    name: tuple(other for other in _NO_OP_FAMILY if other != name)
    for name in _NO_OP_FAMILY
}


@dataclass(frozen=True, slots=True)
class ActionSpec:
    """Each env owns its own Discrete(N) action space with per-game action names.

    The harness does not filter actions by turn-validity; the LLM must reason
    about which actions make sense in the current state. Action names are
    SHOUTY_SNAKE_CASE by convention so the parser can extract them with a
    simple regex as a fallback to JSON parsing.
    """

    names: tuple[str, ...]
    descriptions: tuple[str, ...]
    extra_aliases: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.names) != len(self.descriptions):
            raise ValueError(
                f"ActionSpec.names and ActionSpec.descriptions must have the same length, "
                f"got {len(self.names)} and {len(self.descriptions)}"
            )
        if len(set(self.names)) != len(self.names):
            dupes = [n for n in self.names if self.names.count(n) > 1]
            raise ValueError(f"ActionSpec.names must be unique; duplicates: {dupes}")

    @property
    def n(self) -> int:
        return len(self.names)

    def index_of(self, name: str) -> int:
        """Look up an action by name.

        Tries: exact match → case-insensitive (whitespace-stripped) → spec
        extra_aliases → built-in cross-suite NOOP/WAIT/DONE aliases. Raises
        ``KeyError`` only when nothing resolves to a unique action.
        """
        stripped = name.strip()
        try:
            return self.names.index(stripped)
        except ValueError:
            pass

        folded = stripped.casefold()
        matches = [i for i, n in enumerate(self.names) if n.casefold() == folded]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise KeyError(
                f"Ambiguous action name {name!r} (case-folded matches multiple)."
            )

        # Per-spec custom aliases (e.g. {"FIRE": "SHOOT"}).
        target = self.extra_aliases.get(stripped) or self.extra_aliases.get(stripped.upper())
        if target:
            tfold = target.casefold()
            for i, n in enumerate(self.names):
                if n.casefold() == tfold:
                    return i

        # Built-in cross-suite no-op aliases. Only fire when the env defines
        # exactly one of the alias targets (no ambiguity).
        for canonical, aliases in _DEFAULT_ALIASES.items():
            if folded != canonical.casefold():
                continue
            present = [
                i for i, n in enumerate(self.names) if n.casefold() in {a.casefold() for a in aliases}
            ]
            if len(present) == 1:
                return present[0]

        raise KeyError(
            f"Unknown action name: {name!r}. Valid: {list(self.names)}"
        )

    def render_for_prompt(self) -> str:
        """Rendered once into the system prompt at episode start, not per turn."""
        lines = ["[Actions]", "Pick exactly one per turn:"]
        for name, desc in zip(self.names, self.descriptions, strict=True):
            lines.append(f"  {name} — {desc}")
        return "\n".join(lines)
