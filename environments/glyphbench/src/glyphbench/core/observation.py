"""The frozen observation dataclass returned by every env."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GridObservation:
    """Uniform observation structure across every env in glyphbench.

    All four string fields are required. Use empty string ("") when a field has no
    content this turn. Env code must NOT place metrics-only data in these fields;
    that goes in the env's info dict.
    """

    grid: str
    legend: str
    hud: str
    message: str

    def render(self) -> str:
        """Serialize to the canonical prompt-ready string.

        Section order is always: legend, hud, grid, message. Empty sections are
        omitted from the output except grid, which is always present.
        """
        parts: list[str] = []
        if self.legend:
            parts.append(f"[Legend]\n{self.legend}")
        if self.hud:
            parts.append(f"[HUD]\n{self.hud}")
        parts.append(f"[Grid]\n{self.grid}")
        if self.message:
            parts.append(f"[Message]\n{self.message}")
        return "\n\n".join(parts)
