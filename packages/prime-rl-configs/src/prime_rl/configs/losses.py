"""Composable loss-term config DSL (the unified ``losses`` surface).

A training run defines a list of named loss *terms*; per-env ``enabled_losses`` selects which apply.
Each term is a free composition of two independently-chosen axes, each a discriminated-union preset
(``type`` + kwargs) with a ``custom`` import-path escape hatch:

- ``loss``      — the core (trainer-side): ``dppo_kl`` (RL) · ``ce`` (echo / NLL) · ``custom``.
- ``advantage`` — the per-token advantage_fn (orchestrator-side): ``grpo`` · ``echo`` · ``sft`` ·
  ``custom``. It emits one float per token (``0`` = masked), so a single axis expresses both *which*
  tokens a term trains and *how much* each counts.

Common losses are one-line presets: ``{type = "rl", ...}`` expands a *single* term into the canonical
loss/advantage form (``LossTerm._expand_preset``), while ``{type = "echo", ...}`` is a *compound
recipe* — ``rl ⊕ ce-on-roles`` — that fans out to *two* terms (``expand_compound_recipes``). The
**primary** (rl objective) is the ``grpo``-advantage term, dispatched by ``training_mode``; every
other term is an additive **overlay**. Terms are summed over one shared forward → one backward. Default ``losses``
reproduces today's DPPO+KL. sft/opd remain separate ``training_mode`` paths with fixed cores.
``RLLossConfig`` below is the resolved internal form the trainer consumes.
"""

from typing import Annotated, Any, Literal, TypeAlias

from pydantic import AfterValidator, BeforeValidator, Field, model_validator

from prime_rl.utils.config import BaseConfig

# --------------------------------------------------------------------------------------------------
# Axis 1 — cores (`loss`): trainer-side ``LossInputs -> LossOutputs``.
# --------------------------------------------------------------------------------------------------


class DPPOKLCoreConfig(BaseConfig):
    """DPPO+KL policy-gradient core (the RL objective). Trains its filtered tokens with the weight as
    the per-token advantage; masks trust-region violators and adds a squared-KL regularizer."""

    type: Literal["dppo_kl"] = "dppo_kl"

    dppo_mask_low: float = Field(0.2, ge=0)
    """Lower DPPO masking threshold."""

    dppo_mask_high: float = Field(0.2, ge=0)
    """Upper DPPO masking threshold."""

    kl_tau: float = Field(1e-3, ge=0)
    """Temperature for the KL term."""


class CECoreConfig(BaseConfig):
    """Cross-entropy core (echo / weighted NLL): ``-Σ weightₜ · logprobₜ`` over the filtered tokens."""

    type: Literal["ce"] = "ce"


class CustomCoreConfig(BaseConfig):
    """A user core resolved from a dotted import path (``def core(inputs, **kwargs) -> LossOutputs``)."""

    type: Literal["custom"] = "custom"

    import_path: str
    """Dotted import path to the loss core."""

    kwargs: dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments forwarded to the core as ``**kwargs``."""


LossCoreConfig: TypeAlias = Annotated[
    DPPOKLCoreConfig | CECoreConfig | CustomCoreConfig,
    Field(discriminator="type"),
]

# --------------------------------------------------------------------------------------------------
# Roles — token attribution used by the ``echo`` advantage.
# --------------------------------------------------------------------------------------------------

Role: TypeAlias = Literal["system", "user", "assistant", "tool"]

# --------------------------------------------------------------------------------------------------
# The advantage axis — the per-token advantage_fn (orchestrator-side). Produces one float per token
# (0 = masked); supersedes the filters + weight axes above. Resolved to a callable over a group of
# ``RenderHints`` by ``orchestrator.advantage.resolve_advantage_fn``.
# --------------------------------------------------------------------------------------------------


class GRPOAdvantageConfig(BaseConfig):
    """GRPO advantage (the RL objective / primary): the per-rollout reward-baseline advantage (× tau)
    broadcast over the sampled tokens, 0 elsewhere."""

    type: Literal["grpo"] = "grpo"

    tau: float = Field(1.0, ge=0)
    """Temperature on the advantage."""


class EchoAdvantageConfig(BaseConfig):
    """Echo advantage (overlay): a selection mask — ``1.0`` on role-matched context tokens, ``0``
    elsewhere. ``by_advantage`` multiplies the mask by the rollout's advantage (× ``tau``) for
    advantage-weighted echo. The echo *magnitude* is the term's ``lambda_weight`` — λ owns magnitude,
    the advantage owns selection/shape (non-uniform per-token weights stay reachable via a ``custom``
    advantage)."""

    type: Literal["echo"] = "echo"

    roles: list[Role] = Field(min_length=1)
    """Roles whose content tokens are echoed."""

    tool_names: set[str] | None = Field(None, min_length=1)
    """When ``"tool"`` is among the roles, restrict to these tool function names; None = all tools."""

    by_advantage: bool = False
    """Multiply the selection mask by the rollout's advantage × ``tau`` (advantage-weighted echo)."""

    tau: float = Field(1.0, ge=0)
    """Temperature on the advantage when ``by_advantage``."""

    @model_validator(mode="after")
    def _validate_tool_names(self) -> "EchoAdvantageConfig":
        # ``tool_names`` only narrows the ``tool`` role; setting it without echoing ``tool`` is a
        # silent no-op (``echo_advantage`` only checks names on tool-role tokens), so reject it.
        if self.tool_names is not None and "tool" not in self.roles:
            raise ValueError(
                f"echo advantage: tool_names={sorted(self.tool_names)} only narrows the 'tool' role, but "
                f"'tool' is not in roles={self.roles}; add 'tool' to roles or drop tool_names."
            )
        return self


class SFTAdvantageConfig(BaseConfig):
    """SFT advantage: a constant ``alpha`` on the sampled tokens, 0 elsewhere (masked NLL)."""

    type: Literal["sft"] = "sft"

    alpha: float = Field(1.0, allow_inf_nan=False)
    """Per-token weight."""


class CustomAdvantageFnConfig(BaseConfig):
    """A user advantage_fn resolved from a dotted import path. Signature:
    ``fn(group: list[RenderHints], **kwargs) -> list[list[float]]`` (one list/unit, one float/token;
    ``0`` masks). The group's ``RenderHints`` carry each unit's reward/advantage/rollout + per-token
    attribution, so the fn can compute group-relative, advantage-weighted, or attribution-based signals."""

    type: Literal["custom"] = "custom"

    import_path: str
    """Dotted import path to the advantage_fn."""

    kwargs: dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments forwarded to the fn as ``**kwargs``."""


AdvantageFnConfig: TypeAlias = Annotated[
    GRPOAdvantageConfig | EchoAdvantageConfig | SFTAdvantageConfig | CustomAdvantageFnConfig,
    Field(discriminator="type"),
]

# --------------------------------------------------------------------------------------------------
# The reduce axis — the per-term normalization step (trainer-side). Resolved to a callable by
# ``trainer.rl.loss.setup_reduce``.
# --------------------------------------------------------------------------------------------------


class MeanReduceConfig(BaseConfig):
    """Global per-token mean over the term's eligible tokens (the default; matches today's
    ``loss_scale`` normalization)."""

    type: Literal["mean"] = "mean"


class CustomReduceConfig(BaseConfig):
    """A user reduce resolved from a dotted import path: ``fn(inputs: ReduceInputs, **kwargs) -> Tensor``."""

    type: Literal["custom"] = "custom"

    import_path: str
    """Dotted import path to the reduce callable."""

    kwargs: dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments forwarded to the reduce as ``**kwargs``."""


ReduceConfig: TypeAlias = Annotated[
    MeanReduceConfig | CustomReduceConfig,
    Field(discriminator="type"),
]

# --------------------------------------------------------------------------------------------------
# Hooks — trainer-side, post-core, per-token loss transforms (chainable). Resolved to callables by
# ``trainer.rl.loss.setup_hooks``. Built-in hook presets can join as union members later; for now
# only the ``custom`` import-path form exists.
# --------------------------------------------------------------------------------------------------


class CustomHookConfig(BaseConfig):
    """A trainer-side per-token loss transform resolved from a dotted import path. Signature:
    ``fn(per_token_loss: Tensor, inputs: LossInputs, **kwargs) -> Tensor`` (per-token in, per-token
    out; reduction is the separate ``reduce`` step). For signals only available after the forward —
    current-policy prob/entropy gating, smoothing, penalties."""

    type: Literal["custom"] = "custom"

    import_path: str
    """Dotted import path to the hook callable."""

    kwargs: dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments forwarded to the hook as ``**kwargs``."""


class MinProbFilterConfig(BaseConfig):
    """Built-in hook: zero the per-token loss on tokens the *current policy* already assigns low
    probability (current-policy ``logprob < min_logprob``). A trainer-side filter — it reads the live
    forward, so it can't be precomputed orchestrator-side (e.g. drop low-confidence terminal-task
    warning tokens)."""

    type: Literal["min_prob_filter"] = "min_prob_filter"

    min_logprob: float = Field(allow_inf_nan=False)
    """Keep only tokens whose current-policy logprob is ``>=`` this; below it the per-token loss is zeroed."""


HookConfig: TypeAlias = Annotated[
    CustomHookConfig | MinProbFilterConfig,
    Field(discriminator="type"),
]
"""A loss term's hook: a built-in preset (``min_prob_filter``) or a ``custom`` import path."""

# The single-term ``rl`` preset's default axes (dppo_kl core + grpo advantage). Shared by the
# ``echo`` compound recipe, whose policy-gradient half *is* the ``rl`` term (see ``_expand_echo``).
_RL_PRESET_DEFAULTS: dict[str, Any] = {"name": "rl", "loss": {"type": "dppo_kl"}, "advantage": {"type": "grpo"}}

# --------------------------------------------------------------------------------------------------
# The term: a core, a per-token advantage_fn, and per-term λ + reduce + hooks.
# --------------------------------------------------------------------------------------------------


class LossTerm(BaseConfig):
    """A single loss term — a core and an advantage_fn — summed into the total loss."""

    name: str
    """Unique term name (referenced by per-env ``enabled_losses`` / ``loss_overrides``)."""

    loss: LossCoreConfig
    """The core (trainer-side, required)."""

    advantage: AdvantageFnConfig
    """The advantage_fn (orchestrator-side, required): the per-token signal, ``0`` = masked."""

    lambda_weight: float = Field(1.0, allow_inf_nan=False)
    """Per-term coefficient λ on this term's contribution, applied pre-reduce. Default 1.0."""

    reduce: ReduceConfig = Field(default_factory=MeanReduceConfig)
    """The term's normalization step (trainer-side). Default: global per-token mean (= today)."""

    hooks: list[HookConfig] = Field(default_factory=list)
    """Trainer-side post-core per-token transforms, applied in order between the core and the reduce.
    Default: none (the term's loss is the core's output unchanged)."""

    @model_validator(mode="before")
    @classmethod
    def _expand_preset(cls, data: Any) -> Any:
        """Expand a single-term preset (``{type = "rl", ...}``) into the full loss/advantage form. The
        preset seeds default axes; any explicit ``loss``/``advantage``/``name`` the user sets is
        deep-merged over them with the *same* semantics as a full term. The full form (no top-level
        ``type``) and built terms pass through. Compound recipes (``echo``) fan out to *several* terms
        and are expanded a level up, in ``expand_compound_recipes`` — they cannot resolve to one term."""
        if not isinstance(data, dict) or "type" not in data:
            return data
        data = dict(data)
        preset = data.pop("type")
        if preset == "rl":
            defaults: dict[str, Any] = dict(_RL_PRESET_DEFAULTS)
        elif preset == "echo":
            raise ValueError(
                "the 'echo' recipe expands to multiple terms (rl ⊕ ce-on-roles) at the `losses` list "
                'level, so it can\'t be built as a single term; put `{type = "echo", ...}` directly in '
                "`losses`."
            )
        else:
            raise ValueError(
                f"unknown loss preset type {preset!r}; use 'rl', the 'echo' recipe, or the full "
                f"loss/advantage form."
            )
        unknown = set(data) - {"name", "loss", "advantage", "lambda_weight", "reduce", "hooks"}
        if unknown:
            raise ValueError(
                f"loss preset {preset!r}: override the axes (loss/advantage/lambda_weight/reduce/hooks) "
                f"or name, not {sorted(unknown)}."
            )
        return deep_merge(defaults, data)

    @model_validator(mode="after")
    def validate_supported(self) -> "LossTerm":
        # The primary (rl objective) is the grpo-advantage term with a dppo_kl/custom core; every other
        # advantage (echo / sft / custom) is an additive overlay with a ce/custom core.
        core = self.loss.type
        if self.advantage.type == "grpo":
            if core not in ("dppo_kl", "custom"):
                raise ValueError(f"loss term {self.name!r}: a grpo (primary) advantage needs a dppo_kl or custom core.")
        elif core not in ("ce", "custom"):
            raise ValueError(
                f"loss term {self.name!r}: a {self.advantage.type} (overlay) advantage needs a ce or custom core."
            )
        return self


def is_primary(term: LossTerm) -> bool:
    """The rl objective is the grpo-advantage term; every other advantage is an additive overlay."""
    return term.advantage.type == "grpo"


# --------------------------------------------------------------------------------------------------
# Compound recipes — list-level presets that fan out to *several* terms (vs the single-term ``rl``
# preset handled in ``LossTerm._expand_preset``). ``echo`` is the headline demonstration that the
# framework composes: it is not a primitive but ``rl ⊕ ce-on-roles``. Each emitted term is tagged
# with its source recipe so a name collision can point at the recipe that produced it.
# --------------------------------------------------------------------------------------------------

_ECHO_ADVANTAGE_KNOBS = ("roles", "tool_names", "by_advantage", "tau")  # -> the ce overlay's echo advantage
_ECHO_TERM_KNOBS = ("lambda_weight", "reduce", "hooks")  # -> the ce overlay term itself (λ owns echo magnitude)


def _expand_echo(entry: dict) -> list[dict]:
    """Expand an ``echo`` recipe into its two terms: the ``rl`` primary + a ``ce``-on-roles overlay.

    Echo's advantage knobs (``roles``/``tool_names``/``by_advantage``/``tau``) route to the overlay's
    echo advantage (a 0/1 selection mask, × advantage·tau when ``by_advantage``); ``lambda_weight``/
    ``reduce``/``hooks`` set the overlay term (λ owns the echo magnitude); an ``rl = {...}`` block
    deep-merges into the rl term, so the policy-gradient half is tuned *through* ``echo`` rather than
    as a separate, name-colliding sibling."""
    entry = dict(entry)
    entry.pop("type")
    rl_overrides = entry.pop("rl", {})
    if not isinstance(rl_overrides, dict):
        raise ValueError("loss recipe 'echo': the 'rl' block must be a table of rl-term overrides.")
    unknown = set(entry) - set(_ECHO_ADVANTAGE_KNOBS) - set(_ECHO_TERM_KNOBS)
    if unknown:
        raise ValueError(
            f"loss recipe 'echo': unknown knob(s) {sorted(unknown)}; echo accepts "
            f"{sorted((*_ECHO_ADVANTAGE_KNOBS, *_ECHO_TERM_KNOBS))} plus an 'rl' block "
            f"(tune the policy-gradient half via `rl = {{...}}`)."
        )
    advantage: dict[str, Any] = {"type": "echo", "roles": ["assistant"]}
    overlay: dict[str, Any] = {"name": "echo", "loss": {"type": "ce"}, "advantage": advantage}
    for knob in _ECHO_ADVANTAGE_KNOBS:
        if knob in entry:
            advantage[knob] = entry[knob]
    for knob in _ECHO_TERM_KNOBS:
        if knob in entry:
            overlay[knob] = entry[knob]
    return [deep_merge(_RL_PRESET_DEFAULTS, rl_overrides), overlay]


_RECIPE_EXPANDERS = {"echo": _expand_echo}


def _entry_name(entry: Any) -> str | None:
    """The resolved term name of a raw ``losses`` entry, or None if undetermined (a full-form dict
    without a name — ``LossTerm`` construction will raise the missing-name error)."""
    if isinstance(entry, LossTerm):
        return entry.name
    if isinstance(entry, dict):
        if "name" in entry:
            return entry["name"]
        if entry.get("type") == "rl":  # the single-term preset defaults its name to "rl"
            return "rl"
    return None


def _check_unique_names(entries: list[Any], sources: list[str | None]) -> None:
    """Reject duplicate term names. ``sources[i]`` is the recipe that emitted ``entries[i]`` (else
    None); a clash involving a recipe-emitted term names the recipe so the user knows to tune it
    through the recipe's config rather than as a separate term."""
    seen: dict[str, str | None] = {}
    for entry, source in zip(entries, sources):
        name = _entry_name(entry)
        if name is None:
            continue
        if name not in seen:
            seen[name] = source
            continue
        recipe = source or seen[name]
        if recipe is not None:
            raise ValueError(
                f"Duplicate loss term name {name!r}: emitted by the {recipe!r} compound preset and also "
                f"defined elsewhere in `losses`. Tune the {recipe!r} recipe's sub-terms through the "
                f"{recipe!r} config instead of adding a separate {name!r} term."
            )
        raise ValueError(f"Duplicate loss term names: {[name]}. Each term in `losses` needs a unique name.")


def expand_compound_recipes(losses: Any) -> Any:
    """Expand compound-recipe presets in a ``losses`` list before each entry becomes a ``LossTerm``.

    A list entry like ``{type = "echo", ...}`` fans out to several terms (``echo`` → the ``rl`` primary
    + a ``ce`` overlay); single-term presets (``{type = "rl"}``) and full terms pass through untouched
    (``LossTerm._expand_preset`` handles them). Duplicate names are caught here, where each term's
    source recipe is known, so a clash with a recipe's sub-term names the recipe."""
    if not isinstance(losses, list):
        return losses
    expanded: list[Any] = []
    sources: list[str | None] = []  # parallel to ``expanded``: the recipe that emitted each entry, else None
    for entry in losses:
        recipe = entry.get("type") if isinstance(entry, dict) else None
        if recipe in _RECIPE_EXPANDERS:
            for term in _RECIPE_EXPANDERS[recipe](entry):
                expanded.append(term)
                sources.append(recipe)
        else:
            expanded.append(entry)
            sources.append(None)
    _check_unique_names(expanded, sources)
    return expanded


def validate_loss_list(losses: list[LossTerm]) -> list[LossTerm]:
    """Reject more than one primary (rl objective) term and reserved overlay names. (Duplicate names
    are caught earlier in ``expand_compound_recipes``, where each term's source recipe is known.)"""
    primaries = [term.name for term in losses if is_primary(term)]
    if len(primaries) > 1:
        raise ValueError(f"At most one primary (dppo_kl/custom core) loss term is allowed, got {primaries}.")
    # The trainer dispatches the per-run cores by these keys; an overlay sharing one would silently
    # overwrite the dispatch core (sft/opd) or the rl primary, so reserve them.
    for term in losses:
        if term.name in ("sft", "opd"):
            raise ValueError(f"loss term name {term.name!r} is reserved for the training_mode core; rename it.")
        if term.name == "rl" and not is_primary(term):
            raise ValueError("loss term name 'rl' is reserved for the rl primary; rename the overlay.")
    return losses


LossList: TypeAlias = Annotated[
    list[LossTerm],
    BeforeValidator(expand_compound_recipes),
    AfterValidator(validate_loss_list),
]
"""``list[LossTerm]`` with compound-recipe expansion + the unique-name / single-primary checks; the
field type for ``losses``."""


def check_enabled_losses(loss_names: set[str], enabled: list[str], where: str) -> None:
    """Raise if ``enabled`` references unknown terms."""
    unknown = sorted(set(enabled) - loss_names)
    if unknown:
        raise ValueError(f"{where}: enabled_losses {unknown} not found in losses {sorted(loss_names)}.")


def check_loss_overrides(
    loss_names: set[str],
    overlay_names: set[str],
    enabled: list[str],
    overrides: dict[str, dict],
    where: str,
) -> None:
    """Raise if ``overrides`` references unknown/non-overlay/disabled terms, or overrides anything but
    the ``advantage`` axis. The advantage_fn is resolved per env (orchestrator-side); ``loss`` (the
    core) and ``name`` are global, so overriding them would validate but be silently ignored. An
    override on a term the env doesn't enable is a silent no-op too (``_resolve_overlays`` skips
    disabled terms before applying overrides), so reject that as well."""
    for name, override in overrides.items():
        if name not in loss_names:
            raise ValueError(f"{where}: loss_overrides key {name!r} not found in losses {sorted(loss_names)}.")
        if name not in overlay_names:
            raise ValueError(f"{where}: loss_overrides currently applies only to overlay terms, got {name!r}.")
        if name not in enabled:
            raise ValueError(
                f"{where}: loss_overrides[{name!r}] targets a term not in enabled_losses {sorted(enabled)}; "
                f"the override would be silently ignored. Add it to enabled_losses or drop the override."
            )
        unsupported = set(override) - {"advantage"}
        if unsupported:
            raise ValueError(
                f"{where}: loss_overrides[{name!r}] may only override the 'advantage' axis per env, "
                f"not {sorted(unsupported)}."
            )


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into ``base`` (override wins on leaves)."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def apply_term_override(term: LossTerm, override: dict) -> LossTerm:
    """Deep-merge a per-env ``override`` into a term and rebuild it.

    Reconstructing the ``LossTerm`` validates the merged payload (unknown fields, bad shapes,
    unsupported combos) — at config time from the validator, and at resolve time from the orchestrator.
    """
    return LossTerm(**deep_merge(term.model_dump(), override))


def default_losses() -> list[LossTerm]:
    """Default loss list: RL only (reproduces the pre-``losses`` default)."""
    return [LossTerm(name="rl", loss=DPPOKLCoreConfig(), advantage=GRPOAdvantageConfig())]


# --------------------------------------------------------------------------------------------------
# Resolved internal types + conversions (Phase 1: map the schema onto today's execution engine).
# These are NOT part of the ``losses`` surface; they are produced from ``LossTerm``s at consume time.
# --------------------------------------------------------------------------------------------------


class RLLossConfig(BaseConfig):
    """Resolved DPPO+KL params consumed by ``default_loss_fn`` (built from a dppo_kl primary term)."""

    type: Literal["rl"] = "rl"
    name: str = "rl"

    dppo_mask_low: float = Field(0.2, ge=0)
    dppo_mask_high: float = Field(0.2, ge=0)
    kl_tau: float = Field(1e-3, ge=0)
    # No adv_tau: the advantage weight (× tau) is resolved orchestrator-side, so the core just
    # consumes the already-scaled advantage in LossInputs.advantages.


def to_rl_loss_config(term: LossTerm) -> RLLossConfig:
    """Resolve a ``dppo_kl`` primary term into ``RLLossConfig`` (core knobs only; the advantage
    weight's ``tau`` is applied orchestrator-side, not here)."""
    assert isinstance(term.loss, DPPOKLCoreConfig)
    return RLLossConfig(
        name=term.name,
        dppo_mask_low=term.loss.dppo_mask_low,
        dppo_mask_high=term.loss.dppo_mask_high,
        kl_tau=term.loss.kl_tau,
    )


def overlay_terms(losses: list[LossTerm]) -> list[LossTerm]:
    """The additive (non-primary) overlay terms, in list order."""
    return [term for term in losses if not is_primary(term)]
