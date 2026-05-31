# Echo Filter — Design

User-pluggable per-token gate that runs once per rollout, on top of the role-level echo baseline, before `interleave_rollout` carves the rollout into training samples.

## Motivation

The current role-based echo mask (`_step_echo_alpha`) is purely structural — it knows what's body vs scaffold (renderer's `is_content`), what role each message has, and which tool a tool-message came from. That's all the renderer can tell us.

It can't tell us that **inside** a tool-role message, `"warning: deprecated flag"` is text the user doesn't want to memorize, or that a `submit_code` response containing a stack trace is noise. The ECHO paper (terminal-task training) needs exactly this — selective masking within otherwise-eligible content based on the actual contents of each tool output.

Filter is the escape hatch: user-provided Python code, called per rollout, returning per-token bool masks that further narrow the role-level echo decision.

## Config shape

```python
class EchoFilterConfig(BaseConfig):
    import_path: str
    """e.g. "my_module.filter_warnings"."""

    kwargs: dict[str, Any] = Field(default_factory=dict)
    """Forwarded to the filter as **kwargs."""


class EchoConfig(BaseConfig):
    system:    SystemRoleEchoConfig    | None = None
    user:      UserRoleEchoConfig      | None = None
    assistant: AssistantRoleEchoConfig | None = None
    tool:      ToolRoleEchoConfig      | None = None

    filter: EchoFilterConfig | None = None   # NEW

    # ... model_validator require_at_least_one_role (unchanged)
```

Single global filter, not per-role. Per-role behaviors are recoverable from inside the filter by branching on `prompt_attribution.message_roles[message_indices[k]]`. Per-role config slots would have been redundant.

TOML:

```toml
[orchestrator.train.env.echo.tool]
alpha = 0.05

[orchestrator.train.env.echo.filter]
import_path = "my_module.filter_warnings"

[orchestrator.train.env.echo.filter.kwargs]
warning_patterns = ["^warning:", "DeprecationWarning"]
```

## Filter signature

```python
def my_filter(rollout: vf.RolloutOutput, **kwargs) -> list[list[bool]]:
    """
    Args:
        rollout: The full rollout, exactly as returned by the env server.
            Use ``rollout["trajectory"]`` for per-step data; each step's
            ``tokens`` carries ``prompt_ids``, ``completion_ids``,
            ``completion_logprobs``, ``prompt_mask``, ``completion_mask``,
            ``prompt_attribution`` (with ``message_roles``,
            ``message_indices``, ``is_content``, ``message_tool_names``).
            Reward, error, stop_condition, metrics, info, example_id all
            available on the rollout itself.
        **kwargs: From EchoFilterConfig.kwargs.

    Returns:
        Per-step masks. Outer length must equal ``len(rollout["trajectory"])``.
        Inner mask for step ``i`` must have length
        ``len(step.tokens.prompt_ids) + len(step.tokens.completion_ids)``.

        Per-token semantics:
            True  → keep the role-level echo decision as-is
            False → drop this position back to no-echo (echo_alpha[k] = None)

        The filter never *adds* echo — it only ever narrows. Positions the
        role-level baseline already said "no echo" stay None regardless of
        what the filter returns at those positions.
    """
```

Prompt vs completion boundary inside a step is `len(step.tokens.prompt_ids)` — the filter has both lists separately, no `prompt_len` arg needed.

## Where it runs

In the orchestrator's per-rollout processing path, **before** `interleave_rollout`:

```
env server  →  vf.RolloutOutput  →  process_rollout (orchestrator.py)
                                          │
                                          ├─→ load filter callable (once, cached on env)
                                          ├─→ if echo_config.filter: call filter(rollout, **kwargs)
                                          ├─→ validate shape
                                          ├─→ pass per-step masks into interleave_rollout
                                          │
                                          └─→ interleave_rollout builds echo_alpha per step,
                                              ANDing the filter mask against the role-level
                                              baseline before merging across steps.
```

Single invocation per rollout. The filter sees the entire trajectory at once — can do cross-step reasoning (e.g. "skip echo on this tool output because a later step shows the task failed downstream"). Per-step or per-message granularity is recoverable inside the filter; per-rollout is not recoverable from finer granularities.

## Composition with role-level baseline

For each step's `echo_alpha: list[float | None]` (length `prompt_len + completion_len`):

```
baseline_role_alpha  filter_mask  →  final echo_alpha
─────────────────────────────────────────────────────
None                 *            →  None    (role gate already said no)
float                False        →  None    (filter narrows it back)
float                True         →  float   (kept; both gates approve)
```

Equivalent to: `echo_alpha[k] = baseline[k] if filter_mask[k] else None`.

The filter cannot turn a role-disabled position into an echo position — that would be additive and would defeat the role-config's intent. Strictly narrowing.

## Renderer dependency

Echo already requires renderer attribution to function — `_step_echo_alpha` bails to all-None when `prompt_attribution` is None. The filter implicitly inherits this: with no attribution, the user can't branch on roles inside their filter anyway, so the use case collapses.

If a non-renderer rollout reaches `process_rollout` and echo is configured with a filter, we skip the filter call (echo is already a no-op for that rollout).

## Error handling — fail loud

| condition | behavior |
|---|---|
| filter import fails (bad `import_path`) | `ImportError` from `import_object`, propagated at env setup time (early) |
| filter raises during call | exception propagates, rollout fails, training loop sees the error |
| filter returns wrong outer length (`!= len(trajectory)`) | `ValueError` from validator inside `process_rollout` |
| filter returns wrong inner length on any step | `ValueError` with step index + expected/actual lengths |
| filter returns non-bool element | `TypeError` |

No silent fallbacks. AGENTS.md "errors should never pass silently" — broken filters should kill the run loudly so the user fixes them.

## Determinism contract

The filter **must be deterministic** given `(rollout, kwargs)`. Reasons:

- DP ranks each see the same rollout but call the filter independently. Non-deterministic output → divergent masks → divergent gradients → silent corruption.
- Replays / debugging / resumption assume the masks are reproducible.

This means **no `random` calls without a seed**, **no `time.time()` thresholds**, **no external state lookups that can change mid-training**. The docstring on `EchoFilterConfig.import_path` will say this loudly.

We don't enforce determinism at runtime (would require running the filter twice on every rollout — too expensive). It's a contract the user signs.

## Performance

Filter is user Python, invoked once per rollout. With ~1024 inflight rollouts and a 50-step training cadence, that's ~20 filter calls/second sustained. Per call, the filter sees a full rollout's worth of data (multi-MB on multi-turn agentic envs).

Caveats to document:

- Heavy regex compilation should be at module load, not per-call (use `re.compile` at import time).
- No GPU. If the user wants tokenizer-aware filtering, instantiate the tokenizer once and reuse.
- Filter runs in the orchestrator process, not in the env server. It's not a parallelism boundary — heavy filters block the orchestrator's `process_rollout` step.

If filters become a bottleneck we could move the call into the env-server's `process_rollout` async path, but that's a future optimization, not v0.

## Wire format / `_step_echo_alpha` integration

Two-step build per trajectory step:

```python
# Step 1 (unchanged): build role-level baseline
baseline = _step_echo_alpha(prompt_attribution, prompt_len, completion_len, echo_config)
#   → list[float | None]

# Step 2 (new): if a filter mask exists for this step, AND it in
if filter_mask is not None:  # provided by process_rollout
    assert len(filter_mask) == len(baseline)
    baseline = [a if (a is not None and filter_mask[k]) else None
                for k, a in enumerate(baseline)]
```

`interleave_rollout` signature gains an optional `filter_masks: list[list[bool]] | None` parameter (parallel to trajectory steps); each step's mask flows to `_step_echo_alpha`'s second stage.

The orchestrator's `process_rollout` is the one place that resolves the filter, calls it, validates the result, and passes per-step slices into `interleave_rollout`.

## Caching the filter callable

`import_object` once per env at orchestrator setup, store as `Env.echo_filter_fn: Callable | None`. The hot path in `process_rollout` reads from `Env.echo_filter_fn` instead of resolving the import each time.

## Naming

- Config class: `EchoFilterConfig` ✓
- `EchoConfig` field: `filter` ✓ (shadows the builtin in type-checker quirks but fine as a field name)
- Callable type alias: `EchoFilterFn = Callable[..., list[list[bool]]]` (optional, just for `Env.echo_filter_fn`'s annotation)

## What's NOT in scope (v0)

- **Built-in regex helpers / common filters.** Function-pointer is fully general. Ship that, see what people write, codify common cases later if patterns emerge.
- **Per-role filter slots.** Single global filter sees the role context via attribution and can branch internally. Adding per-role slots adds config surface for zero new expressiveness.
- **Filter access to the tokenizer directly.** User can import + instantiate at module level. Adding a tokenizer argument would couple the filter API to the trainer's tokenizer choice.
- **Async filters.** Sync only in v0. Can revisit if filters need to call out to other services (which they shouldn't anyway — that breaks determinism).
- **Filter aggregating across rollouts.** One rollout per call. No cross-rollout state.

## Implementation plan

1. Add `EchoFilterConfig` pydantic class.
2. Add `EchoConfig.filter: EchoFilterConfig | None = None`. Update docstring.
3. Resolve + cache filter callable at env setup (one `import_object` per env).
4. Add `filter_masks` optional arg to `interleave_rollout`.
5. Update `_step_echo_alpha` to AND the filter mask against the baseline per step.
6. `process_rollout` (orchestrator.py): if filter is set, call it on the rollout, validate shape, pass per-step slices to `interleave_rollout`.
7. Tests:
   - Filter narrows correctly (drops baseline-True positions when filter returns False, keeps True positions where filter returns True)
   - Filter cannot add echo (baseline-None stays None regardless of filter)
   - Wrong outer length → ValueError
   - Wrong inner length → ValueError
   - Non-bool elements → TypeError
   - Filter exception propagates
   - Determinism contract documented in EchoFilterConfig.import_path docstring
   - Renderer-disabled rollouts skip filter call
