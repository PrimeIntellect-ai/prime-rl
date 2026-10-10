"""One module per benchmarked arm, discovered by file name.

Each module defines:

- `LABEL`: how tables describe the arm.
- `FORWARD_ONLY`: true for an arm with no backward; it is timed on the forward alone.
- `EXECUTED_SLOT_TILE`: `{"fwd": tile, "bwd": tile}`, the slot granularity at which the arm reads each
  query's leading tiles, or `None` to count every padded slot as executed.
- `unavailable_reason() -> str | None`: why the arm cannot run here, or `None`.
- `fwd(q, kv, indices, sinks, scale) -> (out, lse)`: `lse` is log2 and includes the sink, or `None` for an
  arm that does not produce it, which skips its LSE gate.
- `install_compile_counter() -> Callable[[], dict[str, int]]`: patch the arm's compile entry points and
  return a reader of how many compiles (and cache loads) they have done.
"""
