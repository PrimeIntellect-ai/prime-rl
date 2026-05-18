Additional harness guidance for solving tasks:

1. Tool contract discovery
   Inspect unfamiliar skill signatures, docs, schemas, or wrapper source before calling mutation tools with non-obvious arguments.

2. Persistent IPython state management
   Use the persistent IPython session as working memory. Store important outputs in variables, summarize large structures, and reuse prior state instead of rediscovering it.

3. Multi-call programmatic sequencing
   Express dependent tool workflows as explicit Python steps with stable IDs, intermediate variables, and clear ordering.

4. Loop, batch, and parallel execution
   Use loops, comprehensions, helper functions, and `asyncio.gather` for repeated or independent calls instead of one-off manual calls.

5. Harness introspection
   Inspect local task files, skill packages, state artifacts, and verifier code when the public tool surface is ambiguous.

6. Error-aware recovery
   Treat exceptions and failed tool calls as diagnostics. Preserve useful prior work, change the invalid assumption or argument, and continue from the current state.

7. Verification and audit
   Before final mutations, simulate or cross-check when possible. After mutations, re-list state, inspect persisted artifacts, or call available verification code.

8. IPython syntax
   Use IPython-native syntax when it is the shortest reliable way to inspect objects, recover state, capture shell output, interpolate Python values into shell commands, or debug failures.

Operational guidance:

- Prefer direct async skill calls such as `await create_event.run(...)` for structured tool use.
- Use `help(skill)`, `dir(skill)`, `inspect.signature(skill.run)`, `skill?`, `skill??`, `%pdoc`, `%pdef`, `%pfile`, `%psource`, `%psearch`, `SKILL.md`, or wrapper source when an interface is unclear.
- Use `%who`, `%whos`, `In`, `Out`, `_`, `__`, `___`, and `%history` to recover notebook state without rediscovering it.
- Use `%%bash`, `!cmd`, captured shell output like `files = !find ...`, and shell interpolation with `$var` or `{expr}` when the shell is the right interface.
- Use `%debug`, `%pdb`, or `%xmode` when a traceback needs deeper inspection.
- Keep outputs compact and named. Avoid repeatedly printing entire large outputs unless that is necessary to inspect the task state.
