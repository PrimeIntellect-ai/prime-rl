Additional harness guidance for solving tasks:

1. Tool contract discovery
   Inspect unfamiliar skill signatures, docs, schemas, or wrapper source before calling mutation tools with non-obvious arguments.

   ```python
   import inspect

   skill_names = ["skill1", "skill2", "skill3"]
   for name in skill_names:
       skill = globals()[name]
       print(name, inspect.signature(skill.run))
       print(inspect.getdoc(skill.run) or "")
   ```

2. Persistent IPython state management
   Use the persistent IPython session as working memory. Store important outputs in variables, summarize large structures, and reuse prior state instead of rediscovering it.

   ```python
   records = await list_records.run()
   by_id = {record["id"]: record for record in records}
   candidates = [record for record in records if predicate(record)]
   print([(record["id"], record.get("name")) for record in candidates[:20]])
   ```

3. Multi-call programmatic sequencing
   Express dependent tool workflows as explicit Python steps with stable IDs, intermediate variables, and clear ordering.

   ```python
   async def step(label, coro):
       try:
           result = await coro
           print(f"{label}: OK", result)
           return result
       except Exception as exc:
           print(f"{label}: ERROR {type(exc).__name__}: {exc}")
           raise

   workflow = [
       ("read current state", read_skill.run(...)),
       ("apply intended mutation", mutate_skill.run(...)),
       ("audit updated state", audit_skill.run(...)),
   ]
   for label, coro in workflow:
       await step(label, coro)
   ```

4. Loop, batch, and parallel execution
   Use loops, comprehensions, helper functions, and `asyncio.gather` for repeated or independent calls instead of one-off manual calls.

   ```python
   import asyncio

   checks = await asyncio.gather(
       *(check_skill.run(candidate["id"]) for candidate in candidates)
   )
   valid_candidates = [
       candidate
       for candidate, check in zip(candidates, checks, strict=True)
       if check["ok"]
   ]
   for candidate in valid_candidates:
       await mutation_skill.run(candidate["id"])
   ```

5. Harness introspection
   Inspect local task files, skill packages, state artifacts, and verifier code when the public tool surface is ambiguous.

   ```python
   from pathlib import Path

   task_name = Path("/workspace/.task_name").read_text().strip()
   workspace = Path.cwd()
   paths = [
       Path("/task/rlm-skills") / "skill_name" / "SKILL.md",
       workspace / "tasks" / task_name / "tools.py",
       Path("/workspace/.solver/db_final.json"),
   ]
   for path in paths:
       if path.exists():
           print(f"\n--- {path} ---")
           print(path.read_text()[:4000])
   ```

6. Error-aware recovery
   Treat exceptions and failed tool calls as diagnostics. Preserve useful prior work, change the invalid assumption or argument, and continue from the current state.

   ```python
   args = {"field": "candidate-value"}
   try:
       result = await mutation_skill.run(**args)
   except Exception as exc:
       print(type(exc).__name__, exc)
       current_state = await read_state_skill.run()
       args = repair_args(args, current_state, exc)
       result = await mutation_skill.run(**args)
   ```

7. Verification and audit
   Before final mutations, simulate or cross-check when possible. After mutations, re-list state, inspect persisted artifacts, or call available verification code.

   ```python
   before = await list_state_skill.run()
   plan = build_plan(before)
   assert preconditions_hold(before, plan)

   result = await mutation_skill.run(**plan)
   after = await list_state_skill.run()
   print(audit_result(before, after, result))
   ```

8. IPython syntax
   Use IPython-native syntax when it is the shortest reliable way to inspect objects, recover state, capture shell output, interpolate Python values into shell commands, or debug failures.

   ```python
   # Object inspection.
   skill?
   skill??
   %pdef skill.run
   %pdoc skill.run
   %pfile skill.run
   %psource skill.run
   %psearch *skill*

   # Namespace and history.
   %who
   %whos
   print(In[-3:])
   print(Out)
   %history -n 1-10

   # Shell capture and interpolation into Python.
   pattern = "needle"
   files = !find . -maxdepth 3 -type f
   matches = !grep -RIn {pattern} .
   print(files[:20], matches[:20])

   # Debugging after an exception.
   %xmode Verbose
   %debug
   ```

Operational guidance:

- Prefer direct async skill calls such as `await create_event.run(...)` for structured tool use.
- Use `help(skill)`, `dir(skill)`, `inspect.signature(skill.run)`, `skill?`, `skill??`, `%pdoc`, `%pdef`, `%pfile`, `%psource`, `%psearch`, `SKILL.md`, or wrapper source when an interface is unclear.
- Use `%who`, `%whos`, `In`, `Out`, `_`, `__`, `___`, and `%history` to recover notebook state without rediscovering it.
- Use `%%bash`, `!cmd`, captured shell output like `files = !find ...`, and shell interpolation with `$var` or `{expr}` when the shell is the right interface.
- Use `%debug`, `%pdb`, or `%xmode` when a traceback needs deeper inspection.
- Keep outputs compact and named. Avoid repeatedly printing entire large outputs unless that is necessary to inspect the task state.
