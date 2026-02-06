# AGENTS.md

## Code Guidelines

- Avoid try/except blocks unless it's really necessary.  It's fine that a program fails if something goes wrong as this helps us to catch non-obvious bugs and unforeseen side-effects earlier. You can add try catch on code that explicitly aims to be fault tolerant like adding retry mechanisms or explicit and intentional robustness. 

- Do not add unnecessary comments. Especially do not try to explain code change that reflect your work process, do not refer to old code. "The code used to do that but now we are doing this" is not a pattern we want. Instead prefer to use targeted comments sparingly to explain ambiguous code.


## Zen of Python
remember the zen of python when writing code.

```
Beautiful is better than ugly.
Explicit is better than implicit.
Simple is better than complex.
Complex is better than complicated.
Flat is better than nested.
Sparse is better than dense.
Readability counts.
Special cases aren't special enough to break the rules.
Although practicality beats purity.
Errors should never pass silently.
Unless explicitly silenced.
In the face of ambiguity, refuse the temptation to guess.
There should be one-- and preferably only one --obvious way to do it.
Although that way may not be obvious at first unless you're Dutch.
Now is better than never.
Although never is often better than *right* now.
If the implementation is hard to explain, it's a bad idea.
If the implementation is easy to explain, it may be a good idea.
Namespaces are one honking great idea -- let's do more of those!
```

## Running code

- All code should be runnable with `uv run` or `uv run <command>`.
- All dependencies should already be installed and pin in the lock file. If not, add it to pyproject.toml and run `uv sync --all-extras` to install it.

## CLI Usage

- Config files use `@` syntax: `uv run sft @ path/to/config.toml`
- For multi-GPU with torchrun: `uv run torchrun --nproc-per-node 2 src/prime_rl/trainer/sft/train.py @ path/to/config.toml`
- Boolean flags don't need `true`: use `--model.optim_cpu_offload` not `--model.optim_cpu_offload true`, use `--no-model.optim_cpu_offload` to pass False.
- Override config values with CLI flags: `--model.name Qwen/Qwen3-0.6B --training.max_steps 100`

## Testing

Write tests as plain functions with pytest fixtures. Don't use class-based tests.

## Git

Branch prefixes: `feature/`, `fix/`, `chore/`

## Releases

When preparing release notes:

1. **Style reference**: check the previous release (`gh release list --limit 1` then `gh release view <tag>`) to match the tone and formatting.
2. **Gather changes**: use `git log <last-tag>..origin/main --oneline --no-merges` to list all commits since the last release.
2. **Check for new commits**: always `git fetch origin main` and re-check right before publishing, since PRs may have been merged while drafting.
3. **Structure**: organize notes into numbered highlight sections (`# 1.`, `# 2.`, ...), then `# Breaking Changes`, `# Bug Fixes`, and `# Misc`.
4. **Highlights**: group related PRs under a single highlight. Use `##` subsections when a highlight contains multiple items (e.g. Performance & Parallelism). Keep the top highlights for the most impactful user-facing features.
5. **Config examples**: when referencing TOML config, verify the exact field names against the actual code or docs — don't guess.
6. **Links**: use clickable links for docs (`[docs/foo.md](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/docs/foo.md)`) and PR references (`[#1234](https://github.com/PrimeIntellect-ai/prime-rl/pull/1234)`).
7. **Contributors**: list all contributors ranked by number of commits, using their GitHub `@username`. Get usernames via the GitHub API, not git author names (which can be inconsistent).
8. **Draft first**: always create releases as `--draft` first, iterate on content, then publish.
