# AGENTS.md

## Code Guidelines

- Avoid try/except blocks unless it's really necessary.  It's fine that a program fails if something goes wrong as this helps us to catch non-obvious bugs and unforeseen side-effects earlier. You can add try catch on code that explicitly aims to be fault tolerant like adding retry mechanisms or explicit and intentional robustness. 

- Do not add unnecessary comments. Especially do not try to explain code change that reflect your work process, do not refer to old code. "The code used to do that but now we are doing this" is not a pattern we want. Instead prefer to use targeted comments sparingly to explain ambiguous code.


## Running code

- All code should be runnable with `uv run` or `uv run <command>`.
- All dependencies should already be installed and pin in the lock file. If not, add it to pyproject.toml and run `uv sync --all-extras` to install it.

## Testing

Write tests as plain functions with pytest fixtures. Don't use class-based tests.

## Git

Branch prefixes: `feature/`, `fix/`, `chore/`
