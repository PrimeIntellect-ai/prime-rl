# AGENTS.md

## code guidelines

- Avoid tryexcept blocks unless its really necessary.  Its fine that a program fails if something goes wrong. You can add try catch on code that expeclity want to be fault tolerant like adding retry mechanisms or robustness. 

- Do not add unnecessary comments. Epsecially do not try to explain code change that you did in the comments section, do not refer to old code. "the code used to do that but now we are doig this" no this is not a pattern we want. Instead use comments if needed but not mandatory to explain ambiguous code

## Testing

Write tests as plain functions with pytest fixtures. Don't use class-based tests.

## Git

Branch prefixes: `feature/`, `fix/`, `chore/`
