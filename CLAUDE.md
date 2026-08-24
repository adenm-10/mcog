# MCOG

Python 3.11, conda env `tsmc`. Run from repo root.
Core is domain-agnostic (`option_graph/`); Dubins-specific code stays in `domains/`.
wandb: org `aden-mckinney10-university-of-central-florida`, project `mcog`.

## Commands
- Test: `pytest -q`
- Single test: `pytest path::test_name -x`
- Lint: `ruff check .`

## Docs — update in the same change that invalidates them
- `docs/TODO.md` — open tasks, next action each.
- `docs/PROGRESS.md` — dated: question → what ran → result → next.
- `docs/STRUCTURE.md` — tree, one line per module.

## Style
Docstrings ≤3 lines/function, ≤7/module. Inline comments sparse, why-only.
