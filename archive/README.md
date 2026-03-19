This folder stores alternate AI-generated coursework runs in a consistent comparison layout.

## Structure

- `model_runs/codex/`
- `model_runs/claude/`
- `model_runs/cursor/`
- `model_runs/copilot/`
- `model_runs/antigravity/`

The repository root remains the canonical `Codex` workflow, and `model_runs/codex/` is a copied snapshot for side-by-side archive comparison.

Each alternate model should keep all of its artifacts inside its own folder under `archive/model_runs/`.

If a model has not been imported yet, its folder can remain empty until its run artifacts are added.
