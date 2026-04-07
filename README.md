# NMF Repository Workflow

This repository contains two separate code tracks:

- `mochi_code/`: base MOCHI model code
- `mochi_code_universal/`: universal variant derived from base MOCHI

Large data artifacts are intentionally excluded from Git tracking.

## Working Rules

Keep base and universal updates as separate commits whenever possible.

- Base model changes: stage only `mochi_code/`
- Universal changes: stage only `mochi_code_universal/`
- Avoid mixed commits unless a shared fix is required

Example:

```bash
# Base model only
git add mochi_code
git commit -m "feat: improve base mochi training loop"

# Universal model only
git add mochi_code_universal
git commit -m "feat: add universal imputer option"
```

## Commit Message Convention

Use a short conventional prefix:

- `feat`: new functionality
- `fix`: bug fix
- `chore`: maintenance/config/docs
- `refactor`: code cleanup without behavior change

Message format:

`<type>: <what changed and why>`

Examples:

- `feat: add universal evaluation script for missing modality`
- `fix: handle empty batch in base dataloader`
- `chore: update ignore rules for local toolkit files`

## Branch Strategy

Recommended branch model:

- `main`: stable branch
- `feature/<topic>`: new work
- `fix/<topic>`: bug fixes

Typical flow:

```bash
git checkout -b feature/base-loss-tuning
# work...
git add mochi_code
git commit -m "feat: tune base loss weighting"
git push -u origin feature/base-loss-tuning
```

Then open a pull request to `main` and merge after review.
