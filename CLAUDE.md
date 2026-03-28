# training-center - Claude Development Guide

## Project Overview

RL training pipeline for alphachu-volleyball — self-play, evaluation, and model export.

### Goals

- Train Pikachu Volleyball AI agents using SB3 PPO with self-play / PFSP
- Evaluate agents with ELO rating and win-rate metrics
- Export trained models as ONNX for web deployment (world-tournament)

### Position in the Ecosystem

```
alphachu-volleyball/
├── pika-zoo              ← RL environment + physics engine
├── training-center       ← this repo: training pipeline
├── world-tournament      ← Web demo (GitHub Pages)
└── vs-recorder           ← Replay analysis (future)
```

- **Upstream**: pika-zoo (Git tag pinned dependency)
- **Downstream**: world-tournament consumes ONNX models via GitHub Releases

This repo is NOT consumed as a library by other repos. Its output is model artifacts, not code.

## Development Environment

- **Python**: 3.10+
- **Package manager**: uv (`pyproject.toml` + `uv.lock`)
- **Linter/Formatter**: ruff
- **Testing**: pytest

### Commands

```bash
uv sync                  # Install dependencies
uv run ruff check .      # Lint
uv run ruff format .     # Format
uv run pytest            # Test
```

### Dependencies

```toml
[project]
dependencies = [
  "pika-zoo @ git+https://github.com/alphachu-volleyball/pika-zoo@v1.1.0",
]
```

Update this tag when upgrading pika-zoo.

## Code Quality

### ruff Configuration

```toml
[tool.ruff]
target-version = "py310"
line-length = 120

[tool.ruff.lint]
select = ["E", "F", "I", "UP"]
```

## Version Control & Git

### Branch Workflow

No release branch, no version tags — two-tier workflow:

```
feat/* ──(squash merge)──► main
fix/*  ──(squash merge)──►
```

- feat/fix → main: squash merge (PR required)
- No tags — experiment reproducibility is handled by commit hash in W&B metadata

### Commit Convention

[Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>(<scope>): <subject>

feat(train): add PFSP opponent sampling
fix(eval): correct ELO calculation
docs(readme): update pipeline diagram
chore(ci): add ruff lint workflow
```

Types: `feat`, `fix`, `docs`, `chore`, `refactor`, `test`, `ci`

## CI/CD (GitHub Actions)

| Trigger | Action |
|---------|--------|
| PR, push to main | ruff lint, pytest |

Training is not run in CI (requires GPU, long-running).

## Experiment Tracking

### Auto-recorded Metadata

Every training run automatically logs the following to W&B / TensorBoard:

- **commit**: `git rev-parse HEAD`
- **dirty**: whether uncommitted changes exist
- **pika_zoo_version**: pinned pika-zoo Git tag

This links each experiment to its exact code version. No per-experiment tags needed.

**Important**: Only run tracked experiments from commits on `main`. Squash merges discard feature branch commit hashes, so commits recorded during pipeline development would become unreachable.

### Tracking Responsibilities

| What | How |
|------|-----|
| Experiment → code version | commit hash in W&B metadata |
| Deployed model | GitHub Release + W&B run link |

## Artifact Management

### Model Files

- Never committed to Git (tens to hundreds of MB)
- `models/checkpoints/`, `models/exported/` are in `.gitignore`
- Deployed models are exported as ONNX and attached to GitHub Releases

## Hardware Notes

- AMD Ryzen 7 3700X (8C/16T), NVIDIA RTX 2080 Super (8GB)
- Low-dimensional vector obs + MLP policy → CPU (env parallelization) is the bottleneck
- SB3 `SubprocVecEnv` with 8-16 parallel environments
