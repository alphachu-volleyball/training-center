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

### Package Structure

```
src/training_center/
├── elo.py                      # ELO rating calculation
├── env_factory.py              # Wrapper chain construction + opponent policy swap
├── game.py                     # Player, play_game, make_player
├── metadata.py                 # Experiment metadata (commit, dirty, pika-zoo version)
├── metrics.py                  # Per-game frame metrics (entropy, court control, etc.)
├── opponent_pool.py            # PFSP opponent pool with sliding-window win-rate
└── scripts/
    ├── train_baseline.py       # Baseline PPO training, fixed opponent (SubprocVecEnv)
    ├── train_selfplay.py       # Self-play with PFSP + curriculum (DummyVecEnv)
    └── evaluate.py             # Round-robin ELO evaluation (p1 pool × p2 pool)
```

### CLI Commands

```bash
uv run train-baseline         # Baseline PPO training (fixed opponent)
uv run train-selfplay         # Self-play training (PFSP + curriculum)
uv run evaluate               # Round-robin ELO evaluation
```

### Wrapper Chain

```
PikachuVolleyballEnv (PettingZoo)
  → SimplifyAction (18 → 13 relative actions)
  → SimplifyObservation (mirror player_2 x-axis, optional)
  → NormalizeObservation ([0, 1])
  → RewardShaping (optional)
  → ConvertSingleAgent (gym.Env for SB3)
```

`env_factory.make_env()` builds this chain. `set_opponent_policy()` swaps opponents in-place for self-play.

### VecEnv Strategy

- **train_ppo**: `SubprocVecEnv` — fixed opponent, maximize CPU parallelism
- **train_selfplay**: `DummyVecEnv` — opponent policy must be swapped in-process each iteration

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
  "pika-zoo @ git+https://github.com/alphachu-volleyball/pika-zoo@v1.3.0",
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

Training is not run in CI (long-running).

## Experiment Tracking

For the full list of tracked metrics, see [README.md § Tracked Metrics](README.md#tracked-metrics).

### Auto-recorded Metadata

Every training run automatically logs the following to W&B:

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

## W&B MCP Server

Claude Code can query W&B experiment data (runs, metrics, sweeps, artifacts) via the [wandb-mcp-server](https://github.com/wandb/wandb-mcp-server).

### Setup

Create `.mcp.json` in the project root (already in `.gitignore`):

```json
{
  "mcpServers": {
    "wandb": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/wandb/wandb-mcp-server.git",
        "wandb_mcp_server"
      ],
      "env": {
        "WANDB_API_KEY": "<your-api-key>"
      }
    }
  }
}
```

Get your API key from https://wandb.ai/authorize.

### Usage

The MCP server exposes GraphQL queries against the W&B API. Example:

```graphql
query {
  project(name: "alphachu-volleyball", entityName: "ootzk") {
    runs(first: 5) {
      edges { node { name displayName config summaryMetrics } }
    }
  }
}
```

This enables Claude Code to inspect experiment results, compare runs, and download artifacts without leaving the conversation.

## Artifact Management

### Model Files

- Never committed to Git (tens to hundreds of MB)
- Training outputs go under `experiments/` (in `.gitignore`)
- All models are also uploaded as W&B Artifacts
- Deployed models are exported as ONNX and attached to GitHub Releases

## Hardware Notes

- AMD Ryzen 7 3700X (8C/16T), NVIDIA RTX 2080 Super (8GB)
- Low-dimensional vector obs + MLP policy → CPU (env parallelization) is the bottleneck
- SB3 `SubprocVecEnv` with 8-16 parallel environments
