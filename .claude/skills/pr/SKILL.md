---
name: pr
description: Create a pull request following the branch conventions (feat/fix → main squash)
---

# Create Pull Request

Create a pull request for the current branch following the project's branch workflow.

## Base Branch Rules

- `feat/*` or `fix/*` → target `main`

No release branches in this repo.

## Steps

1. Run `git branch --show-current` to identify the current branch
2. Base branch is always `main`
3. Push the current branch to remote if not already pushed (`git push -u origin <branch>`)
4. Create the PR using `gh pr create`:
   - Title: concise, under 70 characters, conventional commit style
   - Body: summary bullets + test plan
   - Do NOT set merge method (ruleset enforces squash)
5. Return the PR URL
