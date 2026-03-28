---
name: cleanup
description: Clean up local branches after a PR has been merged on GitHub
---

# Branch Cleanup

Clean up local branches after a PR has been merged on the remote.

## Steps

1. Run `git fetch --prune` to sync remote state and remove stale tracking branches
2. Switch to `main` if currently on a merged branch
3. Delete local branches whose remote tracking branch is gone
4. Report which branches were deleted
