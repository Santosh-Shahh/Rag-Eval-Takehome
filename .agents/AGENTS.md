# Agent Rules

## Git Autoupdate Rule
- **Rule**: After making any code changes, creating new files, or editing existing files in this workspace, the agent must automatically add, commit, and push those changes to the remote GitHub repository.
- **Commit Message Convention**: Keep commits clean, e.g. "Autoupdate: [Brief summary of change]".
- **Execution**: The agent should run git commands (`git add`, `git commit`, `git push`) immediately in the same turn or as a follow-up command.
