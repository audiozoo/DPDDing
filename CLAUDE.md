# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Git & GitHub Workflow

This project uses Git for version control with all changes pushed to GitHub.

### Repository Setup (first time only)
If no `.git` directory exists:
```bash
git init
gh repo create DPDDing --public --source=. --remote=origin --push
```

### Committing Changes
After completing any meaningful unit of work:
1. Stage relevant files (never use `git add -A` blindly — avoid committing secrets or large binaries)
2. Write a clean, imperative commit message: short subject line (≤72 chars), blank line, then optional body explaining *why*
3. Commit locally
4. Push to GitHub immediately so there is always a remote backup

```bash
git add <specific-files>
git commit -m "Short imperative subject

Optional body explaining motivation or context."
git push origin main
```

### Commit Message Style
- Subject line: imperative mood, no trailing period ("Add login form", not "Added login form.")
- Keep subjects ≤72 characters
- If the change needs explanation, add a blank line then a short paragraph in the body
- Never use `--no-verify` to skip hooks

### Branching
- Work directly on `main` for small, self-contained changes
- Create a feature branch for larger work: `git checkout -b feature/short-description`
- Merge via PR on GitHub when the branch is ready
