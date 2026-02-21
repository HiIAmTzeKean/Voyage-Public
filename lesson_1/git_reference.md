---
marp: true
theme: gaia
paginate: true
---

# Git Essential Commands & Workflows

**Technical Onboarding for Data Teams**  
*February 2026*

---

## Installing Git

### macOS

**Method 1: Download from official site**
- Visit git-scm.com and download the installer

**Method 2: Install via Homebrew**
```bash
brew install git
```


### Windows

- Download Git for Windows from git-scm.com
- Run installer and follow wizard
- Verify installation in Git Bash:

```bash
git --version
```


---

## Initial Configuration

Set your name and email (required for commits):

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

These identify your commits in project history.

---

## Repository Setup

**Initialize new repository:**

```bash
git init
```

**Clone existing repository:**

```bash
git clone <repository-url>
```

**Check status:**

```bash
git status
```


---

## Staging and Committing

**Stage files:**

```bash
git add <filename>
# or stage all files
git add .
```

**Commit changes:**

```bash
git commit -m "Descriptive message"
```

**Stage and commit together:**

```bash
git commit -am "Message"
```


---

## Working with Remotes

**Add remote:**

```bash
git remote add origin <url>
```

**View remotes:**

```bash
git remote -v
```

**Push commits:**

```bash
git push origin main
```

**Pull changes:**

```bash
git pull origin main
```


---

## Branching

**Create branch:**

```bash
git branch feature-name
```

**Switch branch:**

```bash
git checkout feature-name
```

**Create and switch in one command:**

```bash
git checkout -b feature-name
```

**List branches:**

```bash
git branch
```


---

## Merging Branches

Switch to target branch, then merge:

```bash
git checkout main
git merge feature-name
```

Combines changes from feature branch into main.

---

## Viewing History

**View commit log:**

```bash
git log
```

**Compact one-line view:**

```bash
git log --oneline
```


---

## Undoing Changes

**Discard unstaged changes:**

```bash
git restore <filename>
```

**Unstage a file:**

```bash
git restore --staged <filename>
```

**View unstaged diff:**

```bash
git diff
```

**View staged diff:**

```bash
git diff --staged
```


---

## Daily Workflow Quick Reference

1. `git pull` - Download from remote
2. `git status` - Check working tree status
3. `git add .` - Stage all changes
4. `git commit -m "Message"` - Commit with message
5. `git push` - Upload to remote

---

## Thank You!

**Remember: Practice makes perfect with Git!**
