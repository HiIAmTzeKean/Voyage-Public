# Using VS Code to Fork, Clone, Edit, Commit, Push, and Open a Pull Request

This guide walks through the full GitHub workflow using **VS Code**:

1. Fork a repository on GitHub  
2. Clone your fork in VS Code  
3. Create a branch and edit files  
4. Commit and push changes to your fork  
5. Open a Pull Request (PR) back to the original repo

---

## Prerequisites

- A GitHub account
- **Git** installed on your machine
- **VS Code** installed
- (Optional but recommended) **GitHub Pull Requests and Issues** extension in VS Code

---

## 1. Fork the repository on GitHub

1. Go to the original repository page on GitHub (e.g. `https://github.com/original-owner/repo-name`).
2. Click the **Fork** button (top right).
3. Choose your GitHub account as the destination.
4. After GitHub creates the fork, you’ll be at `https://github.com/your-username/repo-name`.

You will clone **your fork**, not the original repo.

---

## 2. Clone your fork using VS Code

### Option A: Clone via VS Code UI (recommended)

1. In a browser, go to your fork:  
   `https://github.com/your-username/repo-name`
2. Click the **Code** button → copy the **HTTPS** URL (e.g. `https://github.com/your-username/repo-name.git`).
3. Open **VS Code**.
4. Open the **Command Palette**:
   - Windows/Linux: `Ctrl+Shift+P`  
   - macOS: `Cmd+Shift+P`
5. Type and select: **Git: Clone**
6. Paste the repo URL, press **Enter**.
7. Choose a local folder where VS Code will store the repo.
8. When VS Code prompts: **“Open cloned repository?”**, click **Open**.

### Option B: Clone via terminal, then open in VS Code

In a terminal:

```bash
git clone https://github.com/your-username/repo-name.git
cd repo-name
code .
```


---

## 3. (Recommended) Add the original repo as `upstream`

This makes it easier to sync your fork later.

1. In VS Code, open the **Terminal** (`Ctrl+` or `Ctrl+Shift+` depending on your setup, or via *Terminal → New Terminal*).
2. Add the upstream remote (replace URLs to match your project):
```bash
git remote add upstream https://github.com/original-owner/repo-name.git
git remote -v
```

You should see both `origin` (your fork) and `upstream` (original).

---

## 4. Create and switch to a feature branch

Avoid committing directly to `main` on your fork.

In the **VS Code terminal**:

```bash
git checkout -b feature/my-change
```

Or in the VS Code UI:

1. Click the branch name in the bottom-left status bar (e.g. `main`).
2. Select **Create new branch…**.
3. Enter a name like `feature/my-change`.

---

## 5. Edit files in VS Code

1. Use the **Explorer** (left sidebar) to open the files you want to modify.
2. Make your changes as needed.
3. Save files (`Ctrl+S` / `Cmd+S`).

You’ll see a **blue M (modified)** indicator next to changed files in the Explorer.

---

## 6. Stage and commit your changes in VS Code

1. Click the **Source Control** icon in the left sidebar (or press `Ctrl+Shift+G` / `Cmd+Shift+G`).
2. Under **Changes**, you’ll see all modified files.

### Stage files

- Hover over a file and click the **+** icon to stage it,
or
- Click the **+** next to **Changes** to stage all files.

Staged files move into the **Staged Changes** section.

### Commit

1. At the top of the Source Control view, type a commit message (e.g. `Add new feature X`).
2. Click the checkmark icon (✓) **Commit**.

Alternatively, from the terminal:

```bash
git add .
git commit -m "Add new feature X"
```


---

## 7. Push your branch to your fork (`origin`)

If this is the first push of this branch:

1. In the bottom-left, confirm you’re on `feature/my-change`.
2. In the Source Control view, click **…** (More) → **Push**
or use the top-right **Sync / Push** icon.

VS Code often shows a **Publish Branch** button the first time:

- Click **Publish Branch** to push to your fork as `origin/feature/my-change`.

From the terminal:

```bash
git push -u origin feature/my-change
```

The `-u` sets the upstream so later you can just `git push`.

---

## 8. Open a Pull Request (PR) on GitHub

After pushing, GitHub will know about your branch on your fork.

### Option A: From the GitHub UI

1. Go to your fork on GitHub:
`https://github.com/your-username/repo-name`
2. You may see a banner: **“Compare \& pull request”** for your branch—click it.
If not:
    - Go to the **Pull requests** tab.
    - Click **New pull request**.
    - Set:
        - **base repository**: `original-owner/repo-name`
        - **base branch**: usually `main` (or whatever the project uses)
        - **head repository**: `your-username/repo-name`
        - **compare**: `feature/my-change`
3. Review the diff.
4. Add a PR title and description explaining your changes.
5. Click **Create pull request**.

### Option B: From VS Code (with GitHub extension)

If you have **GitHub Pull Requests and Issues** extension:

1. After pushing, VS Code may show a notification to create a PR.
2. Click **Create Pull Request**.
3. Fill in the title/description and confirm the base and head branches.
4. Submit the PR from within VS Code.

---

## 9. Updating your branch if requested

If maintainers request changes:

1. Make further edits in VS Code.
2. Stage and commit again (as in steps 5–6).
3. Push to the same branch:
```bash
git push
```

The PR updates automatically with your new commits.

---

## 10. Syncing your fork later (optional but useful)

To keep your fork up to date with the original repo:

```bash
# Fetch new changes from upstream
git fetch upstream

# Update your local main
git checkout main
git merge upstream/main   # or `git rebase upstream/main` if the project prefers
git push origin main
```
