---
marp: true
theme: default
paginate: true
---

# Python Setup Using UV Package Manager

**Technical Onboarding for Data Analysts & Scientists**  
*February 2026*

---

## Prerequisites

### macOS
- Terminal access (built-in)

### Windows
- PowerShell or Command Prompt (built-in)

**Note:** No Python installation required - UV manages it for you!

---

## Installing UV: macOS

Open Terminal and run the standalone installer:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Alternative: Install via Homebrew**

```bash
brew install uv
```


---

## Installing UV: Windows

Open PowerShell as Administrator and run:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Alternative: Install via WinGet**

```bash
winget install --id=astral-sh.uv -e
```


---

## Verify Installation

After installation, restart your terminal and verify:

```bash
uv --version
```

You should see output like:

```
uv 0.10.2
```


---

## Installing Python with UV

UV automatically downloads Python when needed. To install specific version:

```bash
uv python install 3.12
```

Check installed Python versions:

```bash
uv python list
```


---

## Creating a New Project

Initialize a new Python project:

```bash
mkdir my-project
cd my-project
uv init
```

This creates `pyproject.toml` and sets up project structure.

---

## Installing Jupyter

Add Jupyter and essential data science packages:

```bash
uv add jupyter ipykernel notebook
```

Add common analysis libraries:

```bash
uv add pandas numpy matplotlib
```


---

## Setting Up VS Code

Install Python and Jupyter extensions:

- Install via Extensions panel (`Ctrl+Shift+X` / `Cmd+Shift+X`)

Open your project in VS Code:

**From terminal:**

```bash
code .
```

**Or via menu:**
File → Open Folder → Select your project folder

---

## Creating a Jupyter Notebook

In VS Code, press `Ctrl+Shift+P` (`Cmd+Shift+P` on Mac)

Type and select:

```
Jupyter: Create New Blank Notebook
```

Save as `notebook.ipynb`

The kernel name will show in top-right corner.

---

## Testing Your Setup

In a notebook cell, run:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("Setup successful!")
print(f"Pandas version: {pd.__version__}")
```

Press `Shift+Enter` to execute.

---

## Essential UV Commands

- `uv add <package>` - Install package
- `uv remove <package>` - Remove package
- `uv sync` - Sync dependencies
- `uv run python script.py` - Run script

---

## Best Practices

✅ **Always use `uv sync`** before starting work

✅ **Use `uv add`** instead of pip for package installation

✅ **Keep UV updated:**

```bash
uv self update
```


---

## Thank You!

**Questions?**
