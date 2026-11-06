# Starting a new Python project

**Goal:** go from an empty repository to a reproducible Python package with useful
checks and one small working feature.

This guide distills this repository's initial commits. It is a recipe for a new
project, not instructions for reinstalling this one; see the [README](../README.md)
for that. The general setup applies to Python projects; the final section adds
the data/ML workflow used here.

## What the first commits established

| Commit | What changed | Reusable lesson |
| --- | --- | --- |
| `80623db` — Initial commit | README, MIT license, Python `.gitignore` | Establish purpose, licensing and ignored files |
| `138bf44` — init repo | `pyproject.toml`, Python pin, `src/` package, uv lockfile | Make the package installable before adding features |
| `6c58d7f` → `ccafd7f` → `de91ef9` | Greeting edits and a revert | Check the editor/Git workflow; these are practice commits, not required setup |
| `f01e6d0` — add data | Raw inputs, notebook, dependencies, VS Code settings | Add the inputs and tools needed for the first task |
| `cb29892` — add linting makefile | Ruff, Pyright, pydoclint and Make shortcuts | Establish repeatable checks early |
| `ba2c6cd` → `b31f796` | EDA, then extracted notebook helpers | Explore first; extract cohesive helpers when needed |
| `60544e8` → `f5672d3` | Project plan, repository guide, agent preferences | Explain goals, navigation and working conventions |
| `01992fb` — correct small details | Test command repair and notebook tooling | Verify commands against real paths and dependencies |

Inspect the evidence locally with `git show <commit>` or
`git log --reverse --oneline`. The sequence below incorporates later corrections
instead of copying every historical choice.

## 1. Define the first outcome

Write a few lines in the README:

- What problem does the project solve, and for whom?
- What is the smallest useful input → output example?
- How will you check that it works?
- What can wait until later?

For this project, the first useful outcome was understanding the banking messages
and building a local classification baseline. The API and LLM comparisons came
later. A short README checklist is enough initially; add a separate plan only
when it helps coordinate real work.

## 2. Create an installable package

Prerequisites: Git, uv and, for the shortcuts below, Make. These examples use a
POSIX shell. Choose a Python version supported by your intended dependencies.
`3.15` below matches this repository's original pin; substitute your chosen
version rather than treating it as a universal requirement.

Run from the directory that will contain the new project:

```bash
uv init --package --python 3.15 my-project
cd my-project
uv sync
uv run my-project
```

`uv init` creates the package scaffold and normally initializes Git. If starting
from a repository you already cloned, run `uv init --package --python 3.15` inside
it instead. Inspect existing files before editing them.

| File/location | Purpose |
| --- | --- |
| `pyproject.toml` | Package metadata, Python compatibility, dependencies, entry point and tool settings |
| `.python-version` | Python version used for local development |
| `uv.lock` | Resolved dependency versions; commit it for this application workflow |
| `src/my_project/` | Importable package; underscores in Python imports |
| `README.md` | Purpose, setup and first working command |
| `.gitignore` | Excludes local/generated files from Git |

The distribution/command name is `my-project`; the import name is `my_project`.
Replace the generated description and inspect the generated entry point. Keep
the greeting only until a useful command takes its place.

Ensure `.gitignore` covers `.venv/`, `__pycache__/`, `.pytest_cache/`,
`.ruff_cache/`, `.env` and notebook checkpoints. Add an appropriate license if the
project will be shared; this repository chose MIT, but that is a project decision.

## 3. Add dependencies when they become useful

Start with the checks used in this repository:

```bash
uv add --dev pytest ruff pyright pydoclint
```

Add runtime dependencies with `uv add <package>` and development-only tools with
`uv add --dev <package>`. For a notebook-based classification project, an initial
selection could be:

```bash
uv add --dev pandas matplotlib ipykernel scikit-learn
```

Move/add a dependency to the runtime set when `src/` needs it. Do not copy this
repository's full dependency list: HTTP serving, model serialization and extra
model families can wait until used.

Commit both `pyproject.toml` and `uv.lock` after dependency changes. On another
machine, `uv sync --frozen` uses the recorded lockfile. Keep it consistent with
`pyproject.toml`; the lockfile does not replace tracking the required Python
version, data or external services.

## 4. Configure checks and the editor

Add these settings to `pyproject.toml` without replacing the generated package
metadata or build configuration:

```toml
[tool.pyright]
venvPath = "."
venv = ".venv"
extraPaths = ["src"]
typeCheckingMode = "standard"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--import-mode=importlib --strict-markers"
markers = [
    "integration: exercises multiple components or external boundaries",
]

[tool.pydoclint]
style = "google"
arg-type-hints-in-signature = true
arg-type-hints-in-docstring = false
check-return-types = false
require-return-section-when-returning-nothing = false
exclude-private-methods = true
```

Select `.venv/bin/python` in the editor and the same environment as the notebook
kernel. On Windows the interpreter is `.venv\Scripts\python.exe`. Confirm imports
through the installed package:

```bash
uv run python -c "import my_project; print(my_project.__file__)"
```

The early VS Code settings injected `src/` into `PYTHONPATH`; begin with the
installed package and selected interpreter, and add editor settings only to solve
an observed problem.

Create a `Makefile` with literal tabs before command lines:

```makefile
.PHONY: lint lint-fix test

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run pyright src/
	uv run pydoclint src/

lint-fix:
	uv run ruff check --fix .
	uv run ruff format .
	uv run pyright src/
	uv run pydoclint src/

test:
	uv run pytest
```

These are fixed shortcuts; use the underlying command for occasional options.
Ruff checks/formats Python code, Pyright checks types, and pydoclint checks
docstrings. The last two commands cover `src/`; expand scope when useful.

## 5. Build and check one useful behavior

Implement the smallest input → output path and add a test for its actual behavior
or an important failure case. Mirror ownership, for example
`src/my_project/parser.py` → `tests/src/my_project/test_parser.py`. Mark integration
tests with `@pytest.mark.integration` instead of creating separate test trees by
scope.

Once there is a real test:

```bash
make lint
make test
git diff --check
git status --short
```

Pytest exits with code 5 when it collects no tests; an empty scaffold is not a
passing test suite. The historical Makefile referenced nonexistent test paths
and coverage options without installing a coverage plugin. Start with plain
`uv run pytest`; introduce coverage only with the dependency and a useful reason.

Update the README with the working command and checks. If using coding agents,
add a short `AGENTS.md` with project boundaries and required checks; keep it
specific rather than copying every convention from this project.

Inspect the staged diff before committing, especially for credentials, generated
files and data that should not be shared:

```bash
git add README.md .gitignore .python-version pyproject.toml uv.lock src/ Makefile tests/
git diff --cached
git commit -m "Set up package and first working feature"
```

Stage `LICENSE` or `AGENTS.md` separately if added. Split setup and feature work
into separate commits if that makes review easier. Before handing off, have a
fresh reviewer challenge correctness and ask what can be removed or simplified.

## 6. For data/ML projects, grow from exploration

Add these locations only as they become useful:

| Location | What belongs there |
| --- | --- |
| `data/raw/` | Unmodified source inputs; record provenance, schema and reuse restrictions |
| `notebooks/` | Exploration, training and model comparisons |
| `notebooks/utils/` | Cohesive experimental helpers extracted from notebooks |
| `config/` | Adjustable experiment/application settings |
| `artifacts/` | Selected models, metrics and provenance worth retaining |
| `src/my_project/` | Selected inference behavior when a runtime is needed |
| `docs/` | Navigation and rationale that have outgrown the README |

Keep private or large datasets outside Git when appropriate; document how to
obtain them. This repository committed its benchmark inputs, which is not a
requirement for every project.

Inspect schema, missing values, label balance and duplicates before training.
Keep duplicate groups together across splits; fit learned preprocessing only on
training data. Compare a trivial reference with one simple baseline on development
data, freeze the choice, and reserve test data for confirmation.

Extract helpers when notebook repetition or length makes them useful. Record
configuration, seed, data/split identity and metrics with retained results. Promote
selected inference into `src/` only when needed, checking that predictions match.
Add an API, Docker, CI or further model families when there is a concrete use for
them; the initial package and baseline should remain runnable locally.
