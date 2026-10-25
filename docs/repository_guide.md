# Repository guide

Start: `AGENTS.md` → relevant file/config/test. Verify docs against actual code.

| Location | Purpose | Status |
| --- | --- | --- |
| `AGENTS.md` | Working preferences for agents | Exists |
| `docs/project_plan.md` | Concise learning roadmap | Exists |
| `notebooks/eda.ipynb`, `notebooks/utils/` | Exploration and experimental helpers | Exist |
| `data/raw/` | Train/test CSVs and category list | Exists |
| `src/banking_query_triage/` | Importable Python package; currently a placeholder | Exists |
| `pyproject.toml`, `uv.lock`, `Makefile` | Packaging, dependencies, development commands | Exist |
| `config/` | TOML experiment/application settings | Planned |
| `data/processed/`, `data/features/` | Saved splits/derived data; optional feature caches | Planned, as needed |
| `tests/unit/`, `tests/integration/` | Focused checks | Planned |
| `artifacts/runs/` | Saved models, resolved settings, metrics | Planned |
| `prompts/` | Versioned prompts for later LLM comparisons | Planned |

Package name: `triage` is the proposed shorter replacement for
`banking_query_triage`; the rename has not been applied.

```text
repository = code + notebooks + data + configuration + tests
Python package = reusable training/inference code inside src/
API = a thin HTTP entry point using that package
```

- Create modules when needed; avoid an empty scaffold for every future step.
- Training, saved-model inference, and HTTP serving are not implemented yet.
- Docker can run the API/integration checks; add a Dockerfile when useful.
- Use bind mounts or named volumes only when they simplify data/model access.
- Current `make test` references `tests/functional/` and coverage options;
  align paths/dependencies when adding integration tests.
- Add decision notes only after meaningful alternatives have been tried.
