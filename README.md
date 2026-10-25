# Banking query triage

**Goal:** compare a few text/model techniques with five-fold validation, then fit on all training data.

- [Notebook](notebooks/experiments.ipynb): concrete component composition, saved results, losses and example predictions.
- **[All results: CSV](artifacts/cv_experiments/results.csv)** · [Detailed record](artifacts/cv_experiments/results.json).
- [Report](docs/experiment_report.md): seven fixed candidates; selected TF-IDF/SVM test accuracy **89.42%**.
- Q&A: [Markdown](docs/experiments_qa.md) · [HTML](docs/experiments_qa.html).
- [Repository guide](docs/repository_guide.md) · [Plan](docs/project_plan.md) · [Decisions](docs/architecture_decisions.md) · [Agent instructions](AGENTS.md).

```bash
uv sync
uv run python -m notebooks.experiments
make test
```

Single words only; no parameter sweep. API/LLM integration is later work.

Replaceable stages implement abstract contracts; native estimators supply fitting and prediction. `Experiment` composes objects, while `ExperimentRunner` owns outer evaluation/refitting. See the [component map](docs/repository_guide.md).
