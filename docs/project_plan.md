# Learning roadmap

**Goal:** an understandable text classifier, then a fake API, then LLM comparisons.

| Stage | Status |
| --- | --- |
| Explore raw data | Existing EDA; preserve user experiments |
| Compare techniques | Seven fixed candidates, five grouped folds across all training |
| Evaluate honestly | Train/validation means + SD; log loss, F1, accuracy, AUC and Brier |
| Fit final models | All 10,003 training rows; select by CV before test |
| Inspect results | One CSV + detailed JSON + notebook views |
| Compose techniques | Abstract contracts, native concrete estimators, separate fitting policies |
| Serve predictions | Next: selected reusable components in `src/`, thin API/client |
| Add LLMs | Later: versioned prompts, same evaluation protocol |

```text
CV comparison → freeze choice → full-data fit → benchmark confirmation
             → saved model → shared inference → fake API
```

- Single-word counts and TF-IDF; no character/phrase features or tuning grid.
- XGBoost stops using an inner training-only subset; the outer fold stays independent.
- Test has already been inspected in this learning project; do not call it pristine.
- Macro groups remain a possible manual mapping; no hierarchy/clustering required initially.
- Compose concrete objects through abstract contracts; native model classes own estimator operations, policies own inner fitting, and the runner owns outer evaluation.
- Keep short notebooks, clear metrics and a small dependency set.
- Use a fresh-context reviewer after each implementation cycle; challenge unnecessary code.

[Results](experiment_report.md) · [Repository guide](repository_guide.md) · [Q&A](experiments_qa.html)
