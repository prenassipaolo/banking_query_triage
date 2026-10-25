# Repository guide

**Goal:** find experiment contracts, implementations, and current evidence.

| Location | Responsibility |
| --- | --- |
| `AGENTS.md` | Working preferences and required fresh-context review |
| `notebooks/eda.ipynb`, `notebooks/utils/utils.py` | Original user exploration/helpers; preserved |
| `notebooks/experiments.ipynb`, `experiments.py` | Object composition and saved-result views; short CLI |
| `config/experiments.toml` | Seven candidates, native estimator settings, explicit fitting policies |
| `notebooks/utils/preprocessing.py` | `BasePreprocessor` → `BasicTextPreprocessor`: raw text → cleaned text |
| `notebooks/utils/features.py` | `BaseFeatureExtractor` → native `CountFeatures`, `TfidfFeatures` |
| `notebooks/utils/models.py` | `BaseMlModel` → native logistic/SVM/NB/forest/XGBoost/dummy classes |
| `notebooks/utils/pipeline.py`, `experiment.py` | `TextPipeline` components; `Experiment` with fit policy/evaluator |
| `notebooks/utils/fitting.py`, `xgboost_training.py` | `BaseFitPolicy` → `StandardFit`, `CalibratedFit`, `EarlyStoppingFit`; decoded `TrainedModel` |
| `notebooks/utils/evaluation.py` | `BaseEvaluator` → `ClassificationEvaluator`; probability validation and fixed `ReviewPolicy` |
| `notebooks/utils/configuration.py` | TOML identifiers → concrete objects; no runtime model-name dispatch |
| `notebooks/utils/data.py` | Inputs, fingerprints, saved grouped folds over all training rows |
| `notebooks/utils/workflow.py`, `validation.py`, `confirmation.py` | Outer CV/comparison, frozen selection, full-training refits, test confirmation |
| `notebooks/utils/artifacts.py`, `results.py` | Save/load checks and notebook-facing result views |
| `tests/unit/`, `tests/integration/` | Contracts/native interoperability, metrics, boundaries, complete workflow |
| `artifacts/cv_experiments/` | Current CSV/JSON, selected full-data model and predictions |
| `artifacts/previous_holdout*` | Compact evidence from the superseded protocol |

```text
BasicTextPreprocessor + CountFeatures/TfidfFeatures + native model
    → TextPipeline + fitting policy + evaluator → Experiment
    → ExperimentRunner: outer CV → frozen choice → full refits → benchmark
    → ResultStore → ExperimentResults
```

Native constructors/get_params/clone remain available. Models only fit feature matrices
and predict encoded labels; policies own inner splits and full-pipeline fitting.
The fitted `TrainedModel` validates probabilities and decodes category names.
Each fit clones components; the original specification stays unfitted. Feature counts
use the native estimator input width, so extractors need no vocabulary attribute.
Saved candidate descriptions record actual object parameters/policy/evaluator settings
separately from the input TOML. Non-JSON settings have descriptive tags; this metadata
is not a reconstruction format.

To extend: implement the relevant abstract contract and compose it directly in Python.
Add a registry entry only when TOML should select it. Extra metrics are callables passed
to `ClassificationEvaluator(labels, extra_metrics={"name": function})`.
Use `ExperimentRunner(config, experiments=[...])` for explicit Python specifications.
Reusable classes can move into `src/` later; refit/reverify bundles when import paths change.

```bash
uv sync
uv run python -m notebooks.experiments
make test
make lint
```

The notebook defaults to saved results; `RUN_EXPERIMENTS=True` reruns training.
CLI reruns overwrite current artifacts; Run All refreshes notebook views/charts.
There is no `make experiments` target. Serving/API/LLM work remains in the
[roadmap](project_plan.md). [Report](experiment_report.md) ·
[Q&A Markdown](experiments_qa.md) · [offline HTML](experiments_qa.html).
