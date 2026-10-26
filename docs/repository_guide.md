# Repository guide

**Goal:** find runtime inference, API behavior, notebook experiments and current evidence.

| Location | Responsibility |
| --- | --- |
| `AGENTS.md` | Working preferences and required fresh-context review |
| `notebooks/eda.ipynb`, `notebooks/utils/utils.py` | Original user exploration/helpers; preserved |
| `notebooks/experiments.ipynb`, `experiments.py` | Object composition and saved-result views; short CLI |
| `config/experiments.toml` | Seven candidates, native estimator settings, explicit fitting policies |
| `notebooks/model_comparison.ipynb`, `model_comparison.py`, `utils/model_comparison.py` | Matched SVM/OpenAI/Qwen evaluation, batch timings and resumable evidence |
| `config/model_comparison.toml` | Pilot sampling, datasets, batch size and comparison API settings |
| `docs/call_flows.md`, `local_llms.md` | Routing/prompt diagrams, metric interpretation, local operations and production concepts |
| `artifacts/model_comparison/` | Pilot predictions/metrics, request evidence, figures and runtime identity |
| `src/banking_query_triage/preprocessing.py` | Stateless text cleaning used by the fitted serving pipeline |
| `src/banking_query_triage/inference.py`, `policy.py`, `service.py` | Validated probabilities, category decoding and route/review decisions |
| `src/banking_query_triage/llm.py` | OpenAI/local LLM inference, validated indexed outputs and explicit offline mock |
| `src/banking_query_triage/artifacts.py`, `settings.py` | Versioned serving bundles, provenance, consistency checks and runtime configuration |
| `src/banking_query_triage/api.py` | HTTP request validation, endpoints, startup and server CLI |
| `notebooks/train_selected.py`, `export_model.py` | Offline selected-model refitting and artifact migration |
| `config/serving.toml`, `examples/` | Local serving limits and a reusable JSON request/client |
| `docs/api_usage.md` | Local run commands, copyable Postman/Bruno payloads and Swagger instructions |
| `notebooks/utils/serving.py` | Offline conversion to native fitted sklearn components and versioned export |
| `notebooks/utils/preprocessing.py`, `features.py`, `models.py`, `pipeline.py`, `fitting.py` | Experiment contracts, candidate models, composition and fitting policies |
| `notebooks/utils/experiment.py`, `xgboost_training.py` | Experiment specifications and training-only XGBoost stopping |
| `notebooks/utils/evaluation.py` | Classification metrics and evaluation contracts |
| `notebooks/utils/configuration.py` | TOML identifiers → concrete objects; no runtime model-name dispatch |
| `notebooks/utils/data.py` | Inputs, fingerprints, saved grouped folds over all training rows |
| `notebooks/utils/workflow.py`, `validation.py`, `confirmation.py` | Outer CV/comparison, frozen selection, full-training refits, test confirmation |
| `notebooks/utils/artifacts.py`, `results.py` | Save/load checks and notebook-facing result views |
| `tests/notebooks/utils/`, `tests/src/banking_query_triage/` | Tests organized by code ownership; `integration` marker selects scope |
| `artifacts/models/tfidf_svm_v1/` | Serving model plus manifest; independent of notebook import paths |
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
Training and its metadata remain in notebooks. Export copies the selected fitted
pipeline state into native sklearn TF-IDF/SVM objects and a stateless runtime
cleaner, with exact prediction/probability checks. The runtime model holds only
the fitted pipeline and category order. Serving does not import notebooks or
experiment-only model families. Experiment fingerprints cover notebook helpers
and runtime code.

```bash
uv sync
uv run python -m notebooks.experiments
make test
make lint
make api
make predict
```

The notebook defaults to saved results; `RUN_EXPERIMENTS=True` reruns training.
CLI reruns overwrite current artifacts; Run All refreshes notebook views/charts.
There is no `make experiments` target.
API requests, configuration and refitting commands are documented in the [README](../README.md).
[Report](experiment_report.md).
