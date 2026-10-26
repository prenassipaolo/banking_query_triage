# Banking query triage

**Goal:** classify banking queries through a locally testable API, backed by a reproducible notebook model.

- [Notebook](notebooks/experiments.ipynb): concrete component composition, saved results, losses and example predictions.
- **[All results: CSV](artifacts/cv_experiments/results.csv)** · [Detailed record](artifacts/cv_experiments/results.json).
- [Report](docs/experiment_report.md): seven fixed candidates; selected TF-IDF/SVM test accuracy **89.42%**.
- [Repository guide](docs/repository_guide.md) · [Decisions](docs/architecture_decisions.md) · [Agent instructions](AGENTS.md).

```bash
uv sync
make api
```

In another terminal:

```bash
make predict
# Optional: make predict PAYLOAD=examples/predict.json PORT=8000
```

Open [interactive API docs](http://127.0.0.1:8000/docs), or use Postman/Bruno:

- Method: `POST`; URL: `http://127.0.0.1:8000/predict`.
- Header: `Content-Type: application/json`.
- Body: the contents of [examples/predict.json](examples/predict.json).
- Response: `model_version` and ordered `predictions`, each with `category`,
  `confidence` and a simulated `route`/`review` action.

`GET /health`, `/ready` and `/model` expose health, readiness and model metadata.
The server loads the fitted model once at startup; prediction requests never train it.
The review threshold is the existing demonstration value of **0.7**, without an
accepted-accuracy guarantee or an out-of-domain detector.

[Serving configuration](config/serving.toml) sets the model directory and request
limits: 32 messages, 4,096 characters each. Invalid inputs return HTTP 422.
`make api PORT=8001 CONFIG=path/to/serving.toml` selects another port/configuration.
Artifact paths are relative to the TOML file. Outside the repository, use
`banking-query-triage --config /absolute/path/serving.toml` or set `BQT_CONFIG`.

The shipped [serving bundle](artifacts/models/tfidf_svm_v1/manifest.json) preserves
the selected model's full fitted pipeline, including vocabulary, IDF values, SVM
weights, calibration and category ordering. Its predictions and probabilities
match the original on all 3,080 benchmark rows. Only load application-owned joblib
artifacts; `uv sync --frozen` reproduces the recorded dependency environment.

```bash
make test
make test ARGS='-m integration'
make test ARGS='-m "not integration"'
make lint
```

Tests mirror `notebooks/` and `src/`; integration is a pytest marker, not a folder.

To fit the selected recipe again, explicitly create a new model version:

```bash
uv run python -m notebooks.train_selected --config config/experiments.toml \
  --output artifacts/models/tfidf_svm_v2 --model-version tfidf_svm_v2
```

Training reads the training CSV and category list only. Existing bundle directories
are never overwritten; update `config/serving.toml` explicitly to serve a new version.
To repeat the historical artifact migration into a new directory:

```bash
uv run python -m notebooks.export_model \
  --output artifacts/models/tfidf_svm_migrated --model-version tfidf_svm_migrated
```

The experiment notebook defaults to saved results. `uv run python -m notebooks.experiments`
reruns all seven comparisons and overwrites their experiment outputs, independently
of the serving bundle. Single-word features only; no parameter sweep.

Deployment and LLM comparisons remain later work.

`src/` contains only serving code. Training, experiment contracts, candidate models
and artifact export stay in `notebooks/`. The serving artifact uses native sklearn
components and the runtime text-cleaning function, with no notebook imports.
See the [component map](docs/repository_guide.md).
