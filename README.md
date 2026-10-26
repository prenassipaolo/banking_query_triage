# Banking query triage

**Goal:** classify banking queries through a locally testable API with a fitted SVM, OpenAI or an optional local LLM.

- [Notebook](notebooks/experiments.ipynb): concrete component composition, saved results, losses and example predictions.
- **[All results: CSV](artifacts/cv_experiments/results.csv)** · [Detailed record](artifacts/cv_experiments/results.json).
- [Report](docs/experiment_report.md): seven fixed candidates; selected TF-IDF/SVM test accuracy **89.42%**.
- [SVM / OpenAI / Qwen comparison notebook](notebooks/model_comparison.ipynb): matched train/test pilot, confidence diagnostics and batch timings.
- [Call flows and prompts](docs/call_flows.md) · [Local LLMs, Ollama, LangChain, tools and MCP](docs/local_llms.md).
- [Repository guide](docs/repository_guide.md) · [Decisions](docs/architecture_decisions.md) · [Agent instructions](AGENTS.md).

**[Start the local system step by step](docs/api_usage.md#start-the-local-system):**
which services to run, terminal commands, OpenAI key setup, Qwen startup and shutdown.

```bash
uv sync
make api
```

In another terminal:

```bash
make predict
uv run python examples/call_api.py --model openai
# Optional: uv run python examples/call_api.py --model local_llm
```

See the **[API usage guide](docs/api_usage.md)** for copy-and-paste Postman/Bruno
payloads, curl commands, response fields, configuration and troubleshooting.
Open [Swagger UI](http://127.0.0.1:8000/docs) to send requests from your browser.
The server must be running for local documentation links to work.

Requests select `model: "svm"` (default), `"openai"` or `"local_llm"`.
OpenAI defaults to `gpt-6-luna`, using an explicitly marked SVM mock without
`OPENAI_API_KEY`. Set the key before starting the server for real calls, or set
`[openai].mode = "mock"` to stay offline. Local LLM serving is optional and requires
a configured OpenAI-compatible server. `GET /models` reports configuration and
mock status. LLM and mock results always require review. Qwen is enabled in the
serving configuration and requires the separate local inference service; see the
[setup instructions](docs/local_llms.md#4-native-ollama-installation).
Ollama runs natively. The failed Docker startup path and the unverified asynchronous
OpenAI Batch workflow were removed; [failure and retry notes](docs/architecture_decisions.md#removed-integrations)
remain. Grouped synchronous requests are still supported. Qwen remains available
for experiments; its pilot accuracy is too low for automatic banking routing.

The shipped [serving bundle](artifacts/models/tfidf_svm_v1/manifest.json) preserves
the selected model's full fitted pipeline, including vocabulary, IDF values, SVM
weights, calibration and category ordering. Its predictions and probabilities
match the original on all 3,080 benchmark rows. Only load application-owned joblib
artifacts; `uv sync --frozen` reproduces the recorded dependency environment.

```bash
make test
uv run pytest -m integration
uv run pytest -m "not integration"
make lint
```

Tests mirror `notebooks/` and `src/`; integration is a pytest marker, not a folder.
Make targets are fixed shortcuts. Use the underlying CLI for options; `make predict`
is a manual HTTP smoke check that prints a response, without asserting its contents.

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

The [three-model pilot](notebooks/model_comparison.ipynb) uses two messages per
class from each split. `uv run python -m notebooks.model_comparison --full` resumes
all rows using matching checkpointed predictions. The notebook reads saved results
by default. Deployment and production validation remain later work.

`src/` contains only serving code. Training, experiment contracts, candidate models
and artifact export stay in `notebooks/`. The serving artifact uses native sklearn
components and the runtime text-cleaning function, with no notebook imports.
See the [component map](docs/repository_guide.md).
