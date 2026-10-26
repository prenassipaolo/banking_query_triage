# Architectural changes after actual experiments

**Goal:** retain reasons for replaced approaches, without documenting every code choice.

| Replaced approach | Evidence / reason for change |
| --- | --- |
| Fixed holdout followed by limited three-fold checks | Replaced by five-fold CV over all 10,003 training rows, then full-data refitting. Clearer use of the separate test benchmark. |
| 22 variants with character/phrase features and C refinements | Earlier selected model achieved 90.06% test accuracy. User prioritizes explanation and technique breadth; keep seven fixed comparisons and single words. This is not a claim of improved accuracy. |
| Normalized per-class sigmoid SVM calibration | Replaced by learned multiclass temperature/softmax calibration. Simpler probability interpretation; no controlled superiority claim. |
| Option-heavy feature/model/pipeline factories and model-name fitting branches | Replaced by abstract stage contracts, thin native estimator/vectorizer subclasses, and object composition. `StandardFit`, `CalibratedFit`, and `EarlyStoppingFit` own inner orchestration; only TOML resolution maps names to classes. Native-first inheritance passed clone, parameter, tag, fitting, and serialization checks. |
| Promoting experiment contracts, candidate models and fitting into `src/` | This put unused training behavior in the serving package. Keep that code in notebooks; export the selected fitted state into native sklearn components for a smaller runtime. Conversion preserves exact benchmark predictions and probabilities without retraining. |
| Separate metric JSON files, obsolete model/prediction bundles and 80/20 assignments | Current run has one detailed JSON and one CSV summary. Removed obsolete bulky files; kept compact historical evidence below. |
| `make experiments` alias | Direct module command is sufficient; keep Makefile for checks/tests. |
| Empty LLM arrays permitted by the generation schema | The first live Qwen pilot returned empty arrays in every batch, correctly rejected by application validation. v2 requires exactly the input count and bounds indices for both providers. Checked on a training batch without label examples or semantic test-error tuning. Failed v1 requests remain in append-only evidence. |

Historical evidence: [previous comparison](../artifacts/previous_holdout.csv) and [metadata/CV/confirmation](../artifacts/previous_holdout_metadata.json). These belong to the superseded protocol, not the current model selection.

Raw data, original EDA, and its helper file remain. Current [results](experiment_report.md) use the new protocol.

## Removed integrations

### Ollama Docker startup

The official `ollama/ollama:0.40.11` image repeatedly failed during layer extraction
on this host. containerd restarted after a runtime heap-pointer error; disk space
was available. This was a host container-runtime failure, not an Ollama requirement
or evidence that Ollama cannot run in Docker. Native Ollama successfully ran Qwen
and remains supported, including the experimental banking backend.

Removed the unverified Compose file and `make ollama`/`make ollama-stop` shortcuts.
The attempted configuration remains in commit `fa5583f` for inspection:

```bash
git show fa5583f:compose.yaml
```

Before restoring it, verify image pull/extraction on a healthy Docker host, localhost
binding, cache ownership, the 8,192-token context and one actual prediction. Avoid
running native and container servers on the same port. The native pilot's timings
do not establish container performance; the attempted container had different CPU limits.

### Asynchronous OpenAI Batch

The former prepare/submit/status/collect CLI had offline tests and prepared a local
input file, but **no live asynchronous job was submitted or collected**. No actual
provider failure was established. Removed the unverified workflow, its dedicated
tests and unused prepared files. Grouped synchronous requests remain supported:
several messages in one `/predict` request make one OpenAI call.

The [provider Batch guide](https://developers.openai.com/api/docs/guides/batch)
describes the asynchronous service. The removed implementation is inspectable
without executing it:

```bash
git show fa5583f:notebooks/utils/openai_batch.py
git show fa5583f:notebooks/openai_batch.py
```

To retry, first verify a small paid job end to end: input hash and row mapping,
upload, saved job ID, status transitions, result/error download, unordered custom-ID
mapping, refusal/completion checks and failure-aware scoring. Test duplicate
submission protection and actual model/account support. Report job turnaround
separately from online latency; keep keys out of input files and Git.

### Retained local-model evidence

Qwen's inference path worked after the output-cardinality fix. Its **5.19% test
accuracy** and approximately **99.7% mean reported confidence** are limitations of
this model/prompt combination, not a broken Ollama integration. Keep it experimental
and review its results; the original notebook, prediction CSVs, request ledger and
runtime record remain unchanged. Standalone Ollama and its cached weights were not removed.
