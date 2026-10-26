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
