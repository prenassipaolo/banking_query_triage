# Architectural changes after actual experiments

**Goal:** retain reasons for replaced approaches, without documenting every code choice.

| Replaced approach | Evidence / reason for change |
| --- | --- |
| Fixed holdout followed by limited three-fold checks | Replaced by five-fold CV over all 10,003 training rows, then full-data refitting. Clearer use of the separate test benchmark. |
| 22 variants with character/phrase features and C refinements | Earlier selected model achieved 90.06% test accuracy. User prioritizes explanation and technique breadth; keep seven fixed comparisons and single words. This is not a claim of improved accuracy. |
| Normalized per-class sigmoid SVM calibration | Replaced by learned multiclass temperature/softmax calibration. Simpler probability interpretation; no controlled superiority claim. |
| Option-heavy feature/model/pipeline factories and model-name fitting branches | Replaced by abstract stage contracts, thin native estimator/vectorizer subclasses, and object composition. `StandardFit`, `CalibratedFit`, and `EarlyStoppingFit` own inner orchestration; only TOML resolution maps names to classes. Native-first inheritance passed clone, parameter, tag, fitting, and serialization checks. |
| Separate metric JSON files, obsolete model/prediction bundles and 80/20 assignments | Current run has one detailed JSON and one CSV summary. Removed obsolete bulky files; kept compact historical evidence below. |
| `make experiments` alias | Direct module command is sufficient; keep Makefile for checks/tests. |

Historical evidence: [previous comparison](../artifacts/previous_holdout.csv) and [metadata/CV/confirmation](../artifacts/previous_holdout_metadata.json). These belong to the superseded protocol, not the current model selection.

Raw data, original EDA, and its helper file remain. Current [results](experiment_report.md) use the new protocol.
