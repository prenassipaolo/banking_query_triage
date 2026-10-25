"""Compose data, pipeline fitting, five-fold evaluation, full refits, and reporting."""

import platform
import tomllib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from threadpoolctl import threadpool_limits

from .artifacts import ResultStore
from .configuration import resolve_experiments
from .confirmation import FinalEvaluator
from .data import ROOT, ExperimentData, fingerprint
from .experiment import Experiment
from .results import ExperimentResults
from .validation import CrossValidator


class ExperimentRunner:
    """Compare configured techniques; freeze CV selection before opening test labels."""

    def __init__(self, config: dict, experiments: list[Experiment] | None = None):
        self.config = config
        self.data = ExperimentData.load(config)
        self.candidates = (
            experiments
            if experiments is not None
            else resolve_experiments(config, self.data.labels)
        )
        if not self.candidates:
            raise ValueError("At least one experiment is required")
        for candidate in self.candidates:
            if not np.array_equal(candidate.evaluator.labels, self.data.labels):
                raise ValueError(
                    "Experiment evaluator labels must match the dataset label order"
                )
        if len({candidate.name for candidate in self.candidates}) != len(
            self.candidates
        ):
            raise ValueError("Experiment names must be unique")
        self.store = ResultStore(ROOT / config["output_dir"])

    @classmethod
    def from_file(cls, path: Path):
        """Read TOML settings; use explicit components in the constructor to customize."""
        return cls(tomllib.loads(Path(path).read_text()))

    def run(self) -> ExperimentResults:
        """Run five-fold validation then full-data fits, with bounded CPU parallelism."""
        with threadpool_limits(limits=self.config["threads"]):
            payload, probabilities = self._validate()
            final = FinalEvaluator(self.config, self.data, self.candidates, self.store)
            models = final.refit(payload)
            self.store.save_model(
                models[payload["selected"]],
                self._candidate(payload["selected"]),
                self.config,
            )
            self.store.save_results(
                payload
            )  # Selected model/rounds already frozen; test not read.
            final.confirm(payload, models, probabilities[payload["selected"]])
            payload["state"] = "complete"
            self.store.save_results(payload)
        return ExperimentResults(self.store.directory)

    def _validate(self) -> tuple:
        records, summary, curves, probabilities, pooled = [], [], [], {}, {}
        validator = CrossValidator()
        for candidate in self.candidates:
            folds, oof, history = validator.run(candidate, self.data)
            records.extend(folds)
            curves.extend(history)
            summary.append(validator.summarize(folds, candidate))
            probabilities[candidate.name] = oof
            pooled[candidate.name] = candidate.evaluator.from_probabilities(
                self.data.frame.category, oof
            )
        selected = max(
            summary,
            key=lambda row: (
                row["cv_validation_macro_f1_mean"],
                row["cv_validation_accuracy_mean"],
            ),
        )["candidate"]
        payload = {
            "state": "selection_frozen",
            "selected": selected,
            "config": self.config,
            "candidates": [c.description() for c in self.candidates],
            "labels": self.data.labels.tolist(),
            "training_rows": len(self.data.frame),
            "fold_metrics": records,
            "summary": summary,
            "learning_curves": curves,
            "out_of_fold_metrics": pooled,
            "fingerprints": self.data.fingerprints,
            "source_sha256": {
                p.relative_to(ROOT).as_posix(): fingerprint(p)
                for p in sorted(Path(__file__).parent.rglob("*.py"))
            },
            "versions": {
                "python": platform.python_version(),
                "sklearn": sklearn.__version__,
                "xgboost": xgboost.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "joblib": joblib.__version__,
            },
            "notes": {
                "selection": "Mean validation macro F1, then accuracy; no test-based selection or parameter grid.",
                "cv_std": "Sample standard deviation across five fold scores (ddof=1), not a confidence interval.",
                "final_std": "Not estimated for full-training/test: one full-data fit and one fixed test set.",
                "test_history": "This benchmark was inspected in earlier experiments; it is not a new untouched test set.",
                "brier": "Mean sum of squared class-probability errors; multiclass scale [0,2].",
            },
        }
        print(f"CV selection frozen: {selected}", flush=True)
        return payload, probabilities

    def _candidate(self, name: str) -> Experiment:
        return next(
            candidate for candidate in self.candidates if candidate.name == name
        )
