"""Refit frozen candidates on all training rows, then confirm on the benchmark."""

import numpy as np
import pandas as pd

from .data import ROOT, ExperimentData, normalize_group
from .evaluation import ReviewPolicy


class FinalEvaluator:
    """Keep final fits and test reporting separate from model selection."""

    def __init__(self, config, data, candidates, store):
        self.config = config
        self.data = data
        self.store = store
        self.policy = ReviewPolicy(config["review_threshold"])
        self.candidates = candidates

    def refit(self, payload: dict) -> dict:
        models = {}
        for row, candidate in zip(payload["summary"], self.candidates):
            fold_rounds = [
                r["rounds"]
                for r in payload["fold_metrics"]
                if r["candidate"] == candidate.name
                and r["phase"] == "train"
                and r["rounds"] is not None
            ]
            rounds = int(np.median(fold_rounds)) if fold_rounds else None
            model = candidate.fit(
                self.data.frame.text, self.data.frame.category, rounds=rounds
            )
            metrics, _ = candidate.evaluator.evaluate(
                model, self.data.frame.text, self.data.frame.category
            )
            row.update({f"final_train_{k}": v for k, v in metrics.items()})
            row.update(
                {
                    "final_fit_seconds": model.fit_seconds,
                    "final_features": model.feature_count,
                    "final_rounds": model.rounds,
                    "final_warnings": model.warnings,
                }
            )
            models[candidate.name] = model
            print(f"Full-training refit complete: {candidate.name}", flush=True)
        return models

    def confirm(self, payload: dict, models: dict, selected_oof: np.ndarray) -> None:
        test = ExperimentData.read_frame(
            ROOT / self.config["test_file"], self.data.labels
        )
        selected_frames = [
            self._predictions(self.data.frame, selected_oof, "out_of_fold")
        ]
        for row, candidate in zip(payload["summary"], self.candidates):
            model = models[candidate.name]
            metrics, probabilities = candidate.evaluator.evaluate(
                model, test.text, test.category
            )
            row.update({f"test_{k}": v for k, v in metrics.items()})
            for metric in metrics:
                if metric != "rows":
                    row[f"test_{metric}_std"] = None
                    row[f"final_train_{metric}_std"] = None
            row.update(
                {
                    f"test_{k}": v
                    for k, v in self.policy.evaluate(
                        test.category, probabilities, self.data.labels
                    ).items()
                }
            )
            if candidate.name == payload["selected"]:
                train_probabilities = model.predict_proba(self.data.frame.text)
                selected_frames.extend(
                    [
                        self._predictions(
                            self.data.frame, train_probabilities, "full_training"
                        ),
                        self._predictions(test, probabilities, "test"),
                    ]
                )
                no_overlap = ~test.text.map(normalize_group).isin(
                    set(self.data.frame.text.map(normalize_group))
                )
                payload["test_excluding_overlap"] = (
                    candidate.evaluator.from_probabilities(
                        test.category[no_overlap], probabilities[no_overlap]
                    )
                )
                payload["test_overlap_rows"] = int((~no_overlap).sum())
        self.store.save_predictions(selected_frames)

    def _predictions(self, frame, probabilities, phase: str) -> pd.DataFrame:
        encoded = np.searchsorted(self.data.labels, frame.category)
        predictions = self.data.labels[probabilities.argmax(axis=1)]
        return pd.DataFrame(
            {
                "phase": phase,
                "row_id": frame.index,
                "text": frame.text,
                "category": frame.category,
                "prediction": predictions,
                "score": probabilities.max(axis=1),
                "true_probability": probabilities[np.arange(len(frame)), encoded],
                "correct": predictions == frame.category,
                "action": self.policy.actions(probabilities),
            }
        )
