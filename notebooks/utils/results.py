"""Small notebook-facing views over the consolidated experiment record."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .evaluation import ReviewPolicy


class ExperimentResults:
    """Load results, compare candidates, inspect folds/errors, and try new messages."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.data = json.loads((self.directory / "results.json").read_text())

    @property
    def selected(self) -> str:
        """Return the candidate chosen using validation only."""
        return self.data["selected"]

    def summary(self) -> pd.DataFrame:
        """Return every candidate's consolidated train/validation/test values."""
        return pd.DataFrame(self.data["summary"]).sort_values(
            "cv_validation_macro_f1_mean", ascending=False
        )

    def comparison(self) -> pd.DataFrame:
        """Show CV mean ± standard deviation beside full-fit training and test metrics."""
        rows = []
        for record in self.summary().to_dict(orient="records"):
            row = {("candidate", "name"): record["candidate"]}
            for phase in ("train", "validation"):
                for metric in ("accuracy", "macro_f1", "log_loss"):
                    prefix = f"cv_{phase}_{metric}"
                    row[(f"CV {phase}", metric)] = (
                        f"{record[prefix + '_mean']:.3f} ± {record[prefix + '_std']:.3f}"
                    )
            for phase in ("final_train", "test"):
                for metric in ("accuracy", "macro_f1", "log_loss"):
                    row[(phase, metric)] = round(record[f"{phase}_{metric}"], 3)
            rows.append(row)
        result = pd.DataFrame(rows)
        result.columns = pd.MultiIndex.from_tuples(result.columns)
        return result

    def folds(self) -> pd.DataFrame:
        """Return individual fold metrics behind the means and standard deviations."""
        return pd.DataFrame(self.data["fold_metrics"])

    def learning_curves(self) -> pd.DataFrame:
        """Return inner-training/early-stopping losses and errors for boosting."""
        return pd.DataFrame(self.data["learning_curves"])

    def errors(self, phase: str = "out_of_fold", limit: int = 10) -> pd.DataFrame:
        """Show selected-model mistakes; out-of-fold errors are the default."""
        frame = pd.read_csv(self.directory / "selected_predictions.csv")
        return (
            frame.loc[(frame.phase == phase) & ~frame.correct]
            .sort_values("score", ascending=False)
            .head(limit)
        )

    def predict_examples(self, texts: list[str]) -> pd.DataFrame:
        """Load the selected bundle and show decoded predictions and review actions."""
        bundle = joblib.load(self.directory / "selected_model.joblib")
        model = bundle["model"]
        probabilities = model.predict_proba(texts)
        policy = ReviewPolicy(bundle["policy"]["threshold"])
        return pd.DataFrame(
            {
                "text": texts,
                "prediction": model.labels[np.argmax(probabilities, axis=1)],
                "score": probabilities.max(axis=1),
                "action": policy.actions(probabilities),
            }
        )
