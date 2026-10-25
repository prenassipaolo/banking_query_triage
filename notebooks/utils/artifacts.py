"""One consolidated results record, a spreadsheet summary, and the selected bundle."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


class ResultStore:
    """Persist experiment evidence without scattering metrics across many JSON files."""

    def __init__(self, directory: Path):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)

    def save_results(self, results: dict) -> None:
        """Write all settings/folds/metrics to JSON and one comparison row per candidate."""
        (self.directory / "results.json").write_text(
            json.dumps(results, indent=2, allow_nan=False) + "\n"
        )
        pd.DataFrame(results["summary"]).to_csv(
            self.directory / "results.csv", index=False
        )

    def save_model(self, model, candidate, config: dict) -> None:
        """Save the selected full-data fit and verify prediction/probability round-trip."""
        path = self.directory / "selected_model.joblib"
        joblib.dump(
            {
                "model": model,
                "candidate": candidate.description(),
                "config": config,
                "policy": {"threshold": config["review_threshold"]},
            },
            path,
            compress=3,
        )
        loaded = joblib.load(path)["model"]
        probes = ["Where is my card?", "I have a problem."]
        np.testing.assert_array_equal(model.predict(probes), loaded.predict(probes))
        np.testing.assert_array_equal(
            model.predict_proba(probes), loaded.predict_proba(probes)
        )

    def save_predictions(self, frames: list[pd.DataFrame]) -> None:
        """Save selected out-of-fold, full-training and test predictions together."""
        pd.concat(frames, ignore_index=True).to_csv(
            self.directory / "selected_predictions.csv", index=False
        )
