"""Fitted inference with explicit probability-column and category ordering."""

from dataclasses import dataclass
from typing import Any

import numpy as np


def validate_probabilities(probabilities, rows: int, classes: int) -> np.ndarray:
    """Validate class probabilities and normalize tiny float32 row-sum drift."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.shape != (rows, classes):
        raise ValueError("Model returned invalid probability shape")
    totals = probabilities.sum(axis=1, keepdims=True)
    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or not np.allclose(totals, 1, atol=1e-6, rtol=1e-6)
    ):
        raise ValueError("Model returned invalid probabilities; rows must sum to one")
    return probabilities / totals


@dataclass
class TrainedModel:
    """Predict with the fitted pipeline and decode its fixed category order."""

    pipeline: Any
    labels: np.ndarray

    def predict_proba(self, texts) -> np.ndarray:
        """Return probabilities in the stored label order."""
        return validate_probabilities(
            self.pipeline.predict_proba(texts), len(texts), len(self.labels)
        )

    def predict(self, texts) -> np.ndarray:
        """Return category names in the fitted probability-column order."""
        return self.labels[self.predict_proba(texts).argmax(axis=1)]
