"""Confidence-based demonstration routing policy."""

import numpy as np


class ReviewPolicy:
    """Use one configured probability threshold; no threshold search is performed."""

    def __init__(self, threshold: float):
        if not 0 <= threshold <= 1:
            raise ValueError("Review threshold must be in [0, 1]")
        self.threshold = threshold

    def actions(self, probabilities) -> np.ndarray:
        """Return simulated route/review actions."""
        return np.where(
            np.asarray(probabilities).max(axis=1) >= self.threshold, "route", "review"
        )
