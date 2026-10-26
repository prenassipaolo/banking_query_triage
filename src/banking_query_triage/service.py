"""HTTP-independent composition of fitted inference and review decisions."""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .artifacts import ModelBundle
from .policy import ReviewPolicy


@dataclass(frozen=True)
class Prediction:
    """One category, backend confidence and simulated routing decision."""

    category: str
    confidence: float
    action: Literal["route", "review"]


@dataclass(frozen=True)
class ServingInfo:
    """Public identity, confidence semantics and availability of a serving backend."""

    model: str
    model_version: str
    candidate: str
    labels: list[str]
    review_threshold: float | None
    confidence_kind: str
    mocked: bool = False
    available: bool = True


class TriageService:
    """Keep the loaded model and policy shared across prediction requests."""

    def __init__(self, bundle: ModelBundle):
        self.bundle = bundle
        self.policy = ReviewPolicy(bundle.manifest.review_threshold)

    @property
    def info(self) -> ServingInfo:
        """Return the fitted artifact identity and calibrated confidence semantics."""
        manifest = self.bundle.manifest
        return ServingInfo(
            model="svm",
            model_version=manifest.model_version,
            candidate=manifest.candidate,
            labels=manifest.labels,
            review_threshold=manifest.review_threshold,
            confidence_kind="calibrated_probability",
        )

    def predict(self, messages: list[str]) -> list[Prediction]:
        """Predict an ordered batch using one probability calculation.

        Args:
            messages: Nonempty list of nonempty messages.

        Returns:
            Predictions in the same order as the supplied messages.

        Raises:
            ValueError: The batch is empty or the model returns invalid probabilities.
        """
        if not messages:
            raise ValueError("At least one message is required")
        probabilities = self.bundle.model.predict_proba(messages)
        indices = np.argmax(probabilities, axis=1)
        actions = self.policy.actions(probabilities)
        return [
            Prediction(
                category=str(self.bundle.model.labels[index]),
                confidence=float(probabilities[row, index]),
                action="route" if actions[row] == "route" else "review",
            )
            for row, index in enumerate(indices)
        ]
