"""Shared probability metrics and a fixed demonstration review policy."""

from abc import ABC, abstractmethod
from time import perf_counter

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.preprocessing import LabelEncoder

from banking_query_triage.inference import validate_probabilities
from banking_query_triage.policy import ReviewPolicy as ServingReviewPolicy


class BaseEvaluator(ABC):
    """Evaluate fitted probability models using one explicit label order."""

    @property
    @abstractmethod
    def labels(self) -> np.ndarray:
        """Return the allowed labels in probability-column order."""

    @property
    @abstractmethod
    def settings(self) -> dict:
        """Return constructor-relevant evaluation settings for provenance."""

    @abstractmethod
    def from_probabilities(self, categories, probabilities: np.ndarray) -> dict:
        """Calculate metrics from truth and class probabilities."""

    @abstractmethod
    def evaluate(self, model, texts, categories) -> tuple:
        """Return shared metrics and probabilities for a fitted model."""


class ClassificationEvaluator(BaseEvaluator):
    """Evaluate all models alike; extra metrics accept encoded truth and probabilities."""

    def __init__(self, labels, extra_metrics: dict | None = None):
        self.encoder = LabelEncoder().fit(labels)
        self._labels = self.encoder.classes_
        self.extra_metrics = extra_metrics or {}

    @property
    def labels(self) -> np.ndarray:
        """Return the sorted label order used by every metric."""
        return self._labels

    @property
    def settings(self) -> dict:
        """Describe labels and extra metrics without serializing encoder internals."""
        return {"labels": self.labels, "extra_metrics": self.extra_metrics}

    def from_probabilities(self, categories, probabilities: np.ndarray) -> dict:
        """Calculate label and probability quality; require all-class probability columns."""
        encoded = self.encoder.transform(categories)
        probabilities = validate_probabilities(
            probabilities, len(encoded), len(self.labels)
        )
        predictions = probabilities.argmax(axis=1)
        correct = int((predictions == encoded).sum())
        auc = (
            roc_auc_score(encoded, probabilities[:, 1])
            if len(self.labels) == 2
            else roc_auc_score(
                encoded, probabilities, multi_class="ovr", average="macro"
            )
        )
        one_hot = np.eye(len(self.labels))[encoded]
        metrics = {
            "rows": len(encoded),
            "errors": len(encoded) - correct,
            "accuracy": float(accuracy_score(encoded, predictions)),
            "macro_f1": float(
                f1_score(encoded, predictions, average="macro", zero_division=0)
            ),
            "log_loss": float(
                log_loss(encoded, probabilities, labels=np.arange(len(self.labels)))
            ),
            "roc_auc_macro": float(auc),
            "brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        }
        metrics.update(
            {
                name: float(function(encoded, probabilities))
                for name, function in self.extra_metrics.items()
            }
        )
        return metrics

    def evaluate(self, model, texts, categories) -> tuple:
        """Return metrics and probabilities, with batched inference time per row."""
        start = perf_counter()
        probabilities = model.predict_proba(texts)
        elapsed = perf_counter() - start
        metrics = self.from_probabilities(categories, probabilities)
        metrics["inference_ms_per_row"] = elapsed * 1000 / len(categories)
        return metrics, probabilities


class ReviewPolicy(ServingReviewPolicy):
    """Measure the notebook review policy while sharing runtime decisions."""

    def evaluate(self, categories, probabilities, labels) -> dict:
        """Return coverage and accepted accuracy; zero coverage has no accepted accuracy."""
        accepted = self.actions(probabilities) == "route"
        predictions = np.asarray(labels)[np.asarray(probabilities).argmax(axis=1)]
        return {
            "coverage": float(accepted.mean()),
            "accepted_accuracy": float(
                (predictions[accepted] == np.asarray(categories)[accepted]).mean()
            )
            if accepted.any()
            else None,
        }
