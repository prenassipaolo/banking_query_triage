"""Pipeline fitting contracts, training policies, and decoded probability models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedGroupKFold

from banking_query_triage.inference import validate_probabilities

from .data import normalize_group
from .pipeline import TextPipeline


@dataclass
class FitResult:
    """Return fitted pipeline state and optional training-only stopping evidence."""

    pipeline: object
    rounds: int | None = None
    history: list = field(default_factory=list)


class BaseFitPolicy(ABC):
    """Orchestrate fitting and inner data boundaries for a component specification."""

    @property
    def settings(self) -> dict:
        """Return policy configuration; policies keep fitted state in FitResult."""
        return vars(self).copy()

    @abstractmethod
    def fit(
        self,
        specification: TextPipeline,
        texts: np.ndarray,
        labels: np.ndarray,
        rounds: int | None = None,
    ) -> FitResult:
        """Fit using only supplied training rows; rounds freezes final boosting fits."""


class StandardFit(BaseFitPolicy):
    """Fit every pipeline stage on all supplied training rows."""

    def fit(self, specification, texts, labels, rounds=None) -> FitResult:
        """Return an ordinary full-training pipeline fit."""
        if rounds is not None:
            raise ValueError("StandardFit does not use boosting rounds")
        pipeline = specification.build()
        pipeline.fit(texts, labels)
        return FitResult(pipeline)


class CalibratedFit(BaseFitPolicy):
    """Calibrate out-of-fold margins while relearning the whole pipeline per fold."""

    def __init__(self, folds: int = 5, seed: int = 42, method: str = "temperature"):
        self.folds = folds
        self.seed = seed
        self.method = method

    def fit(self, specification, texts, labels, rounds=None) -> FitResult:
        """Fit calibration on inner folds, then one base pipeline on all rows."""
        if rounds is not None:
            raise ValueError("CalibratedFit does not use boosting rounds")
        splitter = StratifiedGroupKFold(
            n_splits=self.folds, shuffle=True, random_state=self.seed
        )
        folds = list(splitter.split(texts, labels, [normalize_group(t) for t in texts]))
        pipeline = CalibratedClassifierCV(
            specification.build(), method=self.method, cv=folds, ensemble=False
        )
        pipeline.fit(texts, labels)
        return FitResult(pipeline)


@dataclass
class TrainedModel:
    """Wrap a fitted sklearn estimator with readable labels and training metadata."""

    pipeline: object
    labels: np.ndarray
    fit_seconds: float
    rounds: int | None = None
    history: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def predict_proba(self, texts) -> np.ndarray:
        """Return probabilities in the stored label order."""
        return validate_probabilities(
            self.pipeline.predict_proba(texts), len(texts), len(self.labels)
        )

    def predict(self, texts) -> np.ndarray:
        """Return category names, including for the integer-encoded XGBoost model."""
        return self.labels[self.predict_proba(texts).argmax(axis=1)]

    @property
    def feature_count(self) -> int:
        """Return the feature-matrix width seen by the final native estimator."""
        base = self.pipeline
        if isinstance(base, CalibratedClassifierCV):
            base = base.calibrated_classifiers_[0].estimator
        return int(base.named_steps["model"].n_features_in_)
