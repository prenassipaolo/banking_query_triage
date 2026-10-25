"""Visible experiment objects, independent of TOML names and outer evaluation."""

import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from sklearn.preprocessing import LabelEncoder

from .evaluation import BaseEvaluator
from .fitting import BaseFitPolicy, TrainedModel
from .pipeline import TextPipeline


def _json_setting(value):
    """Describe native settings as strict JSON; this is not a reconstruction format."""
    if isinstance(value, np.ndarray):
        return _json_setting(value.tolist())
    if isinstance(value, np.generic):
        return _json_setting(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_setting(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_setting(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_float": str(value)}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if callable(value) and hasattr(value, "__qualname__"):
        return {"python": f"{value.__module__}.{value.__qualname__}"}
    return {
        "python": f"{type(value).__module__}.{type(value).__qualname__}",
        "repr": repr(value),
    }


def _component_description(component) -> dict:
    """Describe concrete identity and the native constructor parameter values."""
    return {
        "class": f"{type(component).__module__}.{type(component).__qualname__}",
        "parameters": _json_setting(component.get_params(deep=False)),
    }


@dataclass(frozen=True)
class Experiment:
    """Compose actual components, fitting policy, and evaluation behavior."""

    name: str
    pipeline: TextPipeline
    fit_policy: BaseFitPolicy
    evaluator: BaseEvaluator

    def description(self) -> dict:
        """Describe actual components and settings separately from input configuration."""
        return {
            "name": self.name,
            "features": self.pipeline.features.name,
            "model": self.pipeline.model.name,
            "fit_policy": type(self.fit_policy).__name__,
            "pipeline": {
                "preprocessor": _component_description(self.pipeline.preprocessor),
                "features": _component_description(self.pipeline.features),
                "model": _component_description(self.pipeline.model),
            },
            "fitting": {
                "class": f"{type(self.fit_policy).__module__}.{type(self.fit_policy).__qualname__}",
                "settings": _json_setting(self.fit_policy.settings),
            },
            "evaluation": {
                "class": f"{type(self.evaluator).__module__}.{type(self.evaluator).__qualname__}",
                "settings": _json_setting(self.evaluator.settings),
            },
        }

    def fit(self, texts, categories, rounds: int | None = None) -> TrainedModel:
        """Fit only the supplied rows using the explicit policy and label contract."""
        encoder = LabelEncoder().fit(self.evaluator.labels)
        encoded = encoder.transform(categories)
        if not np.array_equal(np.unique(encoded), np.arange(len(encoder.classes_))):
            raise ValueError("Training must contain every allowed category")
        start = perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fitted = self.fit_policy.fit(
                self.pipeline, np.asarray(texts), encoded, rounds=rounds
            )
        if not callable(getattr(fitted.pipeline, "predict_proba", None)):
            raise TypeError(
                "Classification evaluation requires predict_proba; "
                "use CalibratedFit for a margin-only estimator"
            )
        return TrainedModel(
            fitted.pipeline,
            encoder.classes_,
            perf_counter() - start,
            fitted.rounds,
            fitted.history,
            [str(w.message) for w in caught],
        )
