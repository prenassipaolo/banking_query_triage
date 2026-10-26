"""Offline export of the selected model into a notebook-independent serving bundle."""

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC

from banking_query_triage.artifacts import (
    ModelBundle,
    ModelManifest,
    fingerprint,
    load_bundle,
    runtime_versions,
)
from banking_query_triage.inference import TrainedModel
from banking_query_triage.policy import ReviewPolicy
from banking_query_triage.preprocessing import clean_messages

from .features import TfidfFeatures
from .models import LinearSvmModel
from .preprocessing import BasicTextPreprocessor


def _native_pipeline(pipeline):
    """Replace experiment-only subclasses with native sklearn state, without fitting."""
    cleaner = pipeline.named_steps["preprocessing"]
    features = pipeline.named_steps["features"]
    classifier = pipeline.named_steps["model"]
    if not (
        isinstance(cleaner, BasicTextPreprocessor)
        and isinstance(features, TfidfFeatures)
        and isinstance(classifier, LinearSvmModel)
    ):
        raise TypeError("Serving export supports only the selected TF-IDF/SVM pipeline")
    native_features = TfidfVectorizer(**features.get_params(deep=False))
    native_features.__dict__.update(features.__dict__)
    native_model = LinearSVC(**classifier.get_params(deep=False))
    native_model.__dict__.update(classifier.__dict__)
    pipeline.set_params(
        preprocessing=FunctionTransformer(
            clean_messages, kw_args={"lowercase": cleaner.lowercase}, validate=False
        ),
        features=native_features,
        model=native_model,
    )
    return pipeline


def to_serving_model(trained) -> TrainedModel:
    """Copy the selected fitted state into native sklearn components for inference."""
    calibrated = deepcopy(trained.pipeline)
    if not isinstance(calibrated, CalibratedClassifierCV):
        raise TypeError(
            "Serving export requires the selected calibrated TF-IDF/SVM model"
        )
    calibrated.estimator = _native_pipeline(calibrated.estimator)
    for fitted in calibrated.calibrated_classifiers_:
        fitted.estimator = _native_pipeline(fitted.estimator)
    return TrainedModel(calibrated, trained.labels.copy())


def save_bundle(
    directory: Path,
    model: TrainedModel,
    *,
    model_version: str,
    candidate: str,
    threshold: float,
    provenance: dict[str, Any],
) -> ModelBundle:
    """Write a new versioned directory without replacing an existing model.

    Args:
        directory: New artifact directory; must not already exist.
        model: Full fitted preprocessing, features, calibration and estimator state.
        model_version: Explicit model release identifier.
        candidate: Selected experiment identifier.
        threshold: Probability threshold used by the demonstration review policy.
        provenance: Frozen configuration, data fingerprints and source information.

    Returns:
        Reloaded, validated serving bundle.

    Raises:
        FileExistsError: The requested version directory already exists.
    """
    if directory.exists():
        raise FileExistsError(directory)
    ReviewPolicy(threshold)
    directory.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=directory.parent, prefix=".export-") as temporary:
        staged = Path(temporary) / "bundle"
        staged.mkdir()
        path = staged / "model.joblib"
        joblib.dump(model, path, compress=3)
        manifest = ModelManifest(
            model_version=model_version,
            candidate=candidate,
            labels=model.labels.tolist(),
            review_threshold=threshold,
            model_sha256=fingerprint(path),
            versions=runtime_versions(),
            provenance=provenance,
        )
        (staged / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n")
        bundle = load_bundle(staged)
        probes = ["Where is my card?", "I have a problem."]
        np.testing.assert_array_equal(
            model.predict_proba(probes), bundle.model.predict_proba(probes)
        )
        np.testing.assert_array_equal(
            model.predict(probes), bundle.model.predict(probes)
        )
        staged.rename(directory)
    return bundle
