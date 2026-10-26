"""Offline export preserves fitted state and publishes only valid new versions."""

import joblib
import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from banking_query_triage.artifacts import load_bundle
from notebooks.utils.serving import save_bundle, to_serving_model

pytestmark = pytest.mark.integration


def test_save_refuses_to_replace_existing_bundle(bundle_dir):
    bundle = load_bundle(bundle_dir)
    original = (bundle_dir / "model.joblib").read_bytes()
    with pytest.raises(FileExistsError):
        save_bundle(
            bundle_dir,
            bundle.model,
            model_version="new",
            candidate="fixture",
            threshold=0.7,
            provenance={},
        )
    assert (bundle_dir / "model.joblib").read_bytes() == original


def test_failed_export_does_not_leave_partial_version(bundle_dir, tmp_path):
    bundle = load_bundle(bundle_dir)
    bundle.model.pipeline.named_steps["model"].classes_ = np.array([1, 0])
    output = tmp_path / "invalid_version"
    with pytest.raises(ValueError, match="labels"):
        save_bundle(
            output,
            bundle.model,
            model_version="bad",
            candidate="fixture",
            threshold=0.7,
            provenance={},
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".export-*"))


def test_export_converts_to_native_estimators_without_fitting(
    project_root, monkeypatch
):
    original = joblib.load(
        project_root / "artifacts/cv_experiments/selected_model.joblib"
    )["model"]
    probes = ["My card has not arrived", "I have a problem"]
    expected = original.predict_proba(probes)

    def unexpected_fit(*args, **kwargs):
        pytest.fail("Export must retain fitted state without retraining")

    for estimator in (CalibratedClassifierCV, TfidfVectorizer, LinearSVC):
        monkeypatch.setattr(estimator, "fit", unexpected_fit)
    serving = to_serving_model(original)
    assert set(vars(serving)) == {"pipeline", "labels"}
    for pipeline in [serving.pipeline.estimator] + [
        fitted.estimator for fitted in serving.pipeline.calibrated_classifiers_
    ]:
        assert type(pipeline.named_steps["features"]) is TfidfVectorizer
        assert type(pipeline.named_steps["model"]) is LinearSVC
    np.testing.assert_array_equal(serving.predict_proba(probes), expected)
    assert (
        original.pipeline.calibrated_classifiers_[0]
        .estimator.named_steps["model"]
        .__class__.__module__
        == "notebooks.utils.models"
    )


@pytest.mark.parametrize("failure", ["missing_probes", "changed_predictions"])
def test_failed_migration_does_not_publish_a_version(
    project_root, tmp_path, monkeypatch, failure
):
    from notebooks import export_model

    original_converter = export_model.to_serving_model

    def incorrect_labels(trained):
        converted = original_converter(trained)
        converted.labels = converted.labels[::-1]
        return converted

    probes = project_root / "data/raw/test.csv"
    if failure == "missing_probes":
        probes = tmp_path / "missing.csv"
        expected_error = FileNotFoundError
    else:
        monkeypatch.setattr(export_model, "to_serving_model", incorrect_labels)
        expected_error = AssertionError
    output = tmp_path / "invalid_version"
    with pytest.raises(expected_error):
        export_model.export_model(
            project_root / "artifacts/cv_experiments/selected_model.joblib",
            project_root / "artifacts/cv_experiments/results.json",
            probes,
            output,
            "invalid_version",
        )
    assert not output.exists()
