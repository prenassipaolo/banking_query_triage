"""Bundle consistency, preservation, and independence from notebook imports."""

import json
import os
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

from banking_query_triage.artifacts import fingerprint, load_bundle

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    "change", ["checksum", "versions", "labels", "duplicate_labels"]
)
def test_rejects_inconsistent_manifest(bundle_dir, change):
    path = bundle_dir / "manifest.json"
    manifest = json.loads(path.read_text())
    if change == "checksum":
        manifest["model_sha256"] = "0" * 64
    elif change == "versions":
        manifest["versions"]["scikit-learn"] = "0.0.0"
    elif change == "labels":
        manifest["labels"].reverse()
    else:
        manifest["labels"] = ["card", "card"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="checksum|versions|labels"):
        load_bundle(bundle_dir)


def test_rejects_probability_column_order_mismatch(bundle_dir):
    path = bundle_dir / "model.joblib"
    model = joblib.load(path)
    model.pipeline.named_steps["model"].classes_ = np.array([1, 0])
    joblib.dump(model, path)
    manifest_path = bundle_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["model_sha256"] = fingerprint(path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="probability-column order"):
        load_bundle(bundle_dir)


def test_migrated_model_preserves_original_benchmark_predictions(project_root):
    original = joblib.load(
        project_root / "artifacts/cv_experiments/selected_model.joblib"
    )["model"]
    migrated = load_bundle(project_root / "artifacts/models/tfidf_svm_v1")
    messages = pd.read_csv(project_root / "data/raw/test.csv").text.tolist()
    np.testing.assert_array_equal(
        original.predict_proba(messages), migrated.model.predict_proba(messages)
    )
    np.testing.assert_array_equal(
        original.predict(messages), migrated.model.predict(messages)
    )
    assert len(migrated.manifest.labels) == 77


def test_serving_package_loads_without_experiment_dependencies(project_root, tmp_path):
    code = """
import sys
from pathlib import Path
class RejectExperimentDependencies:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"notebooks", "pandas", "xgboost"}:
            raise ModuleNotFoundError(f"Experiment dependency disabled: {fullname}", name=fullname)
sys.meta_path.insert(0, RejectExperimentDependencies())
from banking_query_triage.artifacts import load_bundle
from banking_query_triage.service import TriageService
from banking_query_triage.api import create_app
bundle = load_bundle(Path(sys.argv[1]))
assert len(TriageService(bundle).predict(["My card has not arrived"])) == 1
assert not any(name.split(".")[0] in {"notebooks", "pandas", "xgboost"} for name in sys.modules)
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(project_root / "artifacts/models/tfidf_svm_v1"),
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(project_root / "src")},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
