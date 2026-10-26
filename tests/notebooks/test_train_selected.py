"""Selected-model refitting reads only training data and exports independent versions."""

import json

import pandas as pd
import pytest

from banking_query_triage.artifacts import load_bundle
from notebooks.train_selected import train_selected

pytestmark = pytest.mark.integration


def test_selected_refit_does_not_read_test_or_overwrite_existing_bundle(
    tmp_path, monkeypatch
):
    frame = pd.DataFrame(
        [
            (f"{label} {label} sample unique{i}", label)
            for label in ["card", "cash"]
            for i in range(20)
        ],
        columns=["text", "category"],
    )
    frame.to_csv(tmp_path / "train.csv", index=False)
    (tmp_path / "labels.json").write_text(json.dumps(["cash", "card"]))
    config_path = tmp_path / "training.toml"
    config_path.write_text("""
root_dir = "."
train_file = "train.csv"
labels_file = "labels.json"
test_file = "must_not_be_read.csv"
threads = 1
review_threshold = 0.7
[preprocessing]
lowercase = true
[features.common]
lowercase = false
[features.tfidf]
sublinear_tf = true
[models.svm]
C = 1.0
random_state = 42
[fit_policies.calibrated]
folds = 2
seed = 42
method = "temperature"
[[experiments]]
name = "tfidf_svm"
features = "tfidf"
model = "svm"
fit_policy = "calibrated"
""")
    monkeypatch.chdir("/tmp")
    output = tmp_path / "model_v2"
    bundle = train_selected(config_path, output, "model_v2")
    assert bundle.manifest.provenance["training_rows"] == 40
    assert "notebooks/train_selected.py" in bundle.manifest.provenance["source_sha256"]
    assert (
        "src/banking_query_triage/preprocessing.py"
        in bundle.manifest.provenance["source_sha256"]
    )
    assert bundle.model.predict(["card card", "cash cash"]).tolist() == ["card", "cash"]
    assert load_bundle(output).manifest.model_version == "model_v2"
    assert not (tmp_path / "must_not_be_read.csv").exists()
    assert bundle.model.pipeline.calibrated_classifiers_[0].estimator.named_steps[
        "features"
    ].ngram_range == (1, 1)
    with pytest.raises(FileExistsError):
        train_selected(config_path, output, "replacement")
