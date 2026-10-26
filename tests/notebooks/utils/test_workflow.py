"""Complete model training, calibration boundaries, and frozen test confirmation."""

import json
import tomllib
from copy import deepcopy

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import (
    CountVectorizer,
    HashingVectorizer,
    TfidfVectorizer,
)
from sklearn.model_selection import StratifiedGroupKFold

from notebooks.utils.configuration import resolve_experiments
from notebooks.utils.data import ROOT, ExperimentData, normalize_group
from notebooks.utils.features import BaseFeatureExtractor
from notebooks.utils.workflow import ExperimentRunner

pytestmark = pytest.mark.integration


class HashFeatures(HashingVectorizer, BaseFeatureExtractor):
    """Exercise interchangeable fixed-width features with no learned vocabulary."""

    name = "hashing"


def configuration():
    return deepcopy(tomllib.loads((ROOT / "config/experiments.toml").read_text()))


def fixture_frame():
    return pd.DataFrame(
        [
            (f"{label} {label} sample unique{i}", label)
            for label in ["card", "cash"]
            for i in range(30)
        ],
        columns=["text", "category"],
    )


def test_svm_calibration_refits_features_in_training_folds(monkeypatch, tmp_path):
    config = configuration()
    frame = fixture_frame()
    seen = []
    original = TfidfVectorizer.fit_transform

    def record_fit(self, documents, y=None):
        documents = list(documents)
        seen.append(set(documents))
        return original(self, documents, y)

    monkeypatch.setattr(TfidfVectorizer, "fit_transform", record_fit)
    experiment = resolve_experiments(config, ["card", "cash"])[3]
    model = experiment.fit(frame.text, frame.category)
    folds = StratifiedGroupKFold(5, shuffle=True, random_state=42).split(
        frame.text, frame.category, frame.text.map(normalize_group)
    )
    expected = [set(frame.text.iloc[train]) for train, _ in folds] + [set(frame.text)]
    assert {frozenset(x) for x in seen} == {frozenset(x) for x in expected}
    assert len(seen) == 6
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    loaded = joblib.load(path)
    np.testing.assert_array_equal(model.predict(frame.text), loaded.predict(frame.text))
    np.testing.assert_array_equal(
        model.predict_proba(frame.text), loaded.predict_proba(frame.text)
    )


def test_boosting_stops_internally_and_refits_all_training_rows(monkeypatch):
    config = configuration()
    config["models"]["xgboost"]["n_estimators"] = 5
    config["fit_policies"]["early_stopping"]["patience"] = 2
    frame = fixture_frame()
    seen = []
    original = CountVectorizer.fit_transform

    def record_fit(self, documents, y=None):
        documents = list(documents)
        seen.append(set(documents))
        return original(self, documents, y)

    monkeypatch.setattr(CountVectorizer, "fit_transform", record_fit)
    experiment = resolve_experiments(config, ["card", "cash"])[6]
    model = experiment.fit(frame.text, frame.category)
    assert 1 <= model.rounds <= 5
    assert model.history and "stopping_log_loss" in model.history[0]
    final = experiment.fit(
        frame.text,
        frame.category,
        rounds=model.rounds,
    )
    assert final.history == []
    assert final.pipeline.named_steps["model"].n_estimators == model.rounds
    inner_fit, _ = next(
        StratifiedGroupKFold(5, shuffle=True, random_state=42).split(
            frame.text, frame.category, frame.text.map(normalize_group)
        )
    )
    assert seen == [set(frame.text.iloc[inner_fit]), set(frame.text), set(frame.text)]
    assert (
        final.pipeline.named_steps["model"].get_params()["early_stopping_rounds"]
        is None
    )


def test_full_workflow_selects_before_test_and_refits_full_data(monkeypatch, tmp_path):
    config = configuration()
    frame = fixture_frame()
    for key, filename in [
        ("train_file", "train.csv"),
        ("test_file", "test.csv"),
        ("labels_file", "labels.json"),
        ("fold_file", "folds.csv"),
        ("output_dir", "outputs"),
    ]:
        config[key] = str(tmp_path / filename)
    frame.to_csv(config["train_file"], index=False)
    pd.DataFrame(
        [("card bill", "card"), ("cash withdrawal", "cash")],
        columns=["text", "category"],
    ).to_csv(config["test_file"], index=False)
    (tmp_path / "labels.json").write_text(json.dumps(["card", "cash"]))
    config["experiments"] = [
        {
            "name": "lr",
            "features": "tfidf",
            "model": "logistic_regression",
            "fit_policy": "standard",
        }
    ]
    read_frame = ExperimentData.read_frame

    def check_boundary(path, labels):
        if path == tmp_path / "test.csv":
            record = json.loads((tmp_path / "outputs/results.json").read_text())
            assert record["state"] == "selection_frozen"
            assert record["selected"] == "lr"
            assert (tmp_path / "outputs/selected_model.joblib").exists()
        return read_frame(path, labels)

    monkeypatch.setattr(ExperimentData, "read_frame", staticmethod(check_boundary))
    results = ExperimentRunner(config).run()
    assert len(results.folds()) == 10
    assert results.summary().iloc[0]["final_train_rows"] == len(frame)
    assert results.summary().iloc[0]["test_accuracy"] == 1
    assert results.summary().iloc[0]["test_accuracy_std"] is None
    assert results.data["state"] == "complete"
    assert "src/banking_query_triage/preprocessing.py" in results.data["source_sha256"]
    assert "src/banking_query_triage/inference.py" in results.data["source_sha256"]
    assert "notebooks/utils/workflow.py" in results.data["source_sha256"]


def test_direct_components_record_actual_settings_and_accept_features_without_vocabulary(
    tmp_path,
):
    from dataclasses import replace

    from notebooks.utils.pipeline import TextPipeline

    config = configuration()
    frame = fixture_frame()
    config.update(
        train_file=str(tmp_path / "train.csv"),
        test_file=str(tmp_path / "test.csv"),
        labels_file=str(tmp_path / "labels.json"),
        fold_file=str(tmp_path / "folds.csv"),
        output_dir=str(tmp_path / "outputs"),
    )
    frame.to_csv(config["train_file"], index=False)
    test = frame.iloc[[0, -1]].copy()
    test["text"] = test["text"] + " benchmarkonly"
    test.to_csv(config["test_file"], index=False)
    (tmp_path / "labels.json").write_text(json.dumps(["card", "cash"]))
    template = resolve_experiments(config, ["card", "cash"])[1]
    template.pipeline.model.set_params(C=999)
    experiment = replace(
        template,
        name="direct_hash_logistic",
        pipeline=TextPipeline(
            template.pipeline.preprocessor,
            HashFeatures(n_features=32, alternate_sign=False),
            template.pipeline.model,
        ),
    )
    results = ExperimentRunner(config, experiments=[experiment]).run()
    specification = results.data["candidates"][0]
    assert specification["pipeline"]["model"]["parameters"]["C"] == 999
    assert results.data["config"]["models"]["logistic_regression"]["C"] == 1
    assert specification["pipeline"]["features"]["parameters"]["n_features"] == 32
    assert specification["evaluation"]["settings"]["labels"] == ["card", "cash"]
    assert "encoder" not in specification["evaluation"]["settings"]
    assert results.summary().iloc[0]["final_features"] == 32
    bundle = joblib.load(tmp_path / "outputs/selected_model.joblib")
    assert bundle["candidate"] == specification
    assert bundle["model"].feature_count == 32
