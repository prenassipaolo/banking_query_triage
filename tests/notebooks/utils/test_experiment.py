"""Abstract interfaces, native estimator interoperability, and reusable specifications."""

import inspect
import json
import tomllib
from dataclasses import replace

import joblib
import numpy as np
import pytest
from sklearn.base import clone, is_classifier
from sklearn.utils import get_tags

from notebooks.utils.configuration import resolve_experiments
from notebooks.utils.data import ROOT
from notebooks.utils.evaluation import BaseEvaluator
from notebooks.utils.features import BaseFeatureExtractor
from notebooks.utils.fitting import BaseFitPolicy, StandardFit
from notebooks.utils.models import BaseMlModel, LinearSvmModel
from notebooks.utils.preprocessing import BasePreprocessor
from notebooks.utils.xgboost_training import EarlyStoppingFit


@pytest.mark.parametrize(
    "contract",
    [BasePreprocessor, BaseFeatureExtractor, BaseMlModel, BaseFitPolicy, BaseEvaluator],
)
def test_contracts_reject_missing_implementations(contract):
    assert inspect.isabstract(contract)
    with pytest.raises(TypeError, match="abstract"):
        contract()


@pytest.mark.integration
def test_native_components_clone_fit_serialize_without_mutating_specifications(
    tmp_path,
):
    config = tomllib.loads((ROOT / "config/experiments.toml").read_text())
    config["models"]["xgboost"]["n_estimators"] = 2
    config["models"]["random_forest"]["n_estimators"] = 2
    texts = [
        f"{label} repeated sample{i}" for label in ("card", "cash") for i in range(20)
    ]
    labels = [label for label in ("card", "cash") for _ in range(20)]
    for experiment in resolve_experiments(config, ["card", "cash"]):
        model = experiment.pipeline.model
        assert is_classifier(model)
        assert get_tags(model).estimator_type == "classifier"
        for component in (
            model,
            experiment.pipeline.features,
            experiment.pipeline.preprocessor,
        ):
            copied = clone(component)
            assert type(copied) is type(component)
            assert copied.get_params() == component.get_params()
        first = experiment.fit(texts, labels)
        second = experiment.fit(texts, labels)
        assert first.feature_count > 0
        assert first.pipeline is not second.pipeline
        assert not hasattr(experiment.pipeline.features, "vocabulary_")
        assert not hasattr(model, "classes_")
        assert not hasattr(experiment.pipeline.preprocessor, "n_features_in_")
        assert model.get_params() == clone(model).get_params()
        path = tmp_path / f"{experiment.name}.joblib"
        joblib.dump(first, path)
        loaded = joblib.load(path)
        np.testing.assert_array_equal(
            first.predict_proba(texts), loaded.predict_proba(texts)
        )
    assert not hasattr(LinearSvmModel(), "predict_proba")


def test_incompatible_compositions_report_the_required_capability():
    config = tomllib.loads((ROOT / "config/experiments.toml").read_text())
    experiments = resolve_experiments(config, ["card", "cash"])
    texts = ["card bill", "card bill", "cash cost", "cash cost"]
    labels = ["card", "card", "cash", "cash"]
    with pytest.raises(TypeError, match="CalibratedFit"):
        replace(experiments[3], fit_policy=StandardFit()).fit(texts, labels)
    with pytest.raises(TypeError, match="requires an XGBoost"):
        replace(experiments[1], fit_policy=EarlyStoppingFit()).fit(texts, labels)


def test_early_stopping_checks_required_history_configuration_before_fitting(
    monkeypatch,
):
    config = tomllib.loads((ROOT / "config/experiments.toml").read_text())
    experiment = resolve_experiments(config, ["card", "cash"])[6]
    experiment.pipeline.model.set_params(eval_metric=["mlogloss"])

    def unexpected_fit(*args, **kwargs):
        pytest.fail("Incompatible stopping settings must fail before model fitting")

    monkeypatch.setattr(type(experiment.pipeline.model), "fit", unexpected_fit)
    with pytest.raises(ValueError, match="eval_metric"):
        experiment.fit(["card", "cash"], ["card", "cash"])


def test_component_descriptions_are_strict_json_with_native_and_custom_settings():
    config = tomllib.loads((ROOT / "config/experiments.toml").read_text())
    experiments = resolve_experiments(config, ["card", "cash"])
    experiments[0].evaluator.extra_metrics["custom"] = lambda labels, probabilities: 1
    descriptions = [experiment.description() for experiment in experiments]
    json.dumps(descriptions, allow_nan=False)
    assert descriptions[6]["pipeline"]["model"]["parameters"]["missing"] == {
        "nonfinite_float": "nan"
    }
    assert descriptions[0]["evaluation"]["settings"]["extra_metrics"]["custom"][
        "python"
    ].endswith("<lambda>")
    assert descriptions[3]["fitting"]["settings"] == {
        "folds": 5,
        "seed": 42,
        "method": "temperature",
    }


def test_direct_custom_contract_implementations_have_native_parameter_cloning():
    class LengthFeatures(BaseFeatureExtractor):
        name = "length"

        def __init__(self, scale: float = 1):
            self.scale = scale

        def fit(self, texts, y=None):
            return self

        def transform(self, texts):
            return np.asarray([[len(text) * self.scale] for text in texts])

    class ConstantModel(BaseMlModel):
        name = "constant"

        def __init__(self, label: int = 0):
            self.label = label

        def fit(self, features, labels):
            return self

        def predict(self, features):
            return np.full(len(features), self.label)

    for component in (LengthFeatures(scale=2), ConstantModel(label=1)):
        copied = clone(component)
        assert copied is not component
        assert copied.get_params() == component.get_params()
