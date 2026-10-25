"""Resolve TOML identifiers once; runtime behavior uses concrete objects."""

from .evaluation import ClassificationEvaluator
from .experiment import Experiment
from .features import CountFeatures, TfidfFeatures
from .fitting import CalibratedFit, StandardFit
from .models import (
    DummyModel,
    LinearSvmModel,
    LogisticRegressionModel,
    NaiveBayesModel,
    RandomForestModel,
    XGBoostModel,
)
from .pipeline import TextPipeline
from .preprocessing import BasicTextPreprocessor
from .xgboost_training import EarlyStoppingFit

FEATURES = {"counts": CountFeatures, "tfidf": TfidfFeatures}
MODELS = {
    model.name: model
    for model in (
        DummyModel,
        LogisticRegressionModel,
        LinearSvmModel,
        NaiveBayesModel,
        RandomForestModel,
        XGBoostModel,
    )
}
FIT_POLICIES = {
    "standard": StandardFit,
    "calibrated": CalibratedFit,
    "early_stopping": EarlyStoppingFit,
}


def resolve_experiments(config: dict, labels) -> list[Experiment]:
    """Build concrete experiment specifications from the small explicit registry."""
    # XGBoost requires num_class even for two-class multiclass fixtures.
    model_context = {"xgboost": {"num_class": len(labels)}}
    experiments = []
    for row in config["experiments"]:
        feature_settings = {
            **config["features"]["common"],
            **config["features"][row["features"]],
        }
        specification = TextPipeline(
            BasicTextPreprocessor(**config["preprocessing"]),
            FEATURES[row["features"]](**feature_settings),
            MODELS[row["model"]](
                **model_context.get(row["model"], {}),
                **config["models"][row["model"]],
            ),
        )
        policy_name = row["fit_policy"]
        experiments.append(
            Experiment(
                row["name"],
                specification,
                FIT_POLICIES[policy_name](**config["fit_policies"][policy_name]),
                ClassificationEvaluator(labels),
            )
        )
    return experiments
