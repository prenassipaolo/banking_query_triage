"""Training-only XGBoost stopping followed by a full-training refit."""

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .data import normalize_group
from .fitting import BaseFitPolicy, FitResult


class EarlyStoppingFit(BaseFitPolicy):
    """Choose boosting rounds internally, then relearn features on all training rows."""

    def __init__(self, folds: int = 5, seed: int = 42, patience: int = 10):
        self.folds = folds
        self.seed = seed
        self.patience = patience

    def fit(self, specification, texts, labels, rounds=None) -> FitResult:
        """Use an inner stopping fold, or fit all rows with already frozen rounds."""
        if not isinstance(specification.model, XGBClassifier):
            raise TypeError("EarlyStoppingFit requires an XGBoost estimator")
        parameters = specification.model.get_params()
        if (
            parameters["eval_metric"] != ["merror", "mlogloss"]
            or parameters["objective"] != "multi:softprob"
            or parameters.get("num_class") != len(set(labels))
        ):
            raise ValueError(
                "EarlyStoppingFit requires objective='multi:softprob', "
                "num_class matching the training labels, and "
                "eval_metric=['merror', 'mlogloss'] (stopping monitors the last metric)"
            )
        if rounds is not None:
            pipeline = specification.build()
            pipeline.named_steps["model"].set_params(
                n_estimators=rounds, early_stopping_rounds=None
            )
            pipeline.fit(texts, labels)
            return FitResult(pipeline, rounds)
        splitter = StratifiedGroupKFold(
            n_splits=self.folds,
            shuffle=True,
            random_state=self.seed,
        )
        fit, stop = next(
            splitter.split(texts, labels, [normalize_group(t) for t in texts])
        )
        pipeline = specification.build()
        features = Pipeline(pipeline.steps[:-1])
        fit_x = features.fit_transform(texts[fit], labels[fit])
        stop_x = features.transform(texts[stop])
        model = pipeline.named_steps["model"]
        model.set_params(early_stopping_rounds=self.patience)
        model.fit(
            fit_x,
            labels[fit],
            eval_set=[(fit_x, labels[fit]), (stop_x, labels[stop])],
            verbose=False,
        )
        rounds = model.best_iteration + 1
        history = model.evals_result()
        curve = pd.DataFrame(
            {
                "round": range(1, len(history["validation_0"]["mlogloss"]) + 1),
                "train_log_loss": history["validation_0"]["mlogloss"],
                "stopping_log_loss": history["validation_1"]["mlogloss"],
                "train_error_rate": history["validation_0"]["merror"],
                "stopping_error_rate": history["validation_1"]["merror"],
            }
        )
        curve["selected_rounds"] = rounds
        # Relearn vocabulary on all outer-training rows; outer validation stays unseen.
        refitted = specification.build()
        refitted.named_steps["model"].set_params(
            n_estimators=rounds, early_stopping_rounds=None
        )
        refitted.fit(texts, labels)
        return FitResult(refitted, rounds, curve.to_dict(orient="records"))
