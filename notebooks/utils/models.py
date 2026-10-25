"""Thin estimator contracts: native constructors, parameters, fit, and predict."""

from abc import ABC, abstractmethod

from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier


class BaseMlModel(BaseEstimator, ABC):
    """Common estimator operations; probability support belongs to the fitted pipeline."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the stable model-family identifier used in results."""

    @abstractmethod
    def fit(self, features, labels):
        """Fit this estimator on the supplied feature matrix."""

    @abstractmethod
    def predict(self, features):
        """Predict encoded classes from a feature matrix."""


class LogisticRegressionModel(LogisticRegression, BaseMlModel):
    """Native logistic regression with the shared estimator contract."""

    name = "logistic_regression"


class LinearSvmModel(LinearSVC, BaseMlModel):
    """Native linear SVM; use a calibration policy for probabilities."""

    name = "svm"


class NaiveBayesModel(MultinomialNB, BaseMlModel):
    """Native multinomial naive Bayes."""

    name = "naive_bayes"


class RandomForestModel(RandomForestClassifier, BaseMlModel):
    """Native random forest classification."""

    name = "random_forest"


class XGBoostModel(XGBClassifier, BaseMlModel):
    """Native XGBoost classification; stopping is orchestrated by a fit policy."""

    name = "xgboost"


class DummyModel(DummyClassifier, BaseMlModel):
    """Native dummy classification baseline."""

    name = "dummy"
