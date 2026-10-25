"""Feature contracts backed by native, cloneable sklearn vectorizers."""

from abc import ABC, abstractmethod

from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class BaseFeatureExtractor(BaseEstimator, ABC):
    """Learn a representation on training text and transform unseen text."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the stable representation identifier used in results."""

    @abstractmethod
    def fit(self, texts, y=None):
        """Learn feature state from training messages."""

    @abstractmethod
    def transform(self, texts):
        """Transform messages without updating feature state."""


class CountFeatures(CountVectorizer, BaseFeatureExtractor):
    """Native word-count features; default ngram_range is single words."""

    name = "counts"


class TfidfFeatures(TfidfVectorizer, BaseFeatureExtractor):
    """Native TF-IDF features; default ngram_range is single words."""

    name = "tfidf"
