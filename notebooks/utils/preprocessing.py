"""Raw-text cleaning contract and conservative implementation."""

from abc import ABC, abstractmethod
from collections.abc import Iterable

from sklearn.base import BaseEstimator, TransformerMixin

from banking_query_triage.preprocessing import clean_messages


class BasePreprocessor(TransformerMixin, BaseEstimator, ABC):
    """Learn from training text and transform raw messages into cleaned text."""

    @abstractmethod
    def fit(self, texts: Iterable[str], y=None):
        """Learn preprocessing state from training messages."""

    @abstractmethod
    def transform(self, texts: Iterable[str]) -> list[str]:
        """Clean messages without learning from them."""


class BasicTextPreprocessor(BasePreprocessor):
    """Normalize Unicode and whitespace, preserving words and negation."""

    def __init__(self, lowercase: bool = True):
        self.lowercase = lowercase

    def fit(self, texts: Iterable[str], y=None):
        """Return self; cleaning learns no dataset statistics."""
        return self

    def transform(self, texts: Iterable[str]) -> list[str]:
        """Clean nonempty strings, optionally lowercasing them."""
        return clean_messages(texts, lowercase=self.lowercase)
