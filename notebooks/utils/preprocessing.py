"""Raw-text cleaning contract and conservative implementation."""

import re
import unicodedata
from abc import ABC, abstractmethod
from collections.abc import Iterable

from sklearn.base import BaseEstimator, TransformerMixin


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
        cleaned = []
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Messages must be nonempty strings")
            text = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()
            cleaned.append(text.lower() if self.lowercase else text)
        return cleaned
