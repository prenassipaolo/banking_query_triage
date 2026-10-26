"""Text normalization used by the already-fitted serving pipeline."""

import re
import unicodedata
from collections.abc import Iterable


def clean_messages(texts: Iterable[str], lowercase: bool = True) -> list[str]:
    """Normalize Unicode and whitespace while preserving words and negation.

    Args:
        texts: Nonempty string messages to normalize.
        lowercase: Whether to lowercase normalized text.

    Returns:
        Cleaned messages in the same order.

    Raises:
        ValueError: Any message is not a nonempty string.
    """
    cleaned = []
    for text in texts:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Messages must be nonempty strings")
        text = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()
        cleaned.append(text.lower() if lowercase else text)
    return cleaned
