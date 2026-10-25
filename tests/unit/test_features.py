"""Single-word features and the training vocabulary boundary."""

import tomllib

import pytest

from notebooks.utils.configuration import resolve_experiments
from notebooks.utils.data import ROOT
from notebooks.utils.preprocessing import BasicTextPreprocessor


def test_preprocessing_preserves_negation_and_rejects_empty_messages():
    cleaner = BasicTextPreprocessor()
    assert cleaner.fit_transform(["  I DO NOT\nrecognize this!  "]) == [
        "i do not recognize this!"
    ]
    with pytest.raises(ValueError, match="nonempty"):
        cleaner.transform([""])


def test_single_word_vocabulary_only_learns_training_input():
    config = tomllib.loads((ROOT / "config/experiments.toml").read_text())
    pipeline = resolve_experiments(config, ["a", "b"])[2].pipeline.build()
    pipeline.fit(["card bill", "card cost", "cash bill", "cash cost"], [0, 0, 1, 1])
    features = pipeline.named_steps["features"]
    vocabulary = dict(features.vocabulary_)
    pipeline.predict(["validationsecret card"])
    assert "validationsecret" not in features.vocabulary_
    assert "card bill" not in features.vocabulary_
    assert features.vocabulary_ == vocabulary
    assert features.ngram_range == (1, 1)
