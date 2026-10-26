"""Single-word features and the training vocabulary boundary."""

import pytest

from notebooks.utils.features import TfidfFeatures
from notebooks.utils.models import LogisticRegressionModel
from notebooks.utils.pipeline import TextPipeline
from notebooks.utils.preprocessing import BasicTextPreprocessor


def test_preprocessing_preserves_negation_and_rejects_empty_messages():
    cleaner = BasicTextPreprocessor()
    assert cleaner.fit_transform(["  I DO NOT\nrecognize this!  "]) == [
        "i do not recognize this!"
    ]
    with pytest.raises(ValueError, match="nonempty"):
        cleaner.transform([""])


def test_single_word_vocabulary_only_learns_training_input():
    pipeline = TextPipeline(
        BasicTextPreprocessor(),
        TfidfFeatures(ngram_range=(1, 1)),
        LogisticRegressionModel(),
    ).build()
    pipeline.fit(["card bill", "card cost", "cash bill", "cash cost"], [0, 0, 1, 1])
    features = pipeline.named_steps["features"]
    vocabulary = dict(features.vocabulary_)
    pipeline.predict(["validationsecret card"])
    assert "validationsecret" not in features.vocabulary_
    assert "card bill" not in features.vocabulary_
    assert features.vocabulary_ == vocabulary
    assert features.ngram_range == (1, 1)
