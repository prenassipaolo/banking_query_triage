"""Probability shape, finiteness and normalization at the fitted inference boundary."""

import numpy as np
import pytest

from banking_query_triage.inference import TrainedModel, validate_probabilities


def test_probability_rounding_is_normalized_but_invalid_outputs_are_rejected():
    class ProbabilityFixture:
        def __init__(self, values):
            self.values = values

        def predict_proba(self, texts):
            return np.array(self.values, dtype=np.float32)

    model = TrainedModel(ProbabilityFixture([[0.1, 0.9000001]]), np.array(["a", "b"]))
    probabilities = model.predict_proba(["question"])
    assert probabilities.dtype == np.float64
    assert probabilities.sum() == pytest.approx(1, abs=1e-15)
    model.pipeline = ProbabilityFixture([[0.2, 0.2]])
    with pytest.raises(ValueError, match="invalid probabilities"):
        model.predict_proba(["question"])


@pytest.mark.parametrize(
    "probabilities",
    [
        [0.2, 0.8],
        [[0.2, 0.8], [0.2, 0.8]],
        [[0.2, 0.3, 0.5]],
        [[float("nan"), 0.8]],
        [[-0.1, 1.1]],
    ],
)
def test_probability_validation_rejects_wrong_shapes_and_invalid_values(probabilities):
    with pytest.raises(ValueError, match="invalid probabilit"):
        validate_probabilities(probabilities, 1, 2)
