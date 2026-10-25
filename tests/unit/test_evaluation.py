"""Known answers for label/probability metrics and review decisions."""

import numpy as np
import pytest

from notebooks.utils.evaluation import ClassificationEvaluator, ReviewPolicy
from notebooks.utils.fitting import TrainedModel


def test_metrics_match_hand_calculation_and_accept_additional_extractors():
    evaluator = ClassificationEvaluator(
        ["a", "b"], extra_metrics={"example_metric": lambda y, p: 7.0}
    )
    probabilities = np.array([[0.9, 0.1], [0.2, 0.8]])
    metrics = evaluator.from_probabilities(["a", "b"], probabilities)
    assert metrics["accuracy"] == metrics["macro_f1"] == metrics["roc_auc_macro"] == 1
    assert metrics["log_loss"] == pytest.approx((-np.log(0.9) - np.log(0.8)) / 2)
    assert metrics["brier"] == pytest.approx(0.05)
    assert metrics["example_metric"] == 7


def test_review_zero_coverage_equality_and_invalid_probabilities():
    policy = ReviewPolicy(0.7)
    assert policy.actions([[0.7, 0.3]]).tolist() == ["route"]
    assert policy.evaluate(["a"], [[0.5, 0.5]], ["a", "b"])["accepted_accuracy"] is None
    with pytest.raises(ValueError, match="sum"):
        ClassificationEvaluator(["a", "b"]).from_probabilities(["a"], [[0.2, 0.2]])


def test_unknown_truth_is_rejected_instead_of_mapped_to_a_known_category():
    with pytest.raises(ValueError, match="unseen"):
        ClassificationEvaluator(["a", "b"]).from_probabilities(
            ["unknown"], [[0.9, 0.1]]
        )


def test_probability_rounding_is_normalized_but_invalid_outputs_are_rejected():
    class ProbabilityFixture:
        def __init__(self, values):
            self.values = values

        def predict_proba(self, texts):
            return np.array(self.values, dtype=np.float32)

    model = TrainedModel(
        ProbabilityFixture([[0.1, 0.9000001]]), np.array(["a", "b"]), 0
    )
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
    evaluator = ClassificationEvaluator(["a", "b"])
    with pytest.raises(ValueError, match="invalid probabilit"):
        evaluator.from_probabilities(["a"], probabilities)
