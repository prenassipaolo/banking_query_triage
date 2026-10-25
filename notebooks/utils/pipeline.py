"""An explicit composition of preprocessing, features, and estimator objects."""

from dataclasses import dataclass

from sklearn.base import clone
from sklearn.pipeline import Pipeline

from .features import BaseFeatureExtractor
from .models import BaseMlModel
from .preprocessing import BasePreprocessor


@dataclass(frozen=True)
class TextPipeline:
    """Specify unfitted components; each build returns independent cloned state."""

    preprocessor: BasePreprocessor
    features: BaseFeatureExtractor
    model: BaseMlModel

    def build(self) -> Pipeline:
        """Build a native sklearn pipeline with training-only feature learning."""
        return Pipeline(
            [
                ("preprocessing", clone(self.preprocessor)),
                ("features", clone(self.features)),
                ("model", clone(self.model)),
            ]
        )
