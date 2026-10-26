"""Versioned serving bundles containing a fitted model and frozen provenance."""

import hashlib
import platform
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .inference import TrainedModel


def fingerprint(path: Path) -> str:
    """Return a file's SHA-256 digest.

    Args:
        path: File to fingerprint.

    Returns:
        Hexadecimal SHA-256 digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runtime_versions() -> dict[str, str]:
    """Return Python and the dependencies needed to load a fitted model.

    Returns:
        Runtime version identifiers.
    """
    return {
        "python": platform.python_version(),
        **{
            name: version(name) for name in ("scikit-learn", "numpy", "scipy", "joblib")
        },
    }


class ModelManifest(BaseModel):
    """Readable provenance tied to the exact bytes of a fitted model."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: int = Field(default=1, ge=1, le=1)
    model_version: str = Field(min_length=1)
    candidate: str = Field(min_length=1)
    labels: list[str] = Field(min_length=2)
    review_threshold: float = Field(ge=0, le=1)
    model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    versions: dict[str, str]
    provenance: dict[str, Any]


@dataclass(frozen=True)
class ModelBundle:
    """Validated model and its serving manifest."""

    model: TrainedModel
    manifest: ModelManifest


def load_bundle(directory: Path) -> ModelBundle:
    """Load an application-owned bundle and validate it before serving.

    Args:
        directory: Directory containing model.joblib and manifest.json.

    Returns:
        A model whose labels, encoded classes, probabilities and versions agree.

    Raises:
        TypeError: The artifact does not contain a fitted TrainedModel.
        ValueError: The bundle is inconsistent or incompatible with this runtime.
    """
    manifest = ModelManifest.model_validate_json(
        (directory / "manifest.json").read_text()
    )
    model_path = directory / "model.joblib"
    if fingerprint(model_path) != manifest.model_sha256:
        raise ValueError("Model checksum does not match manifest")
    if manifest.versions != runtime_versions():
        raise ValueError(
            "Model dependency versions do not match this runtime; use the recorded lockfile or retrain"
        )
    # joblib uses pickle. This directory must contain application-owned artifacts.
    model = joblib.load(model_path)
    if not isinstance(model, TrainedModel):
        raise TypeError("Expected a TrainedModel artifact")
    if (
        len(set(manifest.labels)) != len(manifest.labels)
        or any(not label.strip() for label in manifest.labels)
        or not np.array_equal(model.labels, manifest.labels)
        or not np.array_equal(model.pipeline.classes_, np.arange(len(manifest.labels)))
    ):
        raise ValueError(
            "Model labels and probability-column order do not match manifest"
        )
    model.predict_proba(["Model startup check"])
    return ModelBundle(model, manifest)
