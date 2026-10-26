"""Refit the frozen TF-IDF/SVM recipe without rerunning candidate selection."""

import argparse
import json
import tomllib
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from threadpoolctl import threadpool_limits

from banking_query_triage.artifacts import ModelBundle, fingerprint

from .utils.features import TfidfFeatures
from .utils.fitting import CalibratedFit, TrainedModel
from .utils.models import LinearSvmModel
from .utils.pipeline import TextPipeline
from .utils.preprocessing import BasicTextPreprocessor
from .utils.serving import save_bundle, to_serving_model


def train_selected(config_path: Path, output: Path, model_version: str) -> ModelBundle:
    """Train on the training CSV only and export a new versioned serving bundle.

    Args:
        config_path: Experiment TOML; data paths are relative to its root_dir.
        output: New output directory; existing bundles are never replaced.
        model_version: Explicit version for this newly fitted model.

    Returns:
        Exported, reloaded TF-IDF/SVM model and its provenance.

    Raises:
        FileExistsError: The output directory already exists.
        ValueError: The frozen recipe or training data is invalid.
    """
    if output.exists():
        raise FileExistsError(output)
    config_path = config_path.resolve()
    config = tomllib.loads(config_path.read_text())
    root = (config_path.parent / config.get("root_dir", "..")).resolve()
    candidate = next(row for row in config["experiments"] if row["name"] == "tfidf_svm")
    if (candidate["features"], candidate["model"], candidate["fit_policy"]) != (
        "tfidf",
        "svm",
        "calibrated",
    ):
        raise ValueError("The selected training recipe must be calibrated TF-IDF/SVM")
    train_path, labels_path = root / config["train_file"], root / config["labels_file"]
    labels = json.loads(labels_path.read_text())
    if len(set(labels)) != len(labels) or len(labels) < 2:
        raise ValueError(
            "Allowed labels must be unique and contain at least two categories"
        )
    frame = pd.read_csv(train_path)
    if (
        list(frame.columns) != ["text", "category"]
        or frame.isna().any().any()
        or set(frame.category) != set(labels)
    ):
        raise ValueError(
            "Expected non-null text/category columns containing all allowed categories"
        )
    BasicTextPreprocessor().transform(frame.text)
    encoder = LabelEncoder().fit(labels)
    specification = TextPipeline(
        BasicTextPreprocessor(**config["preprocessing"]),
        TfidfFeatures(
            **{**config["features"]["common"], **config["features"]["tfidf"]}
        ),
        LinearSvmModel(**config["models"]["svm"]),
    )
    policy = CalibratedFit(**config["fit_policies"]["calibrated"])
    start = perf_counter()
    with threadpool_limits(limits=config["threads"]):
        fitted = policy.fit(
            specification, frame.text.to_numpy(), encoder.transform(frame.category)
        )
    model = TrainedModel(
        fitted.pipeline, np.asarray(encoder.classes_), perf_counter() - start
    )
    serving = to_serving_model(model)
    np.testing.assert_array_equal(
        model.predict_proba(frame.text), serving.predict_proba(frame.text)
    )
    bundle = save_bundle(
        output,
        serving,
        model_version=model_version,
        candidate="tfidf_svm",
        threshold=config["review_threshold"],
        provenance={
            "operation": "selected_model_refit",
            "training_rows": len(frame),
            "training_config": config,
            "fingerprints": {
                "config": fingerprint(config_path),
                "train_file": fingerprint(train_path),
                "labels_file": fingerprint(labels_path),
            },
            "source_sha256": {
                path.relative_to(
                    Path(__file__).resolve().parents[1]
                ).as_posix(): fingerprint(path)
                for directory in (
                    Path(__file__).resolve().parent,
                    Path(__file__).resolve().parents[1] / "src/banking_query_triage",
                )
                for path in sorted(directory.rglob("*.py"))
            },
        },
    )
    return bundle


def main() -> None:
    """Run the selected-model training command with an explicit output version."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/experiments.toml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-version", required=True)
    args = parser.parse_args()
    bundle = train_selected(args.config, args.output, args.model_version)
    print(f"Saved {bundle.manifest.model_version} to {args.output}")


if __name__ == "__main__":
    main()
