"""Migrate the trusted notebook artifact, retaining all fitted model state."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from banking_query_triage.artifacts import fingerprint

from .utils.serving import save_bundle, to_serving_model


def export_model(
    source: Path,
    results_path: Path,
    probes_path: Path,
    output: Path,
    model_version: str,
) -> None:
    """Re-export historical fitted state and check exact benchmark probability parity."""
    original = joblib.load(source)
    results = json.loads(results_path.read_text())
    if original["candidate"]["name"] != results["selected"]:
        raise ValueError("The saved artifact and selected results disagree")
    probes = pd.read_csv(probes_path).text.tolist()
    serving = to_serving_model(original["model"])
    np.testing.assert_array_equal(
        original["model"].predict_proba(probes), serving.predict_proba(probes)
    )
    np.testing.assert_array_equal(
        original["model"].predict(probes), serving.predict(probes)
    )
    save_bundle(
        output,
        serving,
        model_version=model_version,
        candidate=original["candidate"]["name"],
        threshold=original["policy"]["threshold"],
        provenance={
            "operation": "notebook_artifact_migration_no_refit",
            "source_model_sha256": fingerprint(source),
            "source_results_sha256": fingerprint(results_path),
            "training_config": original["config"],
            "candidate": original["candidate"],
            "training_rows": results["training_rows"],
            "fingerprints": results["fingerprints"],
            "training_versions": results["versions"],
            "training_source_sha256": results["source_sha256"],
        },
    )
    print(
        f"Saved {model_version}; exact predictions/probabilities preserved on {len(probes)} rows"
    )


def main() -> None:
    """Export a new serving version from the existing experiment bundle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("artifacts/cv_experiments/selected_model.joblib"),
    )
    parser.add_argument(
        "--results", type=Path, default=Path("artifacts/cv_experiments/results.json")
    )
    parser.add_argument("--probes", type=Path, default=Path("data/raw/test.csv"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/models/tfidf_svm_v1")
    )
    parser.add_argument("--model-version", default="tfidf_svm_v1")
    args = parser.parse_args()
    export_model(
        args.source, args.results, args.probes, args.output, args.model_version
    )


if __name__ == "__main__":
    main()
