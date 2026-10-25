"""Load raw text and reuse five grouped folds across the entire training file."""

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[2]


def normalize_group(text: str) -> str:
    """Identify repeated text without removing punctuation or words."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def fingerprint(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class ExperimentData:
    """Own the training frame, allowed labels, reusable folds, and provenance."""

    frame: pd.DataFrame
    labels: np.ndarray
    assignments: pd.DataFrame
    fingerprints: dict

    @classmethod
    def load(cls, config: dict):
        """Load training only; preserve all rows and save reproducible fold assignments."""
        paths = {
            name: ROOT / config[name]
            for name in ("train_file", "test_file", "labels_file")
        }
        labels = np.array(sorted(json.loads(paths["labels_file"].read_text())))
        frame = cls.read_frame(paths["train_file"], labels)
        groups = frame.text.map(normalize_group)
        splitter = StratifiedGroupKFold(
            n_splits=config["folds"], shuffle=True, random_state=config["seed"]
        )
        assignments = pd.DataFrame(
            {"row_id": np.arange(len(frame)), "fold": -1, "group": groups}
        )
        for fold, (train, validation) in enumerate(
            splitter.split(frame.text, frame.category, groups)
        ):
            for indices in (train, validation):
                if set(frame.iloc[indices].category) != set(labels):
                    raise ValueError("Every fold must retain all categories")
            assignments.loc[validation, "fold"] = fold
        path = ROOT / config["fold_file"]
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            pd.testing.assert_frame_equal(
                pd.read_csv(path, keep_default_na=False), assignments
            )
        else:
            assignments.to_csv(path, index=False)
        hashes = {name: fingerprint(path) for name, path in paths.items()}
        hashes["folds"] = fingerprint(path)
        return cls(frame, labels, assignments, hashes)

    @staticmethod
    def read_frame(path: Path, labels: np.ndarray) -> pd.DataFrame:
        """Read expected columns and reject empty input or unknown/missing labels."""
        frame = pd.read_csv(path)
        if list(frame.columns) != ["text", "category"] or frame.isna().any().any():
            raise ValueError("Expected non-null text and category columns")
        if not frame.text.map(
            lambda text: isinstance(text, str) and bool(text.strip())
        ).all():
            raise ValueError("Messages must be nonempty strings")
        if set(frame.category) != set(labels):
            raise ValueError("Dataset categories differ from allowed categories")
        return frame

    def splits(self):
        """Yield train/validation indices; each row is validation exactly once."""
        for fold in sorted(self.assignments.fold.unique()):
            validation = np.flatnonzero(self.assignments.fold.to_numpy() == fold)
            train = np.flatnonzero(self.assignments.fold.to_numpy() != fold)
            yield int(fold), train, validation
