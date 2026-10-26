"""Dataset and complete grouped-fold assignment checks."""

import tomllib
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from notebooks.utils.data import ROOT, ExperimentData, normalize_group


def test_every_row_is_validation_once_and_groups_stay_together(tmp_path):
    config = tomllib.loads((ROOT / "config/experiments.toml").read_text())
    config = deepcopy(config)
    config["fold_file"] = str(tmp_path / "folds.csv")
    data = ExperimentData.load(config)
    held = np.concatenate([validation for _, _, validation in data.splits()])
    np.testing.assert_array_equal(np.sort(held), np.arange(len(data.frame)))
    assert data.assignments.groupby("group").fold.nunique().eq(1).all()
    for _, train, validation in data.splits():
        assert not set(train) & set(validation)
        assert len(train) + len(validation) == len(data.frame)
    assert normalize_group("  ＣARD\nquery ") == "card query"
    pd.testing.assert_frame_equal(
        data.assignments, ExperimentData.load(config).assignments
    )


def test_rejects_empty_text_and_unknown_categories(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"text": [" "], "category": ["a"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="nonempty"):
        ExperimentData.read_frame(path, np.array(["a"]))
    pd.DataFrame({"text": ["question"], "category": ["unknown"]}).to_csv(
        path, index=False
    )
    with pytest.raises(ValueError, match="categories"):
        ExperimentData.read_frame(path, np.array(["a"]))
