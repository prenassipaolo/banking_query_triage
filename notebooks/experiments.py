"""Run from the repository root: uv run python -m notebooks.experiments."""

import argparse
from pathlib import Path

from .utils.data import ROOT
from .utils.workflow import ExperimentRunner

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/experiments.toml")
    ExperimentRunner.from_file(parser.parse_args().config).run()
