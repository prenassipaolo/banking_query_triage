"""Run: uv run python -m notebooks.model_comparison; add --full to resume all rows."""

import argparse
from pathlib import Path

from .utils.data import ROOT
from .utils.model_comparison import ComparisonConfig, run_comparison

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "config/model_comparison.toml"
    )
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    print(
        run_comparison(
            ComparisonConfig.load(args.config), args.full, args.retry_failed
        ).to_string(index=False)
    )
