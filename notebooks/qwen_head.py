"""Run the single bounded CPU pilot in config/qwen_head.toml."""

import argparse
import json
import tomllib

from .utils.data import ROOT
from .utils.qwen_tracking import log_saved_experiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-saved",
        action="store_true",
        help="Import saved results into MLflow without training",
    )
    args = parser.parse_args()
    if args.log_saved:
        config = tomllib.loads((ROOT / "config/qwen_head.toml").read_text())
        print("MLflow run:", log_saved_experiment(ROOT / config["output_dir"], config))
    else:
        from .utils.qwen_head import run_experiment

        print(json.dumps(run_experiment(), indent=2))
