"""Run the single bounded CPU pilot in config/qwen_head.toml."""

import json

from .utils.qwen_head import run_experiment

if __name__ == "__main__":
    print(json.dumps(run_experiment(), indent=2))
