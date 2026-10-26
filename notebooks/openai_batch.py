"""Prepare, submit, inspect or collect one asynchronous OpenAI dataset job."""

import argparse
from pathlib import Path

from openai import OpenAI

from .utils.data import ROOT
from .utils.model_comparison import ComparisonConfig
from .utils.openai_batch import batch_status, collect_batch, prepare_batch, submit_batch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "submit", "status", "collect"])
    parser.add_argument(
        "--config", type=Path, default=ROOT / "config/model_comparison.toml"
    )
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    config = ComparisonConfig.load(args.config)
    directory = config.output / (
        "openai_batch_full" if args.full else "openai_batch_pilot"
    )
    if args.action == "prepare":
        print(prepare_batch(config, args.full))
    else:
        with OpenAI(max_retries=0) as client:
            if args.action == "submit":
                print(submit_batch(directory, client))
            elif args.action == "status":
                job = batch_status(directory, client)
                print(
                    {
                        "id": job.id,
                        "status": job.status,
                        "request_counts": job.request_counts,
                    }
                )
            else:
                print(collect_batch(directory, client).to_string(index=False))
