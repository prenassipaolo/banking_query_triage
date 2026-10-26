"""Optional asynchronous OpenAI Batch transport; never pretend turnaround is online latency."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from openai import OpenAI

from banking_query_triage.llm import (
    BackendFailure,
    openai_output,
    openai_request,
    parse_classifications,
)
from banking_query_triage.settings import Settings

from .model_comparison import (
    ComparisonConfig,
    append_record,
    load_splits,
    prediction_rows,
    summarize,
)


def prepare_batch(config: ComparisonConfig, full: bool = False) -> Path:
    """Write one uploadable JSONL job and a row map, without making any provider calls."""
    labels, frames = load_splits(config, full)
    settings = Settings.load(config.root / "config/serving.toml")
    directory = config.output / ("openai_batch_full" if full else "openai_batch_pilot")
    requests, chunks = [], {}
    for split, frame in frames.items():
        for offset in range(0, len(frame), config.batch_size):
            chunk = frame.iloc[offset : offset + config.batch_size]
            custom_id = f"{split}-{offset}"
            chunks[custom_id] = {"split": split, "row_ids": chunk.row_id.tolist()}
            requests.append(
                {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": openai_request(
                        labels,
                        chunk.text.tolist(),
                        settings.openai.model,
                        settings.openai.max_output_tokens,
                    ),
                }
            )
    input_text = "".join(json.dumps(request) + "\n" for request in requests)
    manifest = {
        "labels": labels,
        "model": settings.openai.model,
        "chunks": chunks,
        "signature": hashlib.sha256(input_text.encode()).hexdigest(),
        "truth": {
            split: dict(zip(frame.row_id.astype(str), frame.category, strict=True))
            for split, frame in frames.items()
        },
        "transport": "OpenAI Batch, 24h completion window; no per-request latency measurements",
    }
    if directory.exists():
        if validate_batch_input(directory) == manifest:
            return directory
        raise FileExistsError(
            f"Different Batch evidence exists: {directory}; choose a new output_dir"
        )
    directory.mkdir(parents=True)
    (directory / "input.jsonl").write_text(input_text)
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return directory


def validate_batch_input(directory: Path) -> dict:
    """Bind upload/collection to the prepared file before any paid operation or row join."""
    manifest = json.loads((directory / "manifest.json").read_text())
    with (directory / "input.jsonl").open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != manifest["signature"]:
        raise ValueError(
            "Prepared Batch input changed; refusing mismatched submission or collection"
        )
    return manifest


def submit_batch(directory: Path, client: OpenAI) -> dict:
    """Upload and submit once; preserve IDs to avoid accidental duplicate paid jobs."""
    validate_batch_input(directory)
    job_path = directory / "job.json"
    if job_path.exists():
        raise FileExistsError(
            "Job already submitted; use status/collect rather than resubmit"
        )
    started = directory / "submission_started.json"
    if started.exists():
        raise FileExistsError(
            "A submission was attempted without a saved job ID; inspect the OpenAI dashboard before resubmitting"
        )
    input_path = directory / "input_file.json"
    if input_path.exists():
        file_id = json.loads(input_path.read_text())["id"]
    else:
        with (directory / "input.jsonl").open("rb") as stream:
            uploaded = client.files.create(file=stream, purpose="batch")
        file_id = uploaded.id
        input_path.write_text(json.dumps({"id": file_id}))
    started.write_text(json.dumps({"input_file_id": file_id}))
    job = client.batches.create(
        input_file_id=file_id, endpoint="/v1/responses", completion_window="24h"
    )
    job_path.write_text(job.model_dump_json(indent=2))
    return {"id": job.id, "status": job.status}


def batch_status(directory: Path, client: OpenAI):
    """Refresh the recorded job, without polling in a hidden background loop."""
    path = directory / "job.json"
    job = client.batches.retrieve(json.loads(path.read_text())["id"])
    path.write_text(job.model_dump_json(indent=2))
    return job


def collect_batch(directory: Path, client: OpenAI) -> pd.DataFrame:
    """Match unordered outputs by custom ID, preserving failed/missing rows in metrics."""
    manifest = validate_batch_input(directory)
    job = batch_status(directory, client)
    if job.status not in {"completed", "expired", "failed", "cancelled"}:
        raise ValueError(
            f"Batch is {job.status}; collect again when it reaches a terminal state"
        )
    responses = {}
    for filename, file_id in (
        ("output.jsonl", job.output_file_id),
        ("errors.jsonl", job.error_file_id),
    ):
        if file_id:
            content = client.files.content(file_id).text
            (directory / filename).write_text(content)
            for line in content.splitlines():
                item = json.loads(line)
                custom_id = item["custom_id"]
                if custom_id not in manifest["chunks"] or custom_id in responses:
                    raise ValueError("Unknown or duplicate Batch custom_id")
                responses[custom_id] = item
    records, usage = [], []
    # Rewrite derived evidence from the immutable output files, making collection idempotent.
    evidence = directory / "requests.jsonl"
    evidence.write_text("")
    for custom_id, chunk in manifest["chunks"].items():
        item = responses.get(custom_id, {})
        response = item.get("response") or {}
        predictions, error = [], "missing_or_failed_batch_response"
        body = response.get("body")
        usage.append(
            {
                "custom_id": custom_id,
                "status_code": response.get("status_code"),
                "model": body.get("model") if isinstance(body, dict) else None,
                "usage": body.get("usage") if isinstance(body, dict) else None,
            }
        )
        if response.get("status_code") == 200:
            try:
                body = response["body"]
                predictions = [
                    asdict(prediction)
                    for prediction in parse_classifications(
                        openai_output(body), manifest["labels"], len(chunk["row_ids"])
                    )
                ]
                error = None
            except BackendFailure, TypeError, KeyError:
                error = "invalid_batch_output"
        record = {
            "model": "openai",
            "signature": manifest["signature"],
            "batch_id": custom_id,
            **chunk,
            "predictions": predictions,
            "error": error,
            "elapsed_seconds": None,
        }
        append_record(evidence, record)
        records.append(record)
    frames = {
        split: pd.DataFrame(
            {"row_id": [int(key) for key in truth], "category": list(truth.values())}
        )
        for split, truth in manifest["truth"].items()
    }
    rows = prediction_rows(records, frames, {"openai": manifest["signature"]})
    summary = summarize(
        rows, manifest["labels"], {split: len(frame) for split, frame in frames.items()}
    )
    terminal_time = (
        job.completed_at or job.expired_at or job.cancelled_at or job.failed_at
    )
    summary["job_turnaround_seconds"] = (
        terminal_time - job.created_at if terminal_time else None
    )
    rows.to_csv(directory / "predictions.csv", index=False)
    summary.to_csv(directory / "summary.csv", index=False)
    (directory / "usage.json").write_text(json.dumps(usage, indent=2))
    return summary
