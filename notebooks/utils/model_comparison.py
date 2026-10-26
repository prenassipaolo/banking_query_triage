"""Reproducible API comparison, with append-only batch evidence and failure-aware metrics."""

import hashlib
import json
import platform
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import httpx2
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from banking_query_triage.api import PredictionResponse
from banking_query_triage.llm import prompt_fingerprint

from .data import ROOT, ExperimentData, fingerprint


@dataclass(frozen=True)
class ComparisonConfig:
    """Own dataset selection and request settings, independent of serving configuration."""

    root: Path
    api_url: str
    train_file: str
    test_file: str
    labels_file: str
    output_dir: str
    models: list[str]
    seed: int
    per_class: int
    batch_size: int
    timeout_seconds: float

    @classmethod
    def load(cls, path: Path = ROOT / "config/model_comparison.toml"):
        """Read settings; reject invalid sizes before any paid request."""
        values = tomllib.loads(path.read_text())
        root = (path.resolve().parent / values.pop("root_dir")).resolve()
        config = cls(root=root, **values)
        if config.batch_size < 1 or config.per_class < 1 or config.timeout_seconds <= 0:
            raise ValueError(
                "Batch size, per-class sample and timeout must be positive"
            )
        if (
            not config.models
            or len(set(config.models)) != len(config.models)
            or not set(config.models) <= {"svm", "openai", "local_llm"}
        ):
            raise ValueError("Choose unique supported model names")
        return config

    @property
    def output(self) -> Path:
        """Return the evidence directory shared by pilot and resumable full runs."""
        return self.root / self.output_dir


def select_rows(frame: pd.DataFrame, per_class: int | None, seed: int) -> pd.DataFrame:
    """Keep original row IDs and select the same deterministic rows for every backend."""
    selected = frame.copy()
    selected.insert(0, "row_id", np.arange(len(frame)))
    if per_class is not None:
        selected = pd.concat(
            [
                group.sample(n=min(per_class, len(group)), random_state=seed)
                for _, group in selected.groupby("category", sort=True)
            ]
        )
    return selected.sample(frac=1, random_state=seed).reset_index(drop=True)


def load_splits(config: ComparisonConfig, full: bool = False) -> tuple[list[str], dict]:
    """Read both splits for evaluation only; labels never enter prediction requests."""
    labels = sorted(json.loads((config.root / config.labels_file).read_text()))
    frames = {
        split: select_rows(
            ExperimentData.read_frame(config.root / filename, np.asarray(labels)),
            None if full else config.per_class,
            config.seed,
        )
        for split, filename in (
            ("train", config.train_file),
            ("test", config.test_file),
        )
    }
    return labels, frames


def read_records(path: Path) -> list[dict]:
    """Read completed JSONL records; fail on corruption rather than silently losing evidence."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append_record(path: Path, record: dict) -> None:
    """Save each completed request immediately, without storing messages or credentials."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")


def evidence_signature(config: ComparisonConfig, metadata: dict) -> str:
    """Bind cached predictions to data, model identity, prompt and request configuration."""
    description = {
        "datasets": {
            name: fingerprint(config.root / filename)
            for name, filename in (
                ("train", config.train_file),
                ("test", config.test_file),
                ("labels", config.labels_file),
            )
        },
        "metadata": metadata,
        "batch_size": config.batch_size,
        "timeout_seconds": config.timeout_seconds,
    }
    return hashlib.sha256(json.dumps(description, sort_keys=True).encode()).hexdigest()


def prediction_rows(
    records: list[dict], frames: dict, signatures: dict
) -> pd.DataFrame:
    """Join evidence by row ID, with the newest attempt winning after an explicit retry."""
    rows = {}
    truth = {
        split: frame.set_index("row_id").category.to_dict()
        for split, frame in frames.items()
    }
    for record in records:
        model, split = record["model"], record["split"]
        if record.get("signature") != signatures.get(model) or split not in truth:
            continue
        for index, row_id in enumerate(record["row_ids"]):
            if row_id not in truth[split]:
                continue
            prediction = record["predictions"][index] if record["predictions"] else {}
            rows[model, split, row_id] = {
                "model": model,
                "split": split,
                "row_id": row_id,
                "truth": truth[split][row_id],
                "category": prediction.get("category", "__invalid__"),
                "confidence": prediction.get("confidence"),
                "action": prediction.get("action", "review"),
                "error": record.get("error"),
                "batch_id": record["batch_id"],
                "batch_seconds": record.get("elapsed_seconds"),
                "batch_size": len(record["row_ids"]),
            }
    return pd.DataFrame(rows.values())


def summarize(
    rows: pd.DataFrame,
    labels: list[str],
    expected: dict,
    attempts: list[dict] | None = None,
) -> pd.DataFrame:
    """Score latest outcomes per row and account for every matching request attempt.

    Confidence diagnostics compare top-label confidence to correctness, not to a
    made-up multiclass probability vector. Failed rows are excluded only from those
    diagnostics. Batch latency percentiles and amortized time are distinct metrics.
    """
    summaries = []
    if rows.empty:
        return pd.DataFrame()
    for (model, split), group in rows.groupby(["model", "split"], sort=False):
        valid = group[group.error.isna()]
        correct = valid.category.eq(valid.truth).astype(float)
        precision, recall, f1, _ = precision_recall_fscore_support(
            group.truth, group.category, labels=labels, average="macro", zero_division=0
        )
        if attempts is None:
            batches = group.drop_duplicates("batch_id")
        else:
            selected = set(group.row_id)
            batches = pd.DataFrame(
                [
                    {
                        "batch_seconds": record.get("elapsed_seconds"),
                        "batch_size": len(record["row_ids"]),
                        "error": record.get("error"),
                    }
                    for record in attempts
                    if record["model"] == model
                    and record["split"] == split
                    and selected.intersection(record["row_ids"])
                ]
            )
        timed = batches.batch_seconds.dropna()
        routed = group[group.action.eq("route")]
        confidence = valid.confidence.astype(float)
        ece = 0.0
        bin_ids = np.minimum((confidence * 10).astype(int), 9)
        for bin_id in range(10):
            mask = bin_ids == bin_id
            if mask.any():
                ece += mask.mean() * abs(confidence[mask].mean() - correct[mask].mean())
        summaries.append(
            {
                "model": model,
                "split": split,
                "expected_rows": expected[split],
                "attempted_rows": len(group),
                "complete": len(group) == expected[split],
                "valid_rows": len(valid),
                "failure_rate": 1 - len(valid) / len(group),
                "accuracy": accuracy_score(group.truth, group.category),
                "valid_accuracy": accuracy_score(valid.truth, valid.category)
                if len(valid)
                else None,
                "macro_precision": precision,
                "macro_recall": recall,
                "macro_f1": f1,
                "route_coverage": len(routed) / len(group),
                "routed_accuracy": accuracy_score(routed.truth, routed.category)
                if len(routed)
                else None,
                "mean_confidence": confidence.mean() if len(valid) else None,
                "confidence_correctness_brier": ((confidence - correct) ** 2).mean()
                if len(valid)
                else None,
                "confidence_ece_10_bins": ece if len(valid) else None,
                "requests": len(batches),
                "message_executions": int(batches.batch_size.sum()),
                "request_failure_rate": float(batches.error.notna().mean()),
                "total_request_seconds": timed.sum() if len(timed) else None,
                "batch_p50_seconds": timed.quantile(0.5) if len(timed) else None,
                "batch_p95_seconds": timed.quantile(0.95) if len(timed) else None,
                "amortized_ms_per_message": 1000 * timed.sum() / len(group)
                if len(timed)
                else None,
                "messages_per_second": len(group) / timed.sum()
                if timed.sum()
                else None,
            }
        )
    return pd.DataFrame(summaries)


def validate_saved_run(config: ComparisonConfig, manifest: dict) -> None:
    """Reject stale data/settings/prompts before joining current text to saved row IDs."""
    labels = sorted(json.loads((config.root / config.labels_file).read_text()))
    if manifest.get("status") != "complete":
        raise ValueError(
            "Saved comparison is incomplete or has not finished writing results"
        )
    _, expected_frames = load_splits(config, full=manifest["selection"] == "full")
    if manifest["row_ids"] != {
        split: frame.row_id.tolist() for split, frame in expected_frames.items()
    }:
        raise ValueError("Saved evidence uses a different seed or row selection")
    for filename, expected_hash in manifest["result_hashes"].items():
        if fingerprint(config.output / filename) != expected_hash:
            raise ValueError("Saved result CSV does not match its manifest")
    for model, info in manifest["metadata"].items():
        if evidence_signature(config, info) != manifest["signatures"][model]:
            raise ValueError(
                "Saved evidence uses different data or request settings; choose its matching configuration or rerun"
            )
        if model != "svm" and info.get("prompt_sha256") != prompt_fingerprint(labels):
            raise ValueError(
                "Saved evidence uses a different prompt/schema; do not display it as the current prompt"
            )


def save_results(config, records, frames, labels, signatures, name) -> pd.DataFrame:
    """Refresh derived CSVs from checkpointed predictions and all matching attempts."""
    rows = prediction_rows(records, frames, signatures)
    attempts = [
        record
        for record in records
        if record.get("signature") == signatures.get(record["model"])
    ]
    summary = summarize(
        rows, labels, {split: len(frame) for split, frame in frames.items()}, attempts
    )
    rows.to_csv(config.output / f"{name}_predictions.csv", index=False)
    summary.to_csv(config.output / f"{name}_summary.csv", index=False)
    manifest_path = config.output / f"{name}_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected_groups = {(model, split) for model in config.models for split in frames}
    actual_groups = set(zip(summary.model, summary.split, strict=True))
    manifest["status"] = (
        "complete"
        if actual_groups == expected_groups and summary.complete.all()
        else "partial"
    )
    manifest["result_hashes"] = {
        filename: fingerprint(config.output / filename)
        for filename in (f"{name}_predictions.csv", f"{name}_summary.csv")
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return summary


def discover(
    client, config: ComparisonConfig, labels: list[str], completed: dict | None = None
) -> dict:
    """Require live backends for new calls; retain real metadata for complete cached models."""
    response = client.get("/models")
    response.raise_for_status()
    metadata = {item["model"]: item for item in response.json()}
    for model in config.models:
        info = metadata[model]
        saved = (completed or {}).get(model)
        if saved is not None:
            current_identity = {
                key: value
                for key, value in info.items()
                if key not in {"available", "mocked", "confidence_kind"}
            }
            saved_identity = {
                key: value
                for key, value in saved.items()
                if key not in {"available", "mocked", "confidence_kind"}
            }
            current_identity["model_version"] = current_identity[
                "model_version"
            ].removeprefix("mock:")
            if current_identity == saved_identity:
                metadata[model] = saved
                info = saved
        if info["mocked"] or not info["available"]:
            raise ValueError(
                f"{model} must be available and live; restart/configure the API"
            )
        if info["labels"] != labels or config.batch_size > info["max_messages"]:
            raise ValueError("Label list or batch size differs from the API contract")
        if model != "svm" and not info.get("prompt_sha256"):
            raise ValueError("Restart the API to expose the prompt fingerprint")
        if model != "svm" and info["prompt_sha256"] != prompt_fingerprint(labels):
            raise ValueError(
                "Restart the API: its prompt/schema differs from the comparison code"
            )
    return {model: metadata[model] for model in config.models}


def completed_metadata(config, frames, records, previous, retry_failed) -> dict:
    """Find fully cached real models without requiring their credentials again."""
    completed = {}
    for model, info in previous.get("metadata", {}).items():
        if model not in config.models or info["mocked"] or not info["available"]:
            continue
        signature = evidence_signature(config, info)
        if previous.get("signatures", {}).get(model) != signature:
            continue
        rows = prediction_rows(records, frames, {model: signature})
        if rows.empty:
            continue
        if retry_failed:
            rows = rows[rows.error.isna()]
        if all(
            set(rows.loc[rows.split.eq(split), "row_id"]) == set(frame.row_id)
            for split, frame in frames.items()
        ):
            completed[model] = info
    return completed


def run_comparison(
    config: ComparisonConfig, full: bool = False, retry_failed: bool = False
) -> pd.DataFrame:
    """Evaluate a pilot or all rows through the same API, reusing matching evidence.

    Timing includes gateway, provider and validation. The first batch is retained
    and marked in evidence; model downloads/startup are outside this measurement.
    No retries happen unless retry_failed is explicitly requested on a later run.
    """
    labels, frames = load_splits(config, full)
    config.output.mkdir(parents=True, exist_ok=True)
    path = config.output / "requests.jsonl"
    records = read_records(path)
    name = "full" if full else "pilot"
    manifest_path = config.output / f"{name}_manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    completed = completed_metadata(config, frames, records, previous, retry_failed)
    with httpx2.Client(
        base_url=config.api_url.rstrip("/") + "/", timeout=config.timeout_seconds
    ) as client:
        metadata = discover(client, config, labels, completed)
        signatures = {
            model: evidence_signature(config, info) for model, info in metadata.items()
        }
        manifest = {
            "status": "running",
            "selection": name,
            "seed": config.seed,
            "per_class": None if full else config.per_class,
            "batch_size": config.batch_size,
            "metadata": metadata,
            "reused_completed_backends": [
                model for model, info in completed.items() if metadata[model] == info
            ],
            "signatures": signatures,
            "row_ids": {
                split: frame.row_id.tolist() for split, frame in frames.items()
            },
            "created_at": datetime.now(UTC).isoformat(),
            "hardware": {
                "platform": platform.platform(),
                "processor": platform.processor(),
            },
            "timing": "Sequential HTTP batch wall time; no download/startup time. First batch included.",
        }
        (config.output / f"{name}_manifest.json").write_text(
            json.dumps(manifest, indent=2)
        )
        for model in config.models:
            consecutive_failures = 0
            for split, frame in frames.items():
                cached = prediction_rows(records, frames, signatures)
                done = set()
                if not cached.empty:
                    subset = cached[(cached.model == model) & (cached.split == split)]
                    if retry_failed:
                        subset = subset[subset.error.isna()]
                    done = set(subset.row_id)
                pending = frame[~frame.row_id.isin(done)]
                for offset in range(0, len(pending), config.batch_size):
                    chunk = pending.iloc[offset : offset + config.batch_size]
                    start = perf_counter()
                    predictions, error = [], None
                    http_status = None
                    try:
                        response = client.post(
                            "/predict",
                            json={"model": model, "messages": chunk.text.tolist()},
                        )
                        http_status = response.status_code
                        response.raise_for_status()
                        result = PredictionResponse.model_validate(response.json())
                        if (
                            result.mocked
                            or result.model != model
                            or result.model_version != metadata[model]["model_version"]
                        ):
                            raise RuntimeError(
                                "Backend identity changed; refusing mixed or mock results"
                            )
                        if len(result.predictions) != len(chunk) or any(
                            p.category not in labels
                            or not np.isfinite(p.confidence)
                            or not 0 <= p.confidence <= 1
                            for p in result.predictions
                        ):
                            raise ValueError("Invalid prediction contract")
                        predictions = [
                            prediction.__dict__ for prediction in result.predictions
                        ]
                    except (httpx2.HTTPError, ValueError) as failure:
                        error = type(failure).__name__
                    record = {
                        "model": model,
                        "split": split,
                        "signature": signatures[model],
                        "batch_id": f"{model}-{split}-{len(records)}",
                        "row_ids": chunk.row_id.tolist(),
                        "predictions": predictions,
                        "error": error,
                        "http_status": http_status,
                        "elapsed_seconds": perf_counter() - start,
                        "measured_at": datetime.now(UTC).isoformat(),
                    }
                    append_record(path, record)
                    records.append(record)
                    consecutive_failures = consecutive_failures + 1 if error else 0
                    print(
                        f"{model} {split}: {min(offset + len(chunk), len(pending))}/{len(pending)} pending rows; {record['elapsed_seconds']:.2f}s; {error or 'ok'}",
                        flush=True,
                    )
                    if consecutive_failures >= 3:
                        save_results(config, records, frames, labels, signatures, name)
                        raise RuntimeError(
                            "Stopped after three consecutive failures; inspect HTTP status/evidence before retrying"
                        )
        return save_results(config, records, frames, labels, signatures, name)
