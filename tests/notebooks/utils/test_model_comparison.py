"""Check matched sampling, failure-aware metrics and reuse without paid requests."""

import json
from dataclasses import replace

import httpx2
import pandas as pd
import pytest

from notebooks.utils import model_comparison as comparison


def test_selection_is_stratified_reproducible_and_preserves_original_ids():
    frame = pd.DataFrame(
        {"text": list("abcdefgh"), "category": ["card"] * 4 + ["cash"] * 4}
    )
    selected = comparison.select_rows(frame, 2, 42)
    pd.testing.assert_frame_equal(selected, comparison.select_rows(frame, 2, 42))
    assert selected.category.value_counts().to_dict() == {"card": 2, "cash": 2}
    assert selected.text.tolist() == frame.iloc[selected.row_id].text.tolist()
    assert len(comparison.select_rows(frame, None, 42)) == 8


def test_metrics_count_failed_rows_and_separate_batch_latency_from_amortized_time():
    rows = pd.DataFrame(
        [
            {
                "model": "llm",
                "split": "test",
                "truth": "card",
                "category": "card",
                "confidence": 0.9,
                "error": None,
                "action": "review",
                "batch_id": "a",
                "batch_seconds": 1.0,
                "batch_size": 2,
            },
            {
                "model": "llm",
                "split": "test",
                "truth": "cash",
                "category": "card",
                "confidence": 0.9,
                "error": None,
                "action": "review",
                "batch_id": "a",
                "batch_seconds": 1.0,
                "batch_size": 2,
            },
            {
                "model": "llm",
                "split": "test",
                "truth": "card",
                "category": "__invalid__",
                "confidence": None,
                "error": "invalid",
                "action": "review",
                "batch_id": "b",
                "batch_seconds": 2.0,
                "batch_size": 1,
            },
        ]
    )
    result = comparison.summarize(rows, ["card", "cash"], {"test": 3}).iloc[0]
    assert result.accuracy == pytest.approx(1 / 3)
    assert result.valid_accuracy == 0.5
    assert result.failure_rate == pytest.approx(1 / 3)
    assert result.confidence_correctness_brier == pytest.approx(0.41)
    assert result.batch_p50_seconds == 1.5
    assert result.amortized_ms_per_message == 1000
    assert result.route_coverage == 0
    assert pd.isna(result.routed_accuracy)
    assert result.complete


def test_live_comparison_reuses_pilot_rows_during_full_run(tmp_path, monkeypatch):
    labels = ["card", "cash"]
    frame = pd.DataFrame(
        {"text": ["a", "b", "c", "d"], "category": ["card", "card", "cash", "cash"]}
    )
    for name in ("train.csv", "test.csv"):
        frame.to_csv(tmp_path / name, index=False)
    (tmp_path / "labels.json").write_text(json.dumps(labels))
    config = comparison.ComparisonConfig(
        tmp_path,
        "http://test",
        "train.csv",
        "test.csv",
        "labels.json",
        "results",
        ["svm", "openai", "local_llm"],
        42,
        1,
        2,
        10,
    )
    calls = []
    metadata = [
        {
            "model": model,
            "model_version": "v1",
            "labels": labels,
            "mocked": False,
            "available": True,
            "max_messages": 32,
            "prompt_sha256": comparison.prompt_fingerprint(labels),
        }
        for model in config.models
    ]

    def respond(request):
        if request.url.path == "/models":
            return httpx2.Response(200, json=metadata)
        body = json.loads(request.content)
        calls.append(body)
        assert set(body) == {"model", "messages"}  # No ground truth sent to the model.
        return httpx2.Response(
            200,
            json={
                "model": body["model"],
                "model_version": "v1",
                "mocked": False,
                "predictions": [
                    {"category": "card", "confidence": 0.8, "action": "review"}
                    for _ in body["messages"]
                ],
            },
        )

    original_client = httpx2.Client
    monkeypatch.setattr(
        comparison.httpx2,
        "Client",
        lambda **kwargs: original_client(
            **kwargs, transport=httpx2.MockTransport(respond)
        ),
    )
    first = comparison.run_comparison(config)
    assert len(calls) == 6
    assert first.complete.all()
    pd.testing.assert_frame_equal(first, comparison.run_comparison(config))
    assert len(calls) == 6
    full = comparison.run_comparison(config, full=True)
    assert len(calls) == 12
    assert full.attempted_rows.tolist() == [4] * 6
    assert full.complete.all()
    metadata[1]["mocked"] = True
    metadata[1]["model_version"] = "mock:v1"
    comparison.run_comparison(config)
    assert len(calls) == 12  # Completed real results do not need a fresh key.
    metadata[1]["model_version"] = "mock:changed-model"
    with pytest.raises(ValueError, match="live"):
        comparison.run_comparison(config)
    metadata[1]["model_version"] = "v1"
    metadata[1]["mocked"] = False
    manifest = json.loads((config.output / "full_manifest.json").read_text())
    comparison.validate_saved_run(config, manifest)
    with pytest.raises(ValueError, match="seed or row selection"):
        comparison.validate_saved_run(replace(config, seed=7), manifest)
    (tmp_path / "train.csv").write_text(
        (tmp_path / "train.csv").read_text().replace("a,card", "changed,card")
    )
    with pytest.raises(ValueError, match="different data"):
        comparison.validate_saved_run(config, manifest)
    metadata[1]["mocked"] = True
    with pytest.raises(ValueError, match="live"):
        comparison.run_comparison(config)


@pytest.mark.parametrize(
    "confidence", [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
)
def test_ece_includes_every_bin_boundary(confidence):
    row = {
        "model": "llm",
        "split": "test",
        "truth": "card",
        "category": "card",
        "confidence": confidence,
        "error": None,
        "action": "review",
        "batch_id": "a",
        "batch_seconds": 1.0,
        "batch_size": 1,
    }
    result = comparison.summarize(pd.DataFrame([row]), ["card"], {"test": 1}).iloc[0]
    assert result.confidence_ece_10_bins == pytest.approx(1 - confidence)


def test_retry_time_accounts_for_failed_attempts():
    row = {
        "model": "llm",
        "split": "test",
        "row_id": 0,
        "truth": "card",
        "category": "card",
        "confidence": 0.9,
        "error": None,
        "action": "review",
        "batch_id": "new",
        "batch_seconds": 1.0,
        "batch_size": 1,
    }
    attempts = [
        {
            "model": "llm",
            "split": "test",
            "row_ids": [0],
            "elapsed_seconds": 2.0,
            "error": "failure",
        },
        {
            "model": "llm",
            "split": "test",
            "row_ids": [0],
            "elapsed_seconds": 1.0,
            "error": None,
        },
    ]
    result = comparison.summarize(
        pd.DataFrame([row]), ["card"], {"test": 1}, attempts
    ).iloc[0]
    assert result.requests == 2
    assert result.message_executions == 2
    assert result.total_request_seconds == 3
    assert result.amortized_ms_per_message == 3000
    assert result.request_failure_rate == 0.5
    assert result.failure_rate == 0


def test_missing_model_split_is_never_marked_complete(tmp_path):
    config = comparison.ComparisonConfig(
        tmp_path,
        "http://unused",
        "train.csv",
        "test.csv",
        "labels.json",
        "out",
        ["svm"],
        42,
        1,
        1,
        10,
    )
    config.output.mkdir()
    (config.output / "pilot_manifest.json").write_text("{}")
    frames = {
        split: pd.DataFrame({"row_id": [0], "category": ["card"]})
        for split in ("train", "test")
    }
    records = [
        {
            "model": "svm",
            "split": "train",
            "signature": "s",
            "batch_id": "a",
            "row_ids": [0],
            "predictions": [
                {"category": "card", "confidence": 0.9, "action": "review"}
            ],
            "error": None,
            "elapsed_seconds": 1,
        }
    ]
    comparison.save_results(config, records, frames, ["card"], {"svm": "s"}, "pilot")
    assert (
        json.loads((config.output / "pilot_manifest.json").read_text())["status"]
        == "partial"
    )
