"""Verify live tracking and saved-history imports against a local MLflow store."""

import json
import tomllib

import pandas as pd
import pytest

torch = pytest.importorskip("torch")
mlflow = pytest.importorskip("mlflow")
pytest.importorskip("transformers")
from notebooks.utils import qwen_head
from notebooks.utils.data import ROOT
from notebooks.utils.qwen_tracking import log_saved_experiment


@pytest.mark.integration
def test_live_and_imported_runs_preserve_histories(tmp_path, monkeypatch):
    config = tomllib.loads((ROOT / "config/qwen_head.toml").read_text())
    for split in ("train", "test"):
        frame = pd.DataFrame(
            [
                {"text": f"{label} {split} {i}", "category": label}
                for label in ("a", "b")
                for i in range(4)
            ]
        )
        path = tmp_path / f"{split}.csv"
        frame.to_csv(path, index=False)
        config[f"{split}_file"] = str(path)
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps(["a", "b"]))
    output = tmp_path / "output"
    config.update(
        labels_file=str(labels),
        output_dir=str(output),
        mlflow_store=str(tmp_path / "tracking"),
        mlflow_experiment="test-qwen",
        train_per_class=4,
        test_per_class=1,
        folds=2,
        hidden_sizes=[2, 4],
        max_epochs=2,
        patience=1,
    )
    path = tmp_path / "config.toml"
    path.write_text(
        "\n".join(f"{key} = {json.dumps(value)}" for key, value in config.items())
    )
    monkeypatch.setattr(
        qwen_head,
        "extract_features",
        lambda texts, *_: torch.tensor(
            [[1.0, 0.0] if text.startswith("a") else [0.0, 1.0] for text in texts]
        ),
    )
    summary = qwen_head.run_experiment(path)
    imported_id = log_saved_experiment(output, config)
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name("test-qwen")
    runs = client.search_runs([experiment.experiment_id])
    parents = [run for run in runs if "mlflow.parentRunId" not in run.data.tags]
    assert len(runs) == 12  # Two parents, each with four folds and one refit.
    assert {run.data.tags["source"] for run in parents} == {
        "live-training",
        "saved-artifacts-import",
    }
    assert imported_id in {run.info.run_id for run in parents}
    for parent in parents:
        assert parent.data.metrics["test_accuracy"] == summary["test_accuracy"]
        assert {a.path for a in client.list_artifacts(parent.info.run_id)} >= {
            "head.pt",
            "losses.png",
        }
        children = [
            r
            for r in runs
            if r.data.tags.get("mlflow.parentRunId") == parent.info.run_id
        ]
        for child in children:
            assert child.info.status == "FINISHED"
            history = client.get_metric_history(child.info.run_id, "train_loss")
            assert [m.step for m in history] == list(range(1, len(history) + 1))
            if child.data.tags["mlflow.runName"] == "final-refit":
                test = client.get_metric_history(child.info.run_id, "test_loss")
                assert len(test) == summary["refit_epochs"]
                assert test[-1].value == summary["test_loss"]
            else:
                assert "best_validation_loss" in child.data.metrics
                assert "test_loss" not in child.data.metrics
    assert mlflow.active_run() is None
