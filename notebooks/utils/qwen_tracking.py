"""Local MLflow runs for the Qwen pilot and imports of its saved evidence."""

import json
from contextlib import contextmanager
from pathlib import Path

import mlflow
import pandas as pd

from .data import ROOT


@contextmanager
def tracking_run(config: dict, source: str):
    """Open a parent run in the project's local SQLite tracking store."""
    store = ROOT / config["mlflow_store"]
    store.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{store / 'mlflow.db'}")
    name = config["mlflow_experiment"]
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is None:
        mlflow.create_experiment(name, artifact_location=(store / "runs").as_uri())
    mlflow.set_experiment(name)
    with mlflow.start_run(run_name="qwen-pilot", tags={"source": source}) as run:
        mlflow.log_params(config)
        mlflow.set_tag("backbone", "frozen; final-layer last-token features")
        mlflow.set_tag("test_curves", "retrospective; never used for selection")
        yield run


def log_epoch(row: dict) -> None:
    """Log one epoch to the active fold/refit run when tracking is enabled."""
    if mlflow.active_run() is not None:
        mlflow.log_metrics(
            {key: float(value) for key, value in row.items() if key != "epoch"},
            step=int(row["epoch"]),
        )


def log_summary(summary: dict, output: Path) -> None:
    """Attach final metrics and compact artifacts to the active parent run."""
    mlflow.log_metrics(
        {key: value for key, value in summary.items() if key != "config"}
    )
    for name in (
        "summary.json",
        "cv.csv",
        "cv_history.csv",
        "refit_history.csv",
        "losses.png",
        "head.pt",
    ):
        mlflow.log_artifact(str(output / name))


def log_saved_experiment(output: Path, tracking_config: dict) -> str:
    """Import saved histories without loading Qwen or training another model."""
    summary = json.loads((output / "summary.json").read_text())
    config = summary["config"] | {
        key: tracking_config[key] for key in ("mlflow_store", "mlflow_experiment")
    }
    curves = pd.read_csv(output / "cv_history.csv")
    cv = pd.read_csv(output / "cv.csv")
    final = pd.read_csv(output / "refit_history.csv")
    with tracking_run(config, "saved-artifacts-import") as parent:
        for result in cv.to_dict("records"):
            width, fold = int(result["hidden_size"]), int(result["fold"])
            with mlflow.start_run(
                run_name=f"width-{width}-fold-{fold + 1}", nested=True
            ):
                mlflow.log_params({"hidden_size": width, "fold": fold + 1})
                rows = curves.loc[curves.hidden_size.eq(width) & curves.fold.eq(fold)]
                for row in rows.drop(columns=["hidden_size", "fold"]).to_dict(
                    "records"
                ):
                    log_epoch(row)
                mlflow.log_metrics(
                    {
                        "best_epoch": result["best_epoch"],
                        "best_validation_loss": result["validation_loss"],
                        "best_validation_accuracy": result["validation_accuracy"],
                    }
                )
        with mlflow.start_run(run_name="final-refit", nested=True):
            mlflow.log_params(
                {
                    "hidden_size": summary["selected_hidden_size"],
                    "epochs": summary["refit_epochs"],
                }
            )
            for row in final.to_dict("records"):
                log_epoch(row)
        log_summary(summary, output)
        return parent.info.run_id
