"""One small CPU experiment: frozen Qwen features and a PyTorch classifier."""

import copy
import hashlib
import json
import time
import tomllib
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from transformers import AutoModel, AutoTokenizer

from .data import ROOT, ExperimentData, normalize_group
from .qwen_tracking import log_epoch, log_summary, tracking_run


def sample_frame(frame: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    """Sample a fixed number per class without normalized duplicate messages."""
    frame = frame.assign(group=frame.text.map(normalize_group))
    frame = frame.drop_duplicates("group")
    return pd.concat(
        [
            part.sample(n=count, random_state=seed)
            for _, part in frame.groupby("category")
        ]
    ).reset_index(drop=True)


def extract_features(texts: list[str], config: dict, output: Path) -> torch.Tensor:
    """Save each batch atomically and resume frozen last-token feature extraction."""
    identity = {key: config[key] for key in ("model_id", "revision", "max_length")}
    identity.update(texts=texts, pooling="last_non_padding_token_final_layer")
    identity["backbone_dtype"] = "bfloat16"
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache = output / f"features-{digest}.npz"
    chunks = []
    completed = 0
    if cache.exists():
        with np.load(cache) as saved:
            features = torch.from_numpy(saved["features"].copy())
        completed = len(features)
        if completed == len(texts):
            return features
        chunks.append(features)
        print(f"Resuming Qwen features at {completed}/{len(texts)}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"], revision=config["revision"]
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = (
        AutoModel.from_pretrained(
            config["model_id"],
            revision=config["revision"],
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .cpu()
        .eval()
    )
    model.requires_grad_(False)
    batch_size = config["feature_batch_size"]
    started = time.perf_counter()
    with torch.inference_mode():
        for offset in range(completed, len(texts), batch_size):
            tokens = tokenizer(
                texts[offset : offset + batch_size],
                padding=True,
                truncation=True,
                max_length=config["max_length"],
                return_tensors="pt",
            )
            hidden = model(**tokens, use_cache=False).last_hidden_state
            last = tokens["attention_mask"].sum(dim=1) - 1
            chunks.append(hidden[torch.arange(len(last)), last].float().clone())
            # Replacement keeps the previous checkpoint intact if writing is interrupted.
            temporary = cache.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, features=torch.cat(chunks).numpy())
            temporary.replace(cache)
            if (offset - completed) % (batch_size * 10) == 0:
                print(
                    f"Qwen features {offset + len(last)}/{len(texts)}; "
                    f"{time.perf_counter() - started:.0f}s",
                    flush=True,
                )
    return torch.cat(chunks)


def make_head(width: int, hidden_size: int, classes: int, dropout: float) -> nn.Module:
    """Build one hidden layer with dropout and unnormalized class logits."""
    return nn.Sequential(
        nn.LayerNorm(width),
        nn.Linear(width, hidden_size),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_size, classes),
    )


def score(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    """Evaluate cross-entropy and accuracy with dropout disabled."""
    model.eval()
    with torch.no_grad():
        logits = model(x)
        return (
            nn.functional.cross_entropy(logits, y).item(),
            (logits.argmax(1) == y).float().mean().item(),
        )


def fit_head(
    x: torch.Tensor,
    y: torch.Tensor,
    train: np.ndarray,
    validation: np.ndarray | None,
    hidden_size: int,
    config: dict,
    epochs: int | None = None,
) -> tuple[nn.Module, list[dict], list[dict], int]:
    """Fit a fresh head; stop on validation only, or refit for a fixed duration."""
    torch.manual_seed(config["seed"])
    model = make_head(x.shape[1], hidden_size, config["classes"], config["dropout"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    history, snapshots = [], []
    best_loss, best_epoch, stale = float("inf"), 0, 0
    best_state = copy.deepcopy(model.state_dict())
    for epoch in range(1, (epochs or config["max_epochs"]) + 1):
        model.train()
        order = torch.as_tensor(train)[torch.randperm(len(train))]
        for batch in order.split(config["batch_size"]):
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(x[batch]), y[batch])
            loss.backward()
            optimizer.step()
        row = {"epoch": epoch, "train_loss": score(model, x[train], y[train])[0]}
        if validation is not None:
            val_loss, val_accuracy = score(model, x[validation], y[validation])
            row.update(validation_loss=val_loss, validation_accuracy=val_accuracy)
            if val_loss < best_loss - config["min_delta"]:
                best_loss, best_epoch, stale = val_loss, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
        else:
            # Evaluate test retrospectively after ALL fitting and choices are finished.
            snapshots.append(copy.deepcopy(model.state_dict()))
        log_epoch(row)
        history.append(row)
        if validation is not None and stale >= config["patience"]:
            break
    if validation is not None:
        model.load_state_dict(best_state)
    else:
        best_epoch = len(history)
    return model, history, snapshots, best_epoch


def run_experiment(config_path: Path = ROOT / "config/qwen_head.toml") -> dict:
    """Run bounded grouped CV, freeze choices, refit, then inspect held-out test."""
    config = tomllib.loads(config_path.read_text())
    with tracking_run(config, "live-training"):
        return _run_experiment(config)


def _run_experiment(config: dict) -> dict:
    """Extract cached features, select the head, then evaluate held-out data."""
    torch.set_num_threads(config["threads"])
    output = ROOT / config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    labels = np.array(sorted(json.loads((ROOT / config["labels_file"]).read_text())))
    config["classes"] = len(labels)
    mlflow.log_param("classes", config["classes"])
    train = sample_frame(
        ExperimentData.read_frame(ROOT / config["train_file"], labels),
        config["train_per_class"],
        config["seed"],
    )
    groups = train.text.map(normalize_group)
    splitter = StratifiedGroupKFold(
        config["folds"], shuffle=True, random_state=config["seed"]
    )
    splits = list(splitter.split(train.text, train.category, groups))
    for fit, validation in splits:
        if set(train.iloc[fit].category) != set(labels) or set(
            train.iloc[validation].category
        ) != set(labels):
            raise ValueError(
                "Every fold must retain all classes; increase train_per_class"
            )
    train.to_csv(output / "train_sample.csv", index=False)
    x = extract_features(train.text.tolist(), config, output)
    y = torch.tensor(np.searchsorted(labels, train.category), dtype=torch.long)
    histories, results = [], []
    for hidden_size in config["hidden_sizes"]:
        for fold, (fit, validation) in enumerate(splits):
            with mlflow.start_run(
                run_name=f"width-{hidden_size}-fold-{fold + 1}", nested=True
            ):
                mlflow.log_params({"hidden_size": hidden_size, "fold": fold + 1})
                model, history, _, best_epoch = fit_head(
                    x, y, fit, validation, hidden_size, config
                )
                loss, accuracy = score(model, x[validation], y[validation])
                mlflow.log_metrics(
                    {
                        "best_epoch": best_epoch,
                        "best_validation_loss": loss,
                        "best_validation_accuracy": accuracy,
                    }
                )
                results.append(
                    {
                        "hidden_size": hidden_size,
                        "fold": fold,
                        "best_epoch": best_epoch,
                        "validation_loss": loss,
                        "validation_accuracy": accuracy,
                    }
                )
            histories.extend(
                dict(hidden_size=hidden_size, fold=fold, **row) for row in history
            )
            print(
                f"width={hidden_size}, fold={fold}, best epoch={best_epoch}, "
                f"validation loss={loss:.3f}",
                flush=True,
            )
    cv = pd.DataFrame(results)
    selected = int(cv.groupby("hidden_size").validation_loss.mean().idxmin())
    epochs = max(1, int(cv.loc[cv.hidden_size.eq(selected), "best_epoch"].median()))
    with mlflow.start_run(run_name="final-refit", nested=True):
        mlflow.log_params({"hidden_size": selected, "epochs": epochs})
        model, history, snapshots, _ = fit_head(
            x, y, np.arange(len(y)), None, selected, config, epochs=epochs
        )
        # Test labels and messages enter only after architecture and duration are frozen.
        test_raw = ExperimentData.read_frame(ROOT / config["test_file"], labels)
        test_raw = test_raw.loc[~test_raw.text.map(normalize_group).isin(groups)]
        test = sample_frame(test_raw, config["test_per_class"], config["seed"])
        test.to_csv(output / "test_sample.csv", index=False)
        xt = extract_features(test.text.tolist(), config, output)
        yt = torch.tensor(np.searchsorted(labels, test.category), dtype=torch.long)
        final_state = copy.deepcopy(model.state_dict())
        for row, state in zip(history, snapshots, strict=True):
            model.load_state_dict(state)
            row["test_loss"], row["test_accuracy"] = score(model, xt, yt)
            mlflow.log_metrics(
                {"test_loss": row["test_loss"], "test_accuracy": row["test_accuracy"]},
                step=row["epoch"],
            )
        model.load_state_dict(final_state)
        torch.save(
            {
                "state_dict": final_state,
                "labels": labels.tolist(),
                "config": config,
                "hidden_size": selected,
                "epochs": epochs,
            },
            output / "head.pt",
        )
    curves = pd.DataFrame(histories)
    final = pd.DataFrame(history)
    cv.to_csv(output / "cv.csv", index=False)
    curves.to_csv(output / "cv_history.csv", index=False)
    final.to_csv(output / "refit_history.csv", index=False)
    summary = {
        "config": config,
        "selected_hidden_size": selected,
        "refit_epochs": epochs,
        "train_rows": len(train),
        "test_rows": len(test),
        "test_loss": history[-1]["test_loss"],
        "test_accuracy": history[-1]["test_accuracy"],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_losses(curves, final, selected, output)
    log_summary(summary, output)
    return summary


def plot_losses(curves: pd.DataFrame, final: pd.DataFrame, selected: int, output: Path):
    """Separate fold validation curves from retrospective full-refit test curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for fold, group in curves.loc[curves.hidden_size.eq(selected)].groupby("fold"):
        axes[0].plot(
            group.epoch, group.train_loss, alpha=0.65, label=f"train fold {fold + 1}"
        )
        axes[0].plot(
            group.epoch,
            group.validation_loss,
            "--",
            label=f"validation fold {fold + 1}",
        )
    axes[0].set_title(f"Selected width {selected}: grouped CV")
    axes[1].plot(final.epoch, final.train_loss, label="train (all pilot training rows)")
    axes[1].plot(final.epoch, final.test_loss, label="test (retrospective only)")
    axes[1].set_title("Fixed-duration refit; test never controls training")
    for ax in axes:
        ax.set(xlabel="Epoch", ylabel="Cross-entropy loss")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.savefig(output / "losses.png", dpi=140)
    plt.close(fig)
    return fig
