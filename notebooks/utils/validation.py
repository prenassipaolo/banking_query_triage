"""Five-fold training/validation measurements for each fixed candidate."""

import numpy as np
import pandas as pd

from .experiment import Experiment


class CrossValidator:
    """Fit a fresh model per fold and retain every out-of-fold probability once."""

    def run(self, candidate: Experiment, data) -> tuple:
        """Return individual fold records, pooled probabilities, and stopping curves."""
        out_of_fold = np.full((len(data.frame), len(data.labels)), np.nan)
        records, curves = [], []
        for fold, train, validation in data.splits():
            train_frame, validation_frame = (
                data.frame.iloc[train],
                data.frame.iloc[validation],
            )
            model = candidate.fit(train_frame.text, train_frame.category)
            for phase, frame, indices in [
                ("train", train_frame, train),
                ("validation", validation_frame, validation),
            ]:
                metrics, probabilities = candidate.evaluator.evaluate(
                    model, frame.text, frame.category
                )
                records.append(
                    {
                        "candidate": candidate.name,
                        "fold": fold,
                        "phase": phase,
                        "fit_seconds": model.fit_seconds,
                        "features": model.feature_count,
                        "rounds": model.rounds,
                        "warnings": model.warnings,
                        **metrics,
                    }
                )
                if phase == "validation":
                    out_of_fold[indices] = probabilities
            curves.extend(
                {"candidate": candidate.name, "fold": fold, **row}
                for row in model.history
            )
            print(
                f"{candidate.name}: fold {fold + 1} complete; validation accuracy={records[-1]['accuracy']:.3f}",
                flush=True,
            )
        if not np.isfinite(out_of_fold).all():
            raise ValueError(
                "Every training row must receive one validation prediction"
            )
        return records, out_of_fold, curves

    @staticmethod
    def summarize(records: list[dict], candidate: Experiment) -> dict:
        """Return fold mean/sample standard deviation; folds are not independent replicates."""
        summary = {
            "candidate": candidate.name,
            "feature_extractor": candidate.pipeline.features.name,
            "model": candidate.pipeline.model.name,
        }
        frame = pd.DataFrame(records)
        for phase in ("train", "validation"):
            part = frame.loc[frame.phase == phase]
            for name in part.select_dtypes(include="number").columns:
                if name == "fold" or part[name].isna().all():
                    continue
                summary[f"cv_{phase}_{name}_mean"] = float(part[name].mean())
                summary[f"cv_{phase}_{name}_std"] = float(part[name].std(ddof=1))
        return summary
