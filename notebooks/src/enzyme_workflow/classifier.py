"""Reusable grouped train/test EC classification workflow."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    average_precision_score,
    auc,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, label_binarize

from enzyme_workflow.common import active_manifest, add_project_sources


@dataclass
class ClassificationResult:
    dataset: pd.DataFrame
    class_counts: pd.DataFrame
    class_group_counts: pd.DataFrame
    train_only_classes: list[str]
    train_counts: pd.DataFrame
    test_counts: pd.DataFrame
    report: pd.DataFrame
    confusion: pd.DataFrame
    confusion_fractions: pd.DataFrame
    predictions: pd.DataFrame
    confusion_figure: Any
    confusion_fraction_figure: Any
    roc_figure: Any
    pr_figure: Any
    split_strategy: str
    run_dir: Path
    accuracy: float
    balanced_accuracy: float


def run_classifier(
    project_root: str | Path,
    *,
    test_size: float = 0.20,
    random_state: int = 42,
    node_label_alphabet: str = "aa20",
    base_estimator_kind: str = "linear_sgd",
    max_simplex_dimension: int = 3,
    min_pair_distance: int = 0,
    max_pair_distance: int = 4,
    nbits: int = 14,
    vectorizer_progress: bool = True,
    n_jobs: int | None = -1,
) -> ClassificationResult:
    """Load graph caches, split by UniProt accession, fit, and save a run."""
    root = add_project_sources(project_root)
    from abstractgraph_graphicalizer.protein import ProteinLabelGraphicalizer, read_graph_pickle
    from simplex_pair_vectorizer import SimplexPairGraphVectorizer

    manifest_path, ec_level = active_manifest(root)
    status_path = root / "data" / "processed" / f"graph_build_status_level_{ec_level}.tsv"
    if not status_path.is_file():
        raise FileNotFoundError(f"Run notebook 02 first; missing {status_path}")
    manifest = pd.read_csv(manifest_path, sep="\t", dtype=str).fillna("")
    status = pd.read_csv(status_path, sep="\t", dtype=str).fillna("")
    dataset = manifest.merge(status, on="sample_id", how="inner", suffixes=("", "_build"))
    dataset = dataset[dataset["status"].isin(["created", "skipped_existing"])].copy()
    dataset = dataset[dataset["graph_path"].map(lambda value: Path(value).is_file())].reset_index(drop=True)
    if dataset.empty:
        raise RuntimeError("No completed graph files found for the active manifest")

    graphs = [read_graph_pickle(path) for path in dataset["graph_path"]]
    y = dataset["ec_number"].astype(str).to_numpy()
    groups = dataset["uniprot_accession"].astype(str).to_numpy()
    counts = pd.Series(y).value_counts().sort_index()
    if len(counts) < 2:
        raise ValueError("Classification needs at least two EC classes")
    class_group_counts = dataset.groupby("ec_number")["uniprot_accession"].nunique().sort_index()
    train_only_classes = class_group_counts[class_group_counts < 2].index.astype(str).tolist()

    if class_group_counts.min() >= 5:
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
        train_idx, test_idx = next(splitter.split(np.zeros(len(y)), y, groups))
        split_strategy = "StratifiedGroupKFold first fold"
    else:
        warnings.warn(
            "At least one EC class has fewer than five distinct UniProt groups; "
            "choosing a grouped holdout that retains every class in training. "
            "Singleton classes stay in training and are absent from test.",
            stacklevel=2,
        )
        splitter = GroupShuffleSplit(n_splits=100, test_size=test_size, random_state=random_state)
        candidates = list(splitter.split(np.zeros(len(y)), y, groups))
        all_labels = set(y)
        valid = [
            pair for pair in candidates
            if len(pair[1]) > 0 and set(y[pair[0]]) == all_labels
        ]
        if not valid:
            raise ValueError("Could not create a train split containing every EC class")
        train_idx, test_idx = min(
            valid,
            key=lambda pair: (
                -len(set(y[pair[1]])),
                abs(len(pair[1]) / len(y) - test_size),
            ),
        )
        split_strategy = "best class-covered GroupShuffleSplit"

    steps = [
        ("protein_labels", ProteinLabelGraphicalizer(alphabet=node_label_alphabet)),
        (
            "simplex_pair",
            SimplexPairGraphVectorizer(
                max_simplex_dimension=max_simplex_dimension,
                min_distance=min_pair_distance,
                max_distance=max_pair_distance,
                nbits=nbits,
                dense=False,
                include_self=True,
                n_jobs=n_jobs,
                progress=vectorizer_progress,
            ),
        ),
    ]
    if base_estimator_kind == "linear_sgd":
        steps.extend([
            ("normalize", Normalizer(norm="l2")),
            (
                "classifier",
                SGDClassifier(
                    loss="hinge",
                    alpha=1e-4,
                    max_iter=5000,
                    tol=1e-4,
                    class_weight="balanced",
                    average=True,
                    random_state=random_state,
                    n_jobs=n_jobs,
                ),
            ),
        ])
    elif base_estimator_kind == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "XGBoost classification requires xgboost; install the protein extra with "
                "`pip install -e '.[protein]'`."
            ) from exc

        xgb_class_labels = sorted(set(y))
        xgb_class_to_id = {label: index for index, label in enumerate(xgb_class_labels)}
        y_train_fit = np.asarray(
            [xgb_class_to_id[label] for label in y[train_idx]], dtype=np.int32
        )
        class_frequencies = np.bincount(y_train_fit, minlength=len(xgb_class_labels))
        xgb_sample_weight = len(y_train_fit) / (len(xgb_class_labels) * class_frequencies[y_train_fit])
        steps.append(
            (
                "classifier",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=1.0,
                    tree_method="hist",
                    n_jobs=n_jobs,
                    random_state=random_state,
                    verbosity=0,
                ),
            )
        )
    elif base_estimator_kind == "random_forest":
        steps.append(
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=500,
                    max_features="sqrt",
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    random_state=random_state,
                    n_jobs=n_jobs,
                ),
            )
        )
    else:
        raise ValueError(f"Unknown base_estimator_kind={base_estimator_kind!r}")

    model = Pipeline(steps)
    x_train = [graphs[index] for index in train_idx]
    x_test = [graphs[index] for index in test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    print(f"Fitting classifier on {len(x_train):,} training proteins")
    fit_parameters = {}
    if base_estimator_kind == "xgboost":
        fit_parameters = {
            "classifier__sample_weight": xgb_sample_weight,
        }
        model.fit(x_train, y_train_fit, **fit_parameters)
    else:
        model.fit(x_train, y_train)
    print(f"Predicting {len(x_test):,} test proteins")
    model_predictions = model.predict(x_test)
    if base_estimator_kind == "xgboost":
        y_pred = np.asarray(xgb_class_labels, dtype=object)[model_predictions.astype(int)]
    else:
        y_pred = model_predictions
    labels = sorted(set(y))
    report = pd.DataFrame(
        classification_report(y_test, y_pred, labels=labels, output_dict=True, zero_division=0)
    ).T
    confusion = pd.DataFrame(
        confusion_matrix(y_test, y_pred, labels=labels),
        index=[f"true_{label}" for label in labels],
        columns=[f"pred_{label}" for label in labels],
    )
    accuracy = float(accuracy_score(y_test, y_pred))
    balanced_accuracy = float(balanced_accuracy_score(y_test, y_pred))

    row_totals = confusion.to_numpy().sum(axis=1, keepdims=True)
    fraction_values = np.divide(
        confusion.to_numpy(dtype=float),
        row_totals,
        out=np.zeros_like(confusion.to_numpy(dtype=float)),
        where=row_totals != 0,
    )
    confusion_fractions = pd.DataFrame(
        fraction_values,
        index=confusion.index,
        columns=confusion.columns,
    )

    def confusion_heatmap(matrix: np.ndarray, *, title: str, cmap: str, value_format: str):
        class_count = len(labels)
        figure, axis = plt.subplots(figsize=(8, 8), constrained_layout=True)
        image = axis.imshow(matrix, interpolation="nearest", cmap=cmap, aspect="equal")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        axis.set(
            title=title,
            xlabel="Predicted EC number",
            ylabel="True EC number",
            xticks=np.arange(class_count),
            yticks=np.arange(class_count),
            aspect="equal",
        )
        fontsize = max(4, min(9, 180 // max(class_count, 1)))
        axis.set_xticklabels(labels, rotation=90, fontsize=fontsize)
        axis.set_yticklabels(labels, fontsize=fontsize)
        if class_count <= 24:
            threshold = float(np.nanmax(matrix)) / 2 if matrix.size else 0
            for row_index in range(class_count):
                for column_index in range(class_count):
                    value = matrix[row_index, column_index]
                    axis.text(
                        column_index,
                        row_index,
                        value_format.format(value),
                        ha="center",
                        va="center",
                        fontsize=max(5, fontsize - 1),
                        color="white" if value > threshold else "black",
                    )
        return figure

    confusion_figure = confusion_heatmap(
        confusion.to_numpy(),
        title="Confusion matrix · counts",
        cmap="Blues",
        value_format="{:.0f}",
    )
    confusion_fraction_figure = confusion_heatmap(
        confusion_fractions.to_numpy(),
        title="Confusion matrix · row fractions",
        cmap="Blues",
        value_format="{:.2f}",
    )

    model_classes = list(model.named_steps["classifier"].classes_)
    if base_estimator_kind == "xgboost":
        model_classes = [xgb_class_labels[int(class_id)] for class_id in model_classes]
    if hasattr(model, "decision_function"):
        raw_scores = np.asarray(model.decision_function(x_test))
    else:
        raw_scores = np.asarray(model.predict_proba(x_test))
    if raw_scores.ndim == 1:
        positive_class = model_classes[-1]
        if positive_class != labels[-1]:
            raise ValueError("Binary classifier score orientation does not match sorted EC labels")
        y_micro = (y_test == positive_class).astype(int)
        score_micro = raw_scores
    elif len(labels) == 2 and raw_scores.shape[1] == 2:
        y_micro = (y_test == labels[-1]).astype(int)
        positive_index = model_classes.index(labels[-1])
        score_micro = raw_scores[:, positive_index]
    else:
        y_indicator = label_binarize(y_test, classes=labels)
        class_score_columns = [model_classes.index(label) for label in labels]
        aligned_scores = raw_scores[:, class_score_columns]
        y_micro = y_indicator.ravel()
        score_micro = aligned_scores.ravel()

    roc_figure, roc_axis = plt.subplots(figsize=(8, 8), constrained_layout=True)
    pr_figure, pr_axis = plt.subplots(figsize=(8, 8), constrained_layout=True)
    if np.unique(y_micro).size == 2:
        false_positive_rate, true_positive_rate, _ = roc_curve(y_micro, score_micro)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        roc_axis.plot(
            false_positive_rate,
            true_positive_rate,
            linewidth=2,
            label=f"Micro-average ROC (AUC = {roc_auc:.3f})",
        )
        roc_axis.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Random baseline (AUC = 0.5)")
        roc_axis.legend(loc="lower right")
    else:
        roc_axis.text(0.5, 0.5, "ROC is undefined: test targets contain one class", ha="center", va="center")
    roc_axis.set(
        title="Micro-average ROC curve",
        xlabel="False positive rate",
        ylabel="True positive rate",
        xlim=(0, 1),
        ylim=(0, 1),
        aspect="equal",
    )
    roc_axis.grid(alpha=0.2)

    prevalence = float(np.mean(y_micro))
    if np.any(y_micro):
        precision, recall, _ = precision_recall_curve(y_micro, score_micro)
        average_precision = average_precision_score(y_micro, score_micro)
        pr_axis.plot(recall, precision, linewidth=2, label=f"Micro-average PR (AP = {average_precision:.3f})")
    else:
        pr_axis.text(0.5, 0.5, "PR curve is undefined: test set has no positive targets", ha="center", va="center")
    pr_axis.axhline(
        prevalence,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"Random baseline (prevalence = {prevalence:.3f})",
    )
    pr_axis.legend(loc="lower left")
    pr_axis.set(
        title="Micro-average precision–recall curve",
        xlabel="Recall",
        ylabel="Precision",
        xlim=(0, 1),
        ylim=(0, 1),
        aspect="equal",
    )
    pr_axis.grid(alpha=0.2)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = root / "results" / f"ec_level_{ec_level}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions = dataset.iloc[test_idx][
        ["sample_id", "pdb_id", "label_asym_id", "uniprot_accession", "ec_number", "protein_name", "graph_path"]
    ].copy()
    predictions["y_true"] = y_test
    predictions["y_pred"] = y_pred
    predictions["correct"] = predictions["y_true"] == predictions["y_pred"]
    predictions.to_csv(run_dir / "predictions.tsv", sep="\t", index=False)
    report.to_csv(run_dir / "classification_report.tsv", sep="\t")
    confusion.to_csv(run_dir / "confusion_matrix.tsv", sep="\t")
    confusion_fractions.to_csv(run_dir / "confusion_matrix_row_fractions.tsv", sep="\t")
    confusion_figure.savefig(run_dir / "confusion_matrix_counts.png", dpi=180)
    confusion_fraction_figure.savefig(run_dir / "confusion_matrix_row_fractions.png", dpi=180)
    roc_figure.savefig(run_dir / "roc_curve.png", dpi=180)
    pr_figure.savefig(run_dir / "precision_recall_curve.png", dpi=180)
    metadata = {
        "created_at_utc": run_id,
        "ec_level": ec_level,
        "manifest_path": str(manifest_path),
        "target": "ec_number",
        "split_strategy": split_strategy,
        "test_size_requested": test_size,
        "random_state": random_state,
        "node_label_alphabet": node_label_alphabet,
        "max_simplex_dimension": max_simplex_dimension,
        "min_pair_distance": min_pair_distance,
        "max_pair_distance": max_pair_distance,
        "nbits": nbits,
        "feature_encoding_version": SimplexPairGraphVectorizer.FEATURE_ENCODING_VERSION,
        "base_estimator_kind": base_estimator_kind,
        "train_samples": int(len(train_idx)),
        "test_samples": int(len(test_idx)),
        "train_test_group_overlap": sorted(set(groups[train_idx]) & set(groups[test_idx])),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
    }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    joblib.dump(model, run_dir / "model.joblib")

    return ClassificationResult(
        dataset=dataset,
        class_counts=counts.rename("samples").to_frame(),
        class_group_counts=class_group_counts.rename("uniprot_groups").to_frame(),
        train_only_classes=train_only_classes,
        train_counts=pd.Series(y_train).value_counts().sort_index().rename("train_samples").to_frame(),
        test_counts=pd.Series(y_test).value_counts().sort_index().rename("test_samples").to_frame(),
        report=report,
        confusion=confusion,
        confusion_fractions=confusion_fractions,
        predictions=predictions,
        confusion_figure=confusion_figure,
        confusion_fraction_figure=confusion_fraction_figure,
        roc_figure=roc_figure,
        pr_figure=pr_figure,
        split_strategy=split_strategy,
        run_dir=run_dir,
        accuracy=accuracy,
        balanced_accuracy=balanced_accuracy,
    )

