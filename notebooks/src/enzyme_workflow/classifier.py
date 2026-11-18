"""Reusable grouped train/test EC classification workflow."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

from enzyme_workflow.common import active_manifest, add_project_sources


@dataclass
class ClassificationResult:
    dataset: pd.DataFrame
    class_counts: pd.DataFrame
    train_counts: pd.DataFrame
    test_counts: pd.DataFrame
    report: pd.DataFrame
    confusion: pd.DataFrame
    predictions: pd.DataFrame
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
    if counts.min() < 2:
        raise ValueError(
            "At least one EC class has fewer than two proteins. Increase "
            "PROTEINS_PER_EC in notebook 01 so the class can appear in train and test."
        )

    if counts.min() >= 5:
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
        train_idx, test_idx = next(splitter.split(np.zeros(len(y)), y, groups))
        split_strategy = "StratifiedGroupKFold first fold"
    else:
        warnings.warn(
            "At least one EC class has fewer than five samples; choosing a grouped "
            "holdout that retains every class in training. Some rare classes may not "
            "appear in test.",
            stacklevel=2,
        )
        splitter = GroupShuffleSplit(n_splits=100, test_size=test_size, random_state=random_state)
        candidates = list(splitter.split(np.zeros(len(y)), y, groups))
        all_labels = set(y)
        valid = [pair for pair in candidates if set(y[pair[0]]) == all_labels]
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
                n_jobs=-1,
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
                ),
            ),
        ])
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
                    n_jobs=-1,
                ),
            )
        )
    else:
        raise ValueError(f"Unknown base_estimator_kind={base_estimator_kind!r}")

    model = Pipeline(steps)
    x_train = [graphs[index] for index in train_idx]
    x_test = [graphs[index] for index in test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
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
        train_counts=pd.Series(y_train).value_counts().sort_index().rename("train_samples").to_frame(),
        test_counts=pd.Series(y_test).value_counts().sort_index().rename("test_samples").to_frame(),
        report=report,
        confusion=confusion,
        predictions=predictions,
        split_strategy=split_strategy,
        run_dir=run_dir,
        accuracy=accuracy,
        balanced_accuracy=balanced_accuracy,
    )

