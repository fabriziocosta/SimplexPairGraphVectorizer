"""Reusable graph vectorization, IMSTE embedding, and Plotly display."""

from __future__ import annotations

import hashlib
from html import escape
import json
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from plotly.colors import sample_colorscale
import plotly.graph_objects as go
from scipy import sparse

from enzyme_workflow.common import active_manifest, add_project_sources


@dataclass
class ProteinEmbeddingResult:
    embedding: pd.DataFrame
    figure: go.Figure
    cache_dir: Path
    results_dir: Path


def _stable_key(payload: Any) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _file_signature(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _load_dataset(project_root: Path, manifest_path: Path, ec_level: int) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path, sep="\t", dtype=str).fillna("")
    status_path = project_root / "data" / "processed" / f"graph_build_status_level_{ec_level}.tsv"
    if not status_path.is_file():
        raise FileNotFoundError(f"Run notebook 02 first; missing {status_path}")
    status = pd.read_csv(status_path, sep="\t", dtype=str).fillna("")
    dataset = manifest.merge(status, on="sample_id", how="inner", suffixes=("", "_build"))
    dataset = dataset[dataset["status"].isin(["created", "skipped_existing"])].copy()
    dataset = dataset[dataset["graph_path"].map(lambda value: Path(value).is_file())]
    dataset = dataset.sort_values("sample_id").reset_index(drop=True)
    if len(dataset) < 2:
        raise RuntimeError("At least two completed protein graphs are needed for a 2D embedding")
    return dataset


def _plotly_figure(embedding: pd.DataFrame, ec_level: int) -> go.Figure:
    classes = sorted(embedding["ec_number"].unique())
    positions = np.linspace(0.02, 0.98, max(len(classes), 2))
    colors = sample_colorscale("Turbo", positions[:len(classes)])
    figure = go.Figure()
    for ec_number, color in zip(classes, colors):
        rows = embedding[embedding["ec_number"] == ec_number]
        customdata = rows[
            ["sample_id", "pdb_id", "uniprot_accession", "protein_name"]
        ].fillna("").to_numpy(dtype=object)
        customdata[:, 3] = [
            "<br>".join(textwrap.wrap(escape(str(name)), width=40))
            for name in customdata[:, 3]
        ]
        figure.add_trace(
            go.Scattergl(
                x=rows["imste_1"],
                y=rows["imste_2"],
                customdata=customdata,
                mode="markers",
                name=str(ec_number),
                marker={
                    "color": color,
                    "size": 8,
                    "opacity": 0.8,
                    "line": {"color": "gray", "width": 0.35},
                },
                hoverlabel={"align": "left", "font": {"size": 12}},
                hovertemplate=(
                    "EC: %{fullData.name}<br>"
                    "Protein: %{customdata[3]}<br>"
                    "UniProt: %{customdata[2]}<br>"
                    "PDB: %{customdata[1]}<br>"
                    "Sample: %{customdata[0]}<extra></extra>"
                ),
            )
        )

    coordinates = embedding[["imste_1", "imste_2"]].to_numpy()
    minimum = np.nanmin(coordinates, axis=0)
    maximum = np.nanmax(coordinates, axis=0)
    padding = np.maximum((maximum - minimum) * 0.03, 1e-9)
    figure.update_layout(
        template="plotly_white",
        title=f"Protein graph IMSTE · EC level {ec_level}",
        width=1100,
        height=800,
        margin={"l": 55, "r": 260, "t": 65, "b": 55},
        legend={
            "title": "EC number",
            "x": 1.02,
            "y": 1,
            "xanchor": "left",
            "yanchor": "top",
            "itemclick": "toggle",
            "itemdoubleclick": "toggleothers",
        },
        xaxis={"title": "IMSTE 1", "range": [minimum[0] - padding[0], maximum[0] + padding[0]]},
        yaxis={
            "title": "IMSTE 2",
            "range": [minimum[1] - padding[1], maximum[1] + padding[1]],
            "scaleanchor": "x",
            "scaleratio": 1,
        },
    )
    return figure


def visualize_proteins_2d(
    project_root: str | Path,
    *,
    node_label_alphabet: str = "aa20",
    max_simplex_dimension: int = 3,
    min_pair_distance: int = 0,
    max_pair_distance: int = 4,
    nbits: int = 14,
    include_self: bool = True,
    imste_parameters: dict[str, Any] | None = None,
    rebuild_features: bool = False,
    rebuild_embedding: bool = False,
) -> ProteinEmbeddingResult:
    """Vectorize active protein graphs, fit/load IMSTE, and create a Plotly figure."""
    root = add_project_sources(project_root, include_imste=True)
    from abstractgraph_graphicalizer.protein import ProteinLabelGraphicalizer, read_graph_pickle
    from imste import IteratedMinimumSpanningTreeEmbedder
    from simplex_pair_vectorizer import SimplexPairGraphVectorizer

    manifest_path, ec_level = active_manifest(root)
    dataset = _load_dataset(root, manifest_path, ec_level)
    parameters = {
        "n_components": 2,
        "n_msts": 3,
        "n_epochs": 200,
        "metric": "cosine",
        "mst_method": "prim",
        "gravity_strength": 0.0,
        "mean_shift_iterations": 0,
        "device": "auto",
        "random_state": 42,
        "pca_init": False,
        "verbose": True,
    }
    if imste_parameters is not None:
        parameters.update(imste_parameters)

    vectorizer_settings = {
        "node_label_alphabet": node_label_alphabet,
        "max_simplex_dimension": max_simplex_dimension,
        "min_pair_distance": min_pair_distance,
        "max_pair_distance": max_pair_distance,
        "nbits": nbits,
        "include_self": include_self,
    }
    input_signature = [
        {"sample_id": row.sample_id, **_file_signature(row.graph_path)}
        for row in dataset.itertuples(index=False)
    ]
    feature_key = _stable_key({"inputs": input_signature, "vectorizer": vectorizer_settings})
    cache_dir = root / "data" / "processed" / "embedding_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    feature_cache = cache_dir / f"features_{feature_key}.joblib"

    if feature_cache.is_file() and not rebuild_features:
        X = joblib.load(feature_cache)
        print(f"Loaded cached feature matrix: {X.shape[0]:,} proteins × {X.shape[1]:,} features")
    else:
        graphs = [read_graph_pickle(path) for path in dataset["graph_path"]]
        labeled_graphs = ProteinLabelGraphicalizer(alphabet=node_label_alphabet).fit_transform(graphs)
        vectorizer = SimplexPairGraphVectorizer(
            max_simplex_dimension=max_simplex_dimension,
            min_distance=min_pair_distance,
            max_distance=max_pair_distance,
            nbits=nbits,
            dense=False,
            include_self=include_self,
            n_jobs=-1,
        )
        X = vectorizer.fit_transform(labeled_graphs)
        temporary_feature_cache = feature_cache.with_suffix(".joblib.tmp")
        joblib.dump(X, temporary_feature_cache)
        temporary_feature_cache.replace(feature_cache)
        print(f"Created and cached feature matrix: {X.shape[0]:,} proteins × {X.shape[1]:,} features")

    if not sparse.issparse(X):
        X = sparse.csr_matrix(X)
    if X.shape[0] != len(dataset):
        raise ValueError("Cached feature matrix row count does not match the active manifest")

    embedding_settings = {"feature_key": feature_key, "imste_parameters": parameters}
    embedding_key = _stable_key(embedding_settings)
    coordinates_cache = cache_dir / f"coordinates_{embedding_key}.npz"
    metadata_cache = cache_dir / f"coordinates_{embedding_key}.json"
    if coordinates_cache.is_file() and metadata_cache.is_file() and not rebuild_embedding:
        with np.load(coordinates_cache) as saved:
            coordinates = saved["coordinates"]
        saved_metadata = json.loads(metadata_cache.read_text(encoding="utf-8"))
        if saved_metadata.get("sample_ids") != dataset["sample_id"].tolist():
            raise ValueError("Cached coordinate row order does not match the active dataset")
        print("Loaded cached IMSTE coordinates")
    else:
        mapper = IteratedMinimumSpanningTreeEmbedder(**parameters)
        coordinates = mapper.fit_transform(X)
        np.savez_compressed(coordinates_cache, coordinates=coordinates)
        metadata = {
            "embedding_key": embedding_key,
            "sample_ids": dataset["sample_id"].tolist(),
            "settings": embedding_settings,
        }
        metadata_cache.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Fitted IMSTE and cached {len(coordinates):,} 2D coordinates")

    embedding = dataset[
        ["sample_id", "pdb_id", "uniprot_accession", "ec_number", "protein_name"]
    ].copy()
    embedding["imste_1"] = coordinates[:, 0]
    embedding["imste_2"] = coordinates[:, 1]
    figure = _plotly_figure(embedding, ec_level)
    results_dir = root / "results" / f"ec_level_{ec_level}_imste_embedding"
    results_dir.mkdir(parents=True, exist_ok=True)
    embedding.to_csv(results_dir / "protein_imste_embedding.tsv", sep="\t", index=False)
    figure.write_html(results_dir / "protein_imste_embedding.html", include_plotlyjs=True)
    print(f"Saved interactive plot and coordinates to {results_dir}")
    return ProteinEmbeddingResult(embedding, figure, cache_dir, results_dir)

