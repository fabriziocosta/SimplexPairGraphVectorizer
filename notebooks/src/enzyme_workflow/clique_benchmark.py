"""Benchmark bounded clique enumeration strategies on cached protein graphs."""

from __future__ import annotations

import pickle
import statistics
import time
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd

from enzyme_workflow.common import active_manifest
from simplex_pair_vectorizer.vectorizer import _bounded_cliques


def _networkx_cliques(graph: nx.Graph, max_dimension: int) -> list[tuple[Any, ...]]:
    return list(_bounded_cliques(graph, max_dimension, clique_enumerator="generic"))


def _intersection_cliques(graph: nx.Graph, max_dimension: int) -> list[tuple[Any, ...]]:
    """Enumerate each bounded clique once using the vectorizer's fast path."""
    return list(_bounded_cliques(graph, max_dimension, clique_enumerator="fast"))


def _sample_graph_rows(project_root: Path, sample_count: int) -> pd.DataFrame:
    manifest_path, ec_level = active_manifest(project_root)
    status_path = project_root / "data" / "processed" / f"graph_build_status_level_{ec_level}.tsv"
    manifest = pd.read_csv(manifest_path, sep="\t", dtype=str).fillna("")
    status = pd.read_csv(status_path, sep="\t", dtype=str).fillna("")
    dataset = manifest.merge(status, on="sample_id", how="inner", suffixes=("", "_build"))
    dataset = dataset[dataset["status"].isin(["created", "skipped_existing"])].copy()

    def resolve(path_value: str) -> Path:
        path = Path(path_value)
        return path if path.is_absolute() else project_root / path

    dataset["graph_file"] = dataset["graph_path"].map(resolve)
    dataset = dataset[dataset["graph_file"].map(Path.is_file)].copy()
    if dataset.empty:
        raise RuntimeError("No cached graph files were found for the active protein manifest")
    dataset["graph_file_size"] = dataset["graph_file"].map(lambda path: path.stat().st_size)
    dataset = dataset.sort_values(["graph_file_size", "sample_id"]).reset_index(drop=True)
    if len(dataset) <= sample_count:
        return dataset
    positions = [
        ((index + 1) * len(dataset)) // (sample_count + 1)
        for index in range(sample_count)
    ]
    return dataset.iloc[positions].reset_index(drop=True)


def benchmark_clique_enumeration(
    project_root: str | Path,
    *,
    sample_count: int = 5,
    max_simplex_dimension: int = 3,
    repeats: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare NetworkX enumeration with bounded bitset intersections.

    Samples cached graphs spread across serialized graph-file sizes. Correctness
    is checked by comparing the full clique sets before timing repeated runs.
    The result tables report per-graph medians and a combined timing summary.
    """
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    if max_simplex_dimension < 0:
        raise ValueError("max_simplex_dimension must be nonnegative")
    if repeats < 1:
        raise ValueError("repeats must be positive")

    root = Path(project_root).resolve()
    rows = []
    for row in _sample_graph_rows(root, sample_count).itertuples(index=False):
        graph_path = Path(row.graph_file)
        with graph_path.open("rb") as stream:
            graph = pickle.load(stream)
        if not isinstance(graph, nx.Graph) or graph.is_directed() or graph.is_multigraph():
            raise TypeError(f"Expected a simple undirected NetworkX graph in {graph_path}")

        expected = _networkx_cliques(graph, max_simplex_dimension)
        candidate = _intersection_cliques(graph, max_simplex_dimension)
        if set(expected) != set(candidate) or len(expected) != len(candidate):
            raise AssertionError(f"Clique enumerators disagree for {row.sample_id}")

        # Warm both paths before measurement, and record median wall time.
        _networkx_cliques(graph, max_simplex_dimension)
        _intersection_cliques(graph, max_simplex_dimension)
        networkx_times = []
        intersection_times = []
        for _ in range(repeats):
            started = time.perf_counter()
            _networkx_cliques(graph, max_simplex_dimension)
            networkx_times.append(time.perf_counter() - started)

            started = time.perf_counter()
            _intersection_cliques(graph, max_simplex_dimension)
            intersection_times.append(time.perf_counter() - started)

        nx_median = statistics.median(networkx_times)
        intersection_median = statistics.median(intersection_times)
        dimension_counts = pd.Series(
            [len(clique) - 1 for clique in expected], dtype="int64"
        ).value_counts().sort_index()
        row_values = {
            "sample_id": row.sample_id,
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "cliques": len(expected),
            "networkx_median_seconds": nx_median,
            "intersection_median_seconds": intersection_median,
            "speedup": nx_median / intersection_median if intersection_median else float("inf"),
        }
        for dimension in range(max_simplex_dimension + 1):
            row_values[f"dimension_{dimension}_count"] = int(dimension_counts.get(dimension, 0))
        rows.append(row_values)

    results = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {
                "graphs": len(results),
                "max_simplex_dimension": max_simplex_dimension,
                "repeats": repeats,
                "networkx_total_median_seconds": results["networkx_median_seconds"].sum(),
                "intersection_total_median_seconds": results[
                    "intersection_median_seconds"
                ].sum(),
                "aggregate_speedup": results["networkx_median_seconds"].sum()
                / results["intersection_median_seconds"].sum(),
                "cliques_match": True,
            }
        ]
    )
    return results, summary
