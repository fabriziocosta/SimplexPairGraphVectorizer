"""Compare current feature hashes with AbstractGraph-style SHA-256 hashes."""

from __future__ import annotations

import importlib
import pickle
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import networkx as nx
import pandas as pd

from enzyme_workflow.clique_benchmark import _sample_graph_rows
from simplex_pair_vectorizer import SimplexPairGraphVectorizer
import simplex_pair_vectorizer.vectorizer as vectorizer_module
from simplex_pair_vectorizer.vectorizer import (
    _bounded_cliques,
    _bounded_shortest_path_lengths,
    _node_order,
)


def _abstractgraph_hash_value(project_root: Path) -> Callable[[Any], int]:
    """Load AbstractGraph's canonical SHA-256 helper from its sibling checkout."""
    source = project_root.parent / "abstractgraph-ecosystem" / "repos" / "abstractgraph" / "src"
    if source.is_dir() and str(source) not in sys.path:
        sys.path.insert(0, str(source))
    try:
        return importlib.import_module("abstractgraph.hashing").hash_value
    except ImportError as exc:
        raise ImportError(
            "This experiment uses AbstractGraph's hash_value helper. Install the "
            "abstractgraph package or clone abstractgraph-ecosystem beside this repository."
        ) from exc


def _abstractgraph_style_counts(
    graph: nx.Graph,
    *,
    hash_value: Callable[[Any], int],
    max_simplex_dimension: int,
    max_distance: int,
    nbits: int,
) -> Counter[int]:
    """Hash the same simplex-pair occurrences with typed SHA-256 values."""
    node_keys, ordered_nodes = _node_order(graph, None)
    simplices = list(
        _bounded_cliques(
            graph,
            max_simplex_dimension,
            clique_enumerator="auto",
            node_keys=node_keys,
            ordered_nodes=ordered_nodes,
        )
    )
    by_representative: dict[Any, list[int]] = {node: [] for node in graph.nodes}
    simplex_hashes = []
    for simplex_index, simplex in enumerate(simplices):
        representative = simplex[0]
        node_hashes = {
            node: hash_value(
                (
                    "simplex_node_v1",
                    "representative" if node == representative else "vertex",
                    graph.nodes[node].get("label"),
                )
            )
            for node in simplex
        }
        edge_hashes = []
        for position, left in enumerate(simplex):
            for right in simplex[position + 1 :]:
                edge_label = graph.get_edge_data(left, right, default={}).get("label")
                endpoints = tuple(sorted((node_hashes[left], node_hashes[right])))
                edge_hashes.append(
                    hash_value(("simplex_edge_v1", edge_label, endpoints))
                )
        simplex_hashes.append(
            hash_value(
                (
                    "simplex_atom_v1",
                    tuple(sorted(node_hashes.values())),
                    tuple(sorted(edge_hashes)),
                )
            )
        )
        by_representative[representative].append(simplex_index)

    representatives = [node for node in ordered_nodes if by_representative[node]]
    representative_order = {node: index for index, node in enumerate(ordered_nodes)}
    distances = _bounded_shortest_path_lengths(graph, representatives, max_distance)
    feature_counts: Counter[int] = Counter()
    n_features = 1 << nbits
    for left_representative in representatives:
        left_indices = by_representative[left_representative]
        for right_representative, path_length in distances[left_representative].items():
            right_indices = by_representative.get(right_representative)
            if (
                not right_indices
                or representative_order[right_representative]
                < representative_order[left_representative]
                or path_length > max_distance
            ):
                continue
            for left_position, left_index in enumerate(left_indices):
                start = left_position if left_representative == right_representative else 0
                for right_index in right_indices[start:]:
                    simplex_pair = tuple(
                        sorted((simplex_hashes[left_index], simplex_hashes[right_index]))
                    )
                    feature_hash = hash_value(
                        ("simplex_pair_v1", simplex_pair, float(path_length))
                    )
                    feature_counts[feature_hash % n_features] += 1
    return feature_counts


def benchmark_abstractgraph_hashing(
    project_root: str | Path,
    *,
    sample_count: int = 3,
    max_simplex_dimension: int = 3,
    max_distance: int = 1,
    nbits: int = 14,
    repeats: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare current BLAKE2b/JSON vectorization with AbstractGraph SHA-256."""
    if sample_count < 1 or repeats < 1:
        raise ValueError("sample_count and repeats must be positive")
    if max_simplex_dimension > 3:
        raise ValueError("The experiment uses the fast clique path through dimension 3")
    if max_distance < 0:
        raise ValueError("max_distance must be nonnegative")

    root = Path(project_root).resolve()
    hash_value = _abstractgraph_hash_value(root)
    try:
        import orjson
    except ImportError as exc:
        raise ImportError(
            "Install orjson to compare the accelerated current path: "
            "python -m pip install orjson"
        ) from exc
    original_canonical_json = vectorizer_module._canonical_json

    def native_canonical_json(value: Any) -> str:
        return orjson.dumps(vectorizer_module._normal_form(value)).decode("utf-8")

    vectorizer = SimplexPairGraphVectorizer(
        simplex_dimension=max_simplex_dimension,
        pair_distance=max_distance,
        nbits=nbits,
        n_jobs=1,
    )
    vectorizer.fit([])
    rows = []
    for row in _sample_graph_rows(root, sample_count).itertuples(index=False):
        with Path(row.graph_file).open("rb") as stream:
            graph = pickle.load(stream)
        if not isinstance(graph, nx.Graph) or graph.is_directed() or graph.is_multigraph():
            raise TypeError(f"Expected a simple undirected NetworkX graph in {row.graph_file}")

        vectorizer_module._canonical_json = original_canonical_json
        current = vectorizer._counts(graph)
        vectorizer_module._canonical_json = native_canonical_json
        current_native_json = vectorizer._counts(graph)
        vectorizer_module._canonical_json = original_canonical_json
        abstract_counts = _abstractgraph_style_counts(
            graph,
            hash_value=hash_value,
            max_simplex_dimension=max_simplex_dimension,
            max_distance=max_distance,
            nbits=nbits,
        )
        current_occurrences = sum(current.values())
        abstract_occurrences = sum(abstract_counts.values())
        if current_occurrences != abstract_occurrences:
            raise AssertionError(f"Feature occurrence counts differ for {row.sample_id}")
        if current != current_native_json:
            raise AssertionError(f"orjson changed current feature coordinates for {row.sample_id}")

        current_times = []
        native_json_times = []
        abstract_times = []
        for _ in range(repeats):
            vectorizer_module._canonical_json = original_canonical_json
            started = time.perf_counter()
            vectorizer._counts(graph)
            current_times.append(time.perf_counter() - started)

            vectorizer_module._canonical_json = native_canonical_json
            started = time.perf_counter()
            vectorizer._counts(graph)
            native_json_times.append(time.perf_counter() - started)

            vectorizer_module._canonical_json = original_canonical_json
            started = time.perf_counter()
            _abstractgraph_style_counts(
                graph,
                hash_value=hash_value,
                max_simplex_dimension=max_simplex_dimension,
                max_distance=max_distance,
                nbits=nbits,
            )
            abstract_times.append(time.perf_counter() - started)

        current_median = statistics.median(current_times)
        native_json_median = statistics.median(native_json_times)
        abstract_median = statistics.median(abstract_times)
        rows.append(
            {
                "sample_id": row.sample_id,
                "nodes": graph.number_of_nodes(),
                "edges": graph.number_of_edges(),
                "feature_occurrences": current_occurrences,
                "current_unique_coordinates": len(current),
                "current_orjson_unique_coordinates": len(current_native_json),
                "abstractgraph_unique_coordinates": len(abstract_counts),
                "current_blake2b_median_seconds": current_median,
                "current_blake2b_orjson_median_seconds": native_json_median,
                "abstractgraph_sha256_median_seconds": abstract_median,
                "abstractgraph_speedup": current_median / abstract_median
                if abstract_median
                else float("inf"),
                "abstractgraph_speedup_vs_orjson": native_json_median / abstract_median
                if abstract_median
                else float("inf"),
                "occurrence_counts_match": True,
            }
        )

    results = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {
                "graphs": len(results),
                "dimension_range": f"0–{max_simplex_dimension}",
                "representative_distance_range": f"0–{max_distance}",
                "repeats": repeats,
                "current_total_median_seconds": results[
                    "current_blake2b_median_seconds"
                ].sum(),
                "current_orjson_total_median_seconds": results[
                    "current_blake2b_orjson_median_seconds"
                ].sum(),
                "abstractgraph_total_median_seconds": results[
                    "abstractgraph_sha256_median_seconds"
                ].sum(),
                "abstractgraph_aggregate_speedup": results[
                    "current_blake2b_median_seconds"
                ].sum()
                / results["abstractgraph_sha256_median_seconds"].sum(),
                "abstractgraph_speedup_vs_orjson": results[
                    "current_blake2b_orjson_median_seconds"
                ].sum()
                / results["abstractgraph_sha256_median_seconds"].sum(),
                "occurrence_counts_match": bool(results["occurrence_counts_match"].all()),
            }
        ]
    )
    return results, summary
