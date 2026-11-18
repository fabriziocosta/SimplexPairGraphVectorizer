"""Small Numba experiment for the vectorizer's representative-pair loop."""

from __future__ import annotations

import pickle
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from enzyme_workflow.clique_benchmark import _sample_graph_rows
import simplex_pair_vectorizer.vectorizer as vectorizer_module
from simplex_pair_vectorizer import SimplexPairGraphVectorizer
from simplex_pair_vectorizer.vectorizer import (
    _bounded_cliques,
    _bounded_shortest_path_lengths,
    _node_order,
)


def _prepare_pair_inputs(
    graph: nx.Graph,
    *,
    max_simplex_dimension: int,
    max_distance: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    node_keys, ordered_nodes = _node_order(graph, None)
    simplices_by_representative: dict[Any, list[int]] = {node: [] for node in graph.nodes}
    simplices = list(
        _bounded_cliques(
            graph,
            max_simplex_dimension,
            clique_enumerator="auto",
            node_keys=node_keys,
            ordered_nodes=ordered_nodes,
        )
    )
    for simplex_index, simplex in enumerate(simplices):
        simplices_by_representative[simplex[0]].append(simplex_index)

    representatives = [node for node in ordered_nodes if simplices_by_representative[node]]
    offsets = [0]
    for representative in representatives:
        offsets.append(offsets[-1] + len(simplices_by_representative[representative]))
    representative_index = {node: i for i, node in enumerate(representatives)}
    distances = _bounded_shortest_path_lengths(graph, representatives, max_distance)
    pair_rows = []
    for left in representatives:
        for right, distance in distances[left].items():
            if (
                right in representative_index
                and representative_index[right] >= representative_index[left]
                and distance <= max_distance
            ):
                pair_rows.append((representative_index[left], representative_index[right], distance))

    return (
        np.asarray(offsets, dtype=np.int64),
        np.asarray(pair_rows, dtype=np.int64).reshape((-1, 3)),
        len(simplices),
    )


def _python_pair_indices(
    offsets: np.ndarray,
    representative_pairs: np.ndarray,
    *,
    include_self: bool = True,
) -> np.ndarray:
    result = []
    for left_rep, right_rep, distance in representative_pairs:
        left_start, left_end = offsets[left_rep], offsets[left_rep + 1]
        right_start, right_end = offsets[right_rep], offsets[right_rep + 1]
        for left_index in range(left_start, left_end):
            right_start_for_left = left_index if left_rep == right_rep else right_start
            for right_index in range(right_start_for_left, right_end):
                if not include_self and left_index == right_index:
                    continue
                result.append((left_index, right_index, distance))
    return np.asarray(result, dtype=np.int64).reshape((-1, 3))


def _make_numba_pair_kernel():
    try:
        from numba import njit
    except ImportError as exc:
        raise ImportError(
            "Install Numba to run this experiment: python -m pip install numba"
        ) from exc

    @njit
    def count_pairs(offsets, representative_pairs, include_self):
        count = 0
        for row in range(representative_pairs.shape[0]):
            left_rep = representative_pairs[row, 0]
            right_rep = representative_pairs[row, 1]
            left_size = offsets[left_rep + 1] - offsets[left_rep]
            right_size = offsets[right_rep + 1] - offsets[right_rep]
            if left_rep == right_rep:
                count += left_size * (left_size + 1) // 2 if include_self else left_size * (left_size - 1) // 2
            else:
                count += left_size * right_size
        return count

    @njit
    def emit_pairs(offsets, representative_pairs, include_self):
        count = count_pairs(offsets, representative_pairs, include_self)
        result = np.empty((count, 3), dtype=np.int64)
        cursor = 0
        for row in range(representative_pairs.shape[0]):
            left_rep = representative_pairs[row, 0]
            right_rep = representative_pairs[row, 1]
            distance = representative_pairs[row, 2]
            left_start = offsets[left_rep]
            left_end = offsets[left_rep + 1]
            right_start = offsets[right_rep]
            right_end = offsets[right_rep + 1]
            for left_index in range(left_start, left_end):
                right_start_for_left = left_index if left_rep == right_rep else right_start
                for right_index in range(right_start_for_left, right_end):
                    if not include_self and left_index == right_index:
                        continue
                    result[cursor, 0] = left_index
                    result[cursor, 1] = right_index
                    result[cursor, 2] = distance
                    cursor += 1
        return result

    return emit_pairs


def _profile_graph_counts(graph: nx.Graph) -> dict[str, float | int]:
    original = {}
    totals: dict[str, float] = defaultdict(float)
    calls: dict[str, int] = defaultdict(int)

    def timed(name: str):
        old = getattr(vectorizer_module, name)
        original[name] = old

        def wrapper(*args, **kwargs):
            started = time.perf_counter()
            result = old(*args, **kwargs)
            totals[name] += time.perf_counter() - started
            calls[name] += 1
            return result

        setattr(vectorizer_module, name, wrapper)

    for name in (
        "_bounded_shortest_path_lengths",
        "_simplex_hash",
        "_canonical_json",
        "_hash_index",
    ):
        timed(name)

    old_cliques = vectorizer_module._bounded_cliques
    original["_bounded_cliques"] = old_cliques

    def timed_cliques(*args, **kwargs):
        started = time.perf_counter()
        cliques = list(old_cliques(*args, **kwargs))
        totals["_bounded_cliques"] += time.perf_counter() - started
        calls["_bounded_cliques"] += 1
        return iter(cliques)

    vectorizer_module._bounded_cliques = timed_cliques
    vectorizer = SimplexPairGraphVectorizer(
        simplex_dimension=3, pair_distance=1, n_jobs=1
    )
    vectorizer.fit([])
    started = time.perf_counter()
    try:
        hashed_counts = vectorizer._counts(graph)
    finally:
        for name, function in original.items():
            setattr(vectorizer_module, name, function)
    total_seconds = time.perf_counter() - started
    counted_occurrences = sum(hashed_counts.values())
    accounted = sum(
        totals[name]
        for name in (
            "_bounded_cliques",
            "_simplex_hash",
            "_bounded_shortest_path_lengths",
            "_hash_index",
        )
    )
    return {
        "total_seconds": total_seconds,
        "clique_enumeration_seconds": totals["_bounded_cliques"],
        "simplex_hash_seconds": totals["_simplex_hash"],
        "shortest_path_seconds": totals["_bounded_shortest_path_lengths"],
        "hash_index_seconds": totals["_hash_index"],
        # canonical_json timings overlap because calls nest during normalization.
        "canonical_json_inclusive_seconds": totals["_canonical_json"],
        "pair_generation_and_feature_encoding_seconds": max(0.0, total_seconds - accounted),
        "unique_hashed_coordinates": len(hashed_counts),
        "feature_occurrences": counted_occurrences,
    }


def benchmark_compiled_pair_loop(
    project_root: str | Path,
    *,
    sample_count: int = 3,
    max_simplex_dimension: int = 3,
    max_distance: int = 1,
    repeats: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Profile full vectorization and compare Python/Numba pair-index loops."""
    if sample_count < 1 or repeats < 1:
        raise ValueError("sample_count and repeats must be positive")
    if max_simplex_dimension > 3:
        raise ValueError("This prototype uses the fast clique path through dimension 3")
    if max_distance < 0:
        raise ValueError("max_distance must be nonnegative")

    kernel = _make_numba_pair_kernel()
    rows = []
    graph_rows = _sample_graph_rows(Path(project_root).resolve(), sample_count)
    for row in graph_rows.itertuples(index=False):
        with Path(row.graph_file).open("rb") as stream:
            graph = pickle.load(stream)
        if not isinstance(graph, nx.Graph) or graph.is_directed() or graph.is_multigraph():
            raise TypeError(f"Expected a simple undirected NetworkX graph in {row.graph_file}")

        offsets, representative_pairs, clique_count = _prepare_pair_inputs(
            graph,
            max_simplex_dimension=max_simplex_dimension,
            max_distance=max_distance,
        )
        python_pairs = _python_pair_indices(offsets, representative_pairs)
        # First call includes compilation; exclude it from steady-state timings.
        compiled_pairs = kernel(offsets, representative_pairs, True)
        if not np.array_equal(python_pairs, compiled_pairs):
            raise AssertionError(f"Python and compiled pair loops differ for {row.sample_id}")

        python_times = []
        compiled_times = []
        for _ in range(repeats):
            started = time.perf_counter()
            python_pairs = _python_pair_indices(offsets, representative_pairs)
            python_times.append(time.perf_counter() - started)
            started = time.perf_counter()
            compiled_pairs = kernel(offsets, representative_pairs, True)
            compiled_times.append(time.perf_counter() - started)

        profile = _profile_graph_counts(graph)
        python_median = statistics.median(python_times)
        compiled_median = statistics.median(compiled_times)
        rows.append(
            {
                "sample_id": row.sample_id,
                "nodes": graph.number_of_nodes(),
                "edges": graph.number_of_edges(),
                "cliques": clique_count,
                "eligible_representative_pairs": len(representative_pairs),
                "simplex_pair_occurrences": len(compiled_pairs),
                "python_pair_loop_median_seconds": python_median,
                "numba_pair_loop_median_seconds": compiled_median,
                "pair_loop_speedup": python_median / compiled_median if compiled_median else float("inf"),
                **profile,
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
                "python_pair_loop_total_median_seconds": results[
                    "python_pair_loop_median_seconds"
                ].sum(),
                "numba_pair_loop_total_median_seconds": results[
                    "numba_pair_loop_median_seconds"
                ].sum(),
                "aggregate_pair_loop_speedup": results[
                    "python_pair_loop_median_seconds"
                ].sum()
                / results["numba_pair_loop_median_seconds"].sum(),
                "total_vectorization_seconds": results["total_seconds"].sum(),
                "feature_occurrences": int(results["feature_occurrences"].sum()),
            }
        ]
    )
    return results, summary


def benchmark_native_json(
    project_root: str | Path,
    *,
    sample_count: int = 3,
    repeats: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare stdlib JSON with orjson inside the complete vectorizer count path."""
    try:
        import orjson
    except ImportError as exc:
        raise ImportError(
            "Install orjson to run this experiment: python -m pip install orjson"
        ) from exc
    if sample_count < 1 or repeats < 1:
        raise ValueError("sample_count and repeats must be positive")

    original = vectorizer_module._canonical_json

    def native_canonical_json(value: Any) -> str:
        # _normal_form supplies the canonical type tags and sorted dict entries;
        # orjson then emits the same compact UTF-8 JSON representation.
        return orjson.dumps(vectorizer_module._normal_form(value)).decode("utf-8")

    vectorizer = SimplexPairGraphVectorizer(
        simplex_dimension=3, pair_distance=1, n_jobs=1
    )
    vectorizer.fit([])
    rows = []
    try:
        for row in _sample_graph_rows(Path(project_root).resolve(), sample_count).itertuples(
            index=False
        ):
            with Path(row.graph_file).open("rb") as stream:
                graph = pickle.load(stream)
            if not isinstance(graph, nx.Graph) or graph.is_directed() or graph.is_multigraph():
                raise TypeError(f"Expected a simple undirected NetworkX graph in {row.graph_file}")

            # Compare the exact hashed feature counts before recording timings.
            vectorizer_module._canonical_json = original
            expected = vectorizer._counts(graph)
            vectorizer_module._canonical_json = native_canonical_json
            actual = vectorizer._counts(graph)
            if expected != actual:
                raise AssertionError(f"Native JSON changed feature counts for {row.sample_id}")

            standard_times = []
            native_times = []
            for _ in range(repeats):
                vectorizer_module._canonical_json = original
                started = time.perf_counter()
                vectorizer._counts(graph)
                standard_times.append(time.perf_counter() - started)

                vectorizer_module._canonical_json = native_canonical_json
                started = time.perf_counter()
                vectorizer._counts(graph)
                native_times.append(time.perf_counter() - started)

            standard_median = statistics.median(standard_times)
            native_median = statistics.median(native_times)
            rows.append(
                {
                    "sample_id": row.sample_id,
                    "nodes": graph.number_of_nodes(),
                    "edges": graph.number_of_edges(),
                    "stdlib_json_median_seconds": standard_median,
                    "orjson_median_seconds": native_median,
                    "speedup": standard_median / native_median if native_median else float("inf"),
                    "exact_hashed_counts_match": True,
                }
            )
    finally:
        vectorizer_module._canonical_json = original

    results = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {
                "graphs": len(results),
                "dimension_range": "0–3",
                "representative_distance_range": "0–1",
                "repeats": repeats,
                "stdlib_total_median_seconds": results[
                    "stdlib_json_median_seconds"
                ].sum(),
                "orjson_total_median_seconds": results["orjson_median_seconds"].sum(),
                "aggregate_speedup": results["stdlib_json_median_seconds"].sum()
                / results["orjson_median_seconds"].sum(),
                "exact_hashed_counts_match": bool(
                    results["exact_hashed_counts_match"].all()
                ),
            }
        ]
    )
    return results, summary
