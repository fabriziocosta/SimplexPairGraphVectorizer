"""Scikit-learn transformer for rooted pairs of labeled graph cliques."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import networkx as nx
import numpy as np
from joblib import Parallel, delayed, effective_n_jobs
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


def _normal_form(value: Any) -> Any:
    """Convert common values to a deterministic JSON-safe form."""
    if value is None or isinstance(value, (str, bool, int)):
        return [type(value).__name__, value]
    if isinstance(value, float):
        return ["float", value.hex()] if math.isfinite(value) else ["float", repr(value)]
    if isinstance(value, tuple):
        return ["tuple", [_normal_form(item) for item in value]]
    if isinstance(value, list):
        return ["list", [_normal_form(item) for item in value]]
    if isinstance(value, dict):
        entries = [(_canonical_json(key), _normal_form(item)) for key, item in value.items()]
        entries.sort(key=lambda pair: pair[0])
        return ["dict", entries]
    if isinstance(value, (set, frozenset)):
        entries = sorted((_normal_form(item) for item in value), key=_canonical_json)
        return [type(value).__name__, entries]
    # Custom values can be used when their repr is stable across runs.
    return [f"{type(value).__module__}.{type(value).__qualname__}", repr(value)]


def _canonical_json(value: Any) -> str:
    return json.dumps(_normal_form(value), ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _validate_graph(graph: Any) -> nx.Graph:
    if not isinstance(graph, nx.Graph):
        raise TypeError(f"Expected a NetworkX graph, got {type(graph).__name__}")
    if graph.is_directed():
        raise TypeError("SimplexPairGraphVectorizer requires undirected graphs")
    if graph.is_multigraph():
        raise TypeError("SimplexPairGraphVectorizer requires simple graphs, not multigraphs")
    return graph


def _node_order(
    graph: nx.Graph,
    node_key: Callable[[Any], Any] | None,
) -> tuple[dict[Any, str], list[Any]]:
    keys: dict[Any, str] = {}
    for node in graph.nodes:
        key_value = node if node_key is None else node_key(node)
        keys[node] = _canonical_json(key_value)
    if len(set(keys.values())) != len(keys):
        raise ValueError("node_key must return a unique, stable key for every node in each graph")
    return keys, sorted(graph.nodes, key=keys.__getitem__)


def _simplex_hash(
    graph: nx.Graph,
    simplex: Sequence[Any],
    root: Any,
    node_label_key: str,
    edge_label_key: str,
) -> bytes:
    """Hash a root-marked simplex from unordered node and edge signatures."""
    if root not in simplex:
        raise ValueError("root must be a vertex of the simplex")

    node_hashes = {}
    for node in simplex:
        node_payload = {
            "kind": "simplex_node_v1",
            "role": "root" if node == root else "vertex",
            "label": graph.nodes[node].get(node_label_key),
        }
        node_hashes[node] = hashlib.blake2b(
            _canonical_json(node_payload).encode("utf-8"),
            digest_size=16,
            person=b"simplexnode",
        ).digest()

    edge_hashes = []
    for i, left in enumerate(simplex):
        for right in simplex[i + 1 :]:
            edge_data = graph.get_edge_data(left, right, default={})
            endpoint_hashes = sorted((node_hashes[left].hex(), node_hashes[right].hex()))
            edge_payload = {
                "kind": "simplex_edge_v1",
                "label": edge_data.get(edge_label_key),
                "endpoints": endpoint_hashes,
            }
            edge_hashes.append(
                hashlib.blake2b(
                    _canonical_json(edge_payload).encode("utf-8"),
                    digest_size=16,
                    person=b"simplexedge",
                ).digest()
            )

    signature = _canonical_json(
        {
            "nodes": sorted(value.hex() for value in node_hashes.values()),
            "edges": sorted(value.hex() for value in edge_hashes),
        }
    )
    return hashlib.blake2b(
        signature.encode("utf-8"), digest_size=16, person=b"simplexatom"
    ).digest()


def _graph_counts(
    graph: nx.Graph,
    *,
    max_simplex_dimension: int,
    min_distance: float,
    max_distance: float,
    distance_jitter: str | float | None,
    include_self: bool,
    node_label_key: str,
    edge_label_key: str,
    node_key: Callable[[Any], Any] | None,
) -> Counter[str]:
    graph = _validate_graph(graph)
    node_keys, ordered_nodes = _node_order(graph, node_key)
    if not ordered_nodes:
        return Counter()

    max_size = max_simplex_dimension + 1
    simplices: list[tuple[Any, ...]] = []
    # NetworkX emits cliques in nondecreasing size, so stop at max dimension.
    for clique in nx.enumerate_all_cliques(graph):
        if len(clique) > max_size:
            break
        simplices.append(tuple(sorted(clique, key=node_keys.__getitem__)))

    by_root: dict[Any, list[int]] = {node: [] for node in graph.nodes}
    by_root_representative: dict[Any, dict[Any, list[int]]] = {
        node: {} for node in graph.nodes
    }
    for simplex_index, simplex in enumerate(simplices):
        representative = simplex[0]
        for node in simplex:
            by_root[node].append(simplex_index)
            by_root_representative[node].setdefault(representative, []).append(simplex_index)

    jitter = 0.0 if distance_jitter in (None, "auto") else float(distance_jitter)
    max_hops = math.floor(max_distance - jitter)
    if max_hops < 0:
        return Counter()

    rooted_simplex_hashes = [
        {
            root: _simplex_hash(
                graph,
                simplex,
                root,
                node_label_key,
                edge_label_key,
            )
            for root in simplex
        }
        for simplex in simplices
    ]

    representatives = {simplex[0] for simplex in simplices}
    # Reuse one bounded BFS result for every simplex pair that shares a root.
    distances = {
        source: nx.single_source_shortest_path_length(graph, source, cutoff=max_hops)
        for source in representatives
    }
    counts: Counter[str] = Counter()
    # The common root is part of the feature. A pair is symmetric; each
    # simplex pair is visited once for each possible common root.
    for root in ordered_nodes:
        representative_groups = by_root_representative[root]
        for left_index in by_root[root]:
            left_rep = simplices[left_index][0]
            for right_rep, path_length in distances[left_rep].items():
                right_indices = representative_groups.get(right_rep)
                if not right_indices:
                    continue
                distance = float(path_length) + jitter
                if distance < min_distance or distance > max_distance:
                    continue
                for right_index in right_indices:
                    if right_index < left_index or (right_index == left_index and not include_self):
                        continue
                    simplex_pair = sorted(
                        (
                            rooted_simplex_hashes[left_index][root].hex(),
                            rooted_simplex_hashes[right_index][root].hex(),
                        )
                    )
                    feature = _canonical_json(
                        {
                            "kind": "rooted_simplex_pair_v3",
                            "simplices": simplex_pair,
                            "distance": distance.hex(),
                        }
                    )
                    counts[feature] += 1
    return counts


def _hash_index(feature: str, n_features: int) -> int:
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8, person=b"simplexpair").digest()
    return int.from_bytes(digest, "little", signed=False) % n_features


def _indexed_graph_counts(
    vectorizer: "SimplexPairGraphVectorizer", index: int, graph: nx.Graph
) -> tuple[int, Counter[int]]:
    """Return a graph's input index with compact hashed counts."""
    return index, vectorizer._counts(graph)


class SimplexPairGraphVectorizer(TransformerMixin, BaseEstimator):
    """Map labeled NetworkX graphs to sparse rooted simplex-pair counts.

    A simplex is a clique. A rooted pair contains two simplices sharing a
    vertex. Each simplex is labeled by its node and edge attributes, with the
    root marked. Pair distance is the shortest-path distance between each
    simplex's representative vertex: the vertex with the lowest stable node
    key. Features are deterministically hashed into unsigned coordinates;
    collisions accumulate occurrence counts.

    ``distance_jitter="auto"`` applies no jitter because shortest-path
    distances are discrete. A numeric value is added before distance filtering.
    """

    FEATURE_ENCODING_VERSION = 3

    def __init__(
        self,
        max_simplex_dimension: int = 3,
        min_distance: float = 0,
        max_distance: float = 4,
        distance_mode: str = "representative",
        distance_jitter: str | float | None = "auto",
        pairing_mode: str = "rooted",
        nbits: int = 14,
        dense: bool = False,
        include_self: bool = True,
        cache_graphs: bool = False,
        n_jobs: int | None = -1,
        progress: bool = False,
        node_label_key: str = "label",
        edge_label_key: str = "label",
        node_key: Callable[[Any], Any] | None = None,
    ) -> None:
        self.max_simplex_dimension = max_simplex_dimension
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.distance_mode = distance_mode
        self.distance_jitter = distance_jitter
        self.pairing_mode = pairing_mode
        self.nbits = nbits
        self.dense = dense
        self.include_self = include_self
        self.cache_graphs = cache_graphs
        self.n_jobs = n_jobs
        self.progress = progress
        self.node_label_key = node_label_key
        self.edge_label_key = edge_label_key
        self.node_key = node_key

    def _validate_parameters(self) -> None:
        if not isinstance(self.max_simplex_dimension, int) or self.max_simplex_dimension < 0:
            raise ValueError("max_simplex_dimension must be a nonnegative integer")
        if not isinstance(self.nbits, int) or not 1 <= self.nbits <= 30:
            raise ValueError("nbits must be an integer between 1 and 30")
        if (
            not math.isfinite(float(self.min_distance))
            or not math.isfinite(float(self.max_distance))
            or self.min_distance < 0
            or self.max_distance < self.min_distance
        ):
            raise ValueError("distance limits must satisfy 0 <= min_distance <= max_distance")
        if self.distance_mode != "representative":
            raise ValueError("Only distance_mode='representative' is currently supported")
        if self.pairing_mode != "rooted":
            raise ValueError("Only pairing_mode='rooted' is currently supported")
        if self.distance_jitter not in (None, "auto"):
            try:
                jitter = float(self.distance_jitter)
            except (TypeError, ValueError) as exc:
                raise ValueError("distance_jitter must be 'auto', None, or a number") from exc
            if not math.isfinite(jitter) or jitter < 0:
                raise ValueError("numeric distance_jitter must be finite and nonnegative")
        if not isinstance(self.progress, (bool, np.bool_)):
            raise ValueError("progress must be a boolean")

    def _counts(self, graph: nx.Graph) -> Counter[int]:
        feature_counts = _graph_counts(
            graph,
            max_simplex_dimension=self.max_simplex_dimension,
            min_distance=float(self.min_distance),
            max_distance=float(self.max_distance),
            distance_jitter=self.distance_jitter,
            include_self=self.include_self,
            node_label_key=self.node_label_key,
            edge_label_key=self.edge_label_key,
            node_key=self.node_key,
        )
        hashed_counts: Counter[int] = Counter()
        for feature, count in feature_counts.items():
            hashed_counts[_hash_index(feature, self.n_features_out_)] += count
        return hashed_counts

    def fit(self, X: Iterable[nx.Graph], y: Any = None) -> "SimplexPairGraphVectorizer":
        """Validate the input; hashing requires no learned vocabulary."""
        self._validate_parameters()
        graphs = list(X)
        for graph in graphs:
            _validate_graph(graph)
            _node_order(graph, self.node_key)
        self.n_features_out_ = 1 << self.nbits
        self.n_features_in_ = 1
        self._graph_cache_: dict[int, tuple[nx.Graph, Counter[int]]] = {}
        if self.cache_graphs:
            for graph in graphs:
                self._graph_cache_[id(graph)] = (graph, self._counts(graph))
        return self

    def transform(self, X: Iterable[nx.Graph]) -> csr_matrix | np.ndarray:
        """Transform graphs into a CSR matrix, or dense array if requested."""
        check_is_fitted(self, "n_features_out_")
        graphs = list(X)

        def counts_for(graph: nx.Graph) -> Counter[int]:
            _validate_graph(graph)
            if self.cache_graphs:
                cached = self._graph_cache_.get(id(graph))
                if cached is not None and cached[0] is graph:
                    return cached[1]
                counts = self._counts(graph)
                self._graph_cache_[id(graph)] = (graph, counts)
                return counts
            return self._counts(graph)

        if self.cache_graphs or self.n_jobs == 1 or len(graphs) < 2:
            graph_iterable = graphs
            if self.progress:
                from tqdm.auto import tqdm

                graph_iterable = tqdm(graphs, desc="Vectorizing protein graphs", unit="protein")
            rows = [counts_for(graph) for graph in graph_iterable]
        else:
            # Feature extraction is dominated by Python and NetworkX work, so
            # threads contend on the GIL. Process workers let separate proteins
            # be counted on separate CPU cores.
            results = Parallel(
                n_jobs=self.n_jobs,
                prefer="processes",
                return_as="generator_unordered",
                pre_dispatch=effective_n_jobs(self.n_jobs),
                batch_size=1,
            )(
                delayed(_indexed_graph_counts)(self, index, graph)
                for index, graph in enumerate(graphs)
            )
            if self.progress:
                from tqdm.auto import tqdm

                results = tqdm(
                    results,
                    total=len(graphs),
                    desc="Vectorizing protein graphs",
                    unit="protein",
                )
            rows = [None] * len(graphs)
            for index, row in results:
                rows[index] = row

        data: list[int] = []
        indices: list[int] = []
        indptr = [0]
        for row in rows:
            for index in sorted(row):
                indices.append(index)
                data.append(row[index])
            indptr.append(len(indices))

        matrix = csr_matrix(
            (
                np.asarray(data, dtype=np.float64),
                np.asarray(indices, dtype=np.int32),
                np.asarray(indptr, dtype=np.int64),
            ),
            shape=(len(graphs), self.n_features_out_),
            dtype=np.float64,
        )
        return matrix.toarray() if self.dense else matrix

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Return names for each fixed hashed coordinate."""
        check_is_fitted(self, "n_features_out_")
        return np.asarray(
            [f"simplex_pair_{index}" for index in range(self.n_features_out_)],
            dtype=object,
        )
