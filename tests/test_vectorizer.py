import networkx as nx
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn.base import clone

from simplex_pair_vectorizer import SimplexPairGraphVectorizer


def labeled_path(node_labels=("A", "B", "A"), edge_label="contact"):
    graph = nx.path_graph(3)
    for node, label in enumerate(node_labels):
        graph.nodes[node]["label"] = label
    nx.set_edge_attributes(graph, edge_label, "label")
    return graph


def test_transformer_returns_sparse_count_features_and_clones():
    graph = labeled_path()
    vectorizer = SimplexPairGraphVectorizer(max_simplex_dimension=1, n_jobs=1)

    result = vectorizer.fit_transform([graph, graph])

    assert isinstance(result, csr_matrix)
    assert result.shape == (2, 1 << 14)
    assert result.nnz > 0
    assert (result[0] != result[1]).nnz == 0
    assert clone(vectorizer).get_params() == vectorizer.get_params()


def test_edge_and_node_labels_change_features():
    vectorizer = SimplexPairGraphVectorizer(max_simplex_dimension=1, n_jobs=1)
    graphs = [
        labeled_path(("A", "B", "C"), "contact"),
        labeled_path(("A", "B", "C"), "near"),
        labeled_path(("A", "X", "C"), "contact"),
    ]
    matrix = vectorizer.fit_transform(graphs)

    assert (matrix[0] != matrix[1]).nnz > 0
    assert (matrix[0] != matrix[2]).nnz > 0


def test_distance_limits_and_self_pair_setting():
    graph = labeled_path()
    included = SimplexPairGraphVectorizer(
        max_simplex_dimension=1,
        min_distance=0,
        max_distance=0,
        include_self=True,
        n_jobs=1,
    ).fit_transform([graph])
    excluded = SimplexPairGraphVectorizer(
        max_simplex_dimension=1,
        min_distance=0,
        max_distance=0,
        include_self=False,
        n_jobs=1,
    ).fit_transform([graph])
    distant_only = SimplexPairGraphVectorizer(
        max_simplex_dimension=1,
        min_distance=1,
        max_distance=1,
        n_jobs=1,
    ).fit_transform([graph])

    assert included.sum() > excluded.sum()
    assert distant_only.sum() > 0


def test_isolated_nodes_and_clique_dimensions():
    isolated = nx.Graph()
    isolated.add_node("only", label="A")
    only_self = SimplexPairGraphVectorizer(
        max_simplex_dimension=0,
        include_self=True,
        n_jobs=1,
    ).fit_transform([isolated])
    no_self = SimplexPairGraphVectorizer(
        max_simplex_dimension=0,
        include_self=False,
        n_jobs=1,
    ).fit_transform([isolated])
    assert only_self.sum() == 1
    assert no_self.sum() == 0

    triangle = nx.complete_graph(3)
    nx.set_node_attributes(triangle, "A", "label")
    nx.set_edge_attributes(triangle, "contact", "label")
    counts = [
        SimplexPairGraphVectorizer(
            max_simplex_dimension=dimension,
            n_jobs=1,
        ).fit_transform([triangle]).sum()
        for dimension in range(3)
    ]
    assert counts[0] < counts[1] < counts[2]


def test_stable_node_key_makes_output_independent_of_insertion_order():
    first = nx.Graph()
    first.add_node("b", label="B")
    first.add_node("a", label="A")
    first.add_edge("a", "b", label="x")
    second = nx.Graph()
    second.add_node("a", label="A")
    second.add_node("b", label="B")
    second.add_edge("b", "a", label="x")

    vectorizer = SimplexPairGraphVectorizer(max_simplex_dimension=1, n_jobs=1)
    result = vectorizer.fit_transform([first, second])

    assert (result[0] != result[1]).nnz == 0


def test_custom_node_key_and_dense_output():
    graph = nx.Graph()
    graph.add_node(("node", 2), label="B")
    graph.add_node(("node", 1), label="A")
    graph.add_edge(("node", 1), ("node", 2), label="x")
    vectorizer = SimplexPairGraphVectorizer(
        max_simplex_dimension=1,
        node_key=lambda node: node[1],
        dense=True,
        n_jobs=1,
    )

    result = vectorizer.fit_transform([graph])

    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 1 << 14)
    assert np.count_nonzero(result) > 0


def test_rejects_directed_and_multigraph_inputs():
    vectorizer = SimplexPairGraphVectorizer(n_jobs=1)
    with pytest.raises(TypeError, match="undirected"):
        vectorizer.fit([nx.DiGraph()])
    with pytest.raises(TypeError, match="multigraphs"):
        vectorizer.fit([nx.MultiGraph()])


def test_rejects_non_unique_node_keys():
    graph = nx.path_graph(2)
    vectorizer = SimplexPairGraphVectorizer(node_key=lambda _node: "same", n_jobs=1)

    with pytest.raises(ValueError, match="unique"):
        vectorizer.fit([graph])


def test_parallel_transform_matches_sequential_transform():
    graphs = [labeled_path(), labeled_path(("C", "A", "B")), labeled_path()]
    sequential = SimplexPairGraphVectorizer(max_simplex_dimension=2, n_jobs=1)
    parallel = SimplexPairGraphVectorizer(max_simplex_dimension=2, n_jobs=2)

    sequential_features = sequential.fit_transform(graphs)
    parallel_features = parallel.fit_transform(graphs)

    assert (sequential_features != parallel_features).nnz == 0
