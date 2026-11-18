"""Utilities for interactively inspecting representative-paired simplex features."""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import networkx as nx
import numpy as np
import pandas as pd

from enzyme_workflow.common import active_manifest


@dataclass(frozen=True)
class SimplexPair:
    """Two cliques paired by the distance between their representatives."""

    left_representative: Any
    right_representative: Any
    left: tuple[Any, ...]
    right: tuple[Any, ...]
    distance: int


def load_available_proteins(project_root: str | Path) -> pd.DataFrame:
    """Return active manifest rows with both graph and mmCIF caches available."""
    root = Path(project_root).resolve()
    manifest_path, ec_level = active_manifest(root)
    status_path = root / "data" / "processed" / f"graph_build_status_level_{ec_level}.tsv"
    if not status_path.is_file():
        raise FileNotFoundError(f"Run notebook 02 first; missing {status_path}")

    manifest = pd.read_csv(manifest_path, sep="\t", dtype=str).fillna("")
    status = pd.read_csv(status_path, sep="\t", dtype=str).fillna("")
    dataset = manifest.merge(status, on="sample_id", how="inner", suffixes=("", "_build"))
    dataset = dataset[dataset["status"].isin(["created", "skipped_existing"])].copy()

    def resolve_path(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else root / path

    dataset["graph_file"] = dataset["graph_path"].map(resolve_path)
    dataset["mmcif_file"] = dataset["pdb_id"].map(
        lambda pdb_id: root / "data" / "raw" / "mmcif" / f"{str(pdb_id).lower()}.cif.gz"
    )
    dataset = dataset[
        dataset["graph_file"].map(Path.is_file)
        & dataset["mmcif_file"].map(Path.is_file)
    ].reset_index(drop=True)
    if dataset.empty:
        raise RuntimeError(
            "No active proteins have both graph and mmCIF caches. Run notebooks 01 and 02 first."
        )
    return dataset


def load_protein_geometry(
    row: Mapping[str, Any],
) -> tuple[nx.Graph, dict[Any, tuple[float, float, float]]]:
    """Load one cached graph and its C-alpha coordinates from the cached mmCIF."""
    from abstractgraph_graphicalizer.protein import extract_chain_record, read_graph_pickle

    graph_path = Path(row["graph_file"])
    mmcif_path = Path(row["mmcif_file"])
    graph = read_graph_pickle(graph_path)
    record = extract_chain_record(
        mmcif_path,
        pdb_id=str(row["pdb_id"]),
        sample_id=str(row["sample_id"]),
        label_asym_id=str(row.get("label_asym_id", "")),
        auth_asym_id=str(row.get("auth_asym_id", "")),
    )

    coordinates: dict[Any, tuple[float, float, float]] = {}
    for node, attrs in graph.nodes(data=True):
        residue_index = int(attrs.get("residue_index", node))
        if not 0 <= residue_index < len(record.residues):
            raise ValueError(
                f"Graph residue index {residue_index} is outside the parsed chain "
                f"({len(record.residues)} residues) for {row['sample_id']}"
            )
        residue = record.residues[residue_index]
        graph_label_seq_id = str(attrs.get("label_seq_id", ""))
        if graph_label_seq_id and graph_label_seq_id != residue.label_seq_id:
            raise ValueError(
                f"Graph/mmCIF residue order differs at index {residue_index} for "
                f"{row['sample_id']}: graph={graph_label_seq_id}, "
                f"mmCIF={residue.label_seq_id}"
            )
        coordinates[node] = residue.coord
    return graph, coordinates


def cliques_by_representative(
    graph: nx.Graph,
    max_simplex_dimension: int,
    *,
    clique_enumerator: str = "auto",
) -> dict[Any, list[tuple[Any, ...]]]:
    """Assign each simplex once to its lowest-key vertex."""
    if max_simplex_dimension < 0:
        raise ValueError("max_simplex_dimension must be nonnegative")
    from simplex_pair_vectorizer.vectorizer import _bounded_cliques

    by_representative: dict[Any, list[tuple[Any, ...]]] = {node: [] for node in graph.nodes}
    for simplex in _bounded_cliques(
        graph, max_simplex_dimension, clique_enumerator=clique_enumerator
    ):
        by_representative[simplex[0]].append(simplex)
    return {
        representative: simplices
        for representative, simplices in by_representative.items()
        if simplices
    }


def sample_simplex_pair(
    graph: nx.Graph,
    *,
    min_simplex_dimension: int,
    max_simplex_dimension: int,
    min_distance: int,
    max_distance: int,
    rng: random.Random,
    clique_enumerator: str = "auto",
) -> SimplexPair | None:
    """Sample a representative pair and assigned cliques inside the ranges."""
    from simplex_pair_vectorizer.vectorizer import _bounded_shortest_path_lengths

    if min_distance < 0 or max_distance < min_distance:
        raise ValueError("distance range must satisfy 0 <= min_distance <= max_distance")
    if min_simplex_dimension < 0 or max_simplex_dimension < min_simplex_dimension:
        raise ValueError("simplex dimension range must satisfy 0 <= min <= max")
    by_representative = cliques_by_representative(
        graph,
        max_simplex_dimension,
        clique_enumerator=clique_enumerator,
    )
    by_representative = {
        representative: [
            simplex
            for simplex in simplices
            if min_simplex_dimension <= len(simplex) - 1 <= max_simplex_dimension
        ]
        for representative, simplices in by_representative.items()
    }
    by_representative = {
        representative: simplices
        for representative, simplices in by_representative.items()
        if simplices
    }
    representatives = list(by_representative)
    if not representatives:
        return None
    representative_order = {node: index for index, node in enumerate(graph.nodes)}
    representative_pairs: list[tuple[Any, Any]] = []
    distances = _bounded_shortest_path_lengths(graph, representatives, max_distance)
    for left_representative in representatives:
        for right_representative, path_length in distances[left_representative].items():
            if (
                min_distance <= path_length <= max_distance
                and right_representative in by_representative
                and representative_order[right_representative]
                >= representative_order[left_representative]
            ):
                representative_pairs.append((left_representative, right_representative))
    if not representative_pairs:
        return None
    left_representative, right_representative = rng.choice(representative_pairs)
    return SimplexPair(
        left_representative=left_representative,
        right_representative=right_representative,
        left=rng.choice(by_representative[left_representative]),
        right=rng.choice(by_representative[right_representative]),
        distance=nx.shortest_path_length(graph, left_representative, right_representative),
    )


def _edge_key(left: Any, right: Any) -> frozenset[Any]:
    return frozenset((left, right))


def _edge_trace(
    edges: set[tuple[Any, Any]],
    coordinates: Mapping[Any, tuple[float, float, float]],
    *,
    name: str,
    color: str,
    width: float,
    showlegend: bool,
):
    import plotly.graph_objects as go

    x: list[float | None] = []
    y: list[float | None] = []
    z: list[float | None] = []
    for left, right in sorted(edges, key=lambda edge: (str(edge[0]), str(edge[1]))):
        for axis, value in enumerate(coordinates[left]):
            (x, y, z)[axis].append(value)
        for axis, value in enumerate(coordinates[right]):
            (x, y, z)[axis].append(value)
        x.append(None)
        y.append(None)
        z.append(None)
    return go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="lines",
        name=name,
        line={"color": color, "width": width},
        hoverinfo="skip",
        showlegend=showlegend,
    )


def draw_protein_feature(
    graph: nx.Graph,
    coordinates: Mapping[Any, tuple[float, float, float]],
    feature: SimplexPair | None,
    *,
    title: str,
):
    """Build an orbit-rotatable 3D Plotly view with opaque feature elements."""
    import plotly.graph_objects as go
    from plotly.colors import qualitative

    amino_acids = (
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
        "MSE", "SEC", "PYL",
    )
    amino_colors = dict(zip(amino_acids, qualitative.Dark24))
    selected_nodes: set[Any] = set()
    highlighted_edges_a: set[tuple[Any, Any]] = set()
    highlighted_edges_b: set[tuple[Any, Any]] = set()
    representatives = None
    if feature is not None:
        selected_nodes = set(feature.left) | set(feature.right)
        highlighted_edges_a = set(itertools.combinations(feature.left, 2))
        highlighted_edges_b = set(itertools.combinations(feature.right, 2))
        representatives = (feature.left_representative, feature.right_representative)

    selected_edge_keys = {
        _edge_key(left, right)
        for left, right in highlighted_edges_a | highlighted_edges_b
    }
    background_edges = {
        (left, right)
        for left, right in graph.edges
        if _edge_key(left, right) not in selected_edge_keys
    }
    traces = [
        _edge_trace(
            background_edges,
            coordinates,
            name="Other contact edges",
            color="rgba(0,0,0,0.3)",
            width=1.2,
            showlegend=False,
        )
    ]

    background_nodes = [node for node in graph.nodes if node not in selected_nodes]
    selected_order = sorted(selected_nodes, key=lambda node: int(graph.nodes[node].get("residue_index", node)))

    def add_node_trace(nodes, *, name: str, opacity: float, size: float, outline: str):
        if not nodes:
            return
        xyz = np.asarray([coordinates[node] for node in nodes], dtype=float)
        labels = [str(graph.nodes[node].get("label", "UNK")) for node in nodes]
        customdata = [
            [
                label,
                graph.nodes[node].get("residue_index", node),
                graph.nodes[node].get("label_seq_id", ""),
            ]
            for node, label in zip(nodes, labels)
        ]
        traces.append(
            go.Scatter3d(
                x=xyz[:, 0],
                y=xyz[:, 1],
                z=xyz[:, 2],
                mode="markers",
                name=name,
                marker={
                    "size": size,
                    "color": [amino_colors.get(label, "#A0A0A0") for label in labels],
                    "opacity": opacity,
                    "line": {"color": outline, "width": 0.6 if opacity == 1 else 0},
                },
                customdata=customdata,
                hovertemplate=(
                    "%{customdata[0]} · residue %{customdata[1]}"
                    "<br>label_seq_id %{customdata[2]}<extra></extra>"
                ),
                showlegend=False,
            )
        )

    add_node_trace(background_nodes, name="Other residues", opacity=0.3, size=3.0, outline="rgba(0,0,0,0)")

    edge_a_only = {edge for edge in highlighted_edges_a if edge not in highlighted_edges_b}
    edge_b_only = {edge for edge in highlighted_edges_b if edge not in highlighted_edges_a}
    shared_edges = highlighted_edges_a & highlighted_edges_b
    if edge_a_only:
        traces.append(
            _edge_trace(edge_a_only, coordinates, name="Simplex A edges", color="#FFB000", width=5, showlegend=True)
        )
    if edge_b_only:
        traces.append(
            _edge_trace(edge_b_only, coordinates, name="Simplex B edges", color="#00D5FF", width=5, showlegend=True)
        )
    if shared_edges:
        traces.append(
            _edge_trace(shared_edges, coordinates, name="Shared simplex edges", color="#FFFFFF", width=6, showlegend=True)
        )

    add_node_trace(selected_order, name="Feature residues", opacity=1.0, size=6.0, outline="#FFFFFF")
    if representatives is not None:
        for representative, outline_color, marker_name, symbol in zip(
            representatives,
            ("#FFB000", "#00D5FF"),
            ("Representative A", "Representative B"),
            ("circle", "circle"),
        ):
            xyz = coordinates[representative]
            representative_label = str(graph.nodes[representative].get("label", "UNK"))
            traces.append(
                go.Scatter3d(
                    x=[xyz[0]], y=[xyz[1]], z=[xyz[2]],
                    mode="markers",
                    name=marker_name,
                    marker={
                        "size": 10,
                        "symbol": symbol,
                        "color": amino_colors.get(representative_label, "#A0A0A0"),
                        "opacity": 1.0,
                        "line": {"color": outline_color, "width": 3},
                    },
                    hovertemplate=f"{marker_name} · {representative_label}<extra></extra>",
                    showlegend=True,
                )
            )

    figure = go.Figure(data=traces)
    figure.update_layout(
        title={"text": title, "x": 0.02, "xanchor": "left"},
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=780,
        margin={"l": 0, "r": 0, "t": 90, "b": 0},
        legend={"orientation": "h", "y": 1.02, "x": 0.02},
        scene={
            "aspectmode": "data",
            "dragmode": "orbit",
            "bgcolor": "white",
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "zaxis": {"visible": False},
            "camera": {"eye": {"x": 1.55, "y": 1.55, "z": 1.15}},
        },
    )
    return figure


def display_feature_explorer(
    project_root: str | Path,
    *,
    max_representative_distance: int = 6,
    max_simplex_dimension: int = 3,
    clique_enumerator: str = "auto",
) -> None:
    """Display the widget-driven protein and representative-paired simplex explorer."""
    import ipywidgets as widgets
    from IPython.display import Markdown, clear_output, display

    if not isinstance(max_representative_distance, int) or max_representative_distance < 0:
        raise ValueError("max_representative_distance must be a nonnegative integer")
    if not isinstance(max_simplex_dimension, int) or max_simplex_dimension < 0:
        raise ValueError("max_simplex_dimension must be nonnegative")

    proteins = load_available_proteins(project_root)
    rng = random.Random()
    state: dict[str, Any] = {
        "row": None,
        "graph": None,
        "coordinates": None,
        "feature": None,
    }
    plot_output = widgets.Output(layout=widgets.Layout(width="100%"))
    protein_button = widgets.Button(description="Random protein", button_style="primary")
    feature_button = widgets.Button(description="Random feature")
    distance_slider = widgets.IntRangeSlider(
        value=(0, max_representative_distance),
        min=0,
        max=max_representative_distance,
        step=1,
        description="Representative distance range (hops)",
        continuous_update=False,
        style={"description_width": "initial"},
        layout=widgets.Layout(width="48%"),
    )
    size_slider = widgets.IntRangeSlider(
        value=(0, max_simplex_dimension),
        min=0,
        max=max_simplex_dimension,
        step=1,
        description="Simplex dimension range",
        continuous_update=False,
        style={"description_width": "initial"},
        layout=widgets.Layout(width="48%"),
    )

    def feature_title(row, feature: SimplexPair) -> str:
        graph = state["graph"]
        left_label = ", ".join(str(graph.nodes[node].get("label", "?")) for node in feature.left)
        right_label = ", ".join(str(graph.nodes[node].get("label", "?")) for node in feature.right)
        return (
            f"<b>{row['sample_id']}</b> · EC {row.get('ec_number', '?')} · "
            f"{row.get('protein_name', 'Protein')}<br>"
            f"Feature: {len(feature.left)}-vertex simplex [{left_label}] ↔ "
            f"{len(feature.right)}-vertex simplex [{right_label}] · "
            f"representatives {feature.left_representative} ↔ {feature.right_representative} · "
            f"distance {feature.distance} hops"
        )

    def render(message: str | None = None) -> None:
        with plot_output:
            clear_output(wait=True)
            if message:
                display(Markdown(message))
            if state["graph"] is None:
                return
            title = (
                feature_title(state["row"], state["feature"])
                if state["feature"] is not None
                else f"<b>{state['row']['sample_id']}</b> · No matching feature selected"
            )
            figure = draw_protein_feature(
                state["graph"], state["coordinates"], state["feature"], title=title
            )
            display(figure)

    def sample_feature(_=None) -> None:
        if state["graph"] is None:
            return
        state["feature"] = sample_simplex_pair(
            state["graph"],
            min_simplex_dimension=size_slider.value[0],
            max_simplex_dimension=size_slider.value[1],
            min_distance=distance_slider.value[0],
            max_distance=distance_slider.value[1],
            rng=rng,
            clique_enumerator=clique_enumerator,
        )
        if state["feature"] is None:
            render(
                f"No pair with representative distance **{distance_slider.value[0]}–{distance_slider.value[1]}** "
                f"and simplex dimensions **{size_slider.value[0]}–{size_slider.value[1]}** for "
                f"{state['row']['sample_id']}. Change a slider or pick another protein."
            )
        else:
            render()

    def random_protein(_=None) -> None:
        state.update(row=None, graph=None, coordinates=None, feature=None)
        indices = list(range(len(proteins)))
        rng.shuffle(indices)
        failures = []
        for index in indices[: min(25, len(indices))]:
            row = proteins.iloc[index]
            try:
                graph, coordinates = load_protein_geometry(row)
            except Exception as exc:
                failures.append(f"{row['sample_id']}: {exc}")
                continue
            state.update(row=row, graph=graph, coordinates=coordinates)
            sample_feature()
            return
        details = "<br>".join(failures[:5])
        render(
            "Could not load cached protein geometry from the sampled entries. "
            "Run notebook 02 again for the active manifest. " + details
        )

    protein_button.on_click(random_protein)
    feature_button.on_click(sample_feature)
    distance_slider.observe(sample_feature, names="value")
    size_slider.observe(sample_feature, names="value")
    controls = widgets.VBox(
        [
            widgets.HBox([protein_button, feature_button]),
            widgets.HBox([distance_slider, size_slider]),
            plot_output,
        ],
        layout=widgets.Layout(width="100%"),
    )
    display(controls)
    random_protein()
