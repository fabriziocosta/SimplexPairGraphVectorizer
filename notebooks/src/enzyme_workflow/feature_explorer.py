"""Utilities for interactively inspecting rooted simplex-pair features."""

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
class RootedSimplexPair:
    """A pair of cliques sharing ``root`` and their representative distance."""

    root: Any
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


def cliques_by_root(graph: nx.Graph, max_simplex_size: int) -> dict[Any, list[tuple[Any, ...]]]:
    """Enumerate every simplex up to the selected size and index it by root."""
    if max_simplex_size < 1:
        raise ValueError("max_simplex_size must be at least one vertex")
    by_root: dict[Any, list[tuple[Any, ...]]] = {node: [] for node in graph.nodes}
    for clique in nx.enumerate_all_cliques(graph):
        if len(clique) > max_simplex_size:
            break
        simplex = tuple(sorted(clique))
        for root in simplex:
            by_root[root].append(simplex)
    return {root: simplices for root, simplices in by_root.items() if simplices}


def _representative_distance(
    graph: nx.Graph, left: tuple[Any, ...], right: tuple[Any, ...]
) -> int:
    """Match the vectorizer: shortest path between each simplex's lowest node."""
    left_rep, right_rep = left[0], right[0]
    if left_rep == right_rep:
        return 0
    if graph.has_edge(left_rep, right_rep):
        return 1
    # Both representatives are in cliques that contain the same root, so a
    # two-edge path through that root is always present when they are distinct.
    return 2


def sample_rooted_simplex_pair(
    graph: nx.Graph,
    *,
    max_simplex_size: int,
    distance: int,
    rng: random.Random,
    random_attempts: int = 5000,
) -> RootedSimplexPair | None:
    """Choose a random valid rooted pair, allowing the two simplex sizes to differ."""
    if not isinstance(distance, int) or distance < 0:
        raise ValueError("distance must be a nonnegative integer")
    # In a rooted pair, each representative is either the shared root or one
    # edge away from it, so distances above two cannot occur.
    if distance > 2:
        return None
    by_root = cliques_by_root(graph, max_simplex_size)
    roots = list(by_root)
    if not roots:
        return None

    for _ in range(random_attempts):
        root = rng.choice(roots)
        incident = by_root[root]
        left = rng.choice(incident)
        right = rng.choice(incident)
        if _representative_distance(graph, left, right) == distance:
            return RootedSimplexPair(root=root, left=left, right=right, distance=distance)

    # Exact fallback so a valid but uncommon distance is not missed by random draws.
    seen = 0
    selected: RootedSimplexPair | None = None
    for root, incident in by_root.items():
        for left_index, left in enumerate(incident):
            for right in incident[left_index:]:
                if _representative_distance(graph, left, right) != distance:
                    continue
                seen += 1
                if rng.randrange(seen) == 0:
                    selected = RootedSimplexPair(
                        root=root,
                        left=left,
                        right=right,
                        distance=distance,
                    )
    return selected


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
    feature: RootedSimplexPair | None,
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
    root = None
    if feature is not None:
        selected_nodes = set(feature.left) | set(feature.right)
        highlighted_edges_a = set(itertools.combinations(feature.left, 2))
        highlighted_edges_b = set(itertools.combinations(feature.right, 2))
        root = feature.root

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
            color="rgba(160,175,195,0.12)",
            width=1.4,
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

    add_node_trace(background_nodes, name="Other residues", opacity=0.16, size=3.0, outline="rgba(0,0,0,0)")

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
    if root is not None:
        xyz = coordinates[root]
        root_label = str(graph.nodes[root].get("label", "UNK"))
        traces.append(
            go.Scatter3d(
                x=[xyz[0]], y=[xyz[1]], z=[xyz[2]],
                mode="markers",
                name="Shared root",
                marker={
                    "size": 10,
                    "symbol": "diamond",
                    "color": amino_colors.get(root_label, "#A0A0A0"),
                    "opacity": 1.0,
                    "line": {"color": "#FFFFFF", "width": 2},
                },
                hovertemplate=f"Shared root · {root_label}<extra></extra>",
                showlegend=True,
            )
        )

    figure = go.Figure(data=traces)
    figure.update_layout(
        title={"text": title, "x": 0.02, "xanchor": "left"},
        template="plotly_dark",
        height=780,
        margin={"l": 0, "r": 0, "t": 90, "b": 0},
        legend={"orientation": "h", "y": 1.02, "x": 0.02},
        scene={
            "aspectmode": "data",
            "dragmode": "orbit",
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
    max_simplex_size: int = 4,
):
    """Display the widget-driven protein and rooted simplex-pair explorer."""
    import ipywidgets as widgets
    from IPython.display import Markdown, clear_output, display

    if not isinstance(max_representative_distance, int) or max_representative_distance < 0:
        raise ValueError("max_representative_distance must be a nonnegative integer")
    if not isinstance(max_simplex_size, int) or max_simplex_size < 1:
        raise ValueError("max_simplex_size must be at least one vertex")

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
    distance_slider = widgets.IntSlider(
        value=min(1, max_representative_distance),
        min=0,
        max=max_representative_distance,
        step=1,
        description="Representative distance (hops)",
        continuous_update=False,
        style={"description_width": "initial"},
        layout=widgets.Layout(width="48%"),
    )
    size_slider = widgets.IntSlider(
        value=max_simplex_size,
        min=1,
        max=max_simplex_size,
        step=1,
        description="Maximum simplex size (vertices)",
        continuous_update=False,
        style={"description_width": "initial"},
        layout=widgets.Layout(width="48%"),
    )

    def feature_title(row, feature: RootedSimplexPair) -> str:
        graph = state["graph"]
        left_label = ", ".join(str(graph.nodes[node].get("label", "?")) for node in feature.left)
        right_label = ", ".join(str(graph.nodes[node].get("label", "?")) for node in feature.right)
        return (
            f"<b>{row['sample_id']}</b> · EC {row.get('ec_number', '?')} · "
            f"{row.get('protein_name', 'Protein')}<br>"
            f"Feature: {len(feature.left)}-vertex simplex [{left_label}] ↔ "
            f"{len(feature.right)}-vertex simplex [{right_label}] · "
            f"shared root {feature.root} · representative distance {feature.distance} hops"
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
        state["feature"] = sample_rooted_simplex_pair(
            state["graph"],
            max_simplex_size=size_slider.value,
            distance=distance_slider.value,
            rng=rng,
        )
        if state["feature"] is None:
            render(
                f"No pair at distance **{distance_slider.value}** with simplex sizes up to "
                f"**{size_slider.value}** for {state['row']['sample_id']}. Change a slider or pick another protein."
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
    return controls
