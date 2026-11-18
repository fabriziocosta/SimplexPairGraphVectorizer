# Simplex Pair Vectorizer

This package converts labeled, undirected NetworkX graphs into sparse feature
vectors built from anchored pairs of cliques (simplices). It reconstructs the
SimplexPairGraphVectorizer interface used by the recovered protein graph
notebooks while keeping the implementation domain-neutral.

## Install

```bash
python -m pip install -e '.[test]'
```

## Quick start

```python
import networkx as nx
from simplex_pair_vectorizer import SimplexPairGraphVectorizer

graph = nx.Graph()
graph.add_node("a", label="ALA")
graph.add_node("b", label="GLY")
graph.add_edge("a", "b", label="contact")

vectorizer = SimplexPairGraphVectorizer(n_jobs=1)
features = vectorizer.fit_transform([graph])  # scipy CSR matrix
```

## Enzyme classification notebooks

The `notebooks/` workflow has four stages:

1. `01_select_ec_proteins.ipynb` lists PDB-linked EC classes at a chosen level
   and selects up to a configured number of UniProt proteins per class.
2. `02_build_protein_graph_dataset.ipynb` downloads RCSB mmCIF files and saves
   chain-level protein contact graphs plus resumable build status.
3. `03_train_test_ec_classifier.ipynb` loads the graph cache, clusters protein
   sequences for a homology-aware train/test split, and trains a classifier
   using the simplex-pair features, with square confusion-matrix, ROC, and
   precision–recall plots.
4. `04_visualize_ec_proteins_2d.ipynb` vectorizes the same cached graphs and
   visualizes them with the sibling `IteratedMinimumSpanningTreeEmbedder`.
5. `05_explore_protein_simplex_pair.ipynb` opens a rotatable 3D protein graph
   and interactively highlights a randomly sampled anchored simplex-pair feature.

Notebook 03 defaults to XGBoost's parallel histogram classifier. It consumes
the vectorizer's sparse CSR matrix directly; `N_JOBS = -1` uses all available
CPU cores, and inverse-frequency sample weights balance the EC classes.

Install the experiment dependencies and sibling protein graphicalizer with:

```bash
python -m pip install -e '.[protein]'
python -m pip install -e '../abstractgraph-ecosystem/repos/abstractgraph-graphicalizer'
python -m pip install -e '../IteratedMinimumSpanningTree[notebook]'
```

Notebook 03 also requires [MMseqs2](https://github.com/soedinglab/MMseqs2) on
`PATH`. Install it with one of:

```bash
brew install mmseqs2
# or
conda install -c conda-forge -c bioconda mmseqs2
```

It fetches and caches canonical UniProt sequences, then
groups proteins sharing at least 30% sequence identity across at least 80% of
the shorter sequence. Each sequence cluster is kept entirely in train or test;
the run records the cluster assignments and verifies that the two splits have
no cluster overlap.

Notebook 04 uses IMSTE's transductive `fit_transform` interface on the sparse
simplex-pair feature matrix, then renders an interactive Plotly scatter with EC
class toggles and protein metadata on hover. The feature matrix and coordinates
are cached under `data/processed/`; set `REBUILD_FEATURES` or
`REBUILD_EMBEDDING` in the notebook to recompute either stage.

Notebook 01 publishes its active manifest so notebooks 02–04 follow the latest
selected EC level automatically. Notebook 02 keeps graph caches in a folder
keyed by the contact-distance range, so changing those settings uses a fresh
cache while reruns with the same settings reuse completed graphs. Missing
structures are downloaded concurrently by PDB entry, then contact graphs are
built concurrently by protein chain; adjust `N_JOBS` and `DOWNLOAD_JOBS` in
notebook 02 to fit the machine.

The default EC source is UniProtKB (proteins from any organism with PDB cross-references);
PDBe maps each selected accession and PDB entry to a chain, and the graphicalizer
downloads the corresponding structure from RCSB. The manifest contains one
chain per UniProt accession with exactly one EC prefix at the selected level.
This keeps labels unambiguous and lets the classifier cluster homologous
sequences before splitting.
Edit the first notebook's settings to change organism, EC level, selected ECs,
or number of proteins per EC. UniProt responses, PDBe chain mappings, and a
manifest with matching settings are reused on reruns. Set REBUILD_MANIFEST in
notebook 01 to rerun selection from cached metadata; set both
REFRESH_UNIPROT_CACHE and REBUILD_MANIFEST to refresh UniProt data and selection.
Notebook 02 reuses cached mmCIF files and graph pickles when OVERWRITE=False,
and shows cache counts before building.

Graph caches and experiment outputs are written below `data/` and `results/`;
the `.gitignore` excludes generated raw and processed data.

The notebooks keep configuration and displays in the `.ipynb` files; reusable
workflow code lives in `notebooks/src/enzyme_workflow/`.

The default output has 2**14 hashed coordinates. Hash collisions add counts to
the same unsigned coordinate. The default simplex dimension of 3 includes
cliques of up to four vertices. Every graph vertex anchors all cliques that
contain it. Each unordered anchor-vertex pair in range contributes the
Cartesian product of the associated cliques; the cliques need not share a
vertex. At distance zero, the same anchor supplies both cliques and unordered
clique pairs are used. Self-pairs are included by default.

## Parameters

For the mathematical definition of anchored simplex-pair features and graph-level
count vectors, see [the whitepaper](docs/whitepaper.md).

The vectorizer constructor defaults to dimension 3, pair distance 0–4 inclusive,
representative-vertex distance, vertex-anchor pairing, 14-bit hashing, occurrence counts,
and self-pairs enabled. The enzyme notebooks currently set `NBITS = 11` for
both classification and visualization. Node and edge labels are read from
attributes named label; change node_label_key or edge_label_key to use other
attributes.

Clique vertices are ordered using a stable node key to establish a deterministic
ordering convention; every vertex still anchors all cliques containing it. By
default, the node itself is the key. Supply node_key= for custom node objects
or to control ordering. Keys must be unique within each graph and stable across
runs. distance_jitter="auto" means no jitter for the discrete
shortest-path distances used here; a numeric value adds a constant offset
before distance limits are applied.

fit validates the graph collection and records the fixed hash width; it does
not learn a vocabulary, so held-out graphs can be transformed independently.
transform returns a CSR matrix unless dense=True. Set n_jobs=1 for sequential
processing or use a joblib value such as -1 for parallel graph conversion.
cache_graphs=True retains per-object results for repeated transforms; mutate
graphs only when caching is disabled or refit afterward.
