# EnzymeClassification: Reconstructed Code Specification

## Recovery status and evidence

This specification was reconstructed from the recovered VS Code notebook snapshots and the Codex sessions listed at the end of this document. The original checkout is not currently accessible, so this describes the intended and notebook-observed system; it is not a claim that every component below was present in the last saved checkout.

The strongest evidence is the notebooks for manifest creation, graph building, model fitting, cross-validation, and simplex-pair vectorization. The project workflow notes preserved in Codex history add requirements for leakage prevention, data filtering, and paper-style comparisons. Where those sources differ, the difference is called out.

## Purpose

Classify experimental human protein structures from their three-dimensional residue-contact graphs. The project has two related prediction tasks:

- **Task A:** classify a protein sample as enzyme or non-enzyme.
- **Task B:** classify an enzyme by its single first-level EC class, using classes EC 1–6 for the paper-style experiment.

The modeling stack is classical machine learning over labeled NetworkX graphs, using graph kernels or explicit graph vectorizers. Notebooks orchestrate experiments and display results; reusable code belongs in `src/enzyme_classification/`.

## Data and sample definition

### Source and input format

- Query protein entries and entity metadata from RCSB PDB.
- Download **PDBx/mmCIF**, preferably compressed, to `data/raw/mmcif/{pdb_id}.cif.gz`.
- The project workflow recommends experimental human proteins (`Homo sapiens`, taxonomy ID 9606), protein polymers, and a resolution of at most 3.0 Å where resolution is available.
- Keep candidate selection and review separate from graph extraction. Store the reviewed sample list as a TSV manifest under `data/interim/`.

### Sample unit and manifest

Use one sample per protein polymer instance (chain-level when chain metadata is known). The stable sample ID is:

```text
sample_id = {pdb_id}.{label_asym_id}
```

The minimum manifest fields recovered from the experiment guide are `sample_id`, `pdb_id`, `label_asym_id`, `auth_asym_id`, and `protein_name`. Modeling notebooks also use `uniprot_accessions`, `first_level_ec_numbers`, and `graph_path`. Keep one downloaded mmCIF file per PDB entry and reference it from all chain rows. If no chain is specified, the exploratory graphicalizer may select the longest C-alpha chain; curated chain IDs are preferred for final runs.

### Manifest creation and caching

The saved bulk-download notebook calls `make_human_protein_entity_manifest` and supports explicit PDB or `pdb_id.chain` lists through `make_manifest_from_pdb_ids`. RCSB metadata queries should use retries with exponential backoff and write partial TSVs after each completed batch so interrupted work can resume. A manifest cache metadata JSON records the settings used to build its TSVs; load a current cache by default and rebuild only when explicitly requested.

The notebook's recorded test/full settings are:

| Setting | Test | Full |
|---|---:|---:|
| Maximum entities per task | 100 | 40,000 |
| RCSB page size | 100 | 1,000 |
| Metadata batch size | 50 | 250 |
| Search / metadata workers | 1 / 1 | 4 / 4 |

`PAGE_SIZE` controls request pagination; `*_MAX_ENTITIES` controls the experiment sample cap.

## Protein contact graph

For each selected chain, parse C-alpha coordinates, residue names, and residue sequence identifiers from mmCIF. A protein contact network (PCN) has:

- one node per residue's C-alpha atom;
- node label equal to the amino-acid residue name;
- an undirected edge when the C-alpha distance is in the inclusive interval **[4.0 Å, 8.0 Å]**;
- an optional distance value on edges for weighted or distance-binned experiments.

The paper-style graph is unweighted. Some later graph experiments preserve edge distance and bin contacts (for example, a 5.0 Å threshold creates `contact_close` and `contact_far` labels). The historical workflow note recommends averaging coordinates across alternate C-alpha locations to match the referenced paper, or otherwise recording the alternate-location policy.

The saved notebooks import `ProteinContactNetworkGraphicalizer` and `ProteinLabelGraphicalizer` from the sibling `abstractgraph_graphicalizer.protein` package. An earlier project note proposed implementing the protein graphicalizer locally; the later notebook imports indicate that this boundary changed. Reconstruct against the notebook import unless the original implementation or Git history turns up.

## Labels and filtering

Derive first-level EC labels from RCSB entity metadata and keep filtering counts auditable.

**Task A**

- Positive (`1`): sample has exactly one first-level EC class.
- Negative (`0`): sample has no EC number.
- Exclude samples with multiple first-level EC classes.
- Keep EC 7 as positive in Task A.
- The historical paper-scale target was approximately 21,707 enzyme and 26,312 non-enzyme samples (48,019 total); these are planning counts, not recovered run results.

**Task B**

- Include enzyme samples with exactly one first-level EC class.
- Keep classes EC 1–6 for the paper-style reproduction; exclude EC 7 because it was described as too rare for this task.
- The saved bulk notebook filters `first_level_ec_numbers` to one class and then to `1` through `6`.

## Pipeline and persisted data

The graph builder should be resumable, use one shared graph cache across tasks, and avoid duplicating a graph when a sample occurs in both tasks.

```text
data/
  raw/mmcif/{pdb_id}.cif.gz
  interim/                         reviewed and partial manifests
  processed/graphs/{sample_id}.pkl shared NetworkX graph cache
  processed/task_a/                status, logs, labels, splits
  processed/task_b/                status, logs, labels, splits
```

For each sample, skip an existing graph when `overwrite=False`; retry failed rows on a later run; atomically rewrite the status TSV and append a JSONL event after each sample. A prior `done` row is reusable only while its graph file still exists. The notebook calls `build_graphs_from_manifest` and inspects `summarize_status` rather than implementing the loop itself.

The bulk notebook records these outputs:

- `data/processed/task_a/graph_build_status.tsv`
- `data/processed/task_a/graph_build_events.jsonl`
- `data/processed/task_b/graph_build_status.tsv`
- `data/processed/task_b/graph_build_events.jsonl`
- shared graph files in `data/processed/graphs/`

The historical cache policy set default byte budgets of 5 GiB for mmCIF, 10 GiB for standard graphs, and 1 GiB for weighted display graphs; default cache age was 30 days and the minimum free-space floor 10 GiB. These are configurable via `EC_MMCIF_CACHE_MAX_BYTES`, `EC_GRAPHS_CACHE_MAX_BYTES`, `EC_GRAPHS_DISPLAY_WEIGHTED_CACHE_MAX_BYTES`, `EC_CACHE_MAX_AGE_SECONDS`, `EC_CACHE_MIN_FREE_BYTES`, and `EC_CACHE_MAINTENANCE_INTERVAL_SECONDS`.

## Graph representations

### Simplex-pair / pairwise-clique vectorization

The recovered `04_simplex_pair_edge_removal_similarity.ipynb` uses `SimplexPairGraphVectorizer` from the sibling `simplex_pair_vectorizer` package. It builds explicit sparse graph features from pairs of simplices (cliques), with rooted pairing and representative distance handling. The experiment notebook configuration is:

```text
max_simplex_dimension = 2
min_distance = 0
max_distance = 2
distance_mode = representative
distance_jitter = auto
pairing_mode = rooted
nbits = 14
dense = false
include_self = true
cache_graphs = false
n_jobs = -1
```

The train/test and UMAP notebooks also record a broader configuration with maximum simplex dimension 3 and maximum pair distance 4. Keep the distance and simplex limits as experiment parameters rather than hard-coding them into data construction. A separate edge-removal notebook uses PDB chain `1ubq.A`, removes 25 random edges per step (seed 13), vectorizes each graph, and compares their vectors with cosine similarity.

The `ProteinLabelGraphicalizer` can remap residue labels using one of `aa20`, `hp2`, `charge4`, `physchem5`, `chem7`, or `dayhoff6`; optional edge-distance thresholds can turn distances into contact labels before vectorization.

### Other recorded representations

- **NSPPK:** `radius=1`, `distance=4`, `connector=1`, `nbits=14`, sparse output, parallel processing enabled.
- **Simplicial/clique histogram:** the project workflow notes describe residue-labeled cliques as symbolic features and graph-level histograms. Build the feature dictionary from training data only and apply it unchanged to held-out data. Historical expected dimensionalities were about 16,000 features for Task A and 13,000 for Task B.
- **INDVAL selection:** a planned step to retain class-associated motifs, with approximate paper-style retained dimensions of 1,600 (Task A) and 1,300 (Task B). Selection must be fitted without test labels.
- **Graph kernels:** histogram cosine and weighted Jaccard similarities with a precomputed-kernel nu-SVM were listed in the experiment guide.

The clique histogram, INDVAL, and graph-kernel items are part of the recovered experiment plan. The saved notebook sequence directly confirms the SimplexPair and NSPPK workflows; it does not establish that every planned representation was completed.

## Models and evaluation

The notebooks wrap graph preprocessing and a scikit-learn estimator in `abstractgraph_ml.estimators.GraphEstimator`. The observed selectable base estimators are:

- **Linear SGD / hinge loss:** L2 normalization, `alpha=1e-4`, `max_iter=5000`, `tol=1e-4`, `n_iter_no_change=20`, balanced class weights, averaged coefficients, and a seeded run.
- **Random forest:** 1,000 trees, square-root feature sampling, `min_samples_leaf=2`, balanced-subsample class weights, and a seeded run.

An earlier workflow note also names linear SVM and nu-SVM paper baselines. The saved model notebook confirms the SGD hinge and random forest options; treat the SVM list as an intended comparison unless recovered run code confirms it.

### Leakage control

Group related structures by `uniprot_accessions` before splitting. Normalize a multi-accession value by splitting on semicolons, trimming, lowercasing, sorting, and joining. If missing, fall back in this order: `protein_name`, `entity_id`, `pdb_id`, then `sample_id`. Assert that no group crosses train, validation, and test. Fit graph dictionaries, feature selection, kernels, scaling, and hyperparameters on training data only.

The saved generic train/test notebook uses a 20% held-out split. The later cross-validation notebook uses `StratifiedGroupKFold` with five folds. An earlier project workflow note instead describes five grouped, stratified 60/20/20 train/validation/test runs. This is a recorded protocol change; select and document one final protocol when rebuilding rather than combining them silently.

Report accuracy, balanced accuracy, adjusted balanced accuracy, macro precision/recall/F1, per-class metrics, confusion matrices, and predictions. Keep validation and test results distinct and report mean and spread across folds/runs. Record split membership, class support, group overlap, vectorizer settings, estimator settings, and software version with each run.

## Results layout

The model notebooks write timestamped runs under `results/task_a/` or `results/task_b/`. Recorded artifacts include:

- `predictions.tsv`
- `metrics.tsv`
- `classification_report.tsv`
- `confusion_matrix.tsv`
- `run_metadata.json`

The five-fold notebook writes the same kinds of artifacts per fold and aggregates fold metrics and confusion matrices for display.

## Recovered module and notebook boundaries

The following module paths are referenced directly in the saved notebooks or session logs:

```text
src/enzyme_classification/
  data/protein_list.py
  experiments/edge_removal.py
  modeling/datasets.py
  modeling/results.py
  modeling/splits.py
  modeling/vectorizers.py
  pipelines/build_graphs.py
  representations/simplex_pair.py
  representations/amino_acid_alphabet.py
notebooks/
  _bootstrap.py
  01_showcase_protein_graphicalizer.ipynb
  02_bulk_download_task_a_task_b.ipynb
  03_train_test_graph_estimator_vectorizers.ipynb
  04_cross_validation_graph_estimator_vectorizers.ipynb
  05_simplex_pair_edge_removal_similarity.ipynb  # renamed to 05 in a later session
  06_umap_graph_vectorizer_embedding.ipynb       # renamed to 06 in a later session
```

The history snapshots include earlier notebook names (`04_simplex_pair_edge_removal_similarity.ipynb` and `05_umap_graph_vectorizer_embedding.ipynb`); the cross-validation session records moving those to 05 and 06. The recovered folder keeps the names present in VS Code history rather than rewriting the notebooks.

Keep notebooks thin: configuration, function calls, small inspections, plots, and result tables only. RCSB clients, parsing, graph construction, split generation, vectorizer assembly, metrics, and persistence belong in reusable Python modules.

## Open reconstruction decisions

1. Recover the missing checkout or source snapshots before treating this as an exact restoration. The original project path pointed into an unavailable `/mnt/data/fc334/Projects` storage mount.
2. Confirm the final location of `ProteinContactNetworkGraphicalizer` and whether the project-local implementation was replaced by the sibling package.
3. Confirm whether the final evaluation protocol is five-fold grouped CV or five 60/20/20 runs.
4. Verify the exact RCSB filtering query and metadata source used to build the final manifests.
5. Keep paper-plan representations (clique histogram, INDVAL, precomputed kernels) distinct from vectorizers confirmed in recovered notebooks (SimplexPair and NSPPK).

## Codex sessions used

- **Write experiment setup guide** — `019dafba-c15a-7890-92e5-585e72e826a3`
- **allow the choice for base_estimator between the lienar sgd and a random forest** — `019db135-c347-7492-8b5f-2a1e4f89c802`
- **Locate PDB graph module** — `019db4c6-1bde-7ca3-9232-2c98a57f4546`
- **Check UniProt stratification** — `019dc024-70e4-7910-86ca-3d924108a0cc`
- **Add stratified 5-fold CV** — `019dc09d-c74f-73e0-a484-c66bbc06c12c`
