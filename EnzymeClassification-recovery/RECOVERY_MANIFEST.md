# Recovery manifest

## Notebook snapshots

Each notebook below is the latest saved VS Code Local History snapshot found for its original resource. The restored copy is in `notebooks/`.

| Restored file | History snapshot | Original resource |
|---|---|---|
| `01_showcase_protein_graphicalizer.ipynb` | `uqvE.ipynb` | `notebooks/01_showcase_protein_graphicalizer.ipynb` |
| `02_bulk_download_task_a_task_b.ipynb` | `ZlQW.ipynb` | `notebooks/02_bulk_download_task_a_task_b.ipynb` |
| `03_train_test_graph_estimator_nsppk.ipynb` | `gweN.ipynb` | `notebooks/03_train_test_graph_estimator_nsppk.ipynb` |
| `03_train_test_graph_estimator_vectorizers.ipynb` | `inn0.ipynb` | `notebooks/03_train_test_graph_estimator_vectorizers.ipynb` |
| `04_cross_validation_graph_estimator_vectorizers.ipynb` | `aHEw.ipynb` | `notebooks/04_cross_validation_graph_estimator_vectorizers.ipynb` |
| `04_simplex_pair_edge_removal_similarity.ipynb` | `C4xE.ipynb` | `notebooks/04_simplex_pair_edge_removal_similarity.ipynb` |
| `05_umap_graph_vectorizer_embedding.ipynb` | `n0HQ.ipynb` | `notebooks/05_umap_graph_vectorizer_embedding.ipynb` |

The source snapshots were copied from `/home/fabrizio/.config/Code/User/History/`. Their resource records identify the original project as `/home/fabrizio/code/EnzymeClassification`. A later Codex session renumbered the simplex-pair and UMAP notebooks to 05 and 06; the recovered snapshots retain their recorded Local History filenames.

## Codex session logs

The raw files in `sessions/raw/` are copied from `/home/fabrizio/.codex/sessions/2026/`. They are complete compressed rollout logs; each record is JSON Lines. Verify copied archives with:

```bash
cd sessions/raw
sha256sum -c SHA256SUMS
zstdcat rollout-*.jsonl.zst
```

The session index maps filenames to titles, dates, and relevance. The five project sessions were identified by their original working directory, and the plotting-helper session is included because the recovered UMAP notebook imports that helper from the AbstractGraph ecosystem.

## Search scope and limitations

Checked local project checkouts, the accessible Trash folders and project ZIP, VS Code Local History, shell history, connected GitHub repositories, and Codex session records. No project source files or Git remote URL were recovered. The old system image contains a root-owned `/media/fc334` mount point, but its contents were inaccessible to this user session; it may be the remaining storage lead.
