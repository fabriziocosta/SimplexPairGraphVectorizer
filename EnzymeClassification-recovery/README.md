# EnzymeClassification recovery archive

This folder collects the recoverable project material found on this computer. The original source checkout is missing, so this is an archival/reconstruction package, not a runnable repository.

## Contents

```text
EnzymeClassification-recovery/
├── CODE_SPEC.md                    reconstructed architecture and experiment spec
├── RECOVERY_MANIFEST.md             provenance for recovered notebook snapshots
├── notebooks/                       seven latest VS Code history snapshots
└── sessions/
    ├── SESSION_INDEX.md             relevant Codex sessions and archive guide
    └── raw/                          compressed original Codex rollout logs + checksums
```

## What was recovered

- Seven notebook snapshots covering graph construction, PDB/mmCIF downloads, Task A/Task B manifests, train/test, grouped cross-validation, simplex-pair edge-removal analysis, and UMAP visualization.
- Six complete Codex session logs for the project and a directly related AbstractGraph plotting helper session. They include the saved user/assistant/tool event stream; `.jsonl.zst` is Zstandard-compressed JSON Lines.
- A code specification rebuilt from those notebooks and from design text visible in the Codex session records.

## What is still missing

No `src/enzyme_classification/` Python source, project Git metadata, PDB/mmCIF cache, processed graphs, manifests, or result files were found in the accessible checkout, Trash, or VS Code local history. The old checkout was recorded under `/home/fabrizio/code/EnzymeClassification`; its `Projects` storage pointed to `/mnt/data/fc334/Projects`, which is unavailable in this environment.

The recovered notebook snapshots preserve their original names and are not guaranteed to run without the missing modules and sibling dependencies. See `CODE_SPEC.md` for confirmed behavior, historical plans, and unresolved protocol differences.
