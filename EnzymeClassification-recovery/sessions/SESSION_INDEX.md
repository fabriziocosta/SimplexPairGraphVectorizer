# Codex session index

The full event logs are in `raw/` as `.jsonl.zst` files. They can be viewed with `zstdcat`, searched with `zstdgrep`, or decompressed with `zstd -d`. The five project sessions use the original EnzymeClassification checkout as their working directory. One supporting session is from the sibling AbstractGraph ecosystem because its plotting helper is imported by the recovered UMAP notebook.

| Date | Session title | Session ID | Why it is included | Raw log |
|---|---|---|---|---|
| 21 Apr 2026 | **Write experiment setup guide** | `019dafba-c15a-7890-92e5-585e72e826a3` | Detailed protocol for RCSB data, chain-level manifests, PCN construction, Task A/B labels, caches, representations, and repository layout. | `raw/rollout-2026-04-21T12-09-05-019dafba-c15a-7890-92e5-585e72e826a3.jsonl.zst` |
| 22 Apr 2026 | **allow the choice for base_estimator between the lienar sgd and a random forest** | `019db135-c347-7492-8b5f-2a1e4f89c802` | Model and vectorizer options, notebook wiring, and SimplexPair import recovery. | `raw/rollout-2026-04-21T19-03-03-019db135-c347-7492-8b5f-2a1e4f89c802.jsonl.zst` |
| 22 Apr 2026 | **Locate PDB graph module** | `019db4c6-1bde-7ca3-9232-2c98a57f4546` | PDB-to-graph module boundaries, notebook bootstrap, graph pipeline imports, and project module references. | `raw/rollout-2026-04-22T11-39-35-019db4c6-1bde-7ca3-9232-2c98a57f4546.jsonl.zst` |
| 22 Apr 2026 | **Add plot_embedding_2d** | `019db482-fefe-7892-a75a-c87de15d3fe0` | Supporting AbstractGraph helper used by the recovered embedding notebook. | `raw/rollout-2026-04-22T10-26-17-019db482-fefe-7892-a75a-c87de15d3fe0.jsonl.zst` |
| 24 Apr 2026 | **Check UniProt stratification** | `019dc024-70e4-7910-86ca-3d924108a0cc` | Leakage discussion about splitting by UniProt accession. | `raw/rollout-2026-04-24T16-38-27-019dc024-70e4-7910-86ca-3d924108a0cc.jsonl.zst` |
| 24 Apr 2026 | **Add stratified 5-fold CV** | `019dc09d-c74f-73e0-a484-c66bbc06c12c` | Notebook renumbering and creation of a stratified grouped five-fold experiment. | `raw/rollout-2026-04-24T18-50-59-019dc09d-c74f-73e0-a484-c66bbc06c12c.jsonl.zst` |

## Reading a raw log

From the archive root:

```bash
zstdcat sessions/raw/rollout-2026-04-21T12-09-05-019dafba-c15a-7890-92e5-585e72e826a3.jsonl.zst
```

Each line is a JSON event with a timestamp, ordinal, and payload. Message, tool-call, command-output, and file-change records are preserved. `CODE_SPEC.md` summarizes the code-relevant conclusions without reproducing the entire event stream.
