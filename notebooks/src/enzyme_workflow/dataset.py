"""Notebook-facing orchestration for building cached protein contact graphs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from enzyme_workflow.common import active_manifest, add_project_sources


@dataclass
class GraphDatasetResult:
    manifest: pd.DataFrame
    status: pd.DataFrame
    class_counts: pd.DataFrame
    status_counts: pd.DataFrame
    manifest_path: Path
    graph_dir: Path


def build_graph_dataset(
    project_root: str | Path,
    *,
    min_contact_distance: float = 4.0,
    max_contact_distance: float = 8.0,
    overwrite: bool = False,
    sample_limit: int | None = None,
    download_timeout_seconds: int = 60,
    download_max_retries: int = 4,
    download_retry_backoff_seconds: float = 2.0,
    n_jobs: int = 8,
    download_jobs: int = 128,
    graph_executor: str = "process",
) -> GraphDatasetResult:
    """Build graphs for the active manifest, reporting cache and progress."""
    root = add_project_sources(project_root)
    from enzyme_classification.graphs import build_protein_graph_dataset as build

    manifest_path, ec_level = active_manifest(root)
    manifest = pd.read_csv(manifest_path, sep="\t", dtype=str).fillna("")
    if sample_limit is not None:
        manifest = manifest.head(sample_limit).copy()
    if manifest.empty:
        raise RuntimeError(f"No protein samples in manifest {manifest_path}")

    data_dir = root / "data"
    raw_dir = data_dir / "raw" / "mmcif"
    cache_tag = f"contact_{min_contact_distance:g}_{max_contact_distance:g}A"
    graph_dir = data_dir / "processed" / "graphs" / cache_tag
    status_path = data_dir / "processed" / f"graph_build_status_level_{ec_level}.tsv"
    event_log_path = data_dir / "processed" / f"graph_build_events_level_{ec_level}.jsonl"

    def graph_path(sample_id: str) -> Path:
        safe_id = str(sample_id).replace("/", "_").replace(chr(92), "_")
        return graph_dir / f"{safe_id}.pkl"

    cached_graphs = sum(
        path.is_file() and path.stat().st_size > 0
        for path in (graph_path(sample_id) for sample_id in manifest["sample_id"])
    )
    unique_pdb_ids = set(manifest["pdb_id"].str.lower())
    cached_mmcif = sum((raw_dir / f"{pdb_id}.cif.gz").is_file() for pdb_id in unique_pdb_ids)
    pending_rows = [
        row for row in manifest.to_dict(orient="records")
        if overwrite or not (
            graph_path(row["sample_id"]).is_file()
            and graph_path(row["sample_id"]).stat().st_size > 0
        )
    ]
    missing_pending_pdbs = {
        str(row["pdb_id"]).lower()
        for row in pending_rows
        if not (raw_dir / f"{str(row['pdb_id']).lower()}.cif.gz").is_file()
    }
    print(f"Manifest source: {manifest_path} (EC level {ec_level})")
    print(f"Manifest samples: {len(manifest):,}")
    print(f"Existing graph caches: {cached_graphs:,}/{len(manifest):,}")
    print(f"Existing mmCIF files: {cached_mmcif:,}/{len(unique_pdb_ids):,} unique PDB entries")
    print(f"Graph cache: {graph_dir} (contact range {min_contact_distance:g}–{max_contact_distance:g} Å)")
    print(
        f"Parallel workers: {n_jobs} {graph_executor} graph workers; "
        f"{download_jobs} structure-download workers"
    )

    planned = len(manifest)
    status_rows = []
    with (
        tqdm(total=len(missing_pending_pdbs), desc="Downloading missing mmCIFs", unit="PDB", position=0) as download_bar,
        tqdm(total=planned, desc="Building contact graphs", unit="protein", position=1) as bar,
    ):

        def report_progress(result: dict) -> None:
            bar.update(1)
            bar.set_postfix(status=result["status"], sample=result["sample_id"], refresh=True)

        def report_download(result: dict) -> None:
            download_bar.update(1)
            download_bar.set_postfix(status=result["status"], pdb=result["pdb_id"], refresh=True)

        status_rows = build(
            manifest_path,
            raw_mmcif_dir=raw_dir,
            graph_dir=graph_dir,
            status_path=status_path,
            event_log_path=event_log_path,
            min_distance=min_contact_distance,
            max_distance=max_contact_distance,
            overwrite=overwrite,
            download_timeout_seconds=download_timeout_seconds,
            download_max_retries=download_max_retries,
            download_retry_backoff_seconds=download_retry_backoff_seconds,
            n_jobs=n_jobs,
            download_jobs=download_jobs,
            graph_executor=graph_executor,
            sample_limit=sample_limit,
            progress=report_progress,
            download_progress=report_download,
            verbose=False,
        )

    status = pd.DataFrame(status_rows)
    status_counts = status["status"].value_counts(dropna=False).rename("proteins").to_frame()
    class_counts = manifest.groupby("ec_number").size().rename("samples").to_frame()
    return GraphDatasetResult(
        manifest=manifest,
        status=status,
        class_counts=class_counts,
        status_counts=status_counts,
        manifest_path=manifest_path,
        graph_dir=graph_dir,
    )

