"""Resumable graph-cache construction for a chain-level protein manifest."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


def read_tsv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _write_tsv_atomic(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def build_protein_graph_dataset(
    manifest_path: str | Path,
    *,
    raw_mmcif_dir: str | Path,
    graph_dir: str | Path,
    status_path: str | Path,
    event_log_path: str | Path,
    min_distance: float = 4.0,
    max_distance: float = 8.0,
    overwrite: bool = False,
    download_timeout_seconds: int = 60,
    download_max_retries: int = 4,
    download_retry_backoff_seconds: float = 2.0,
    sleep_seconds: float = 0.0,
    sample_limit: int | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Download/build cached PCNs, resuming successful rows from status TSV."""
    try:
        from abstractgraph_graphicalizer.protein import ProteinContactNetworkGraphicalizer
    except ImportError as exc:
        raise ImportError(
            "Install abstractgraph-graphicalizer and add its src directory to PYTHONPATH"
        ) from exc

    manifest = read_tsv(manifest_path)
    if sample_limit is not None:
        manifest = manifest[:sample_limit]
    if not manifest:
        raise ValueError("Manifest is empty")

    status_file = Path(status_path)
    log_file = Path(event_log_path)
    old_rows = read_tsv(status_file) if status_file.exists() else []
    old_by_id = {row.get("sample_id", ""): row for row in old_rows}
    results: list[dict[str, Any]] = []
    fieldnames = [
        "sample_id", "pdb_id", "label_asym_id", "auth_asym_id", "graph_path",
        "status", "error", "updated_at_utc",
    ]
    graphicalizer = ProteinContactNetworkGraphicalizer(
        mmcif_dir=raw_mmcif_dir,
        graph_dir=graph_dir,
        min_distance=min_distance,
        max_distance=max_distance,
        overwrite=overwrite,
        return_graphs=False,
        download_timeout_seconds=download_timeout_seconds,
        download_max_retries=download_max_retries,
        download_retry_backoff_seconds=download_retry_backoff_seconds,
        sleep_seconds=sleep_seconds,
    )

    for row in manifest:
        sample_id = str(row["sample_id"])
        graph_path = graphicalizer.graph_path_for(sample_id)
        old = old_by_id.get(sample_id)
        if not overwrite and graph_path.exists() and old and old.get("status") in {"created", "skipped_existing"}:
            result = {
                **{key: row.get(key, "") for key in ("sample_id", "pdb_id", "label_asym_id", "auth_asym_id")},
                "graph_path": str(graph_path),
                "status": "skipped_existing",
                "error": "",
            }
        else:
            try:
                _, build_result = graphicalizer.transform_one(row)
                result = {
                    **{key: row.get(key, "") for key in ("sample_id", "pdb_id", "label_asym_id", "auth_asym_id")},
                    "graph_path": str(build_result.graph_path),
                    "status": build_result.status,
                    "error": "",
                }
            except Exception as exc:  # keep the notebook run resumable after one bad entry
                result = {
                    **{key: row.get(key, "") for key in ("sample_id", "pdb_id", "label_asym_id", "auth_asym_id")},
                    "graph_path": str(graph_path),
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
        result["updated_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        results.append(result)
        _write_tsv_atomic(status_file, results, fieldnames)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, sort_keys=True) + "\n")
        if progress is not None:
            progress(result)
        if verbose:
            print(f"{result['status']}: {sample_id} -> {result['graph_path']}")
    return results
