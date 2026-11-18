"""Resumable graph-cache construction for a chain-level protein manifest."""

from __future__ import annotations

import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def _worker_count(requested: int) -> int:
    available = os.cpu_count() or 1
    if requested == -1:
        return available
    if requested < 1:
        raise ValueError("worker counts must be positive integers or -1 for all available CPUs")
    return min(requested, available)


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
    n_jobs: int = 8,
    download_jobs: int = 8,
    sample_limit: int | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
    download_progress: Callable[[dict[str, Any]], None] | None = None,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Download structures and build PCNs concurrently, resuming graph caches.

    Structure downloads are parallelized by unique PDB identifier, then graph
    construction is parallelized by protein chain. The download phase finishes
    first so multiple workers never race to create the same mmCIF cache file.
    """
    try:
        from abstractgraph_graphicalizer.protein import (
            ProteinContactNetworkGraphicalizer,
            download_mmcif,
        )
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
    sample_ids = [str(row.get("sample_id", "")) for row in manifest]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Manifest contains duplicate sample_id values; concurrent graph writes require unique IDs")

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

    def make_result(row: dict[str, str], *, status: str, graph_path: Path, error: str = "") -> dict[str, Any]:
        result = {
            **{key: row.get(key, "") for key in ("sample_id", "pdb_id", "label_asym_id", "auth_asym_id")},
            "graph_path": str(graph_path),
            "status": status,
            "error": error,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        return result

    def record_result(result: dict[str, Any]) -> None:
        results.append(result)
        _write_tsv_atomic(status_file, results, fieldnames)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, sort_keys=True) + "\n")
        if progress is not None:
            progress(result)
        if verbose:
            print(f"{result['status']}: {result['sample_id']} -> {result['graph_path']}")

    build_rows: list[dict[str, str]] = []
    for row in manifest:
        sample_id = str(row["sample_id"])
        graph_path = graphicalizer.graph_path_for(sample_id)
        if not overwrite and graph_path.is_file() and graph_path.stat().st_size > 0:
            record_result(make_result(row, status="skipped_existing", graph_path=graph_path))
        else:
            build_rows.append(row)

    workers = _worker_count(n_jobs)
    download_workers = _worker_count(download_jobs)
    raw_dir = Path(raw_mmcif_dir)
    pdb_ids = sorted({str(row["pdb_id"]).strip().lower() for row in build_rows})
    missing_pdb_ids = [pdb_id for pdb_id in pdb_ids if not (raw_dir / f"{pdb_id}.cif.gz").is_file()]
    download_errors: dict[str, str] = {}

    def download_one(pdb_id: str) -> tuple[str, str | None]:
        try:
            download_mmcif(
                pdb_id,
                raw_dir,
                overwrite=False,
                timeout_seconds=download_timeout_seconds,
                max_retries=download_max_retries,
                retry_backoff_seconds=download_retry_backoff_seconds,
                sleep_seconds=sleep_seconds,
            )
            return pdb_id, None
        except Exception as exc:
            return pdb_id, f"{type(exc).__name__}: {exc}"

    if missing_pdb_ids:
        with ThreadPoolExecutor(max_workers=min(download_workers, len(missing_pdb_ids))) as executor:
            futures = {executor.submit(download_one, pdb_id): pdb_id for pdb_id in missing_pdb_ids}
            for future in as_completed(futures):
                pdb_id = futures[future]
                _, error = future.result()
                if error:
                    download_errors[pdb_id] = error
                if download_progress is not None:
                    download_progress({
                        "pdb_id": pdb_id,
                        "status": "failed" if error else "downloaded",
                        "error": error or "",
                    })

    ready_rows: list[dict[str, str]] = []
    for row in build_rows:
        pdb_id = str(row["pdb_id"]).strip().lower()
        if pdb_id in download_errors:
            record_result(
                make_result(
                    row,
                    status="failed",
                    graph_path=graphicalizer.graph_path_for(str(row["sample_id"])),
                    error=download_errors[pdb_id],
                )
            )
        else:
            ready_rows.append(row)

    def build_one(row: dict[str, str]) -> dict[str, Any]:
        graph_path = graphicalizer.graph_path_for(str(row["sample_id"]))
        try:
            _, build_result = graphicalizer.transform_one(row)
            return make_result(
                row,
                status=build_result.status,
                graph_path=Path(build_result.graph_path or graph_path),
            )
        except Exception as exc:  # keep the notebook run resumable after one bad entry
            return make_result(
                row,
                status="failed",
                graph_path=graph_path,
                error=f"{type(exc).__name__}: {exc}",
            )

    if ready_rows:
        with ThreadPoolExecutor(max_workers=min(workers, len(ready_rows))) as executor:
            futures = [executor.submit(build_one, row) for row in ready_rows]
            for future in as_completed(futures):
                record_result(future.result())

    return results
