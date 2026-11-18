"""Probe safe RCSB mmCIF download concurrency without touching the project cache.

The benchmark sends one burst per concurrency level using distinct manifest PDB
IDs. It stops at the first HTTP/network failure and removes all downloaded data
when it exits. Keep the default upper bound conservative; pass --levels to change
the ladder explicitly.
"""

from __future__ import annotations

import argparse
import csv
import random
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests

RCSB_URL = "https://files.rcsb.org/download/{pdb_id}.cif.gz"


def read_pdb_ids(manifest_path: Path) -> list[str]:
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle, delimiter="\t")
        return sorted({str(row["pdb_id"]).strip().lower() for row in rows if row.get("pdb_id")})


def download_one(pdb_id: str, out_dir: Path, timeout: float) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        response = requests.get(RCSB_URL.format(pdb_id=pdb_id), timeout=timeout)
        response.raise_for_status()
        path = out_dir / f"{pdb_id}.cif.gz"
        path.write_bytes(response.content)
        return {
            "pdb_id": pdb_id,
            "ok": True,
            "status_code": response.status_code,
            "bytes": len(response.content),
            "error": "",
            "seconds": time.perf_counter() - started,
        }
    except Exception as exc:  # report connection, timeout and HTTP errors uniformly
        response = getattr(exc, "response", None)
        return {
            "pdb_id": pdb_id,
            "ok": False,
            "status_code": getattr(response, "status_code", None),
            "bytes": 0,
            "error": f"{type(exc).__name__}: {exc}",
            "seconds": time.perf_counter() - started,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/interim/ec_protein_manifest_level_1.tsv"))
    parser.add_argument("--levels", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=20261118)
    args = parser.parse_args()

    levels = args.levels
    if not levels or any(level < 1 for level in levels) or levels != sorted(set(levels)):
        parser.error("--levels must be positive, unique, and in increasing order")
    pdb_ids = read_pdb_ids(args.manifest)
    required = sum(levels)
    if len(pdb_ids) < required:
        parser.error(f"Need {required} distinct PDB IDs for these levels; manifest has {len(pdb_ids)}")
    random.Random(args.seed).shuffle(pdb_ids)

    print("DOWNLOAD_JOBS | requests | successes | failures | seconds | PDB/s | MiB/s | result")
    cursor = 0
    highest_clean: int | None = None
    with tempfile.TemporaryDirectory(prefix="rcsb-download-jobs-") as temp:
        out_dir = Path(temp)
        for workers in levels:
            batch = pdb_ids[cursor : cursor + workers]
            cursor += workers
            started = time.perf_counter()
            results: list[dict[str, Any]] = []
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(download_one, pdb_id, out_dir, args.timeout) for pdb_id in batch]
                for future in as_completed(futures):
                    results.append(future.result())
            elapsed = time.perf_counter() - started
            successes = sum(result["ok"] for result in results)
            failures = len(results) - successes
            total_bytes = sum(result["bytes"] for result in results)
            mib_per_second = total_bytes / (1024 * 1024) / elapsed if elapsed else 0.0
            blocked = any(result["status_code"] in (403, 429) for result in results)
            server_error = any(
                isinstance(result["status_code"], int) and result["status_code"] >= 500
                for result in results
            )
            label = "OK" if failures == 0 else "STOP: throttled" if blocked else "STOP: server/network error"
            print(
                f"{workers:12d} | {len(results):8d} | {successes:9d} | {failures:8d} | "
                f"{elapsed:7.2f} | {successes / elapsed:5.2f} | {mib_per_second:7.2f} | {label}"
            )
            for result in results:
                if not result["ok"]:
                    print(
                        f"  failed pdb={result['pdb_id']} http={result['status_code']} "
                        f"{result['error']}"
                    )
            if failures or blocked or server_error:
                break
            highest_clean = workers
    print(f"Highest level completed without errors: {highest_clean or 'none'}")


if __name__ == "__main__":
    main()
