"""Shared paths and data loading for the enzyme-classification notebooks."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def add_project_sources(project_root: str | Path, *, include_imste: bool = False) -> Path:
    """Add this checkout and the required sibling source checkouts to sys.path."""
    root = Path(project_root).resolve()
    source_paths = [root / "src"]
    graphicalizer = root.parent / "abstractgraph-ecosystem/repos/abstractgraph-graphicalizer/src"
    if graphicalizer.is_dir():
        source_paths.append(graphicalizer)
    if include_imste:
        imste = root.parent / "IteratedMinimumSpanningTree"
        if imste.is_dir():
            source_paths.append(imste)
    for source_path in reversed(source_paths):
        value = str(source_path)
        if value in sys.path:
            sys.path.remove(value)
        sys.path.insert(0, value)
    return root


def active_manifest(project_root: str | Path) -> tuple[Path, int]:
    """Resolve notebook 01's active manifest, with newest-manifest fallback."""
    interim = Path(project_root) / "data" / "interim"
    pointer = interim / "active_ec_protein_manifest.json"
    if pointer.is_file():
        payload = json.loads(pointer.read_text(encoding="utf-8"))
        manifest_path = Path(payload["manifest_path"])
        ec_level = int(payload["ec_level"])
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Active protein manifest does not exist: {manifest_path}")
        return manifest_path, ec_level

    manifests = list(interim.glob("ec_protein_manifest_level_*.tsv"))
    if not manifests:
        raise FileNotFoundError("Run notebook 01 first to create an EC protein manifest")
    manifest_path = max(manifests, key=lambda path: path.stat().st_mtime_ns)
    ec_level = int(manifest_path.stem.rsplit("_", 1)[1])
    return manifest_path, ec_level

