"""Notebook-facing orchestration for selecting proteins by EC prefix."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.auto import tqdm

from enzyme_workflow.common import add_project_sources


@dataclass
class SelectionResult:
    available_ec_numbers: pd.DataFrame
    manifest: pd.DataFrame
    selected_counts: pd.DataFrame
    manifest_path: Path
    reused_manifest: bool


def run_selection(
    project_root: str | Path,
    *,
    ec_level: int,
    proteins_per_ec: int,
    human_only: bool = True,
    selected_ec_numbers: list[str] | None = None,
    random_state: int = 42,
    refresh_uniprot_cache: bool = False,
    rebuild_manifest: bool = False,
) -> SelectionResult:
    """Fetch cached UniProt metadata, select proteins, and publish the manifest."""
    root = add_project_sources(project_root)
    from enzyme_classification.rcsb import (
        ec_protein_counts,
        fetch_uniprot_ec_pdb_records,
        select_ec_protein_manifest,
    )

    interim = root / "data" / "interim"
    interim.mkdir(parents=True, exist_ok=True)
    uniprot_cache = interim / f"uniprot_ec_pdb_human_{human_only}.tsv"
    manifest_path = interim / f"ec_protein_manifest_level_{ec_level}.tsv"
    metadata_path = interim / f"ec_protein_manifest_level_{ec_level}.json"
    mapping_cache = interim / "pdbe_uniprot_chain_mappings.json"
    active_pointer = interim / "active_ec_protein_manifest.json"
    settings: dict[str, Any] = {
        "ec_level": ec_level,
        "proteins_per_ec": proteins_per_ec,
        "human_only": human_only,
        "selected_ec_numbers": selected_ec_numbers,
        "random_state": random_state,
    }

    records = fetch_uniprot_ec_pdb_records(
        uniprot_cache,
        human_only=human_only,
        refresh=refresh_uniprot_cache,
    )
    print(f"Loaded {len(records):,} UniProt records from {uniprot_cache}")
    available = pd.DataFrame(ec_protein_counts(records, ec_level))
    if available.empty:
        raise RuntimeError(f"No EC classes with PDB-linked proteins found at level {ec_level}")

    cached_settings = None
    if metadata_path.is_file():
        cached_settings = json.loads(metadata_path.read_text(encoding="utf-8")).get("settings")
    can_reuse = (
        manifest_path.is_file()
        and cached_settings == settings
        and not rebuild_manifest
        and not refresh_uniprot_cache
    )
    if can_reuse:
        manifest = pd.read_csv(manifest_path, sep="\t", dtype=str).fillna("")
        print(f"Reusing saved manifest: {manifest_path}")
    else:
        total_classes = len(selected_ec_numbers) if selected_ec_numbers else len(available)
        with tqdm(total=total_classes, desc="Mapping EC classes to PDB chains", unit="EC") as bar:

            def update_progress(event: dict[str, Any]) -> None:
                bar.update(1)
                bar.set_postfix(
                    ec=event["ec_number"],
                    proteins=f"{event['selected']}/{event['requested']}",
                )

            rows = select_ec_protein_manifest(
                records,
                level=ec_level,
                proteins_per_ec=proteins_per_ec,
                selected_ec_numbers=selected_ec_numbers,
                random_state=random_state,
                mapping_cache_path=mapping_cache,
                output_path=manifest_path,
                progress=update_progress,
                verbose=False,
            )
        manifest = pd.DataFrame(rows)
        metadata = {"settings": settings, "sample_count": len(manifest)}
        temporary = metadata_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(metadata_path)
        print(f"Saved manifest: {manifest_path}")

    active = {
        "manifest_path": str(manifest_path.resolve()),
        "ec_level": ec_level,
        "settings": settings,
    }
    temporary_pointer = active_pointer.with_suffix(".json.tmp")
    temporary_pointer.write_text(json.dumps(active, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_pointer.replace(active_pointer)

    selected_counts = manifest.groupby("ec_number").size().rename("selected_proteins").to_frame()
    print(f"Selected {len(manifest):,} chain-level samples from {manifest_path}")
    return SelectionResult(
        available_ec_numbers=available,
        manifest=manifest,
        selected_counts=selected_counts,
        manifest_path=manifest_path,
        reused_manifest=can_reuse,
    )

