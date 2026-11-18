"""EC-labelled protein selection using UniProt records and PDBe chain maps."""

from __future__ import annotations

import csv
import json
import random
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
PDBE_UNIPROT_MAPPING_URL = "https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id}"
UNIPROT_FIELDS = "accession,ec,xref_pdb,protein_name,organism_name"
UNIPROT_COLUMNS = ["Entry", "EC number", "PDB", "Protein names", "Organism"]


def _session() -> requests.Session:
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )
    session = requests.Session()
    session.headers.update({"User-Agent": "simplex-pair-enzyme-notebook/0.1"})
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def ec_prefix(ec_number: str, level: int) -> str | None:
    """Return an EC prefix when the requested level is fully specified."""
    if level not in (1, 2, 3, 4):
        raise ValueError("EC level must be between 1 and 4")
    parts = str(ec_number).strip().split(".")
    if len(parts) < level or any(not part.isdigit() for part in parts[:level]):
        return None
    return ".".join(parts[:level])


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def fetch_uniprot_ec_pdb_records(
    cache_path: str | Path,
    *,
    human_only: bool = True,
    refresh: bool = False,
    page_size: int = 500,
    timeout_seconds: int = 60,
    sleep_seconds: float = 0.05,
) -> list[dict[str, str]]:
    """Fetch EC-annotated UniProt proteins with PDB cross-references.

    The tab-separated response is cached together with the query used to
    produce it. On a cache miss, the UniProt cursor pagination is followed
    until every result page has been saved.
    """
    cache = Path(cache_path)
    metadata_path = cache.with_suffix(cache.suffix + ".json")
    query = "ec:* AND database:pdb"
    if human_only:
        query += " AND organism_id:9606"
    if not refresh and cache.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("query") == query and metadata.get("fields") == UNIPROT_FIELDS:
            return _read_tsv(cache)

    session = _session()
    response = session.get(
        UNIPROT_SEARCH_URL,
        params={"query": query, "format": "tsv", "fields": UNIPROT_FIELDS, "size": page_size},
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    text = response.text
    header, _, first_body = text.partition("\n")
    rows: list[dict[str, str]] = []
    if first_body.strip():
        import io

        rows.extend(csv.DictReader(io.StringIO(text), delimiter="\t"))

    next_url = response.links.get("next", {}).get("url")
    while next_url:
        if sleep_seconds:
            time.sleep(sleep_seconds)
        response = session.get(next_url, timeout=timeout_seconds)
        response.raise_for_status()
        import io

        rows.extend(csv.DictReader(io.StringIO(response.text), delimiter="\t"))
        next_url = response.links.get("next", {}).get("url")

    if header.split("\t") != UNIPROT_COLUMNS:
        raise RuntimeError(f"Unexpected UniProt TSV columns: {header}")
    _write_tsv(cache, rows, UNIPROT_COLUMNS)
    metadata_path.write_text(
        json.dumps(
            {
                "query": query,
                "fields": UNIPROT_FIELDS,
                "retrieved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "row_count": len(rows),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return rows


def ec_protein_counts(records: list[dict[str, str]], level: int) -> list[dict[str, Any]]:
    """Count unambiguous UniProt accessions with PDB links per EC prefix."""
    accessions_by_ec: dict[str, set[str]] = defaultdict(set)
    for row in records:
        accession = row.get("Entry", "").strip()
        if not accession or not row.get("PDB", "").strip():
            continue
        prefixes = {
            prefix
            for term in row.get("EC number", "").split(";")
            if (prefix := ec_prefix(term, level)) is not None
        }
        if len(prefixes) == 1:
            accessions_by_ec[next(iter(prefixes))].add(accession)
    return [
        {"ec_number": ec_number, "uniprot_proteins_with_pdb": len(accessions)}
        for ec_number, accessions in sorted(accessions_by_ec.items())
    ]


def _pdb_ids(value: str) -> list[str]:
    ids = re.findall(r"[A-Za-z0-9]{4,8}", value or "")
    return list(dict.fromkeys(pdb_id.lower() for pdb_id in ids))


def _load_mapping_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_mapping_cache(path: Path, cache: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _mapping_for_accession(
    session: requests.Session,
    pdb_id: str,
    accession: str,
    cache: dict[str, Any],
    cache_path: Path,
    timeout_seconds: int,
) -> dict[str, Any] | None:
    if pdb_id not in cache:
        response = session.get(
            PDBE_UNIPROT_MAPPING_URL.format(pdb_id=pdb_id),
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        cache[pdb_id] = response.json()
        _save_mapping_cache(cache_path, cache)

    mapping_data = cache[pdb_id].get(pdb_id, {}).get("UniProt", {}).get(accession, {})
    mappings = mapping_data.get("mappings", [])
    if not mappings:
        return None
    best = max(
        mappings,
        key=lambda item: (
            float(item.get("coverage") or 0.0),
            float(item.get("identity") or 0.0),
            str(item.get("struct_asym_id", "")),
        ),
    )
    label_asym_id = str(best.get("struct_asym_id") or "").strip()
    if not label_asym_id:
        return None
    return {
        "pdb_id": pdb_id,
        "entity_id": str(best.get("entity_id") or ""),
        "label_asym_id": label_asym_id,
        "auth_asym_id": str(best.get("chain_id") or ""),
        "mapping_identity": float(best.get("identity") or 0.0),
        "mapping_coverage": float(best.get("coverage") or 0.0),
    }


def select_ec_protein_manifest(
    records: list[dict[str, str]],
    *,
    level: int,
    proteins_per_ec: int,
    selected_ec_numbers: list[str] | None = None,
    random_state: int = 42,
    mapping_cache_path: str | Path,
    output_path: str | Path,
    timeout_seconds: int = 30,
    sleep_seconds: float = 0.05,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Choose at most N UniProt proteins per EC prefix and map to PDB chains.

    A UniProt accession must have exactly one distinct EC prefix at the chosen
    level. PDB structures for that accession are tried in seeded random order;
    the first mapped chain with the highest within-entry coverage is selected.
    """
    if proteins_per_ec < 1:
        raise ValueError("proteins_per_ec must be at least one")
    if level not in (1, 2, 3, 4):
        raise ValueError("level must be between 1 and 4")

    candidates_by_ec: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in records:
        accession = row.get("Entry", "").strip()
        if not accession or not row.get("PDB", "").strip():
            continue
        prefixes = {
            prefix
            for term in row.get("EC number", "").split(";")
            if (prefix := ec_prefix(term, level)) is not None
        }
        if len(prefixes) == 1:
            candidates_by_ec[next(iter(prefixes))].append(row)

    ec_numbers = sorted(selected_ec_numbers) if selected_ec_numbers else sorted(candidates_by_ec)
    rng = random.Random(random_state)
    session = _session()
    cache_path = Path(mapping_cache_path)
    mapping_cache = _load_mapping_cache(cache_path)
    output = Path(output_path)
    manifest: list[dict[str, Any]] = []
    fieldnames = [
        "sample_id", "pdb_id", "entity_id", "label_asym_id", "auth_asym_id",
        "ec_number", "ec_level", "uniprot_accession", "ec_numbers",
        "protein_name", "organism", "mapping_identity", "mapping_coverage",
    ]

    for ec_number in ec_numbers:
        candidate_rows = list(candidates_by_ec.get(ec_number, []))
        rng.shuffle(candidate_rows)
        class_rows: list[dict[str, Any]] = []
        for row in candidate_rows:
            if len(class_rows) >= proteins_per_ec:
                break
            accession = row["Entry"].strip()
            pdb_ids = _pdb_ids(row.get("PDB", ""))
            rng.shuffle(pdb_ids)
            chosen_mapping = None
            for pdb_id in pdb_ids:
                try:
                    chosen_mapping = _mapping_for_accession(
                        session,
                        pdb_id,
                        accession,
                        mapping_cache,
                        cache_path,
                        timeout_seconds,
                    )
                except requests.RequestException as exc:
                    if verbose:
                        print(f"Mapping failed for {pdb_id}/{accession}: {exc}")
                    continue
                if chosen_mapping is not None:
                    break
                if sleep_seconds:
                    time.sleep(sleep_seconds)
            if chosen_mapping is None:
                continue
            sample_id = f"{chosen_mapping['pdb_id']}.{chosen_mapping['label_asym_id']}"
            if any(existing["sample_id"] == sample_id for existing in manifest + class_rows):
                continue
            ec_terms = sorted(term.strip() for term in row.get("EC number", "").split(";") if term.strip())
            class_rows.append(
                {
                    "sample_id": sample_id,
                    **chosen_mapping,
                    "ec_number": ec_number,
                    "ec_level": level,
                    "uniprot_accession": accession,
                    "ec_numbers": ";".join(ec_terms),
                    "protein_name": row.get("Protein names", ""),
                    "organism": row.get("Organism", ""),
                }
            )
        manifest.extend(class_rows)
        _write_tsv(output, manifest, fieldnames)
        if verbose:
            print(f"EC {ec_number}: selected {len(class_rows)}/{proteins_per_ec} proteins")

    return manifest
