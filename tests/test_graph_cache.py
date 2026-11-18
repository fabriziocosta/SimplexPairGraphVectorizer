import csv
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

from enzyme_classification.graphs import build_protein_graph_dataset


def test_graph_build_writes_one_status_snapshot_and_resumes_from_graph_files(tmp_path, monkeypatch):
    package = types.ModuleType("abstractgraph_graphicalizer")
    protein = types.ModuleType("abstractgraph_graphicalizer.protein")

    class FakeGraphicalizer:
        def __init__(self, *, graph_dir, **_kwargs):
            self.graph_dir = Path(graph_dir)
            self.graph_dir.mkdir(parents=True, exist_ok=True)

        def graph_path_for(self, sample_id):
            return self.graph_dir / f"{sample_id}.pkl"

        def transform_one(self, row):
            path = self.graph_path_for(row["sample_id"])
            path.write_bytes(b"graph")
            return None, SimpleNamespace(status="created", graph_path=str(path))

    protein.ProteinContactNetworkGraphicalizer = FakeGraphicalizer
    protein.download_mmcif = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("cached test structures should not be downloaded")
    )
    package.protein = protein
    monkeypatch.setitem(sys.modules, "abstractgraph_graphicalizer", package)
    monkeypatch.setitem(sys.modules, "abstractgraph_graphicalizer.protein", protein)

    manifest = tmp_path / "manifest.tsv"
    manifest.write_text(
        "sample_id\tpdb_id\tlabel_asym_id\tauth_asym_id\n"
        "pdb1.A\tpdb1\tA\tA\n"
        "pdb2.B\tpdb2\tB\tB\n",
        encoding="utf-8",
    )
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    for pdb_id in ("pdb1", "pdb2"):
        (raw_dir / f"{pdb_id}.cif.gz").write_bytes(b"cached")

    status_path = tmp_path / "status.tsv"
    event_log = tmp_path / "events.jsonl"
    arguments = {
        "raw_mmcif_dir": raw_dir,
        "graph_dir": tmp_path / "graphs",
        "status_path": status_path,
        "event_log_path": event_log,
        "n_jobs": 2,
        "download_jobs": 2,
        "verbose": False,
    }

    first_run = build_protein_graph_dataset(manifest, **arguments)
    with status_path.open(encoding="utf-8", newline="") as handle:
        first_status = list(csv.DictReader(handle, delimiter="\t"))
    assert len(first_status) == 2
    assert {row["status"] for row in first_status} == {"created"}
    assert len(first_run) == 2

    second_run = build_protein_graph_dataset(manifest, **arguments)
    with status_path.open(encoding="utf-8", newline="") as handle:
        second_status = list(csv.DictReader(handle, delimiter="\t"))
    assert len(second_status) == 2
    assert {row["status"] for row in second_status} == {"skipped_existing"}
    assert len(second_run) == 2
    events = [json.loads(line) for line in event_log.read_text(encoding="utf-8").splitlines()]
    assert len(events) == 4
