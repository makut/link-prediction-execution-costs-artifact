"""Tiny independent serialization fixtures; no large graph or torch dependency."""
import importlib.util
import json
from pathlib import Path
import platform
import struct
import subprocess
import sys

import numpy as np
import pytest

SCRIPT = Path(__file__).with_name("memory_benchmark.py")
SPEC = importlib.util.spec_from_file_location("seal_revision_memory", SCRIPT)
memory = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(memory)


def fixture_graph(root: Path, endian="little", timestamps=True):
    root.mkdir()
    prefix = "<" if endian == "little" else ">"
    # Three nodes: 0->{1,2}, 1->{0}, 2->{}. Starts omit terminal sentinel.
    (root / "edge_ends.bin").write_bytes(struct.pack(prefix + "q3i", 3, 1, 2, 0))
    (root / "edge_starts.bin").write_bytes(struct.pack(prefix + "3q", 0, 2, 3))
    (root / "timestamps.bin").write_bytes(struct.pack(prefix + "q3i", 3, 10, 20, 30) if timestamps else b"")
    return root


@pytest.mark.parametrize("endian", ["big", "little"])
def test_standalone_formats_byte_counts_and_endian(tmp_path, endian):
    root = fixture_graph(tmp_path / "graph", endian)
    meta = memory.inspect_graph(root, endian)
    assert (meta["nodes"], meta["adjacency_entries"]) == (3, 3)
    mixed = memory.logical_bytes(meta, "mixed-mmap", False)
    scipy = memory.logical_bytes(meta, "scipy-int64", True)
    assert mixed["components_bytes"] == {"indices": 12, "offsets": 24, "values": 0, "timestamps": 0}
    assert mixed["timestamp_bytes_on_disk"] == 12
    assert scipy["components_bytes"] == {"indices": 24, "offsets": 32, "values": 3, "timestamps": 12}
    arrays, matrix = memory.load_scipy(meta, True, 8)
    assert matrix.indices.dtype == matrix.indptr.dtype == np.int64
    assert np.array_equal(matrix.toarray(), [[0, 1, 1], [1, 0, 0], [0, 0, 0]])
    assert arrays["offsets"].tolist() == [0, 2, 3, 3]
    assert arrays["timestamps"].tolist() == [10, 20, 30]
    assert all(not isinstance(a, np.memmap) for a in arrays.values())
    assert {k: a.nbytes for k, a in arrays.items()} == scipy["components_bytes"]


def test_conversion_result_survives_closing_source_and_has_no_alias(tmp_path):
    meta = memory.inspect_graph(fixture_graph(tmp_path / "graph"), "little")
    source = memory.load_mixed(meta, False, True, 8)
    arrays, matrix = memory.load_scipy(meta, True, 8, source=source)
    assert not np.shares_memory(source["indices"], arrays["indices"])
    assert not np.shares_memory(source["offsets"], arrays["offsets"])
    assert not np.shares_memory(source["timestamps"], arrays["timestamps"])
    memory.close_mixed(source)
    assert source == {}
    assert matrix.sum() == 3
    assert arrays["indices"].tolist() == [1, 2, 0]


def test_corrupt_header_rejected_before_allocation(tmp_path):
    root = fixture_graph(tmp_path / "graph")
    (root / "edge_ends.bin").write_bytes(struct.pack("<q", 1_000_000_000))
    with pytest.raises(ValueError, match="count/size"):
        memory.inspect_graph(root, "little")


def test_conversion_preflight_includes_source_once_not_once_per_worker(tmp_path):
    meta = memory.inspect_graph(fixture_graph(tmp_path / "graph"), "little")
    one = memory.estimate_peak_bytes(meta, "scipy-int64", 1, False, 8)
    conversion = memory.estimate_peak_bytes(meta, "conversion-peak", 1, False, 8)
    assert conversion - one == 36
    sixteen = memory.estimate_peak_bytes(meta, "mixed-mmap", 16, False, 8)
    assert sixteen == 36 + 16 * memory.PROCESS_RESERVE + 16


def test_budget_refusal_writes_reviewable_report_without_child(tmp_path, monkeypatch):
    root = fixture_graph(tmp_path / "graph")
    out = tmp_path / "out"
    monkeypatch.setattr(memory.platform, "platform", lambda: "test-platform")
    monkeypatch.setattr(memory.subprocess, "Popen", lambda *a, **kw: pytest.fail("preflight must prevent allocation"))
    assert memory.main(["--graph-path", str(root), "--out", str(out),
                        "--memory-budget-gib", ".001", "--representations", "conversion-peak"]) == 1
    report = json.loads((out / "report.json").read_text())
    assert report["runs"][0]["status"] == "skipped_preflight"
    assert report["runs"][0]["preflight"]["estimated_peak_bytes"] > .001 * memory.GIB


def test_all_representation_processes_and_report_contract(tmp_path):
    root = fixture_graph(tmp_path / "graph")
    out = tmp_path / "out"
    run = subprocess.run([sys.executable, str(SCRIPT), "--graph-path", str(root), "--out", str(out),
                          "--memory-budget-gib", "2", "--include-timestamps", "--hold-seconds", ".15",
                          "--sample-interval", ".04", "--chunk-bytes", "8"],
                         capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr + run.stdout
    report = json.loads((out / "report.json").read_text())
    assert report["schema_version"] == 1
    assert len(report["runs"]) == 4
    assert len({r["child_pid"] for r in report["runs"]}) == 4
    assert "non-unique" in report["measurement_contract"]["rss"]
    for result in report["runs"]:
        assert result["status"] == "complete"
        assert "warm_shared_consumers" in result["phases"]
        loaded = [e for e in result["events"] if e.get("phase", "").startswith("loaded_")][0]
        assert loaded["source_mappings_live"] is False
        assert loaded["array_nbytes"] == {k: v for k, v in result["logical"]["components_bytes"].items() if v}
        assert result["events"][-1]["gpu_memory_bytes"] is None
        assert result["events"][-1]["torch_imported"] is False
        assert Path(result["samples_file"]).exists()
    mixed = report["runs"][0]
    assert "loaded_unfaulted" in mixed["phases"]
    conversion = report["runs"][3]
    assert "conversion_source_and_destination_live" in conversion["phases"]
    if platform.system() != "Linux":
        assert all(r["peak"]["tree_pss_bytes"] is None for r in report["runs"])


def test_timeout_leaves_failed_scenario_in_report(tmp_path):
    root = fixture_graph(tmp_path / "graph")
    out = tmp_path / "out"
    run = subprocess.run([sys.executable, str(SCRIPT), "--graph-path", str(root), "--out", str(out),
                          "--representations", "mixed-mmap", "--memory-budget-gib", "2",
                          "--timeout-seconds", ".1", "--hold-seconds", "2", "--sample-interval", ".02"],
                         capture_output=True, text=True, timeout=10)
    assert run.returncode == 1, run.stderr + run.stdout
    result = json.loads((out / "report.json").read_text())["runs"][0]
    assert result["status"] == "failed"
    assert result["abort_reason"] == "scenario timeout"
    assert result["returncode"] != 0


@pytest.mark.skipif(platform.system() != "Linux", reason="PSS requires Linux /proc")
def test_linux_four_readonly_consumers_use_proportional_tree_pss(tmp_path):
    root = fixture_graph(tmp_path / "graph", timestamps=False)
    out = tmp_path / "out"
    run = subprocess.run([sys.executable, str(SCRIPT), "--graph-path", str(root), "--out", str(out),
                          "--representations", "mixed-resident", "--workers", "4",
                          "--memory-budget-gib", "2", "--hold-seconds", ".4", "--sample-interval", ".05"],
                         capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr + run.stdout
    report = json.loads((out / "report.json").read_text())
    samples = [json.loads(s) for s in Path(report["runs"][0]["samples_file"]).read_text().splitlines()]
    shared = [s for s in samples if s["phase"] == "warm_shared_consumers" and s["process_count"] == 4]
    assert shared, "must include loader plus three live fork consumers"
    assert all(s["pss_complete"] for s in shared)
    assert all(s["tree_pss_bytes"] < s["tree_rss_sum_nonunique_bytes"] for s in shared)
    assert all(s["tree_pss_bytes"] == sum(p["pss_bytes"] for p in s["processes"]) for s in shared)
