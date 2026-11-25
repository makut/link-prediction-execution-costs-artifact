"""Orchestration contracts only: no real GPU query, training, cluster or large inputs."""
import importlib.util
import json
from pathlib import Path
import struct
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).with_name("run_pipeline_suite_v4.py")
SPEC = importlib.util.spec_from_file_location("run_pipeline_suite_v4_tested", SCRIPT)
suite_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(suite_module)


def tiny_repo(path):
    for relative in suite_module.SOURCE_FILES:
        target = path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# frozen tiny source fixture\n")
    return path


def controller(tmp_path):
    base = tmp_path / "base"
    repo = tiny_repo(base / "code")
    output = base / "results/pipeline"
    output.mkdir(parents=True)
    plan = suite_module.build_plan(base, repo, "/fake/python")
    plan["source_hashes"] = suite_module.source_hashes(repo)
    args = SimpleNamespace(poll_seconds=.01, quality_wait_hours=1, input_wait_hours=1, gpu_idle_wait_minutes=1)
    return suite_module.Suite(args, plan)


def test_plan_has_exact_35_jobs_and_common_8000_pair_protocol(tmp_path):
    plan = suite_module.build_plan(tmp_path, tmp_path / "code", "/fake/python")
    assert len(plan["jobs"]) == 35  # 2 GPU smokes + 3*(manifest + 10 controls)
    assert [job["kind"] for job in plan["jobs"][:2]] == ["expected_gpu_refusal", "gpu_smoke"]
    controls = [job for job in plan["jobs"] if job["kind"] == "pipeline"]
    assert len(controls) == 30
    for job in controls:
        argv = job["argv"]
        for flag, value in [("--batch-sizes", "32"), ("--steps", "100"), ("--warmup-steps", "8"),
                            ("--prefetch-factor", "1"), ("--n-positives", "4000"), ("--cap", "200"),
                            ("--memory-budget-gib", "180"), ("--gpu-memory-budget-gib", "24"),
                            ("--max-neighbors", "20000000"), ("--timeout-seconds", "3600"),
                            ("--stock-index-dtype", "auto")]:
            assert argv[argv.index(flag) + 1] == value
        assert argv[argv.index("--workers") + 1] == str(job["workers"])
        assert job["workers"] in (1, 8)
    assert plan["inputs"]["friendster"]["graph"].endswith("data/friendster/full_csr")
    assert plan["protocol"]["max_neighbors"] == 20_000_000
    assert plan["protocol"]["scenario_timeout_seconds"] == 3600
    assert plan["protocol"]["stock_index_dtype"] == "auto"
    assert plan["protocol_sha256"] == suite_module.digest_json(plan["protocol"])


def test_existing_outputs_refused_without_overwriting(tmp_path):
    output = tmp_path / "pipeline"
    suite_module.reserve_output(output)
    original = output / "important.json"
    original.write_text("keep")
    with pytest.raises(RuntimeError, match="Refusing existing"):
        suite_module.reserve_output(output)
    assert original.read_text() == "keep"


def test_quality_gate_waits_for_six_results_and_controller_finalizer(tmp_path):
    assert suite_module.quality_gate(tmp_path) is None
    (tmp_path / "GPU_SUITE_COMPLETE.json").write_text(json.dumps({"quality_complete": True}))
    rows = [{"name": f"{variant}{seed}", "variant": variant, "seed": seed} for variant in ("union", "pv") for seed in (42, 43, 44)]
    (tmp_path / "QUALITY_COMPLETE.json").write_text(json.dumps({"results": rows}))
    lock = tmp_path / "GPU_SUITE_RUNNING.lock"
    lock.write_text("123")
    assert suite_module.quality_gate(tmp_path) is None
    lock.unlink()
    assert suite_module.quality_gate(tmp_path)["quality_results"] == 6
    (tmp_path / "QUALITY_COMPLETE.json").write_text(json.dumps({"results": rows[:5]}))
    with pytest.raises(RuntimeError, match="six unique"):
        suite_module.quality_gate(tmp_path)


def test_failed_quality_is_blocking_not_an_idle_permission(tmp_path):
    (tmp_path / "gpu_status.json").write_text(json.dumps({"stage": "failed"}))
    with pytest.raises(RuntimeError, match="Quality suite failed"):
        suite_module.quality_gate(tmp_path)


def test_gpu_uuid_selection_and_compute_parser_are_fail_closed():
    inventory = suite_module.parse_inventory("0, GPU-aaa, NVIDIA H100 80GB HBM3, 81559, 570.124\n1, GPU-bbb, NVIDIA H100 80GB HBM3, 81559, 570.124\n")
    permitted = suite_module.permitted_inventory(inventory, {"NVIDIA_VISIBLE_DEVICES": "GPU-bbb"})
    assert suite_module.select_target(permitted, {})["uuid"] == "GPU-bbb"
    assert suite_module.select_target(inventory, {"CUDA_VISIBLE_DEVICES": "GPU-bbb"})["uuid"] == "GPU-bbb"
    assert suite_module.parse_compute_apps("") == []
    apps = suite_module.parse_compute_apps('GPU-bbb, 123, "python, custom name", 1024\n')
    assert apps[0]["pid"] == 123 and apps[0]["process_name"] == "python, custom name"
    with pytest.raises(RuntimeError, match="refusing to assume"):
        suite_module.parse_compute_apps("NVIDIA query was not understood")
    with pytest.raises(RuntimeError, match="requires H100"):
        suite_module.select_target([{**inventory[0], "name": "NVIDIA RTX A6000"}], {})


def test_changed_runtime_source_blocks_next_job(tmp_path):
    repo = tiny_repo(tmp_path / "repo")
    frozen = suite_module.source_hashes(repo)
    (repo / suite_module.SOURCE_FILES[-1]).write_text("changed")
    with pytest.raises(RuntimeError, match="sources changed"):
        suite_module.assert_frozen_sources(repo, frozen)


def test_input_snapshot_ignores_other_download_records_but_detects_topology_change(tmp_path):
    graph = tmp_path / "data/friendster/full_csr"
    graph.mkdir(parents=True)
    (graph / "edge_ends.bin").write_bytes(struct.pack("<q2i", 2, 1, 0))
    (graph / "edge_starts.bin").write_bytes(struct.pack("<2q", 0, 1))
    download = tmp_path / "data/input_download_manifest.json"
    download.write_text(json.dumps({str(graph / "edge_ends.bin"): {"sha256": "frozen", "bytes": 16}}))
    settings = {"graph": str(graph), "positive_graph": None}
    before = suite_module.snapshot_input(settings, tmp_path)
    download.write_text(json.dumps({str(graph / "edge_ends.bin"): {"sha256": "frozen", "bytes": 16},
                                    "unrelated download": {"bytes": 100}}))
    assert suite_module.snapshot_input(settings, tmp_path) == before
    (graph / "edge_ends.bin").write_bytes(struct.pack("<q2i", 3, 1, 0))
    with pytest.raises(RuntimeError, match="byte sizes"):
        suite_module.snapshot_input(settings, tmp_path)


def test_expected_gpu_guard_refusal_is_retained_not_treated_as_successful_training(tmp_path, monkeypatch):
    suite = controller(tmp_path)
    job = suite.plan["jobs"][0]
    monkeypatch.setattr(suite, "wait_idle", lambda _job: None)

    class FakeProcess:
        pid = 999

        def __init__(self, *_args, **_kwargs):
            directory = Path(job["out"])
            directory.mkdir(parents=True)
            (directory / "report.json").write_text(json.dumps({"runs": [{"status": "failed"}]}))
            (directory / "process.log").write_text("RuntimeError: GPU preflight refused enormous budget\n")

        def poll(self):
            return 1

        def wait(self):
            return 1

    monkeypatch.setattr(suite_module.subprocess, "Popen", FakeProcess)
    record = suite.execute(job)
    assert record["status"] == "expected_guard_refusal"
    assert record["returncode"] == 1
    assert (suite.output / "jobs/guard_refusal_exit.json").exists()
    assert len(suite.records) == 1


def test_existing_job_output_rejected_before_gpu_query(tmp_path, monkeypatch):
    suite = controller(tmp_path)
    job = suite.plan["jobs"][0]
    Path(job["out"]).mkdir(parents=True)
    monkeypatch.setattr(suite, "wait_idle", lambda _job: pytest.fail("existing outputs must fail before touching GPU"))
    with pytest.raises(RuntimeError, match="existing job output"):
        suite.execute(job)


def test_busy_gpu_waits_for_two_clean_snapshots_and_never_kills_apps(tmp_path, monkeypatch):
    suite = controller(tmp_path)
    for name in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"):
        suite.environment.pop(name, None)
    busy_then_idle = ["GPU-aaa, 123, existing_training, 1024\n", "", ""]

    def fake_query(query):
        if query.startswith("--query-gpu"):
            return "0, GPU-aaa, NVIDIA H100, 81559, 570.124\n"
        return busy_then_idle.pop(0)

    monkeypatch.setattr(suite_module, "nvidia_query", fake_query)
    monkeypatch.setattr(suite_module.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(suite_module.os, "kill", lambda *_args: pytest.fail("must not kill GPU apps"))
    monkeypatch.setattr(suite_module.os, "killpg", lambda *_args: pytest.fail("must not kill GPU apps"))
    suite.wait_idle(suite.plan["jobs"][0])
    assert busy_then_idle == []
    assert suite.environment["CUDA_VISIBLE_DEVICES"] == "GPU-aaa"
    observations = [json.loads(line) for line in (suite.output / "gpu_idle_observations.jsonl").read_text().splitlines()]
    assert observations[0]["busy_apps"][0]["pid"] == 123
    assert observations[-1]["busy_apps"] == []


def test_v4_routes_and_hashes_are_explicit_without_changing_job_protocol(tmp_path):
    plan = suite_module.build_plan(tmp_path, tmp_path / "code", "/fake/python")
    assert plan["protocol"]["protocol_version"] == 4
    assert plan["protocol"]["routing_version"] == "production_matched_vectorized_selected_degree_once_v4_reference_bridge"
    assert "scripts/seal_revision/pipeline_benchmark_v4.py" in suite_module.SOURCE_FILES
    assert "scripts/seal_revision/exact_benchmark.py" in suite_module.SOURCE_FILES
    assert "seal/graph_csr_extraction.py" in suite_module.SOURCE_FILES
    assert "scripts/seal_revision/pipeline_benchmark.py" not in suite_module.SOURCE_FILES
    assert all("pipeline_benchmark_v4.py" in job["argv"][2] for job in plan["jobs"])


def test_v4_controller_refuses_unversioned_or_old_routing_report(tmp_path):
    suite = controller(tmp_path)
    with pytest.raises(RuntimeError, match="V4 routing metadata"):
        suite.validate_gpu_report(suite.plan["jobs"][1], {"runs": []})


def test_v4_keeps_every_existing_v3_job_and_adds_only_six_bridge_controls(tmp_path):
    spec = importlib.util.spec_from_file_location("old_v3_for_comparison", SCRIPT.with_name("run_pipeline_suite_v3.py"))
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    expected = old.build_plan(tmp_path, tmp_path / "code", "/fake/python")["jobs"]
    plan = suite_module.build_plan(tmp_path, tmp_path / "code", "/fake/python")
    bridge = [j for j in plan["jobs"] if j.get("arm") == "reference_union_compiled"]
    assert len(bridge) == 6 and all(j["kind"] == "pipeline" for j in bridge)
    existing = [json.loads(json.dumps(j)) for j in plan["jobs"] if j not in bridge]
    # The five-arm fixture adds one bridge argument; existing measured jobs remain identical.
    for job in existing:
        job["argv"] = [x.replace("pipeline_benchmark_v4.py", "pipeline_benchmark_v3.py") for x in job["argv"]]
        if job["kind"] == "gpu_smoke":
            job["argv"].remove("reference_union_compiled")
    assert existing == expected


def test_five_arm_smoke_requires_all_three_reference_bridge_audits(tmp_path):
    from itertools import combinations
    suite = controller(tmp_path)
    refs = ("reference_union_stock", "reference_union_corrected", "reference_union_compiled")
    report = {"benchmark_version": 4, "routing_contract": {
        "version": suite.plan["protocol"]["routing_version"], "bridge_arm": "reference_union_compiled"},
        "runs": [{"config": {"arm": arm}, "device": {"type": "cuda"},
                  "gpu_memory": {"reserved_peak_gib": 1, "guard": {
                      "status": "configured_before_model_allocation", "allocator_budget_bytes": 24*2**30, "allocator_fraction": .3}},
                  "model": {"sortpool_k": 281, "use_feature": False}} for arm in suite_module.ARMS],
        "equality_audits": [{"reference_arm": a, "candidate_arm": b,
                             "same_nodes_edges_drnl_and_initial_model": True} for a,b in combinations(refs,2)]}
    suite.validate_gpu_report(suite.plan["jobs"][1], report)
    report["equality_audits"][-1]["same_nodes_edges_drnl_and_initial_model"] = False
    with pytest.raises(RuntimeError, match="five-arm"):
        suite.validate_gpu_report(suite.plan["jobs"][1], report)
