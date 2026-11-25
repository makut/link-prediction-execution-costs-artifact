"""Semantic equality, workload freezing, bounded allocation and timer accounting."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.seal_revision import pipeline_benchmark_v4 as pipeline


def config(path, arm="reference_union_stock", workers=0):
    return {"fixture": True, "graph_path": None, "file_endian": "little", "seed": 42,
            "hops": 2, "cap": 3, "oversample": 1.5, "auto_ratio": 1,
            "max_neighbors": 10000, "warmup_pairs": 4, "steps": 2,
            "prefetch_factor": 1, "audit_pairs": 4, "device": "cpu", "torch_threads": 1,
            "pairs_path": str(path), "batch_size": 2, "num_workers": workers, "arm": arm,
            "stock_index_dtype": "auto"}


@pytest.mark.parametrize("negative_kind", ["uniform", "twohop"])
def test_manifest_is_frozen_balanced_and_train_nonedges(tmp_path, negative_kind):
    a, b = tmp_path / "a.npy", tmp_path / "b.npy"
    settings = config(a)
    first = pipeline.prepare_manifest(a, settings, 20, 10, None, negative_kind)
    second = pipeline.prepare_manifest(b, settings, 20, 10, None, negative_kind)
    assert first["sha256"] == second["sha256"]
    assert first["positive_occurrences"] == 10
    graph = pipeline.exact.fixture_graph()
    for u, v, label in np.load(a):
        assert graph.has_edge(int(u), int(v)) == bool(label)
        assert u != v


def test_reference_correction_changes_only_induction_and_construct_preserves_drnl(tmp_path):
    from seal.utils import construct_pyg_graph
    path = tmp_path / "pairs.npy"
    settings = config(path)
    pipeline.prepare_manifest(path, settings, 20, 10, None, "twohop")
    reference = pipeline.PipelineDataset(settings)
    corrected = pipeline.PipelineDataset({**settings, "arm": "reference_union_corrected"})
    for i in range(4):
        a, a_timing, a_audit, a_width = reference[i]
        b, b_timing, b_audit, b_width = corrected[i]
        assert a_audit == b_audit
        assert a_width == "int32" and b_width is None
        for key in ("edge_index", "edge_weight", "z", "node_id", "y"):
            assert np.array_equal(getattr(a, key), getattr(b, key))
        graph = pipeline.exact.oracle_induced(reference.graph, a.node_id.numpy())
        original = construct_pyg_graph(a.node_id.tolist(), graph, np.zeros(a.num_nodes), None, int(a.y[0]), "drnl")
        assert np.array_equal(original.z, a.z)
        assert (a_timing >= 0).all() and (b_timing >= 0).all()


@pytest.mark.parametrize("pv", [False, True])
def test_staged_production_sampler_matches_actual_extractor(tmp_path, pv):
    from graph_csr.graph import GraphCSR
    from graph_csr.large_int32 import LargeInt32Array
    from graph_csr.large_int64_raw import LargeInt64RawArray
    from seal.graph_csr_extraction import _k_hop_subgraph_graph_csr
    graph = pipeline.exact.fixture_graph()
    adapter = GraphCSR(LargeInt32Array(graph.m, graph.indices), LargeInt64RawArray(graph.starts))
    settings = config(tmp_path / "unused.npy")
    for pair in [(0, 1), (0, 4), (3, 5)]:
        staged = pipeline.sample_production(graph, pair, settings, pv)
        seed = pipeline.pair_seed(settings["seed"], *pair)
        np.random.seed(seed)
        pipeline.exact.kernels.seed_numba_rng(seed)
        actual, adjacency, _, _, _ = _k_hop_subgraph_graph_csr(
            *pair, settings["hops"], adapter, max_nodes_per_hop=settings["cap"],
            per_vertex_oversample=settings["oversample"], graph_csr_use_per_vertex_sampling=pv,
            graph_csr_use_pairwise_subgraph="auto")
        assert staged.tolist() == actual
        own = pipeline.exact.production_induced(graph, staged, "auto")
        assert np.array_equal(pipeline.exact.canonical_edges(own), pipeline.exact.canonical_edges(adjacency))


def test_stock_preflight_counts_one_n_scratch_per_worker_no_whole_scipy(tmp_path):
    settings = config(tmp_path / "pairs.npy", workers=8)
    meta = {"nodes": 178_614_236, "adjacency_entries": 12_827_611_818}
    stock = pipeline.preflight(meta, settings)
    local = pipeline.preflight(meta, {**settings, "arm": "reference_union_corrected"})
    assert stock["stock_global_scratch_bytes_per_worker"] == 8 * meta["nodes"]
    assert stock["estimated_peak_bytes"] - local["estimated_peak_bytes"] == 8 * 8 * meta["nodes"]
    assert stock["graph_logical_bytes_once"] == 8 * meta["nodes"] + 4 * meta["adjacency_entries"]
    assert stock["stock_expected_column_scratch_bytes_per_worker"] == 4 * meta["nodes"]
    assert stock["stock_expected_row_index_dtype"] == "int32"


def test_native_scipy_row_restriction_downcasts_even_from_forced_int64_full_csr():
    from scipy import sparse
    full = sparse.csr_matrix((3, 3), dtype=np.int8)
    full.indptr = np.array([0, 2, 3, 4], dtype=np.int64)
    full.indices = np.array([1, 2, 0, 0], dtype=np.int64)
    full.data = np.ones(4, dtype=np.int8)
    assert full.indices.dtype == full.indptr.dtype == np.dtype(np.int64)
    selected = full[np.array([0, 2]), :]
    assert selected.indices.dtype == selected.indptr.dtype == np.dtype(np.int32)
    assert pipeline.resolve_stock_index_dtype("auto", 3, 3) == selected.indices.dtype


def test_native_int32_and_explicit_int64_stock_have_identical_edges_and_drnl():
    graph = pipeline.exact.fixture_graph()
    nodes = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    native, native_width = pipeline.stock_induced(graph, nodes, "auto", selected_nnz=pipeline.selected_degree_sum(graph, nodes))
    wide, wide_width = pipeline.stock_induced(graph, nodes, "int64", selected_nnz=pipeline.selected_degree_sum(graph, nodes))
    assert native_width == "int32" and wide_width == "int64"
    assert np.array_equal(pipeline.exact.canonical_edges(native), pipeline.exact.canonical_edges(wide))
    assert np.array_equal(pipeline.exact.drnl_labels(native), pipeline.exact.drnl_labels(wide))
    assert pipeline.resolve_stock_index_dtype("auto", 178_614_236, 20_000_000) == np.dtype(np.int32)
    assert pipeline.resolve_stock_index_dtype("auto", 1 << 31, 3) == np.dtype(np.int64)
    assert pipeline.resolve_stock_index_dtype("auto", 3, 1 << 31) == np.dtype(np.int64)
    with pytest.raises(pipeline.exact.WorkspaceLimit):
        pipeline.resolve_stock_index_dtype("int32", 1 << 31, 3)


def test_cli_default_is_native_auto_in_manifest_only_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "train_child", lambda *_args: pytest.fail("manifest preparation must not train"))
    out = tmp_path / "manifest_only"
    assert pipeline.main(["--fixture", "--device", "cpu", "--manifest-only", "--out", str(out)]) == 0
    report = json.loads((out / "report.json").read_text())
    assert report["stock_index_dtype"] == "auto"
    assert report["runs"] == []
    assert report["benchmark_version"] == 4
    assert report["routing_contract"]["bridge_arm"] == "reference_union_compiled"
    assert "keeps reference sampling/S/DRNL/model" in report["contract"]["bridge"]


def test_gpu_budget_snapshots_free_memory_before_setting_allocator_cap():
    calls = []

    class FakeCuda:
        def mem_get_info(self, device):
            calls.append(("memory_snapshot", device))
            return 64 * pipeline.memory.GIB, 80 * pipeline.memory.GIB

        def set_per_process_memory_fraction(self, fraction, device):
            calls.append(("allocator_cap", fraction, device))

    device = SimpleNamespace(type="cuda")
    guard = pipeline.configure_gpu_budget(SimpleNamespace(cuda=FakeCuda()), device, 24)
    assert [call[0] for call in calls] == ["memory_snapshot", "allocator_cap"]
    assert calls[1][1] == pytest.approx(.3)
    assert guard["allocator_budget_bytes"] == 24 * pipeline.memory.GIB
    assert guard["free_gib_before_model"] == 64
    assert "external" in guard["scope"]


@pytest.mark.parametrize("budget,free,total", [(24, 20, 80), (81, 80, 80)])
def test_gpu_budget_refuses_insufficient_headroom_or_total_capacity(budget, free, total):
    class FakeCuda:
        def mem_get_info(self, _device):
            return free * pipeline.memory.GIB, total * pipeline.memory.GIB

        def set_per_process_memory_fraction(self, *_args, **_kwargs):
            pytest.fail("failed preflight must not configure an allocator/model")

    with pytest.raises(RuntimeError, match="GPU preflight refused"):
        pipeline.configure_gpu_budget(SimpleNamespace(cuda=FakeCuda()), SimpleNamespace(type="cuda"), budget)


def test_cpu_budget_configuration_never_calls_cuda():
    class ForbiddenCuda:
        def __getattr__(self, name):
            pytest.fail(f"CPU fixture called CUDA API {name}")

    guard = pipeline.configure_gpu_budget(SimpleNamespace(cuda=ForbiddenCuda()), SimpleNamespace(type="cpu"), 24)
    assert guard["status"] == "not_applicable_cpu"
    assert guard["cuda_api_called"] is False


def test_cpu_five_arm_training_counts_exclude_warmup_and_match_audits(tmp_path):
    out = tmp_path / "run"
    process = subprocess.run([sys.executable, str(Path(pipeline.__file__)), "--fixture", "--device", "cpu",
                              "--workers", "0", "--batch-sizes", "2", "--steps", "2", "--warmup-steps", "1",
                              "--cap", "3", "--memory-budget-gib", "8", "--gpu-memory-budget-gib", "7",
                              "--stock-index-dtype", "int64", "--out", str(out)],
                             capture_output=True, text=True, timeout=120)
    assert process.returncode == 0, process.stdout + process.stderr
    report = json.loads((out / "report.json").read_text())
    assert len(report["runs"]) == 5
    assert report["equality_audits"][0]["same_nodes_edges_drnl_and_initial_model"] is True
    for run in report["runs"]:
        assert run["status"] == "complete"
        assert run["timing"]["measured_graphs"] == 4
        assert run["timing"]["warmup_graphs_excluded"] == 2
        assert run["timing"]["measured_steps"] == 2
        assert run["model"]["sortpool_k"] == 281 and run["model"]["use_feature"] is False
        assert set(run["timing"]["worker_service_seconds_sum"]) == {"sampling", "induction", "drnl", "construct"}
        assert run["timing"]["graphs_per_second"] == pytest.approx(4 / run["timing"]["steady_seconds"])
        assert run["gpu_memory"]["allocated_peak_gib"] is None
        assert run["config"]["gpu_memory_budget_gib"] == 7
        assert run["config"]["stock_index_dtype"] == "int64"
        assert run["gpu_memory"]["guard"]["status"] == "not_applicable_cpu"
    assert len(report["equality_audits"]) == 3
    assert all(a["same_nodes_edges_drnl_and_initial_model"] for a in report["equality_audits"])
    assert report["benchmark_version"] == 4
    a, b = report["runs"][:2]
    assert a["stock_physical_route"]["measured_temporary_row_dtype_counts"] == {"int64": 4}
    # With the same CPU tensors, order, seeds and optimizer steps, this pair has
    # identical trained weights too; a test against accidentally changed labels.
    assert a["model"]["final_state_sha256"] == b["model"]["final_state_sha256"] == report["runs"][-1]["model"]["final_state_sha256"]


def test_spawn_workers_preserve_manifest_and_per_pair_extraction(tmp_path):
    from torch.utils.data import DataLoader
    path = tmp_path / "pairs.npy"
    settings = config(path, workers=2)
    pipeline.prepare_manifest(path, settings, 12, 6, None, "uniform")
    dataset = pipeline.PipelineDataset(settings)
    expected = [dataset[i][2] for i in range(4)]
    # A fresh Dataset contains only paths, not already-open graph mmap arrays.
    spawned = pipeline.PipelineDataset(settings)
    assert spawned.graph is None and spawned.pairs is None
    loader = DataLoader(spawned, batch_size=2, num_workers=2, multiprocessing_context="spawn",
                        worker_init_fn=pipeline.worker_init, collate_fn=pipeline.Collate(), prefetch_factor=1)
    iterator = iter(loader)
    try:
        actual = next(iterator)["audits"] + next(iterator)["audits"]
        assert actual == expected
    finally:
        iterator._shutdown_workers()


@pytest.mark.parametrize("nodes", [[], [8], [0, 8], [0, 1, 2, 4, 8]])
def test_vectorized_degree_sum_matches_bounds_including_last_row(nodes):
    graph = pipeline.exact.fixture_graph()
    selected = np.array(nodes, dtype=np.int64)
    assert pipeline.selected_degree_sum(graph, selected) == graph.degree_sum(selected)


def test_vectorized_degree_sum_uses_int64_and_preserves_offset_guards():
    graph = SimpleNamespace(starts=np.array([0, 2**31], dtype=np.int64), m=2**31 + 7)
    assert pipeline.selected_degree_sum(graph, np.array([0, 1], dtype=np.int64)) == 2**31 + 7
    for nodes in [np.array([-1]), np.array([2]), np.array([.5])]:
        with pytest.raises(ValueError):
            pipeline.selected_degree_sum(graph, nodes)
    with pytest.raises(ValueError, match="invalid selected"):
        pipeline.selected_degree_sum(SimpleNamespace(starts=np.array([2, 1]), m=3), np.array([0]))


@pytest.mark.parametrize("arm", pipeline.ARMS)
def test_getitem_collects_selected_degree_once_and_never_calls_python_degree_or_auto(tmp_path, monkeypatch, arm):
    graph = pipeline.exact.fixture_graph()
    nodes = np.array([0, 1, 2, 4, 8], dtype=np.int64)
    settings = config(tmp_path / "unused.npy", arm=arm)
    dataset = pipeline.PipelineDataset(settings)
    dataset.graph = graph
    dataset.pairs = np.array([[0, 1, 1]], dtype=np.int64)
    monkeypatch.setattr(pipeline.exact, "sample_node_set", lambda *_a, **_k: nodes)
    monkeypatch.setattr(pipeline, "sample_production", lambda *_a, **_k: nodes)
    monkeypatch.setattr(graph, "degree_sum", lambda *_a: pytest.fail("Python selected degree must not be recomputed"))
    original_degree = pipeline.selected_degree_sum
    original_induced = pipeline.exact.production_induced
    calls = []
    def degree_once(*args):
        calls.append("degree")
        return original_degree(*args)
    def fixed_induced(g, selected, mode, **kwargs):
        assert mode in ("scan", "pairwise")
        return original_induced(g, selected, mode, **kwargs)
    monkeypatch.setattr(pipeline, "selected_degree_sum", degree_once)
    monkeypatch.setattr(pipeline.exact, "production_induced", fixed_induced)
    data, _, audit, width = dataset[0]
    assert calls == ["degree"]
    oracle = pipeline.exact.oracle_induced(graph, nodes)
    assert audit["edges_sha256"] == pipeline.exact.digest(pipeline.exact.canonical_edges(oracle))
    assert audit["drnl_sha256"] == pipeline.exact.digest(pipeline.exact.drnl_labels(oracle))
    assert width == ("int32" if arm == "reference_union_stock" else None)


@pytest.mark.parametrize("arm", ["reference_union_stock", "reference_union_corrected", "reference_union_compiled", "production_union"])
def test_known_degree_guards_refuse_before_induction(tmp_path, monkeypatch, arm):
    graph = pipeline.exact.fixture_graph()
    nodes = np.array([0, 1, 2, 4], dtype=np.int64)
    settings = {**config(tmp_path / "unused.npy", arm=arm), "max_neighbors": 1, "auto_ratio": .001}
    degree = pipeline.selected_degree_sum(graph, nodes)
    with pytest.raises(pipeline.exact.WorkspaceLimit):
        pipeline.induce_selected(graph, nodes, settings, degree)


def test_pairwise_route_keeps_its_existing_no_scan_workspace_behavior(tmp_path):
    graph = pipeline.exact.fixture_graph()
    nodes = np.array([0, 1, 2, 4], dtype=np.int64)
    settings = {**config(tmp_path / "unused.npy", arm="production_pv"), "max_neighbors": 1, "auto_ratio": 100}
    actual, width = pipeline.induce_selected(graph, nodes, settings, pipeline.selected_degree_sum(graph, nodes))
    assert width is None
    assert np.array_equal(pipeline.exact.canonical_edges(actual), pipeline.exact.canonical_edges(pipeline.exact.oracle_induced(graph, nodes)))


def test_bridge_matches_both_reference_arms_when_cap_binds(tmp_path):
    path = tmp_path / "pairs.npy"
    settings = {**config(path), "cap": 1}
    pipeline.prepare_manifest(path, settings, 20, 10, None, "twohop")
    refs = [pipeline.PipelineDataset({**settings, "arm": arm}) for arm in
            ("reference_union_stock", "reference_union_corrected", "reference_union_compiled")]
    for index in range(4):
        samples = [dataset[index] for dataset in refs]
        first = samples[0][0]
        assert first.num_nodes <= 2 + settings["hops"] * settings["cap"] < refs[0].graph.n
        assert samples[0][2] == samples[1][2] == samples[2][2]
        for sample, _, _, _ in samples[1:]:
            for field in ("edge_index", "edge_weight", "node_id", "z", "y"):
                assert np.array_equal(getattr(first, field), getattr(sample, field))
    assert pipeline.ROUTING_CONTRACT["bridge_arm"] == "reference_union_compiled"
