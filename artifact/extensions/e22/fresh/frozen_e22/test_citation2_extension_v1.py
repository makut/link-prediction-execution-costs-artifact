"""Real tiny numerical/native checks; no full graph, Torch simulation or performance claims."""
from pathlib import Path
import ast
import collections
import copy
import hashlib
import json
import os
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cpu_common_v1 as common
import cpu_coordinator_v1 as coordinator
import cpu_worker_v1 as worker
import cohort_v1 as cohort


def actual_runtime():
    return worker.load_runtime()


def tiny_graph(width=32):
    np, exact, prepare, replay, functions = actual_runtime()
    rows = [set() for _ in range(16)]
    for u in range(16):
        for v in ((u + 1) % 16, (u + 3) % 16):
            rows[u].add(v); rows[v].add(u)
    ptr = np.asarray([0, *np.cumsum([len(x) for x in rows])], dtype='<i8')
    col = np.asarray([v for row in rows for v in sorted(row)], dtype='<i'+str(width//8))
    return exact.CSRGraph(ptr[:-1], col, 'tiny actual fixture'), ptr, col


def test_source_inventory_and_all_python_syntax_are_python310_compatible():
    bindings = common.verify_sources()
    assert len(bindings) >= 30
    for item in common.read(common.HERE/'SOURCE_MANIFEST.json')['files']:
        if item['path'].endswith('.py'):
            ast.parse((common.HERE/item['path']).read_text(), feature_version=(3, 10))


def test_new_candidate_selection_is_deterministic_stratified_and_blind_to_timings():
    _, _, prepare, _, _ = actual_runtime()
    protocol = common.read(common.HERE/'COHORT_PROTOCOL.json')
    selected, _ = cohort.choose_candidates(protocol, common.sha(common.HERE/'COHORT_PROTOCOL.json'), prepare)
    assert len(selected) == 128
    counts = collections.Counter((m['sampler'], m['cap'], m['workload']) for _, m in selected)
    assert len(counts) == 12 and set(counts.values()) <= {10, 11}
    assert selected == cohort.choose_candidates(protocol, common.sha(common.HERE/'COHORT_PROTOCOL.json'), prepare)[0]
    changed = copy.deepcopy(protocol); changed['candidate_selection_seed'] += 1
    assert selected != cohort.choose_candidates(changed, common.sha(common.HERE/'COHORT_PROTOCOL.json'), prepare)[0]


def test_actual_tiny_cohort_generation_and_independent_oracles_are_reproducible():
    np, exact, prepare, _, _ = actual_runtime()
    graph, _, _ = tiny_graph()
    protocol = common.read(common.HERE/'COHORT_PROTOCOL.json')
    protocol.update(nodes=graph.n, stored_entries=graph.m)
    archive = cohort.build_archive(graph, protocol, 'fixture-only', np, exact, prepare)
    graph.origin = 'different local fixture path, no effect on cohort'
    assert archive == cohort.build_archive(graph, protocol, 'fixture-only', np, exact, prepare)
    assert archive['configuration']['timing_methods_executed'] == 0 and len(archive['records']) == 128
    for row in archive['records']:
        prepare.valid_record(row, graph.n)
        nodes = np.asarray(row['nodes'], dtype=np.int64)
        result = exact.oracle_induced(graph, nodes)
        assert exact.digest(exact.canonical_edges(result)) == row['canonical_edges_sha256']
        assert exact.digest(exact.drnl_labels(result)) == row['drnl_sha256']
        assert not result[0, 1] and not result[1, 0]


def test_independent_streamed_int32_promotion_hashes_and_source_corruption(tmp_path):
    import numpy as np
    graph, ptr, col = tiny_graph()
    folder = tmp_path/'tiny graph'; folder.mkdir()
    (folder/'edge_starts.bin').write_bytes(ptr[:-1].tobytes())
    (folder/'edge_ends.bin').write_bytes(graph.m.to_bytes(8, 'little', signed=True)+col.tobytes())
    pins = {p.name: common.info(p) for p in folder.iterdir()}
    admitted = common.admit_original(folder, pins, (graph.n, graph.m))
    expected = {'rowptr_sha256': hashlib.sha256(ptr.tobytes()).hexdigest(),
                'col_sha256': hashlib.sha256(col.astype('<i8').tobytes()).hexdigest()}
    assert admitted['expected_native_hashes'] == expected
    common.stable_original(admitted)
    p = folder/'edge_ends.bin'; before = p.stat(); raw = bytearray(p.read_bytes()); raw[-1] ^= 1; p.write_bytes(raw)
    os.utime(p, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(RuntimeError, match='payload differs'):
        common.admit_original(folder, pins, (graph.n, graph.m))


@pytest.mark.parametrize('width', [32, 64])
def test_actual_fixed_numba_kernels_match_independent_graph_and_BFS(width):
    np, exact, prepare, replay, functions = actual_runtime()
    from fixtures_v1 import bfs_labels
    graph, ptr, col = tiny_graph(width)
    for index, ids in enumerate(([7, 0, 1, 3, 4, 5, 8], [13, 2, 6, 14], [10, 8])):
        nodes = np.asarray(ids, dtype=np.int64)
        oracle = exact.oracle_induced(graph, nodes)
        edges = exact.canonical_edges(oracle); labels = bfs_labels(edges, len(ids), (0, 1))
        row = {'nodes': list(ids), 'sum_out_degrees': graph.degree_sum(nodes),
               'node_set_sha256': exact.digest(nodes), 'canonical_edges_sha256': exact.digest(edges),
               'drnl_sha256': exact.digest(labels), 'induced_directed_entries': int(oracle.nnz)}
        report = functions.one(row, index, graph, np, exact)
        assert report['exact_archived_oracles'] and set(report['methods']) == {'scan', 'pairwise'}
        for result in report['methods'].values():
            assert result['exact_edges'] and result['exact_drnl']
            assert coordinator.finite_times(result['edge_seconds'])
            assert coordinator.finite_times(result['edge_plus_drnl_seconds'])
    assert all(getattr(exact.kernels, n).nopython_signatures for n in
               ('_build_induced_edges_pairwise', '_fill_neighbors_with_owner_numba', '_build_induced_edges_neighbor_scan'))


def test_actual_compiled_fullclear_and_persistent_native_retain_order_and_reset(tmp_path):
    np, exact, prepare, replay, functions = actual_runtime()
    from build_native_v1 import build
    from native_adapter_v1 import NativeCSR, ordered_oracle, pack_local, canonical_edges, drnl
    library, _ = build(tmp_path/'actual standalone build')
    graph, ptr, col = tiny_graph(64)
    ids = np.asarray([7, 0, 1, 3, 4, 5, 8], dtype=np.int64)
    expected = ordered_oracle(ptr, col, ids)
    oracle = exact.oracle_induced(graph, ids)
    for mode in ('reference_full_clear', 'persistent_touched_reset'):
        with NativeCSR(ptr, col, library, mode=mode, max_assoc_bytes=4096) as native:
            assert native.initialization['persistent_assoc_bytes_per_worker'] == (8 * graph.n if mode == 'persistent_touched_reset' else 0)
            for _ in range(2):
                with native.induce(ids) as result:
                    assert all(np.array_equal(a, b) for a, b in zip((result.rows, result.cols, result.edge_ids), expected))
                    local = pack_local(result.rows, result.cols, len(ids))
                    assert np.array_equal(canonical_edges(local), exact.canonical_edges(oracle))
                    assert np.array_equal(drnl(local), exact.drnl_labels(oracle))
                assert native.live_touched == 0


def test_plan_only_is_real_source_check_without_numerical_imports_or_outputs():
    result = subprocess.run([sys.executable, '-B', str(common.HERE/'cpu_coordinator_v1.py')],
                            check=True, capture_output=True, text=True, timeout=20)
    value = json.loads(result.stdout)
    assert value['status'] == 'plan_only' and value['runtime_or_graph_read'] is False
    assert value['native_bytes'] == 509051728 and value['count'] == 128
    assert value['graph']['path'].endswith('/dataset/prepared/train_csr')
    command = value['native_argv']
    assert command[command.index('--warmup')+1] == '1' and command[command.index('--repeats')+1] == '3'
    assert command[command.index('--conversion-max-native-gib')+1] == '1'
    assert command[command.index('--max-peak-gib')+1] == '8'
    assert command[command.index('--max-assoc-gib')+1] == '0.05'


def test_same_host_boot_and_affinity_are_required():
    host = {'hostname': 'fixture', 'boot_id': 'boot', 'cpu_models': ['CPU'], 'cpu_affinity': [0, 1]}
    common.verify_host(host, dict(host))
    for key in host:
        changed = copy.deepcopy(host); changed[key] = 'different'
        with pytest.raises(RuntimeError, match='same documented CPU'):
            common.verify_host(host, changed)
