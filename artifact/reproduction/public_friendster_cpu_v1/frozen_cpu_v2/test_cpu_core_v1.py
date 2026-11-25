"""Bounded contract/admission tests and real tiny unchanged scientific kernels."""
from pathlib import Path
import copy, hashlib, json, os, sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cpu_common_v1 as common
import cpu_coordinator_v1 as queue


def tiny_full(path):
    import numpy as np
    path.mkdir()
    starts = np.array([0, 1, 3], dtype='<i8')
    columns = np.array([1, 0, 2, 1], dtype='<i4')
    (path/'edge_starts.bin').write_bytes(starts.tobytes())
    (path/'edge_ends.bin').write_bytes((4).to_bytes(8, 'little', signed=True)+columns.tobytes())
    (path/'num_nodes.txt').write_text('3\n')
    pins = {p.name: common.info(p) for p in path.iterdir()}
    return pins


def test_full_SHA_admission_and_actual_stat_continuity(tmp_path):
    folder = tmp_path/'full'; pins = tiny_full(folder)
    admission = common.admit_original(folder, pins, (3, 4)); common.stable_original(admission)
    p = folder/'edge_ends.bin'; before = p.stat(); data = bytearray(p.read_bytes()); data[-1] ^= 1; p.write_bytes(data)
    os.utime(p, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(RuntimeError, match='whole public fullCSR'): common.admit_original(folder, pins, (3, 4))


def test_train_graph_header_is_rejected_even_with_self_consistent_file_hash(tmp_path):
    folder = tmp_path/'full'; pins = tiny_full(folder)
    with pytest.raises(RuntimeError, match='trainCSR rejected'): common.admit_original(folder, pins, (3, 5))


def test_full_graph_symlink_is_rejected(tmp_path):
    folder = tmp_path/'full'; pins = tiny_full(folder)
    p = folder/'edge_starts.bin'; p.rename(folder/'elsewhere'); p.symlink_to(folder/'elsewhere')
    with pytest.raises(RuntimeError, match='regular numeric'): common.admit_original(folder, pins, (3, 4))


@pytest.mark.parametrize('change', ['nodes', 'seed', 'order', 'oracle', 'M', 'count'])
def test_exact_frozen128_selection_mutations_rejected(change):
    plan = common.read(common.HERE/'FROZEN_128_SELECTION.json'); common.frozen_plan_match(plan)
    if change == 'nodes': plan['selected'][0]['node_set_sha256'] = '0'*64
    elif change == 'seed': plan['selection_seed'] += 1
    elif change == 'order': plan['selected'].reverse()
    elif change == 'oracle': plan['selected'][0]['drnl_sha256'] = '0'*64
    elif change == 'M': plan['graph']['stored_entries'] -= 1
    else: plan['count'] -= 1
    with pytest.raises(RuntimeError, match='frozen128'): common.frozen_plan_match(plan)


def test_native_command_has_correct_full_graph_and_original_budgets():
    command = queue.native_command(common.BASE, Path('/fresh/bin/python'))
    def arg(name): return command[command.index(name)+1]
    assert arg('--graph-path').endswith('/new_data/full_csr')
    assert arg('--count') == '128' and arg('--conversion-max-native-gib') == '28'
    assert arg('--max-peak-gib') == '48' and arg('--max-assoc-gib') == '0.5'
    assert arg('--warmup') == '1' and arg('--repeats') == '3'
    assert all(not s.startswith('--yt') and not s.startswith('//home/') and '/vk/' not in s for s in command)


@pytest.mark.parametrize('bad', [[1., 2.], [1., 2., float('nan')], [1., 2., -1.], [1., 2., True]])
def test_all_three_strict_finite_repeat_times_required(bad):
    assert not queue.finite_times(bad)


def test_boolean_or_guarded_process_status_is_not_clean():
    clean = {'returncode': 0, 'timed_out': False, 'owned_tree_quiescent': True, 'orphan_cleanup': False, 'guard_abort_reason': None}
    assert common.clean(clean)
    for key, value in [('returncode', False), ('timed_out', True), ('owned_tree_quiescent', False), ('guard_abort_reason', 'budget')]:
        assert not common.clean({**clean, key: value})


def test_default_plan_performs_no_runtime_probe_or_input_admission(monkeypatch, capsys):
    monkeypatch.setattr(queue, 'verify_sources', lambda: {})
    monkeypatch.setattr(queue, 'sha', lambda _: '0'*64)
    monkeypatch.setattr(queue, 'idle', lambda *_: (_ for _ in ()).throw(AssertionError('runtime scan forbidden in default plan')))
    assert queue.main([]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value['status'] == 'plan_only' and value['runtime_or_graph_read'] is False
    assert value['native_bytes'] == 29421941096 and value['graph']['stored_entries'] == 3612134270


def test_only_source_manifest_members_enter_local_raw_packet(tmp_path, monkeypatch):
    source = tmp_path/'source'; source.mkdir(); (source/'kept.txt').write_text('public\n')
    (source/'unlisted.bin').write_text('not selected\n')
    common.save(source/'SOURCE_MANIFEST.json', {'files': [{'path': 'kept.txt', **common.info(source/'kept.txt')}]})
    out = tmp_path/'result'; out.mkdir(); common.save(out/'SCIENTIFIC_TERMINAL.json', {'success': True})
    monkeypatch.setattr(queue, 'HERE', source)
    packet = tmp_path/'result.tgz'; receipt = queue.pack_local(out, packet)
    import tarfile
    with tarfile.open(packet) as archive:
        names = archive.getnames()
    assert len(names) == 4 and not any('unlisted' in name for name in names)
    assert receipt['YT_writes'] == 0


def test_proc_scan_detects_only_live_other_project_jobs(tmp_path):
    proc = tmp_path/'proc'; proc.mkdir()
    for pid, state, filename in [('999991', 'S', 'cpu_worker_v1.py'), ('999992', 'Z', 'cpu_worker_v1.py'), ('999993', 'S', 'unrelated.py')]:
        folder = proc/pid; folder.mkdir(); (folder/'stat').write_text(pid+' (python) '+state+' 0 0 0')
        (folder/'cmdline').write_bytes(('python\0'+str(common.BASE/filename)+'\0').encode())
    assert queue.scan_active(common.BASE, proc) == [{'pid': 999991, 'entrypoints': ['cpu_worker_v1.py']}]


@pytest.mark.parametrize('change', ['wrong_global_hash', 'wrong_source', 'wrong_dimensions', 'missing_sentinel', 'boolean_exit'])
def test_consistent_but_wrong_conversion_receipts_cannot_replace_whole_graph_anchor(tmp_path, change):
    parent, native = tmp_path/'parent', tmp_path/'native'; parent.mkdir(); native.mkdir()
    files = {str(tmp_path/name): {'bytes': common.FULL_FILES[name]['bytes'], 'mtime_ns': 222}
             for name in ('edge_starts.bin', 'edge_ends.bin')}
    meta = {'graph': {'n': common.N, 'm': common.M,
                      'source_files': [{'path': p, **v} for p, v in files.items()]},
            **common.NATIVE_HASHES, 'sentinel_added': True, 'native_dtype': 'little-endian int64'}
    exit = {'returncode': 0, 'timed_out': False}
    if change == 'wrong_global_hash': meta['col_sha256'] = '0'*64
    elif change == 'wrong_source': meta['graph']['source_files'][0]['mtime_ns'] += 1
    elif change == 'wrong_dimensions': meta['graph']['m'] -= 1
    elif change == 'missing_sentinel': meta['sentinel_added'] = False
    else: exit['returncode'] = False
    common.save(native/'manifest.json', meta)
    common.save(parent/'native_conversion_receipt.json', {'status': 'complete', 'conversion': meta})
    common.save(parent/'native_conversion_exit.json', exit)
    with pytest.raises(RuntimeError): common.conversion_binding(native, parent, {'files': files})


def clean_exit():
    return {'returncode': 0, 'timed_out': False, 'owned_tree_quiescent': True, 'orphan_cleanup': False, 'guard_abort_reason': None}


@pytest.mark.parametrize('change', [None, 'count', 'time', 'flag', 'cohort', 'ordered', 'configuration'])
def test_native_all128_three_arms_and_independent_semantic_gates(tmp_path, monkeypatch, change):
    plan = common.read(common.HERE/'FROZEN_128_SELECTION.json'); parent = tmp_path/'native'; parent.mkdir()
    common.save(parent/'replay_plan.json', plan); converted = {'anchor': 'verified'}
    monkeypatch.setattr(queue, 'conversion_binding', lambda *_: converted)
    arms = []
    for index, mode in enumerate(queue.NATIVE_MODES):
        records = [{'archive_record_index': r['archive_record_index'], 'node_set_sha256': r['node_set_sha256'],
                    'status': 'ok', 'exact_edges': True, 'exact_drnl': True, 'exact_ordered_rows_cols_and_original_eids': True,
                    'root_positions': [0, 1], 'edge_seconds': [.01, .02, .03], 'drnl_seconds': [.01, .02, .03],
                    'raw_operator_seconds': [.01, .02, .03], 'ordered_output_sha256': ['1'*64, '2'*64, '3'*64]}
                   for r in plan['selected']]
        report = {'status': 'complete', 'record_count': 128, 'all_executed_outputs_exact': True,
                  'provenance': {'library_sha256': '4'*64}, 'native_setup': {'original_conversion_manifest': converted},
                  'configuration': {'mode': mode, 'warmup': 1, 'repeats': 3, 'max_query_neighbors': 20000000,
                                    'max_assoc_gib': .5, 'max_peak_gib': 48, 'native_dir': str(tmp_path/'arrays')}, 'records': records}
        if index == 0:
            if change == 'count': report['record_count'] = 127
            elif change == 'time': records[0]['edge_seconds'][1] = -1
            elif change == 'flag': records[0]['exact_drnl'] = False
            elif change == 'cohort': records[0]['archive_record_index'] += 1
            elif change == 'ordered': records[0]['ordered_output_sha256'][0] = '0'*64
            elif change == 'configuration': report['configuration']['repeats'] = 1
        path = parent/(mode+'.json'); common.save(path, report)
        arms.append({'mode': mode, 'eligible': True, 'exit': {'returncode': 0, 'timed_out': False}, 'report_sha256': common.sha(path)})
    common.save(parent/'COMPLETE.json', {'complete': True, 'scientific_terminal': True, 'success': True,
                                        'replay_plan_sha256': common.sha(parent/'replay_plan.json'), 'arms': arms})
    if change is None:
        assert queue.classify_native(parent, clean_exit(), '4'*64, {}, tmp_path/'arrays')['eligible'] is True
    else:
        with pytest.raises(RuntimeError): queue.classify_native(parent, clean_exit(), '4'*64, {}, tmp_path/'arrays')


@pytest.mark.parametrize('change', [None, 'configuration', 'width', 'oracle', 'missing_method', 'time'])
def test_numba_all128_exact_cohort_two_fixed_modes_and_width_gate(tmp_path, change):
    parent = tmp_path/'native'; parent.mkdir(); plan = common.read(common.HERE/'FROZEN_128_SELECTION.json')
    common.save(parent/'replay_plan.json', plan)
    host = {'hostname': 'test', 'boot_id': 'test', 'cpu_models': ['test'], 'cpu_affinity': [0]}
    methods = {name: {'status': 'ok', 'exact_edges': True, 'exact_drnl': True,
                     'edge_seconds': [.01, .02, .03], 'drnl_seconds': [.01, .02, .03], 'edge_plus_drnl_seconds': [.02, .04, .06]}
               for name in common.METHODS}
    rows = [{k: r[k] for k in ('archive_record_index', 'node_set_sha256', 'canonical_edges_sha256', 'drnl_sha256')}
            for r in plan['selected']]
    for row in rows: row.update(exact_archived_oracles=True, methods=copy.deepcopy(methods))
    report = {'status': 'complete', 'record_count': 128, 'all_executed_outputs_exact': True, 'numba_threads': 1,
              'neighbor_width': 64, 'torch_imported': False, 'host': host, 'records': rows,
              'compiled_kernels': {name: ['synthetic nopython signature'] for name in
                                   ('_build_induced_edges_pairwise', '_fill_neighbors_with_owner_numba', '_build_induced_edges_neighbor_scan')},
              'configuration': {'selection_count': 128, 'selection_seed': 20261125, 'warmup': 1, 'repeats': 3,
                                'methods': list(common.METHODS), 'worker_count': 1, 'cpu_threads': 1, 'max_query_neighbors': 20000000},
              'physical_arrays': {'offset_dtype': 'int64', 'neighbor_dtype': 'int64', 'offset_count': common.N,
                                  'neighbor_count': common.M, 'native_rowptr_view_excludes_existing_final_sentinel': True,
                                  'full_graph_copy_or_conversion_performed': False}}
    if change == 'configuration': report['configuration']['warmup'] = 0
    elif change == 'width': report['physical_arrays']['neighbor_dtype'] = 'int32'
    elif change == 'oracle': rows[0]['canonical_edges_sha256'] = '0'*64
    elif change == 'missing_method': rows[0]['methods'].pop('pairwise')
    elif change == 'time': rows[0]['methods']['scan']['edge_seconds'][0] = True
    path = tmp_path/'numba.json'; common.save(path, report)
    if change is None:
        assert queue.classify_numba(path, clean_exit(), 64, parent, host)['eligible'] is True
    else:
        with pytest.raises(RuntimeError): queue.classify_numba(path, clean_exit(), 64, parent, host)


@pytest.mark.parametrize('width', [32, 64])
def test_real_tiny_original_fixed_kernels_match_independent_edges_and_BFS(width):
    import cpu_worker_v1
    np, exact, prepare, replay, functions = cpu_worker_v1.load_runtime()
    from fixtures_v1 import bfs_labels
    rows = [set() for _ in range(9)]
    for u, v in [(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4), (5, 6)]:
        rows[u].add(v); rows[v].add(u)
    ptr = np.asarray([0, *np.cumsum([len(row) for row in rows])], dtype=np.int64)
    col = np.asarray([v for row in rows for v in sorted(row)], dtype=np.int64)
    cases = [([4, 0, 2, 1, 3], (2, 0)), ([6, 0, 5, 1], (1, 0)), ([8, 7], (0, 1))]
    graph = exact.CSRGraph(ptr[:-1], col.astype('int'+str(width)), 'tiny independent fixture')
    for index, (selected, roots) in enumerate(cases[:3]):
        selected = list(selected); ordered = [selected[roots[0]], selected[roots[1]]]
        ordered += [n for i, n in enumerate(selected) if i not in roots]
        local = {int(n): i for i, n in enumerate(ordered)}; edges = []
        for u, i in local.items():
            for v in col[ptr[u]:ptr[u+1]]:
                j = local.get(int(v))
                if j is not None and (i, j) not in ((0, 1), (1, 0)): edges.append((i, j))
        edges = np.asarray(sorted(edges), dtype=np.int64).reshape(-1, 2)
        labels = bfs_labels(edges, len(ordered), (0, 1))
        record = {'nodes': ordered, 'sum_out_degrees': sum(int(ptr[u+1]-ptr[u]) for u in ordered),
                  'node_set_sha256': exact.digest(np.asarray(ordered, dtype=np.int64)),
                  'canonical_edges_sha256': exact.digest(edges), 'drnl_sha256': exact.digest(labels),
                  'induced_directed_entries': len(edges)}
        result = functions.one(record, index, graph, np, exact)
        assert result['exact_archived_oracles'] is True and set(result['methods']) == {'scan', 'pairwise'}
        assert all(m['exact_edges'] and m['exact_drnl'] and queue.finite_times(m['edge_seconds']) for m in result['methods'].values())
