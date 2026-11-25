"""Protocol equivalence and false-success/resource regressions; no real GPU jobs."""
import argparse
import ast
import copy
from itertools import combinations
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from . import systems_common as c
from . import reproduce_pipeline_v4 as v4
from . import reproduce_native_friendster as native
from . import reproduce_pencil as pencil
from . import pencil_validation as pv


def args(tmp_path):
    return argparse.Namespace(out=tmp_path / 'new', execute=False,
        citation2_root=tmp_path / 'c2', friendster_root=tmp_path / 'fr',
        validation_root=None, native_dir=None, coordination_dir=None,
        memory_budget_gib=180., max_output_gib=32., max_hours=12.)


def normalized(argv):
    return ['<path>' if str(x).startswith('/') else x for x in argv]


def test_v4_is_exact_public_projection_of_frozen_builder(tmp_path):
    a = args(tmp_path)
    plan = v4.make_plan(a)
    original = c.load_file(v4.SUITE, 'tested_frozen_suite').build_plan(a.out, c.CODE, sys.executable)
    expected = [j for j in original['jobs'] if j['dataset'] in (None, 'citation2', 'friendster')]
    assert len(plan['jobs']) == len(expected) == 24
    assert plan['original_protocol_sha256'] == original['protocol_sha256']
    for actual, frozen in zip(plan['jobs'], expected):
        assert {k: actual[k] for k in ('name', 'kind', 'dataset', 'gpu')} == {k: frozen[k] for k in ('name', 'kind', 'dataset', 'gpu')}
        assert normalized(actual['argv']) == normalized(frozen['argv'])
    assert all(j['dataset'] != 'vk' for j in plan['jobs'])
    assert sum(j['kind'] == 'pipeline' for j in plan['jobs']) == 20
    assert not a.out.exists()


def test_pencil_grid_matches_frozen_protocol_json(tmp_path):
    plan = pencil.make_plan(args(tmp_path))
    spec = c.read(c.ROOT / 'metadata/protocols/pencil_queue_v2_original.json')
    jobs = plan['jobs']
    assert len(jobs) == spec['scientific_jobs'] == 48
    assert sum(j['kind'] == 'benchmark' for j in jobs) == 36
    primary = [j for j in jobs if j.get('mode') == 'online']
    assert {(j['arm'], j['batch'], j['workers']) for j in primary} == {
        (arm, batch, workers) for arm in pv.ARMS for batch in (128, 512) for workers in (0, 1)}
    audits = [j for j in jobs if j['kind'] == 'audit' and j['gpu']]
    assert {(j['precision'], j['batch'], j['workers']) for j in audits} == {
        (precision, batch, workers) for precision in ('fp32', 'bf16') for batch in (128, 512) for workers in (0, 1)}
    assert len(plan['setup_jobs']) == 2
    for job in jobs:
        assert '--quality-plan' in job['argv'] or job['name'].startswith('tiny_')
        assert not any('vk_' in token for token in job['argv'])


def test_every_pencil_argument_matches_unchanged_frozen_job_builder(tmp_path):
    a = args(tmp_path)
    plan = pencil.make_plan(a)
    path = c.ROOT / 'metadata/protocols/pencil_original_scientific_builder.py'
    namespace = {'Path': Path, 'BRIDGE': pencil.BRIDGE, 'ARMS': pv.ARMS, 'REPLAY_ARMS': pv.REPLAY_ARMS,
                 'layout': lambda base: {'out': a.out,
                     'cpp_module': pencil.BRIDGE / 'corrected_native/native_adapter_v1.py',
                     'cpp_library': a.out / 'cpp_build/native_assoc_v1.so',
                     'cpp_receipt': a.out / 'cpp_build/build_receipt_v1.json'}}
    exec(compile(path.read_text(), str(path), 'exec'), namespace)
    original = namespace['make_jobs'](tmp_path / 'original_base', sys.executable)
    assert len(original) == len(plan['jobs']) == 48
    for actual, frozen in zip(plan['jobs'], original):
        assert set(actual) == set(frozen)
        assert {k: v for k, v in actual.items() if k not in ('argv', 'output')} == {
            k: v for k, v in frozen.items() if k not in ('argv', 'output')}
        assert normalized(actual['argv']) == normalized(frozen['argv'])


def test_native_exact_public_archive_modes_and_bounds(tmp_path):
    plan = native.make_plan(args(tmp_path))
    command = plan['jobs'][0]['argv']
    assert command[command.index('--count') + 1] == '128'
    assert command[command.index('--repeats') + 1] == '3'
    assert command[command.index('--warmup') + 1] == '1'
    assert '--include-upstream' in command and '--prepare-native' in command
    assert command[command.index('--archive') + 1].endswith('/data/friendster_exact_int32_reference.json')
    assert not any('/vk' in token or 'vk_' in token for token in command)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), 0., -1., 13.])
def test_deadlines_fail_before_any_output(tmp_path, value):
    a = args(tmp_path); a.max_hours = value
    with pytest.raises(RuntimeError, match='finite'):
        c.validate_arguments(a)
    assert not a.out.exists()


def test_output_cannot_overlap_input_or_source(tmp_path):
    a = args(tmp_path); a.out = a.citation2_root / 'nested'
    with pytest.raises(RuntimeError, match='disjoint'):
        c.validate_arguments(a, (a.citation2_root,))
    a.out = c.CODE / 'bad_output'
    with pytest.raises(RuntimeError, match='outside'):
        c.validate_arguments(a)


@pytest.mark.parametrize('field,value', [('returncode', 1), ('timed_out', True), ('owned_tree_quiescent', False),
                                       ('orphan_cleanup', True), ('guard_abort_reason', 'memory')])
def test_failed_owned_phase_never_clean(field, value):
    result = {'returncode': 0, 'timed_out': False, 'owned_tree_quiescent': True,
              'orphan_cleanup': False, 'guard_abort_reason': None}
    assert c.clean_exit(result)
    result[field] = value
    assert not c.clean_exit(result)


def test_v4_twelve_prefix_comparisons_require_exact_initial_state(tmp_path):
    reports = {}
    for dataset in ('citation2', 'friendster'):
        for workers in (1, 8):
            for arm in v4.REFERENCES:
                reports[f'{dataset}_{arm}_workers{workers}'] = {'runs': [{
                    'audits': [{'index': i, 'nodes_sha256': str(i)} for i in range(32)],
                    'model': {'initial_state_sha256': 'same'}, 'config': {'manifest_sha256': dataset}}]}
    assert len(v4.comparisons(reports)) == 12
    reports['friendster_reference_union_compiled_workers8']['runs'][0]['model']['initial_state_sha256'] = 'different'
    with pytest.raises(RuntimeError, match='equivalence'):
        v4.comparisons(reports)


def test_native_report_success_does_not_hide_failed_arm(tmp_path):
    folder = tmp_path / 'replay'; folder.mkdir()
    c.save(folder / 'COMPLETE.json', {'complete': True, 'scientific_terminal': True, 'success': True,
                                    'arms': [{'mode': 'persistent_touched_reset', 'eligible': True}]})
    good = {'returncode': 0, 'timed_out': False, 'owned_tree_quiescent': True, 'orphan_cleanup': False, 'guard_abort_reason': None}
    with pytest.raises(RuntimeError, match='all3'):
        native.validate_result(tmp_path, good)


def test_pencil_crossjob_scores_rejects_one_mismatched_sweep(tmp_path):
    jobs = []
    fingerprint = {'kind': 'tensor', 'shape': [1024], 'sha256': 'a' * 64}
    for mode, batch, workers, arms in [('online', b, w, pv.ARMS) for b in (128, 512) for w in (0, 1)] + [('replay', 128, 0, pv.REPLAY_ARMS)]:
        for arm in arms:
            path = tmp_path / f'{mode}_{batch}_{workers}_{arm}.json'
            c.save(path, {'result': {'model': {'checkpoint': None, 'initialization_seed': 42, 'tiny_fixture_model': False},
                                    'sweeps': [{'scores': fingerprint} for _ in range(4)]}})
            jobs.append({'kind': 'benchmark', 'precision': 'bf16', 'batch': batch, 'workers': workers,
                         'mode': mode, 'arm': arm, 'output': str(path)})
    assert len(pv.crossjob_scores(jobs)['groups']) == 5
    path = Path(jobs[-1]['output']); bad = c.read(path)
    bad['result']['sweeps'][-1]['scores']['sha256'] = 'b' * 64
    c.save(path, bad)
    with pytest.raises(RuntimeError, match='fingerprints'):
        pv.crossjob_scores(jobs)


def test_validation_subset_rejects_mutated_public_candidate(tmp_path):
    (tmp_path / 'valid_edge.npy').write_bytes(b'not the pinned validation array')
    with pytest.raises(RuntimeError, match='exact original'):
        pencil.verify_validation(tmp_path)


def test_pencil_setup_selects_only_validation_without_test_payload_access(tmp_path, monkeypatch):
    from . import prepare_pencil_validation as prep
    observed = []
    class FrozenHelper:
        def freeze_source_subset(self, **kw):
            observed.append(kw)
            return {'literal_fixture': True}
    monkeypatch.setattr(c, 'load_file', lambda *values: FrozenHelper())
    assert prep.prepare(tmp_path / 'input', tmp_path / 'out') == {'literal_fixture': True}
    assert observed[0]['pos_path'].name == 'valid_edge.npy'
    assert observed[0]['neg_path'].name == 'valid_edge_neg.npy'
    assert observed[0]['name'] == 'valid' and observed[0]['count'] == 512 and observed[0]['seed'] == 20261124
    plan = pencil.make_plan(args(tmp_path))
    assert '-m' in plan['setup_jobs'][0]['argv']
    assert 'artifact_public.prepare_pencil_validation' in plan['setup_jobs'][0]['argv']


def test_pencil_input_hashing_never_opens_test_arrays(tmp_path, monkeypatch):
    from . import verify_payloads
    observed = []
    monkeypatch.setattr(verify_payloads, 'verify', lambda root, definition: observed.append(definition) or {'passed': True})
    assert pencil.verify_public_inputs(tmp_path) == {'passed': True}
    assert len(observed[0]['files']) == 5
    assert all(not item['path'].startswith('test') for item in observed[0]['files'])


def test_source_pin_guard_detects_local_source_change(tmp_path, monkeypatch):
    root = tmp_path / 'artifact'; (root / 'metadata').mkdir(parents=True)
    source = root / 'frozen.py'; source.write_text('original')
    c.save(root / 'metadata/SYSTEMS_FROZEN_INPUTS.json', {'files': {'frozen.py': c.sha(source)}})
    c.save(root / 'metadata/SYSTEMS_SOURCE_SELECTION.json', {'sources': {}})
    monkeypatch.setattr(c, 'ROOT', root)
    assert c.frozen_sources()
    source.write_text('changed')
    with pytest.raises(RuntimeError, match='differs'):
        c.frozen_sources()


def test_public_adapter_snapshot_cannot_change_behind_frozen_kernel_pins(tmp_path, monkeypatch):
    root = tmp_path / 'artifact'; (root / 'metadata').mkdir(parents=True)
    source = root / 'adapter.py'; source.write_text('original')
    c.save(root / 'metadata/SYSTEMS_FROZEN_INPUTS.json', {'files': {}})
    c.save(root / 'metadata/SYSTEMS_SOURCE_SELECTION.json', {'sources': {'adapter.py': {'packaged_sha256': c.sha(source)}}})
    monkeypatch.setattr(c, 'ROOT', root)
    c.frozen_sources()
    source.write_text('changed')
    with pytest.raises(RuntimeError, match='public systems source differs'):
        c.frozen_sources()


def test_external_guard_reason_survives_clean_child_exit(tmp_path, monkeypatch):
    a = args(tmp_path); a.out.mkdir()
    execution = c.Execution(a, {'scope': 'test'})
    execution.deadline = c.time.monotonic() + 100
    execution.check_disk = lambda: None
    execution.memory = type('Memory', (), {
        'sample_tree': staticmethod(lambda owner: {}),
        'cgroup_memory': staticmethod(lambda: {}),
        'budget_decision': staticmethod(lambda *values: {'abort_reason': 'literal test budget exceeded'}),
        'GUARD_POLICY': {'version': 'test'}})()
    monkeypatch.setattr(c, 'frozen_sources', lambda: {})
    monkeypatch.setattr(c.supervision, 'owned_groups', lambda owner: {})
    def child(*args, **kw):
        kw['groups_reader'](123)
        return {'returncode': 0, 'timed_out': False, 'owned_tree_quiescent': True, 'orphan_cleanup': False}
    monkeypatch.setattr(c.supervision, 'run_owned', child)
    result = execution.run({'name': 'test_guard', 'argv': ['never_launched']})
    assert result['returncode'] == 0 and result['guard_abort_reason'] == 'literal test budget exceeded'
    assert not c.clean_exit(result)


def test_e16_complete_public_raw_aggregation():
    from .verify_dispatch_e16 import verify
    result = verify()
    assert result['records'] == 768
    assert result['python_auto_over_vectorized_auto_sum_ratio'] == pytest.approx(1.1728193694365459)


@pytest.mark.parametrize('module,inputs,count', [
    ('reproduce_pipeline_v4', ['--citation2-root', '/public/c2', '--friendster-root', '/public/fr'], 24),
    ('reproduce_native_friendster', ['--friendster-root', '/public/fr'], 3),
    ('reproduce_pencil', ['--citation2-root', '/public/c2'], 48)])
def test_standalone_plan_cli_needs_no_graph_gpu_network_or_writable_output(tmp_path, module, inputs, count):
    output = tmp_path / module
    result = subprocess.run([sys.executable, '-m', 'artifact_public.' + module, *inputs, '--out', str(output)],
        cwd=c.CODE, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan['execute'] is False and plan['input_identity_checked'] is False
    assert len(plan['jobs']) == count and not output.exists()


def test_failed_context_writes_explicit_false_science_and_keeps_outputs(tmp_path):
    a = args(tmp_path); a.out.mkdir()
    marker = a.out / 'kept_partial.json'; marker.write_text('{"partial":true}')
    execution = c.Execution(a, {'scope': 'literal cleanup test'})
    execution.cleanup = lambda: True
    execution.lock = (tmp_path / 'literal.lock').open('w')
    assert execution.__exit__(RuntimeError, RuntimeError('literal failure'), None) is False
    terminal = c.read(a.out / 'COMPLETE.json')
    assert terminal['success'] is False and terminal['failure']['type'] == 'RuntimeError'
    assert marker.exists()


def test_incomplete_owned_cleanup_never_marks_controller_finished(tmp_path):
    a = args(tmp_path); a.out.mkdir()
    execution = c.Execution(a, {'scope': 'literal unreaped-child test'})
    execution.cleanup = lambda: False
    execution.lock = (tmp_path / 'literal.lock').open('w')
    with pytest.raises(RuntimeError, match='cleanup'):
        execution.__exit__(None, None, None)
    terminal = c.read(a.out / 'COMPLETE.json')
    assert terminal['success'] is False and terminal['controller_finished'] is False
    assert terminal['owned_tree_quiescent'] is False


def test_sigterm_becomes_owned_cleanup_and_restores_previous_handler(tmp_path):
    import signal
    a = args(tmp_path); a.out.mkdir()
    execution = c.Execution(a, {'scope': 'literal SIGTERM test'})
    execution.lock = (tmp_path / 'literal.lock').open('w')
    calls = []
    execution.cleanup = lambda: calls.append('owned cleanup') or True
    previous = signal.getsignal(signal.SIGTERM)
    execution.install_signals()
    try:
        with pytest.raises(KeyboardInterrupt, match='SIGTERM'):
            try:
                signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            except BaseException as exc:
                execution.__exit__(type(exc), exc, None)
                raise
        assert calls == ['owned cleanup'] and signal.getsignal(signal.SIGTERM) == previous
        assert c.read(a.out / 'COMPLETE.json')['success'] is False
    finally:
        execution.restore_signals()


def test_failed_enter_restores_sigterm_handler(tmp_path, monkeypatch):
    import signal
    previous = signal.getsignal(signal.SIGTERM)
    execution = c.Execution(args(tmp_path), {'scope': 'literal failed admission'})
    monkeypatch.setattr(c.sys, 'platform', 'linux')
    monkeypatch.setattr(c, 'frozen_sources', lambda: {})
    monkeypatch.setattr(c.shutil, 'disk_usage', lambda path: type('Disk', (), {'free': 16 * c.GIB})())
    monkeypatch.setattr(c.supervision, 'enable_subreaper', lambda: (_ for _ in ()).throw(RuntimeError('literal failure')))
    with pytest.raises(RuntimeError, match='literal failure'):
        execution.__enter__()
    assert signal.getsignal(signal.SIGTERM) == previous and execution.lock.closed


@pytest.mark.parametrize('mutation', ['raw_timer', 'ordered_eids', 'roots', 'hashes', 'nested_timer'])
def test_native_raw_and_ordered_output_contract_rejects_missing_or_invalid_evidence(mutation):
    row = {'status': 'ok', 'exact_edges': True, 'exact_drnl': True, 'edge_seconds': [2.] * 3,
           'drnl_seconds': [1.] * 3, 'raw_operator_seconds': [1.] * 3,
           'exact_ordered_rows_cols_and_original_eids': True, 'root_positions': [0, 1],
           'ordered_output_sha256': ['a' * 64] * 3}
    native.validate_times(row, raw=True)
    if mutation == 'raw_timer': row['raw_operator_seconds'] = [float('nan')] * 3
    elif mutation == 'ordered_eids': row['exact_ordered_rows_cols_and_original_eids'] = False
    elif mutation == 'roots': row['root_positions'] = [1, 0]
    elif mutation == 'hashes': row['ordered_output_sha256'].pop()
    else: row['raw_operator_seconds'] = [3.] * 3
    with pytest.raises(RuntimeError): native.validate_times(row, raw=True)


def test_samehost_matrix_has_only_two_public_followups_without_new_conversion(tmp_path):
    a = args(tmp_path)
    plan = native.make_plan(a)
    assert len(plan['jobs']) == 3 and plan['scientific_native_arms'] == 3
    assert plan['scientific_numba_width_workers'] == 2
    for width, job in zip((32, 64), plan['jobs'][1:]):
        assert job['name'] == f'numba_friendster{width}' and job['timeout_seconds'] == 3600
        assert job['argv'][job['argv'].index('--width') + 1] == str(width)
        assert not any('prepare-native' in token or '/vk' in token for token in job['argv'])
    a.native_dir = tmp_path / 'existing_conversion'
    plan = native.make_plan(a)
    assert '--prepare-native' not in plan['jobs'][0]['argv']
    assert all(str(a.native_dir) in job['argv'] for job in plan['jobs'])


@pytest.mark.parametrize('key', ['hostname', 'boot_id', 'cpu_models', 'cpu_affinity'])
def test_samehost_rejects_every_host_identity_change(key):
    from .samehost_friendster_worker import verify_host
    host = {'hostname': 'literal', 'boot_id': 'boot', 'cpu_models': ['test CPU'], 'cpu_affinity': [0]}
    verify_host(host, host)
    changed = {**host, key: ['changed'] if isinstance(host[key], list) else 'changed'}
    with pytest.raises(RuntimeError, match='share'):
        verify_host(host, changed)
    with pytest.raises(RuntimeError, match='8480'):
        verify_host(host, host, reference=True)


def test_samehost_selection_and_pregraph_guards(tmp_path):
    from . import samehost_friendster_worker as worker
    science = worker.scientific_functions()
    plan = {'archive': {'original_local_path': 'old', 'sha256': 'literal'}, 'selected': [{'index': 2}]}
    original = tmp_path / 'plan.json'; c.save(original, plan)
    science.verify_same_selection({**plan, 'archive': {**plan['archive'], 'original_local_path': 'new'}}, original)
    with pytest.raises(RuntimeError, match='selection differs'):
        science.verify_same_selection({**plan, 'selected': [{'index': 3}]}, original)
    rows = [(i, {'sum_out_degrees': 0, 'nodes': [0, 1]}) for i in range(128)]
    worker.precheck_records(rows)
    rows[2][1]['sum_out_degrees'] = 20_000_001
    with pytest.raises(RuntimeError, match='before graph'):
        worker.precheck_records(rows)


def test_samehost_missing_or_rebound_conversion_never_calls_graph_open(tmp_path):
    from . import samehost_friendster_worker as worker
    called = []
    replay = type('Replay', (), {'open_native': lambda *values: called.append(values)})
    with pytest.raises(RuntimeError, match='bounded'):
        worker.open_existing_conversion(tmp_path, {}, {}, None, replay)
    c.save(tmp_path / 'manifest.json', {'graph': {'source_files': []}})
    with pytest.raises(RuntimeError, match='not bound'):
        worker.open_existing_conversion(tmp_path, {}, {'different': {'bytes': 1, 'mtime_ns': 2}}, None, replay)
    assert called == []


def test_skipped_samehost_report_is_never_scientific_success(tmp_path):
    c.save(tmp_path / 'numba_friendster64.json', {'status': 'skipped_native_conversion_unavailable'})
    good = {'returncode': 0, 'timed_out': False, 'owned_tree_quiescent': True,
            'orphan_cleanup': False, 'guard_abort_reason': None}
    with pytest.raises(RuntimeError, match='skips'):
        native.validate_followup(tmp_path, 64, good)


@pytest.mark.parametrize('width', [32, 64])
def test_samehost_unchanged_science_executes_tiny_both_physical_widths(width):
    from . import samehost_friendster_worker as worker
    np, exact, _, _, _ = worker.load_runtime()
    source = exact.fixture_graph()
    graph = exact.CSRGraph(source.starts.astype(np.int64), source.indices.astype(f'int{width}'), 'tiny_public_test')
    nodes = np.arange(graph.n, dtype=np.int64)
    oracle = exact.oracle_induced(graph, nodes)
    record = {'nodes': nodes.tolist(), 'sum_out_degrees': graph.degree_sum(nodes),
        'node_set_sha256': exact.digest(nodes), 'canonical_edges_sha256': exact.digest(exact.canonical_edges(oracle)),
        'drnl_sha256': exact.digest(exact.drnl_labels(oracle)), 'induced_directed_entries': oracle.nnz}
    actual = worker.scientific_functions().one(record, 7, graph, np, exact)
    assert actual['exact_archived_oracles'] is True and set(actual['methods']) == {'scan', 'pairwise'}
    for result in actual['methods'].values():
        native.validate_times(result)
    assert graph.starts.dtype.itemsize == 8 and graph.indices.dtype.itemsize == width // 8
