"""Public Friendster projection of the frozen same-host fixed-mode worker.

Reuses an existing native conversion. Only unchanged exact.measure and
exact.production_induced perform scientific measurements; no acquisition,
conversion, selector, sampler, auto timer or new kernel is introduced.
"""
from __future__ import annotations
import argparse
import importlib
import os
from pathlib import Path
import sys
import time

from . import systems_common as c

NATIVE = c.ROOT / 'work/vldb_corrected_native'
ARCHIVE = c.ROOT / 'data/friendster_exact_int32_reference.json'
METHODS = ('scan', 'pairwise')
THREADS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS')
HOST_KEYS = ('hostname', 'boot_id', 'cpu_models', 'cpu_affinity')


def verify_host(expected, actual, reference=False):
    c.require(all(actual.get(key) for key in HOST_KEYS), 'documented Linux host identity required')
    c.require(all(actual[key] == expected.get(key) for key in HOST_KEYS),
              'native and Numba jobs must share hostname/boot/CPUmodel/affinity')
    c.require(not reference or all('8480+' in model for model in actual['cpu_models']),
              'strict reference hardware requires8480+')


def scientific_functions():
    module = c.load_file(c.ROOT / 'metadata/protocols/numba_samehost_scientific_functions.py',
                         'public_samehost_scientific_functions')
    module.require, module.read, module.GIB, module.METHODS = c.require, c.read, c.GIB, METHODS
    return module


def load_runtime():
    c.frozen_sources()
    sys.path.insert(0, str(NATIVE))
    import numpy as np
    from scripts.seal_revision import exact_benchmark as exact
    prepare = importlib.import_module('prepare_replay_v1')
    replay = importlib.import_module('replay_worker_v1')
    expected = ((exact, c.CODE / 'scripts/seal_revision/exact_benchmark.py'),
                (exact.kernels, c.CODE / 'seal/graph_csr_kernels.py'),
                (prepare, NATIVE / 'prepare_replay_v1.py'), (replay, NATIVE / 'replay_worker_v1.py'),
                (sys.modules['native_adapter_v1'], NATIVE / 'native_adapter_v1.py'),
                (sys.modules['fixtures_v1'], NATIVE / 'fixtures_v1.py'))
    for value, path in expected:
        c.require(Path(value.__file__).resolve() == path, 'actual imported runtime origin differs: ' + str(path))
    return np, exact, prepare, replay, {str(path): c.sha(path) for _, path in expected}


def precheck_records(records):
    c.require(len(records) == 128 and all(
        row['sum_out_degrees'] <= 20_000_000
        and row['sum_out_degrees'] * 256 + len(row['nodes']) * 256 <= 6 * c.GIB
        and len(row['nodes']) * (len(row['nodes']) - 1) // 2 <= 20_000_000
        for _, row in records), 'all128 selected-row/workspace/pairwise guards required before graph calls')


def open_existing_conversion(native_dir, plan, original_identity, exact, replay):
    metadata = c.read(native_dir / 'manifest.json')
    c.require(metadata['graph']['source_files'] == [
        {'path': str(path), 'bytes': value['bytes'], 'mtime_ns': value['mtime_ns']}
        for path, value in original_identity.items()], 'native conversion not bound to current original GraphCSR')
    ptr, col, setup = replay.open_native(native_dir, plan)
    return exact.CSRGraph(ptr[:-1], col, str(native_dir)), metadata, setup


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--friendster-root', type=Path, required=True)
    parser.add_argument('--native-dir', type=Path, required=True)
    parser.add_argument('--replay-dir', type=Path, required=True)
    parser.add_argument('--expected-host', type=Path, required=True)
    parser.add_argument('--width', type=int, choices=(32, 64), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference-hardware', action='store_true')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    c.frozen_sources()
    if not args.execute:
        print('{"execute":false,"graph":"friendster","count":128,"conversion_performed":false}')
        return 0
    c.require(sys.platform.startswith('linux'), 'execute requires Linux')
    for name in ('friendster_root', 'native_dir', 'replay_dir', 'expected_host', 'output'):
        setattr(args, name, getattr(args, name).resolve())
    c.require(not args.output.exists() and not args.output.with_name(args.output.name + '.tmp').exists(), 'fresh worker output required')
    c.require(not args.output.is_relative_to(c.ROOT), 'output must be outside artifact sources')
    for folder in (args.friendster_root, args.native_dir, args.replay_dir):
        c.require(not args.output.is_relative_to(folder), 'worker output must be disjoint from input trees')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {'schema_version': 1, 'kind': 'public_numba_samehost_worker_v1', 'status': 'initializing',
        'graph': 'friendster', 'neighbor_width': args.width, 'records': [],
        'configuration': {'selection_count': 128, 'selection_seed': 20261125, 'warmup': 1, 'repeats': 3,
            'methods': list(METHODS), 'worker_count': 1, 'cpu_threads': 1, 'max_query_neighbors': 20000000},
        'timing_contract': 'Unchanged exact.measure edge timer includes exact.production_induced invocation, compiled fixed-mode extraction, target removal/local CSR; DRNL separate. Validation/setup/JIT warmup excluded. No auto/dispatcher timer.',
        'scope': 'Same documented host as preceding native128 replay; mixed64offset/32neighbor versus existing64offset/64neighbor. Different hardware does not reproduce published timing values.',
        'all_executed_outputs_exact': False}
    c.save(args.output, report)
    try:
        host, expected_host = c.host_identity(), c.read(args.expected_host)
        verify_host(expected_host, host, args.reference_hardware)
        c.require(all(os.environ.get(key) == '1' for key in THREADS)
                  and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'one-thread CPU-only environment required')
        from . import reproduce_native_friendster as native
        native.validate_result(args.replay_dir.parent, c.read(args.replay_dir.parent / 'native_friendster128.exit.json'))
        report['host'] = host
        np, exact, prepare, replay, bindings = load_runtime()
        scientific = scientific_functions()
        plan = prepare.prepare(ARCHIVE, 128, 20261125)
        records = prepare.load_selected(plan, ARCHIVE)
        precheck_records(records)
        scientific.verify_same_selection(plan, args.replay_dir / 'replay_plan.json')
        report.update(replay_plan=plan, replay_plan_sha256=c.sha(args.replay_dir / 'replay_plan.json'),
                      archive_sha256=c.sha(ARCHIVE), source_bindings=bindings)
        original = exact.load_graph(args.friendster_root, 'little')
        identity = scientific.original_identity(args.friendster_root, original)
        c.require((original.n, original.m) == (plan['graph']['nodes'], plan['graph']['stored_entries']), 'original graph dimensions changed')
        report['original_input_stat_header_proof'] = identity
        setup_start = time.perf_counter()
        metadata, setup = None, None
        if args.width == 64:
            graph, metadata, setup = open_existing_conversion(args.native_dir, plan, identity, exact, replay)
            report['native_setup'] = setup
        else:
            graph = original
        report['setup_seconds_excluded'] = time.perf_counter() - setup_start
        report['physical_arrays'] = {'offset_dtype': str(graph.starts.dtype), 'neighbor_dtype': str(graph.indices.dtype),
            'offset_count': graph.n, 'neighbor_count': graph.m, 'full_graph_copy_or_conversion_performed': False,
            'native_rowptr_view_excludes_existing_final_sentinel': args.width == 64}
        c.require(graph.starts.dtype.itemsize == 8 and graph.indices.dtype.itemsize == args.width // 8,
                  'actual physical array widths differ')
        for mode in METHODS:
            exact.production_induced(graph, np.asarray(records[0][1]['nodes'], dtype=np.int64), mode)
        import numba
        import scipy
        report['numba_threads'] = numba.get_num_threads()
        c.require(report['numba_threads'] == 1, 'actual Numba thread count changed')
        report['numerical_environment'] = {name: {'version': value.__version__, 'origin': value.__file__}
            for name, value in (('numpy', np), ('scipy', scipy), ('numba', numba))}
        report['status'] = 'running'
        c.save(args.output, report)
        for index, record in records:
            report['records'].append(scientific.one(record, index, graph, np, exact))
            c.save(args.output, report)
        c.require(scientific.original_identity(args.friendster_root, original) == identity, 'original graph changed during worker')
        c.frozen_sources()
        c.require(all(c.sha(path) == value for path, value in bindings.items()), 'runtime source changed')
        c.require(c.sha(ARCHIVE) == report['archive_sha256'], 'frozen archive changed')
        scientific.verify_same_selection(plan, args.replay_dir / 'replay_plan.json')
        verify_host(expected_host, c.host_identity(), args.reference_hardware)
        c.require(c.read(args.expected_host) == expected_host, 'expected host receipt changed')
        if setup:
            c.require(c.read(args.native_dir / 'manifest.json') == metadata, 'conversion manifest changed')
            for item in setup['runtime_files']:
                p = Path(item['path']); stat = p.stat()
                c.require(not p.is_symlink() and (stat.st_size, stat.st_mtime_ns) == (item['bytes'], item['mtime_ns']),
                          'native payload changed during worker')
        report.update(status='complete', record_count=len(records), all_executed_outputs_exact=True,
                      memory_final=replay.memory())
        c.save(args.output, report)
        return 0
    except Exception as error:
        report.update(status='failed', all_executed_outputs_exact=False,
                      error={'type': type(error).__name__, 'message': str(error)[:1000]})
        c.save(args.output, report)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
