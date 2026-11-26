#!/usr/bin/env python3
"""Public fullCSR128S Numba32/64 adapter; original scientific functions retained."""
from cpu_common_v1 import *
import argparse, time


def load_runtime():
    sys.path.insert(0, str(HERE/'original/native'))
    sys.path.insert(0, str(HERE/'scientific_repo'))
    import numpy as np
    from scripts.seal_revision import exact_benchmark as exact
    import prepare_replay_v1 as prepare
    import replay_worker_v1 as replay
    import scientific_functions_v1 as functions
    require(Path(exact.__file__).resolve() == HERE/'scientific_repo/scripts/seal_revision/exact_benchmark.py' and
            Path(exact.kernels.__file__).resolve() == HERE/'scientific_repo/seal/graph_csr_kernels.py', 'actual exact/kernel origins changed')
    for obj, name in ((prepare, 'prepare_replay_v1.py'), (replay, 'replay_worker_v1.py'),
                      (sys.modules['native_adapter_v1'], 'native_adapter_v1.py'),
                      (sys.modules['fixtures_v1'], 'fixtures_v1.py')):
        require(Path(obj.__file__).resolve() == HERE/'original/native'/name, 'actual original replay import origin changed')
    return np, exact, prepare, replay, functions


def execute(base, width, output):
    L = layout(base); bindings = verify_sources(); output = Path(output)
    require(not output.exists() and width in (32, 64) and
            output.resolve() == L['out']/('citation2_neighbors'+str(width)+'.json'), 'exact fresh32/64 worker output required')
    admission_path = L['out']/'admission.json'; admission = read(admission_path)
    require(admission['kind'] == 'citation2_cpu_fresh_admission_v1' and admission['complete'] is True, 'genuine CPU admission required')
    original = admission['full_original']; expected_host = read(L['out']/'host.json')
    verify_host(expected_host, host_receipt()); stable_original(original)
    require(all(os.environ.get(k) == '1' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS')) and
            os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'actual one-thread CPU-only environment required')
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {'schema_version': 1, 'kind': 'public_strict_clean_citation2_matched_numba_v1', 'status': 'initializing',
              'neighbor_width': width, 'host': expected_host, 'configuration': {'selection_count': 128, 'selection_seed': 20261125,
                  'warmup': 1, 'repeats': 3, 'methods': list(METHODS), 'worker_count': 1, 'cpu_threads': 1, 'max_query_neighbors': 20000000},
              'admission_sha256': sha(admission_path), 'source_bindings': bindings, 'records': [],
              'timing_contract': 'Original exact.measure/production_induced edge timer, target removal/local CSR; DRNL separate. Setup, validation, JIT and warmup excluded.',
              'scope': 'New same-host public strictcleanCitation2 physical comparison on128existing orderedS, not a trained quality or two-graph recovery result.'}
    save(output, report)
    try:
        np, exact, prepare, replay, functions = load_runtime()
        archive = L['cohort_archive']
        require(info(archive) == admission['cohort']['archive'], 'admitted pre-timing cohort archive changed')
        plan = prepare.prepare(archive, 128, 20261125); frozen_plan_match(plan); records = prepare.load_selected(plan, archive)
        require((plan['graph']['nodes'], plan['graph']['stored_entries']) == (N, M) and len(records) == 128, 'strictcleanCitation2 frozen cohort changed')
        previous = L['out']/'native'
        functions.verify_same_selection(plan, previous/'replay_plan.json')
        require(all(r['sum_out_degrees'] <= 20000000 and r['sum_out_degrees']*256+len(r['nodes'])*256 <= 6*GIB and
                    len(r['nodes'])*(len(r['nodes'])-1)//2 <= 20000000 for _, r in records), 'original selected-work guards exceeded')
        original_graph = exact.load_graph(L['full'], 'little')
        identity = functions.original_identity(L['full'], original_graph)
        require((original_graph.n, original_graph.m) == (N, M), 'preparedtrain/other graph cannot substitute')
        meta = conversion_binding(L['native'], previous, original)
        started = time.perf_counter()
        if width == 64:
            ptr, col, native_setup = replay.open_native(L['native'], plan)
            graph = exact.CSRGraph(ptr[:-1], col, str(L['native']))
            report['native_setup'] = native_setup
        else:
            graph = original_graph
        report['setup_seconds_excluded'] = time.perf_counter()-started
        report['replay_plan'] = plan
        report['original_full_payload_admission'] = original
        report['original_input_stat_header_proof'] = identity
        report['physical_arrays'] = {'offset_dtype': str(graph.starts.dtype), 'neighbor_dtype': str(graph.indices.dtype),
                                     'offset_count': graph.n, 'neighbor_count': graph.m, 'full_graph_copy_or_conversion_performed': False,
                                     'native_rowptr_view_excludes_existing_final_sentinel': width == 64}
        require(graph.starts.dtype.itemsize == 8 and graph.indices.dtype.itemsize == width//8, 'physical index widths changed')
        for mode in METHODS:
            exact.production_induced(graph, np.asarray(records[0][1]['nodes'], dtype=np.int64), mode)
        import numba, scipy
        require(not numba.config.DISABLE_JIT, 'actual Numba JIT must be enabled')
        kernel_names = ('_build_induced_edges_pairwise', '_fill_neighbors_with_owner_numba', '_build_induced_edges_neighbor_scan')
        report['compiled_kernels'] = {name: [str(s) for s in getattr(getattr(exact.kernels, name), 'nopython_signatures', ())]
                                      for name in kernel_names}
        require(all(report['compiled_kernels'].values()), 'actual fixed kernels require compiled nopython signatures')
        report['numerical_environment'] = {name: {'version': obj.__version__, 'origin': obj.__file__}
                                           for name, obj in (('numpy', np), ('scipy', scipy), ('numba', numba))}
        require([report['numerical_environment'][k]['version'] for k in ('numpy', 'scipy', 'numba')] == ['1.24.4', '1.10.1', '0.58.0'], 'pinned CPU numerical versions changed')
        report['numba_threads'] = numba.get_num_threads(); require(report['numba_threads'] == 1, 'actual Numba threads changed')
        report['status'] = 'running'; save(output, report)
        for index, record in records:
            report['records'].append(functions.one(record, index, graph, np, exact))
            save(output, report)
        require(functions.original_identity(L['full'], original_graph) == identity, 'original graph header/stat changed')
        stable_original(original); verify_host(expected_host, host_receipt()); assert_bindings(bindings)
        require(conversion_binding(L['native'], previous, original) == meta, 'conversion receipt changed')
        if width == 64:
            for item in native_setup['runtime_files']:
                value = Path(item['path']).stat()
                require((value.st_size, value.st_mtime_ns) == (item['bytes'], item['mtime_ns']), 'native array stat changed')
        report.update(status='complete', record_count=128, all_executed_outputs_exact=True,
                      torch_imported='torch' in sys.modules, memory_final=replay.memory())
        require(report['torch_imported'] is False, 'Numba-only worker unexpectedly imported Torch')
        save(output, report); return 0
    except BaseException as error:
        report.update(status='failed', all_executed_outputs_exact=False,
                      error={'type': type(error).__name__, 'message': str(error)[:1200]})
        save(output, report); return 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, default=BASE); p.add_argument('--width', type=int, choices=(32, 64), required=True)
    p.add_argument('--output', type=Path, required=True); a = p.parse_args(argv)
    require(a.base.resolve() == BASE and sys.platform.startswith('linux'), 'canonical reviewed CPU base required')
    return execute(a.base, a.width, a.output)


if __name__ == '__main__': raise SystemExit(main())
