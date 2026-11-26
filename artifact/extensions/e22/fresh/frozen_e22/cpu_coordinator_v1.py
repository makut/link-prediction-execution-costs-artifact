#!/usr/bin/env python3
"""New public strictcleanCitation2-only matched CPU coordinator; default plan only."""
from cpu_common_v1 import *
import argparse, shutil, signal, tarfile, time
import cpu_environment_v1 as environment

LOCKS = ('GPU_SUITE_RUNNING.lock', 'PIPELINE_SUITE_RUNNING.lock', 'ADDITIONAL_TIMING_RUNNING.lock')
NATIVE_MODES = ('reference_full_clear', 'persistent_touched_reset', 'upstream_torch_sparse_direct')


def scan_active(base, proc=Path('/proc')):
    names = {'cpu_coordinator_v1.py', 'cpu_worker_v1.py', 'run_public_fresh_pair_v1.py', 'run_public.py',
             'friendster_public_execution_v1.py', 'cpu_recovery_controller_v1.py', 'queue_v2.py', 'worker_v2.py',
             'run_parent_replay_v1.py', 'replay_worker_v1.py', 'prepare_native_only_v1.py',
             'external_operator_benchmark.py', 'runner.py', 'exact_benchmark.py', 'run_pipeline_suite_v4.py'}
    found = []
    for p in proc.iterdir():
        if not p.name.isdigit() or int(p.name) == os.getpid(): continue
        try:
            if (p/'stat').read_text().rsplit(')', 1)[1].split()[0] == 'Z': continue
            argv = [v.decode(errors='replace') for v in (p/'cmdline').read_bytes().split(b'\0') if v]
            if str(base) in ' '.join(argv) and any(Path(x).name in names for x in argv):
                found.append({'pid': int(p.name), 'entrypoints': [Path(x).name for x in argv if Path(x).name in names]})
        except (OSError, ValueError, IndexError): continue
    return found


def idle(base, owned_lock=None):
    for name in LOCKS:
        p = Path(base)/'results'/name
        require(not os.path.lexists(p) or p == owned_lock, 'existing timing lock; no overlap or stale-lock removal')
    require(not scan_active(base), 'existing project process; no overlapping measurement')


def native_command(base, python):
    L = layout(base)
    return [str(python), '-B', '-u', str(HERE/'original/native/run_parent_replay_v1.py'),
            '--archive', str(L['cohort_archive']), '--count', '128',
            '--output-dir', str(L['out']/'native'), '--execute', '--library', str(L['build']/'native_assoc_v1.so'),
            '--native-dir', str(L['native']), '--prepare-native', '--graph-path', str(L['full']),
            '--admission-marker', str(L['out']/'admission.json'), '--include-upstream',
            '--external-helper', str(HERE/'original/external_operator_benchmark.py'),
            '--external-helper-sha256', SCIENCE_PINS['original/external_operator_benchmark.py'],
            '--conversion-max-native-gib', '1', '--max-assoc-gib', '0.05', '--max-peak-gib', '8',
            '--max-query-neighbors', '20000000', '--warmup', '1', '--repeats', '3',
            '--timeout-seconds', '3600', '--conversion-timeout-seconds', '1800', '--order-seed', '20261125']


def finite_times(values):
    return isinstance(values, list) and len(values) == 3 and all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in values)


def classify_native(folder, exit, library_sha, original, native):
    require(clean(exit), 'native parent owned process did not exit cleanly')
    folder = Path(folder); done = read(folder/'COMPLETE.json'); plan = read(folder/'replay_plan.json')
    require(done['complete'] is True and done['scientific_terminal'] is True and done['success'] is True and
            len(done['arms']) == 3 and {a['mode'] for a in done['arms']} == set(NATIVE_MODES), 'all three native arms required')
    require(done['replay_plan_sha256'] == sha(folder/'replay_plan.json') and plan['count'] == 128 and
            plan['archive']['sha256'] == sha(layout(BASE)['cohort_archive']) and (plan['graph']['nodes'], plan['graph']['stored_entries']) == (N, M), 'native replay identity changed')
    frozen_plan_match(plan)
    meta = conversion_binding(native, folder, original); ordered = None
    for arm in done['arms']:
        path = folder/(arm['mode']+'.json'); report = read(path)
        require(arm['eligible'] is True and type(arm['exit']['returncode']) is int and arm['exit']['returncode'] == 0 and
                arm['exit']['timed_out'] is False and sha(path) == arm['report_sha256'], 'native arm raw hash/exit differs')
        require(report['status'] == 'complete' and report['record_count'] == 128 and report['all_executed_outputs_exact'] is True and
                report['provenance']['library_sha256'] == library_sha and report['native_setup']['original_conversion_manifest'] == meta,
                'native scientific/source/whole-array proof incomplete')
        config = report['configuration']
        require(config['mode'] == arm['mode'] and config['warmup'] == 1 and config['repeats'] == 3 and
                config['max_query_neighbors'] == 20000000 and config['max_assoc_gib'] == 0.05 and config['max_peak_gib'] == 8 and
                Path(config['native_dir']) == Path(native), 'native executed configuration differs from fixed protocol')
        require([(r['archive_record_index'], r['node_set_sha256']) for r in report['records']] ==
                [(r['archive_record_index'], r['node_set_sha256']) for r in plan['selected']], 'native record order differs')
        fingerprints = []
        for row in report['records']:
            require(row['status'] == 'ok' and all(row[k] is True for k in ('exact_edges', 'exact_drnl', 'exact_ordered_rows_cols_and_original_eids')) and
                    row['root_positions'] == [0, 1] and all(finite_times(row[k]) for k in ('edge_seconds', 'drnl_seconds', 'raw_operator_seconds')),
                    'all native repeat timers/semantic proofs required')
            fingerprints.append(row['ordered_output_sha256'])
            require(len(row['ordered_output_sha256']) == 3 and all(valid_sha(x) for x in row['ordered_output_sha256']), 'all three ordered native output hashes required')
        require(ordered is None or ordered == fingerprints, 'native raw ordered outputs differ across arms'); ordered = fingerprints
    return {'eligible': True, 'record_count': 128, 'modes': list(NATIVE_MODES), 'global_native_hashes': original['expected_native_hashes']}


def classify_numba(path, exit, width, parent, expected_host):
    require(clean(exit), 'Numba owned worker did not exit cleanly')
    r = read(path); plan = read(Path(parent)/'replay_plan.json')
    require(r['status'] == 'complete' and r['record_count'] == 128 and r['all_executed_outputs_exact'] is True and
            r['numba_threads'] == 1 and r['neighbor_width'] == width and r['torch_imported'] is False, 'Numba scientific completion incomplete')
    require(r['configuration'] == {'selection_count': 128, 'selection_seed': 20261125, 'warmup': 1, 'repeats': 3,
             'methods': list(METHODS), 'worker_count': 1, 'cpu_threads': 1, 'max_query_neighbors': 20000000}, 'Numba fixed configuration differs')
    require(set(r['compiled_kernels']) == {'_build_induced_edges_pairwise', '_fill_neighbors_with_owner_numba', '_build_induced_edges_neighbor_scan'} and
            all(isinstance(v, list) and v and all(isinstance(s, str) and s for s in v) for v in r['compiled_kernels'].values()), 'actual compiled nopython kernels required')
    verify_host(expected_host, r['host'])
    physical = r['physical_arrays']
    require(physical['offset_dtype'] == 'int64' and physical['neighbor_dtype'] == 'int'+str(width) and
            physical['offset_count'] == N and physical['neighbor_count'] == M and
            physical['native_rowptr_view_excludes_existing_final_sentinel'] is (width == 64) and
            physical['full_graph_copy_or_conversion_performed'] is False, 'actual physical width/copy proof differs')
    require([(x['archive_record_index'], x['node_set_sha256'], x['canonical_edges_sha256'], x['drnl_sha256']) for x in r['records']] ==
            [(x['archive_record_index'], x['node_set_sha256'], x['canonical_edges_sha256'], x['drnl_sha256']) for x in plan['selected']], 'Numba frozen ordered cohort differs')
    for row in r['records']:
        require(row['exact_archived_oracles'] is True and set(row['methods']) == set(METHODS), 'both fixed Numba modes required')
        for method in row['methods'].values():
            require(method['status'] == 'ok' and method['exact_edges'] is True and method['exact_drnl'] is True and
                    all(finite_times(method[k]) for k in ('edge_seconds', 'drnl_seconds', 'edge_plus_drnl_seconds')), 'Numba all repeat timers/semantics required')
    return {'eligible': True, 'record_count': 128, 'width': width, 'methods': list(METHODS)}


def pack_local(out, archive):
    require(not archive.exists() and not archive.with_suffix('.partial').exists(), 'fresh local archive only')
    selected = []
    for prefix, folder in (('results', out), ('sources', HERE)):
        names = ([p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts]
                 if prefix == 'results' else [HERE/item['path'] for item in read(HERE/'SOURCE_MANIFEST.json')['files']]+[HERE/'SOURCE_MANIFEST.json'])
        for p in names:
            require(not p.is_symlink() and p.stat().st_size <= 16*MIB, 'bounded regular public raw/source file required')
            selected.append((prefix+'/'+str(p.relative_to(folder)), p))
    require(sum(p.stat().st_size for _, p in selected) <= 96*MIB and len(selected) <= 256, 'local raw packet size/member bound')
    require(shutil.disk_usage(out).free >= GIB, 'local archive1GiB free reserve')
    files = {name: info(p) for name, p in selected}; tmp = archive.with_suffix('.partial')
    manifest = json.dumps({'kind': 'public_strict_clean_citation2_matched_cpu_v1_raw', 'files': files, 'YT_writes': 0}, indent=2).encode()+b'\n'
    import io
    with tarfile.open(tmp, 'w:gz') as tf:
        for name, p in selected: tf.add(p, arcname=NAME+'/'+name, recursive=False)
        member = tarfile.TarInfo(NAME+'/MANIFEST.json'); member.size = len(manifest); tf.addfile(member, io.BytesIO(manifest))
    require(tmp.stat().st_size <= 64*MIB and all(info(p) == files[name] for name, p in selected), 'bounded immutable final raw archive required')
    tmp.rename(archive)
    return {'path': str(archive), **info(archive), 'files': len(selected), 'manifest_sha256': hashlib.sha256(manifest).hexdigest(), 'YT_writes': 0}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--base', type=Path, default=BASE)
    p.add_argument('--graph-path', type=Path, default=DEFAULT_GRAPH_PATH,
                   help='Existing public strict-clean CSR; files admitted by exact whole SHA, no copying or preparation')
    p.add_argument('--execute', action='store_true'); p.add_argument('--source-manifest-sha256')
    p.add_argument('--max-hours', type=float, default=6); a = p.parse_args(argv); base = a.base.resolve()
    require(a.graph_path.is_absolute(), 'explicit absolute public graph input path required')
    os.environ['LPP_C2_MATCHED_GRAPH_PATH'] = str(a.graph_path.resolve())
    L = layout(base)
    require(math.isfinite(a.max_hours) and 0 < a.max_hours <= 6, 'finite total deadline at most6hours required')
    bindings = verify_sources(); env_layout = environment.layout(base)
    if not a.execute:
        print(json.dumps({'status': 'plan_only', 'graph': {'nodes': N, 'stored_entries': M, 'path': str(L['full'])},
                          'count': 128, 'native_argv': native_command(base, env_layout['python']),
                          'Numba_widths': [32, 64], 'methods': list(METHODS), 'native_bytes': 8*(N+1+M),
                          'source_manifest_sha256': sha(HERE/'SOURCE_MANIFEST.json'), 'remote_actions': 0, 'runtime_or_graph_read': False})); return 0
    require(base == BASE and HERE == base/'citation2_matched_cpu_v1_code' and sys.platform.startswith('linux') and platform.machine() == 'x86_64', 'explicit canonical idle CPU host/source/base required')
    require(L['full'].is_relative_to(base), 'existing graph input must be under canonical base')
    require(a.source_manifest_sha256 == sha(HERE/'SOURCE_MANIFEST.json'), 'root-pinned source manifest required')
    for key in ('out', 'archive', 'venv', 'wheels', 'native', 'build', 'cache'):
        require(not os.path.lexists(L[key]), 'preserve existing/failed CPU output: '+key)
    idle(base); require(shutil.disk_usage(base).free >= 8*GIB, '8GiB free disk admission required')
    L['out'].mkdir(parents=True); L['cache'].mkdir()
    os.environ.update(PYTHONDONTWRITEBYTECODE='1', NUMBA_CACHE_DIR=str(L['cache']), CUDA_VISIBLE_DEVICES='',
                      OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMBA_NUM_THREADS='1')
    supervisor = module(HERE/'third_party/owned_supervision_v3.py', 'public_cpu_supervisor')
    memory = module(HERE/'third_party/memory_guard_v3.py', 'public_cpu_memory')
    execution = module(HERE/'execution_v1.py', 'public_cpu_execution')
    supervisor.enable_subreaper(); started = time.monotonic(); deadline = started+3600*a.max_hours
    own = supervisor.process_identity(os.getpid()); require(own, 'actual controller PID/start required')
    token = {'kind': NAME, 'pid': os.getpid(), 'start_ticks': own['start_ticks'], 'entrypoint': str(Path(__file__).resolve()),
             'source_sha256': sha(Path(__file__)), 'source_manifest_sha256': a.source_manifest_sha256}
    lock = base/'results/ADDITIONAL_TIMING_RUNNING.lock'; owned = False; failure = None; scientific = []; stages = []; pointer = None
    original_handler = signal.getsignal(signal.SIGTERM)
    def interrupt(sig, frame): raise KeyboardInterrupt('received signal '+str(sig))
    signal.signal(signal.SIGTERM, interrupt)
    save(L['out']/'plan.json', {'kind': NAME, 'controller_identity': token, 'source_bindings': bindings, 'max_hours': a.max_hours,
                               'native_job': native_command(base, env_layout['python']), 'numba_widths': [32, 64], 'YT_writes': 0})
    def run(name, argv, timeout):
        remaining = deadline-time.monotonic()-120
        require(remaining >= 30, 'global finite deadline reached before '+name)
        assert_bindings(bindings)
        if owned: require(read(lock) == token, 'owned CPU timing lock changed')
        result = execution.guarded_run(supervisor, memory, [str(x) for x in argv], L['out']/(name+'.log'), HERE, min(timeout, remaining))
        stages.append({'name': name, 'exit': result}); save(L['out']/'stages.json', {'stages': stages})
        require(clean(result), 'failed/guarded owned phase: '+name)
        assert_bindings(bindings); return result
    try:
        idle(base); execution.acquire_lock(lock, token); owned = True; idle(base, lock)
        before = L['out']/'base_environment_before.json'
        run('base_environment_before', [sys.executable, '-B', HERE/'cpu_environment_v1.py', '--probe', 'base', '--output', before], 120)
        wheel_receipt = environment.unpack_wheels(L['wheel_archive'], L['wheels']); save(L['out']/'wheel_extraction.json', wheel_receipt)
        run('create_isolated_env', env_layout['create_argv'], 120)
        route = environment.add_base_site(L['venv'], read(before)); save(L['out']/'base_site_binding.json', route)
        run('install_offline_native', env_layout['install_argv'], 600)
        isolated = L['out']/'isolated_environment.json'
        run('isolated_environment', [env_layout['python'], '-B', HERE/'cpu_environment_v1.py', '--probe', 'isolated', '--venv', L['venv'], '--output', isolated], 120)
        base_after_install = L['out']/'base_environment_after_install.json'
        run('base_environment_after_install', [sys.executable, '-B', HERE/'cpu_environment_v1.py', '--probe', 'base', '--output', base_after_install], 120)
        preliminary = environment.verify(read(before), read(isolated), read(base_after_install)); save(L['out']/'environment_admission.json', preliminary)
        bindings.update(environment.environment_bindings(read(isolated), L['venv']))
        run('build_standalone_native', [env_layout['python'], '-B', HERE/'original/native/build_native_v1.py', '--output-dir', L['build']], 180)
        build = read(L['build']/'build_receipt_v1.json')
        require(build['source_sha256'] == SCIENCE_PINS['original/native/native_assoc_v1.cpp'] and
                build['library_sha256'] == sha(L['build']/'native_assoc_v1.so'), 'actual standalone source/build/library differs')
        for path in (L['build']/'native_assoc_v1.so', L['build']/'build_receipt_v1.json'): bindings[str(path)] = info(path)
        run('full_input_before', [env_layout['python'], '-B', HERE/'cpu_input_v1.py', '--base', base, '--phase', 'before'], 600)
        original = read(L['out']/'full_input_before.json'); host = host_receipt(); save(L['out']/'host.json', host)
        run('freeze_citation2_cohort', [env_layout['python'], '-B', '-u', HERE/'cohort_v1.py', '--base', base], 1800)
        cohort = read(L['cohort']/'COHORT_READY.json')
        require(cohort['status'] == 'FROZEN_BEFORE_ALL_MEASUREMENTS' and cohort['record_count'] == 128, 'fresh deterministic cohort required')
        require(cohort['archive'] == info(L['cohort_archive']) and cohort['plan'] == info(L['cohort_plan']) and
                cohort['protocol'] == info(HERE/'COHORT_PROTOCOL.json'), 'generated cohort/protocol bindings differ')
        for file in (L['cohort_archive'], L['cohort_plan'], L['cohort']/'COHORT_READY.json'):
            bindings[str(file)] = info(file)
        save(L['out']/'admission.json', {'kind': 'citation2_cpu_fresh_admission_v1', 'complete': True,
             'controller_identity': token, 'host': host, 'full_original': original, 'source_manifest_sha256': a.source_manifest_sha256,
             'environment_probe_sha256': sha(isolated), 'offline_wheels': wheel_receipt, 'cohort': cohort,
             'meaning': 'Fresh CPU source/environment/fullinput admission complete; not a fabricated predecessor completion.'})
        native_exit = run('native_citation2', native_command(base, env_layout['python']), 13000)
        scientific.append({'name': 'native_citation2', **classify_native(L['out']/'native', native_exit, build['library_sha256'], original, L['native'])})
        save(L['out']/'scientific_jobs.json', {'jobs': scientific})
        for width in (32, 64):
            stable_original(original); verify_host(host, host_receipt()); idle(base, lock)
            output = L['out']/('citation2_neighbors'+str(width)+'.json')
            result = run('numba'+str(width), [env_layout['python'], '-B', '-u', HERE/'cpu_worker_v1.py', '--base', base,
                                            '--width', str(width), '--output', output], 1800)
            scientific.append({'name': 'numba'+str(width), **classify_numba(output, result, width, L['out']/'native', host)})
            save(L['out']/'scientific_jobs.json', {'jobs': scientific})
        run('full_input_after', [env_layout['python'], '-B', HERE/'cpu_input_v1.py', '--base', base, '--phase', 'after'], 600)
        require(read(L['out']/'full_input_after.json') == original, 'whole original fullCSR changed across scientific execution')
        iso_after, base_after = L['out']/'isolated_environment_after.json', L['out']/'base_environment_after.json'
        run('isolated_environment_after', [env_layout['python'], '-B', HERE/'cpu_environment_v1.py', '--probe', 'isolated', '--venv', L['venv'], '--output', iso_after], 120)
        run('base_environment_after', [sys.executable, '-B', HERE/'cpu_environment_v1.py', '--probe', 'base', '--output', base_after], 120)
        require(read(isolated) == read(iso_after), 'isolated source/version/native environment changed')
        save(L['out']/'environment_verification.json', environment.verify(read(before), read(isolated), read(base_after)))
        verify_host(host, host_receipt()); assert_bindings(bindings)
    except BaseException as error:
        failure = {'type': type(error).__name__, 'message': str(error)[:1600]}
    finally:
        quiescent = execution.cleanup_owned(supervisor)
        success = failure is None and quiescent and len(scientific) == 3 and all(r['eligible'] for r in scientific)
        save(L['out']/'SCIENTIFIC_TERMINAL.json', {'controller_jobs_finished': quiescent, 'success': success, 'jobs': scientific,
             'failure': failure, 'owned_tree_quiescent': quiescent, 'timing_lock_retained': owned, 'runtime_seconds': time.monotonic()-started,
             'private_or_GPU_jobs': 0, 'new_original_input_copies': 0, 'YT_writes': 0})
    try:
        require(owned and quiescent and read(lock) == token, 'owned quiescent lock required for final local packet')
        pointer = pack_local(L['out'], L['archive']); save(L['out']/'ARCHIVE_READY.json', pointer)
    except BaseException as error:
        failure = failure or {'type': type(error).__name__, 'message': str(error)[:1600]}
    finally:
        released = False
        if owned and quiescent:
            execution.release_lock(lock, token, True); released = True
        save(L['out']/'COMPLETE.json', {'controller_finished': True, 'controller_identity': token,
             'scientific_complete': success, 'success': success and failure is None and pointer is not None,
             'failure': failure, 'owned_tree_quiescent': quiescent, 'timing_lock_released': released,
             'local_archive': pointer, 'YT_writes': 0, 'base_package_installs': 0, 'GPU_or_pod_lifecycle_actions': 0})
        signal.signal(signal.SIGTERM, original_handler)
    return 0 if success and failure is None and pointer is not None else 1


if __name__ == '__main__': raise SystemExit(main())
