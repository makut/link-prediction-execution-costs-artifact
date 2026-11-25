"""Explicit FAKE unit-test exports; never scientific or actual lifecycle evidence.

Only small exact public sources/128-row oracles are reused. Every timing, host,
environment observation and process event below is synthetic. The production
auditor must reject the synthetic_fixture flag unless its Python test API is
explicitly invoked with synthetic=True. No graph, checkpoint or native binary
is loaded, generated or installed.
"""
from pathlib import Path
import ast, copy, hashlib, io, json, random, tarfile, types

BASE = Path('/home/jovyan/seal_revision_20261124')
NAME = 'public_fr_cpu_matched_v2_20261125'
SOURCE = BASE/'public_fr_cpu_v2_code'
OUT = BASE/'results'/NAME
FULL = BASE/'public_friendster_repro_v1/new_data/full_csr'
NATIVE = BASE/'public_fr_cpu_matched_v2_data/native_i64'
BUILD = BASE/'public_fr_cpu_matched_v2_build'
VENV = BASE/'public_fr_cpu_matched_v2_env'
N, M, GIB = 65608366, 3612134270, 2**30
FLAG = 'NOT_SCIENTIFIC_EVIDENCE'
MODES = ('reference_full_clear', 'persistent_touched_reset', 'upstream_torch_sparse_direct')


def encode(value): return (json.dumps(value, indent=2, allow_nan=False)+'\n').encode()
def digest(data): return hashlib.sha256(data).hexdigest()
def identity(data): return {'bytes': len(data), 'sha256': digest(data)}
def put(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data); return path
def save(path, value): return put(path, encode(value))
def read(path): return json.loads(Path(path).read_text())
def clean_exit(pid=51000):
    return {'pid': pid, 'returncode': 0, 'timed_out': False, 'timeout_seconds': 600,
            'owned_tree_quiescent': True, 'orphan_cleanup': False, 'guard_abort_reason': None,
            'guard_policy': 'unchanged stable_pss_individual_rss_cgroup90_v3 externally sampled at2s;180GiB; CPU-only'}


def constants(source):
    result = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try: result[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError): pass
    return result


def environment_receipts(source_files, wheel_members):
    """Reuse only the frozen synthetic factory, not any measured environment."""
    text = source_files['test_cpu_environment_v1.py'].decode()
    fn = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == 'fake_receipts')
    values = constants(source_files['cpu_environment_v1.py'].decode())
    env = types.SimpleNamespace(**values)
    env.WHEEL_MEMBERS = wheel_members
    env.THREADS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS')
    scope = {'Path': Path, 'copy': copy, 'e': env}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), '<frozen_synthetic_environment_factory>', 'exec'), scope)
    before, isolated, after = scope['fake_receipts'](VENV)
    # macOS resolves /home through /System/Volumes/Data; fake target receipts
    # deliberately retain the canonical Linux path without touching that host.
    resolved_venv=str(VENV.resolve())
    def target_paths(value):
        if isinstance(value,str): return value.replace(resolved_venv,str(VENV))
        if isinstance(value,list): return [target_paths(v)for v in value]
        if isinstance(value,dict): return {k:target_paths(v)for k,v in value.items()}
        return value
    before,isolated,after=map(target_paths,(before,isolated,after))
    origin = SOURCE/'cpu_environment_v1.py'
    for record in (before, isolated, after):
        record['schema_version'] = 1; record['kind'] = 'public_fr_cpu_environment_probe_v1'
        record['probe_source'] = {'path': str(origin), **identity(source_files['cpu_environment_v1.py'])}
        record.update(GPU_API_calls_by_probe=0, models_loaded=0, package_installations=0)
    return before, isolated, after


def make_fixture(root, anchors):
    """Return {export_dir, root_launch_file, expected, extracted, anchors}."""
    root = Path(root)
    if root.exists(): raise FileExistsError('fresh synthetic test directory required')
    root.mkdir(parents=True)
    source_files = anchors['source_files']
    manifest_raw = source_files['SOURCE_MANIFEST.json']; manifest_sha = digest(manifest_raw)
    plan = copy.deepcopy(anchors['frozen_plan'])
    plan['archive']['original_local_path'] = str(SOURCE/'records/friendster_exact_int32.json')
    selected = plan['selected']
    if len(selected) != 128: raise ValueError('actual frozen128 cohort required for fake unit fixtures')
    e1 = anchors['e1']; records = e1['records'] if isinstance(e1, dict) else e1
    ordered = anchors['ordered_hashes']
    def oracle(index): return ordered[index] if index in ordered else ordered[str(index)]
    extracted = root/'extracted'/NAME; results = extracted/'results'
    results.mkdir(parents=True)
    for name, raw in source_files.items(): put(extracted/'sources'/name, raw)
    source_sha = digest(source_files['cpu_coordinator_v1.py'])
    packet_sha = 'f79cd610e8f24830b896ed4e8b3a37035ab31d0626349699c3b568ebe69c118f'
    launch = {'pid': 42042, 'start_ticks': 424242, 'source_sha256': source_sha,
              'entrypoint': str(SOURCE/'cpu_coordinator_v1.py'),
              'argv': ['/opt/python/bin/python3', '-B', '-u', str(SOURCE/'cpu_coordinator_v1.py'), '--execute',
                       '--source-manifest-sha256', manifest_sha, '--max-hours', '6'],
              'source_packet_sha256': packet_sha, 'source_manifest_sha256': manifest_sha,
              'started_unix': 1., 'host': 'lpp-cpu-0', 'YT_writes': 0, 'GPU_allocations': 0,
              'CPU_lifecycle_actions': 0, 'stdin_dependency': False}
    token = {'kind': NAME, 'pid': launch['pid'], 'start_ticks': launch['start_ticks'], 'entrypoint': launch['entrypoint'],
             'source_sha256': source_sha, 'source_manifest_sha256': manifest_sha}
    launch_path = save(root/'synthetic_root_launch.json', launch)
    host = {'hostname': 'lpp-cpu-0', 'boot_id': '00000000-0000-0000-0000-000000000001',
            'cpu_models': ['SYNTHETIC TEST CPU; NOT OBSERVED'], 'cpu_affinity': [0]}
    save(results/'host.json', host)
    full_files = json.loads(source_files['FULL_INPUT_IDENTITY.json'])['files']
    full_lookup = {r['path']: r for r in full_files}
    full = {'kind': 'whole_public_fullCSR_SHA_v1', 'nodes': N, 'stored_entries': M, 'files': {}}
    for inode, name in enumerate(('edge_starts.bin', 'edge_ends.bin', 'num_nodes.txt'), 10):
        pin = full_lookup[name]
        full['files'][str(FULL/name)] = {'bytes': pin['bytes'], 'sha256': pin['sha256'], 'mtime_ns': 100,
                                       'device': 1, 'inode': inode}
    for phase in ('before', 'after'): save(results/('full_input_'+phase+'.json'), full)
    before, isolated, after = environment_receipts(source_files, anchors['wheel_members'])
    for name, value in [('base_environment_before.json', before), ('base_environment_after_install.json', after),
                        ('base_environment_after.json', after), ('isolated_environment.json', isolated),
                        ('isolated_environment_after.json', isolated)]: save(results/name, value)
    verify = {'status': 'PASS', 'base_before_after_exact': True, 'inherited_numerical_origins_exact': True,
              'native_wheel_source_and_binary_hashes_exact': True, 'saint_wrapper_git_blob': '440455853e0309b00376cbe9ad25d7226cfa5e0e',
              'two_official_wheels_only': True, 'no_base_package_install_or_patch': True, 'CUDA_VISIBLE_DEVICES': '', 'GPU_API_calls_by_probe': 0}
    save(results/'environment_admission.json', verify); save(results/'environment_verification.json', verify)
    env_constants = constants(source_files['cpu_environment_v1.py'].decode())
    wheels = env_constants['WHEELS']
    wheel_receipt = {'status': 'PASS', 'packet': env_constants['PACKET'], 'download_manifest_sha256': env_constants['MANIFEST_SHA'],
                     'members': {'DOWNLOAD_MANIFEST.json': {'bytes': 1105, 'sha256': env_constants['MANIFEST_SHA']},
                                 **{k: {x: v[x] for x in ('bytes', 'sha256')} for k, v in wheels.items()}},
                     'destination': str(BASE/'public_fr_cpu_matched_v2_wheels'), 'package_installations': 0,
                     'base_packages_changed': False, 'network_calls': 0, 'subprocesses_started': 0}
    save(results/'wheel_extraction.json', wheel_receipt)
    route = {'path': isolated['base_site_pth']['path'], 'identity': {k: isolated['base_site_pth'][k] for k in ('bytes', 'sha256')},
             'base_sites': before['base_sites'], 'plain_path_lines_only': True, 'import_statements': 0, 'base_packages_modified': False}
    save(results/'base_site_binding.json', route)
    admission = {'kind': 'public_fr_cpu_fresh_admission_v2', 'complete': True, 'controller_identity': token, 'host': host,
                 'full_original': full, 'source_manifest_sha256': manifest_sha,
                 'environment_probe_sha256': digest(encode(isolated)), 'offline_wheels': wheel_receipt,
                 'meaning': 'FAKE unit fixture; no actual CPU admission or event.'}
    save(results/'admission.json', admission)
    source_bindings = {str(SOURCE/name): identity(raw) for name, raw in source_files.items()}
    save(results/'plan.json', {'kind': NAME, 'controller_identity': token, 'source_bindings': source_bindings,
                             'max_hours': 6., 'numba_widths': [32, 64], 'YT_writes': 0})
    native = results/'native'; native.mkdir(); save(native/'replay_plan.json', plan)
    source_metadata = [{'path': p, 'bytes': r['bytes'], 'mtime_ns': r['mtime_ns']} for p, r in full['files'].items()
                       if Path(p).name in ('edge_starts.bin', 'edge_ends.bin')]
    native_hashes = {'rowptr_sha256': '28060771441d71ba71877e55c0a48230c54d1b2715dd4edd08042019cf4a7a81',
                     'col_sha256': '012e1d0a907d2eb6b46d72f31be8b3ca722d88432b2c0509162c4ff487f25a5d'}
    graph = {'n': N, 'm': M, 'endian': 'little', 'source_files': source_metadata,
             'source_storage_bytes': 14973404016, 'native_int64_storage_bytes': 29421941096}
    conversion = {'schema_version': 1, 'graph': graph, 'native_dtype': 'little-endian int64', 'sentinel_added': True,
                  'conversion_seconds': 1., **native_hashes}
    native_setup = {'original_conversion_manifest': conversion, 'content_verification_seconds': 1.,
                    'runtime_files': [{'path': str(NATIVE/name), 'bytes': size, 'mtime_ns': 200, 'sha256': native_hashes[key]}
                                      for name, size, key in [('rowptr.i8', 8*(N+1), 'rowptr_sha256'), ('col.i8', 8*M, 'col_sha256')]],
                    'conversion_performed_by_this_worker': False,
                    'scope': 'Synthetic receipt; no native payload exists or was read.'}
    fixture_proof = {'status': 'pass', 'literal_cases': 4, 'ordered_raw_edges_and_eids': True,
                     'arbitrary_root_packaging_and_independent_BFS_labels': True}
    conversion_receipt = {'schema_version': 1, 'kind': 'corrected_native_conversion_only_v1', 'status': 'complete',
                          'external_helper_sha256': digest(source_files['original/external_operator_benchmark.py']),
                          'graph': graph, 'conversion': conversion, 'actual_native_fixture': fixture_proof,
                          'setup_wall_seconds': 1., 'runtime_native_content_verification': {'status':'pass','seconds':1.,'files':native_setup['runtime_files']},
                          'estimate': {'source_storage_bytes':8*N+8+4*M,'native_int64_storage_bytes':8*(N+1+M),
                           'per_query_assoc_bytes':8*N,'public_api_global_row_bytes':0,'conversion_chunk_upper_bytes':24000000,
                           'conservative_peak_bytes':8*N+8+4*M+8*(N+1+M)+8*N+24000000+GIB}}
    save(native/'native_conversion_receipt.json', conversion_receipt)
    save(native/'native_conversion_exit.json', {'pid': 50001, 'returncode': 0, 'timed_out': False, 'timeout_seconds': 1800})
    library_sha = '7'*64
    build = {'schema_version': 1, 'kind': 'corrected_native_build_v1', 'source': str(SOURCE/'original/native/native_assoc_v1.cpp'),
             'source_sha256': digest(source_files['original/native/native_assoc_v1.cpp']), 'compiler': '/usr/bin/c++',
             'compiler_version': 'SYNTHETIC TEST COMPILER', 'argv': ['/usr/bin/c++', '-std=c++17', '-O3', '-DNDEBUG', '-fPIC', '-pthread',
                '-shared', str(SOURCE/'original/native/native_assoc_v1.cpp'), '-o', str(BUILD/'native_assoc_v1.so.building')],
             'platform': 'Linux SYNTHETIC', 'library': str(BUILD/'native_assoc_v1.so'), 'library_sha256': library_sha,
             'scope': 'FAKE fixture: no compiler or binary was executed.'}
    put(results/'build_standalone_native.log', encode(build))
    software = {'torch': '2.2.2+cu121', 'torch_sparse': '0.6.18+pt22cu121', 'torch_num_threads': 1,
                'installed_wrapper_sha256': anchors['wheel_members']['torch_sparse/saint.py']['sha256'],
                'installed_wrapper_git_blob': '440455853e0309b00376cbe9ad25d7226cfa5e0e',
                'extension_binaries': [{'path': str(VENV/'lib/python3.10/site-packages'/name), 'sha256': pin['sha256']}
                                      for name, pin in anchors['wheel_members'].items() if name.startswith('torch_sparse/_saint')]}
    arm_receipts = []
    expected = {'native_reference_full_clear': 256., 'native_persistent_touched_reset': 128., 'native_upstream_torch_sparse_direct': 512.,
                'numba32_scan': 64., 'numba32_pairwise': 128., 'numba64_scan': 128., 'numba64_pairwise': 256.}
    for arm_index, mode in enumerate(MODES):
        factor = [1., .5, 2.][arm_index]
        native_rows = []
        for frozen in selected:
            index = frozen['archive_record_index']
            native_rows.append({'archive_record_index': index, 'node_set_sha256': frozen['node_set_sha256'], 'status': 'ok',
                'raw_operator_seconds': [factor/2, factor, factor*1.5], 'edge_seconds': [factor, 2*factor, 3*factor],
                'drnl_seconds': [.1, .2, .3], 'exact_ordered_rows_cols_and_original_eids': True,
                'ordered_output_sha256': list(oracle(index)), 'exact_edges': True, 'exact_drnl': True, 'root_positions': [0, 1]})
        state = {'persistent_assoc_bytes_per_worker': 8*N if mode == 'persistent_touched_reset' else 0,
                 'assoc_bytes_during_each_call': 8*N, 'initialization_seconds': .25,
                 'mode': mode, 'nodes': N, 'stored_entries': M, 'owner_pid': 52000+arm_index,
                 'initialization_policy': 'once per worker' if mode == 'persistent_touched_reset' else 'full-N allocation/init inside every native call',
                 'library_sha256': library_sha,'library':str(BUILD/'native_assoc_v1.so'),'version':'persistent_assoc_touched_reset_v1'}
        if mode=='upstream_torch_sparse_direct':
            state={'persistent_assoc_bytes_per_worker':0,'assoc_bytes_during_each_call':8*N,
                   'policy':'unmodified upstream full-N association allocation/init inside each call'}
        report = {'schema_version': 1, 'kind': 'corrected_native_replay_worker_v1', 'status': 'complete', 'mode': mode,
                  'configuration': {'mode': mode, 'warmup': 1, 'repeats': 3, 'max_assoc_gib': .5, 'max_peak_gib': 48.,
                                    'max_query_neighbors': 20000000, 'native_dir': str(NATIVE), 'library': str(BUILD/'native_assoc_v1.so'),
                                    'archive': str(SOURCE/'records/friendster_exact_int32.json'), 'replay_plan': str(OUT/'native/replay_plan.json')},
                  'provenance': {'python': '3.10.12 SYNTHETIC', 'platform': 'Linux SYNTHETIC',
                     'worker_sha256': digest(source_files['original/native/replay_worker_v1.py']),
                     'adapter_sha256': digest(source_files['original/native/native_adapter_v1.py']), 'library_sha256': library_sha},
                  'archive_sha256': digest(source_files['records/friendster_exact_int32.json']),
                  'replay_plan_sha256': digest(encode(plan)), 'actual_native_admission': fixture_proof, 'native_setup': native_setup,
                  'state_initialization': state, 'records': native_rows, 'record_count': 128, 'all_executed_outputs_exact': True}
        report['configuration'].update(output=str(OUT/'native'/(mode+'.json')),
           external_helper=str(SOURCE/'original/external_operator_benchmark.py')if mode=='upstream_torch_sparse_direct'else None,
           external_helper_sha256=digest(source_files['original/external_operator_benchmark.py'])if mode=='upstream_torch_sparse_direct'else None)
        local = 160*max(records[r['archive_record_index']]['induced_directed_entries']for r in selected)+256*max(len(records[r['archive_record_index']]['nodes'])for r in selected)
        report['budget']={'native_int64_storage_and_potential_file_working_set_bytes':8*(N+1+M),
           'association_bytes_per_worker':8*N,'local_output_upper_estimate_bytes':local,'conservative_peak_estimate_bytes':8*(N+1+M)+8*N+local+GIB}
        if mode == 'upstream_torch_sparse_direct': report['upstream_software'] = software
        path = save(native/(mode+'.json'), report)
        exit = {'pid': 52000+arm_index, 'returncode': 0, 'timed_out': False, 'timeout_seconds': 3600}
        save(native/(mode+'_exit.json'), exit)
        arm_receipts.append({'mode': mode, 'eligible': True, 'exit': exit, 'report_status': 'complete', 'report_sha256': digest(path.read_bytes())})
    mode_order=list(MODES);random.Random(20261125).shuffle(mode_order)
    source_hashes={Path(name).name:digest(raw)for name,raw in source_files.items()if name.startswith('original/native/')}
    save(native/'parent_plan.json',{'source_hashes':source_hashes,'replay_plan_sha256':digest(encode(plan)),
      'mode_order_seed':20261125,'modes':mode_order,'configuration':{'count':128,'prepare_native':True,'include_upstream':True,
       'warmup':1,'repeats':3,'max_query_neighbors':20000000,'max_assoc_gib':.5,'max_peak_gib':48,
       'conversion_max_native_gib':28,'native_dir':str(NATIVE),'library':str(BUILD/'native_assoc_v1.so'),
       'archive':str(SOURCE/'records/friendster_exact_int32.json'),'output_dir':str(OUT/'native'),'execute':True,'build_dir':None,
       'admission_marker':str(OUT/'admission.json'),'external_helper':str(SOURCE/'original/external_operator_benchmark.py'),
       'external_helper_sha256':digest(source_files['original/external_operator_benchmark.py']),'graph_path':str(FULL),
       'conversion_timeout_seconds':1800,'timeout_seconds':3600,'order_seed':20261125}})
    runtime={str(SOURCE/'original/native'/n):s for n,s in source_hashes.items()}
    runtime.update({str(BUILD/'native_assoc_v1.so'):library_sha,str(SOURCE/'records/friendster_exact_int32.json'):digest(source_files['records/friendster_exact_int32.json']),
       str(OUT/'native/replay_plan.json'):digest(encode(plan)),str(SOURCE/'original/external_operator_benchmark.py'):digest(source_files['original/external_operator_benchmark.py'])})
    runtime_raw=encode(runtime);put(native/'runtime_bindings.json',runtime_raw)
    save(native/'COMPLETE.json', {'complete': True, 'scientific_terminal': True, 'success': True, 'terminal_status': 'complete',
          'arms': sorted(arm_receipts,key=lambda row:mode_order.index(row['mode'])), 'replay_plan_sha256': digest(encode(plan)),
          'runtime_bindings_sha256':digest(runtime_raw),'no_remote_or_YT_actions': True, 'no_performance_advantage_inferred': True})
    fixture_report=copy.deepcopy(anchors['upstream_fixture']);fixture_report.update(status='complete',fixture=True,
        all_executed_arms_exact=True,provenance=software,native_admission_fixture={'status':'pass','cases_per_arm':4,'arms':['direct']})
    for row in fixture_report['records']:
        row['methods']={'direct':{'status':'ok','exact_edges':True,'exact_drnl':True,
            'raw_operator_seconds':[.1,.2,.3],'edge_seconds':[.2,.3,.4],'drnl_seconds':[.1,.2,.3]}}
    save(native/'upstream_native_admission.json',fixture_report)
    save(native/'admission.json',{'path':str(OUT/'admission.json'),'sha256':digest(encode(admission)),'content':admission})
    save(native/'upstream_native_admission_exit.json', {'pid': 50000, 'returncode': 0, 'timed_out': False, 'timeout_seconds': 120})
    for width in (32, 64):
        numba_rows = []
        for frozen in selected:
            index = frozen['archive_record_index']; record = records[index]
            methods = {}
            for method_index, method in enumerate(('scan', 'pairwise')):
                factor = (.25 if width == 32 else .5)*(1 if method_index == 0 else 2)
                methods[method] = {'status': 'ok', 'edge_seconds': [factor, 2*factor, 3*factor], 'drnl_seconds': [.1, .2, .3],
                                   'edge_plus_drnl_seconds': [factor+.1, 2*factor+.2, 3*factor+.3], 'exact_edges': True, 'exact_drnl': True}
            numba_rows.append({k: frozen[k] for k in ('archive_record_index', 'node_set_sha256', 'canonical_edges_sha256', 'drnl_sha256')})
            numba_rows[-1].update(sum_out_degrees=record['sum_out_degrees'], induced_directed_entries=record['induced_directed_entries'],
                                 method_order=['scan', 'pairwise'], methods=methods, exact_archived_oracles=True)
        report = {'schema_version': 1, 'kind': 'public_full_friendster_matched_numba_v1', 'status': 'complete', 'neighbor_width': width,
                  'host': host, 'configuration': {'selection_count': 128, 'selection_seed': 20261125, 'warmup': 1, 'repeats': 3,
                    'methods': ['scan', 'pairwise'], 'worker_count': 1, 'cpu_threads': 1, 'max_query_neighbors': 20000000},
                  'admission_sha256': digest(encode(admission)), 'source_bindings': source_bindings, 'replay_plan': plan,
                  'original_full_payload_admission': full,
                  'original_input_stat_header_proof': {p: {k: r[k] for k in ('bytes', 'mtime_ns')} for p, r in full['files'].items() if Path(p).name != 'num_nodes.txt'},
                  'physical_arrays': {'offset_dtype': 'int64', 'neighbor_dtype': 'int'+str(width), 'offset_count': N, 'neighbor_count': M,
                                     'full_graph_copy_or_conversion_performed': False, 'native_rowptr_view_excludes_existing_final_sentinel': width == 64},
                  'setup_seconds_excluded': 1., 'compiled_kernels': {k: ['synthetic nopython signature; NOT ACTUALLY COMPILED'] for k in
                    ('_build_induced_edges_pairwise', '_fill_neighbors_with_owner_numba', '_build_induced_edges_neighbor_scan')},
                  'numerical_environment': {k: {'version': v, 'origin': before['numerical_modules'][k]['origin']}
                                            for k, v in [('numpy', '1.24.4'), ('scipy', '1.10.1'), ('numba', '0.58.0')]},
                  'numba_threads': 1, 'records': numba_rows, 'record_count': 128, 'all_executed_outputs_exact': True, 'torch_imported': False}
        if width == 64: report['native_setup'] = native_setup
        save(results/('friendster_neighbors'+str(width)+'.json'), report)
    phase_names = ['base_environment_before', 'create_isolated_env', 'install_offline_native', 'isolated_environment',
                   'base_environment_after_install', 'build_standalone_native', 'full_input_before', 'native_friendster',
                   'numba32', 'numba64', 'full_input_after', 'isolated_environment_after', 'base_environment_after']
    stages = []
    for index, name in enumerate(phase_names):
        exit = clean_exit(53000+index)
        exit['timeout_seconds']={'base_environment_before':120,'create_isolated_env':120,'install_offline_native':600,
          'isolated_environment':120,'base_environment_after_install':120,'build_standalone_native':180,'full_input_before':600,
          'native_friendster':13000,'numba32':1800,'numba64':1800,'full_input_after':600,'isolated_environment_after':120,'base_environment_after':120}[name]
        stages.append({'name': name, 'exit': exit}); save(results/(name+'.exit.json'), exit)
        if not (results/(name+'.log')).exists(): put(results/(name+'.log'), b'FAKE SYNTHETIC UNIT FIXTURE; no process ran.\n')
        decision = {'abort_reason': None, 'memory_budget_bytes': 180*GIB, 'own_pss_estimate_bytes': 1,
                    'cgroup_current_bytes': 1, 'cgroup_max_bytes': 256*GIB}
        put(results/(name+'.guard.jsonl'), (json.dumps({'unix': 1., 'tree_pss_bytes': 1, 'pss_complete': True,
                 'membership_stable': True, 'max_process_rss_bytes': 1, 'decision': decision})+'\n').encode())
    save(results/'stages.json', {'stages': stages})
    jobs = [{'name': 'native_friendster', 'eligible': True, 'record_count': 128, 'modes': list(MODES), 'global_native_hashes': native_hashes}]
    jobs += [{'name': 'numba'+str(w), 'eligible': True, 'record_count': 128, 'width': w, 'methods': ['scan', 'pairwise']} for w in (32, 64)]
    save(results/'scientific_jobs.json', {'jobs': jobs})
    save(results/'SCIENTIFIC_TERMINAL.json', {'controller_jobs_finished': True, 'success': True, 'jobs': jobs, 'failure': None,
         'owned_tree_quiescent': True, 'timing_lock_retained': True, 'runtime_seconds': 1., 'private_or_GPU_jobs': 0,
         'new_original_input_copies': 0, 'YT_writes': 0})
    fixture = {'export_dir': root, 'root_launch_file': launch_path, 'root_launch': launch_path, 'extracted': extracted,
               'results': results, 'anchors': anchors, 'launch': launch, 'token': token, 'expected': expected,
               'expected_edge_sums_of128_per_case_medians': expected, 'synthetic_fixture': FLAG}
    refresh(fixture)
    return fixture


def refresh(fixture):
    """Repack after an explicit test mutation; never edits immutable source pins."""
    root, extracted, results = fixture['export_dir'], fixture['extracted'], fixture['results']
    rows = {str(p.relative_to(extracted)): identity(p.read_bytes()) for p in sorted(extracted.rglob('*'))
            if p.is_file() and p.name != 'MANIFEST.json'}
    manifest = {'kind': 'public_full_friendster_matched_cpu_v1_raw', 'files': rows, 'YT_writes': 0}
    raw_manifest = encode(manifest); put(extracted/'MANIFEST.json', raw_manifest)
    packet = root/'final_snapshot.tgz'
    with tarfile.open(packet, 'w:gz', compresslevel=1) as archive:
        for name in rows: archive.add(extracted/name, arcname=NAME+'/'+name, recursive=False)
        archive.add(extracted/'MANIFEST.json', arcname=NAME+'/MANIFEST.json', recursive=False)
    archive_pin = identity(packet.read_bytes())
    pointer = {'path': str(BASE/(NAME+'.tgz')), **archive_pin, 'files': len(rows), 'manifest_sha256': digest(raw_manifest), 'YT_writes': 0}
    complete = {'controller_finished': True, 'controller_identity': fixture['token'], 'scientific_complete': True,
                'success': True, 'failure': None, 'owned_tree_quiescent': True, 'timing_lock_released': True,
                'local_archive': pointer, 'YT_writes': 0, 'base_package_installs': 0, 'GPU_or_pod_lifecycle_actions': 0}
    source_files = fixture['anchors']['source_files']
    receipts = {'public_fr_cpu_matched_v2_launch.json': fixture['root_launch_file'].read_bytes(),
                'results/'+NAME+'/COMPLETE.json': encode(complete), 'results/'+NAME+'/ARCHIVE_READY.json': encode(pointer),
                'public_fr_cpu_v2_code/SOURCE_MANIFEST.json': source_files['SOURCE_MANIFEST.json'],
                'public_fr_cpu_v2_code/cpu_coordinator_v1.py': source_files['cpu_coordinator_v1.py'],
                'public_fr_cpu_matched_v2_build/build_receipt_v1.json':(results/'build_standalone_native.log').read_bytes()}
    for name, data in receipts.items(): put(root/'remote_receipts'/name, data)
    manifest_sha = digest(source_files['SOURCE_MANIFEST.json']); source_sha = digest(source_files['cpu_coordinator_v1.py'])
    view = {'owner': {'pid': fixture['launch']['pid'], 'expected_start_ticks': fixture['launch']['start_ticks'],
                      'exists': False, 'quiescent': True},
            'timing_locks': {name: False for name in ('GPU_SUITE_RUNNING.lock', 'PIPELINE_SUITE_RUNNING.lock', 'ADDITIONAL_TIMING_RUNNING.lock')},
            'receipt_sha256': {name: digest(data) for name, data in receipts.items()},
            'source_observed_sha256': source_sha, 'source_manifest_observed_sha256': manifest_sha, 'hostname': 'lpp-cpu-0',
            'standalone_library_observed':{'path':str(BUILD/'native_assoc_v1.so'),'bytes':100000,'sha256':read(results/'build_standalone_native.log')['library_sha256']}}
    export = {'kind': 'public_fr_cpu_matched_v2_final_readonly_export',
              'archive': pointer,
              'remote_receipts': {name: identity(data) for name, data in receipts.items()},
              'source_observed_sha256': source_sha, 'source_manifest_observed_sha256': manifest_sha,
              'before_view': copy.deepcopy(view), 'after_view': copy.deepcopy(view), 'synthetic_fixture': FLAG}
    save(root/'export.json', export)
    return fixture


make_export = make_fixture
repack = refresh
