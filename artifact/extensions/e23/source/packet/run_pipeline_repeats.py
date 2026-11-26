#!/usr/bin/env python3
"""Predeclared fresh-process repetitions of the unchanged public V4 control.

Plan mode is portable and does not create output. Execute mode requires Linux,
one idle whole H100, pinned dependencies and pre-existing public graph inputs.
This program performs no downloads, installs, remote storage or scheduler work.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import importlib.metadata
import importlib.util
from itertools import combinations
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import signal
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
CODE = HERE / 'frozen_src'
BENCHMARK = CODE / 'scripts/seal_revision/pipeline_benchmark_v4.py'
REFERENCES = ('reference_union_stock', 'reference_union_corrected', 'reference_union_compiled')
DATASETS = ('citation2', 'friendster')
WORKERS = (1, 8)
GIB = 1 << 30
PROTOCOL = {
    'name': 'public_v4_fresh_process_repetitions_v1', 'repetitions': 3,
    'runtime_seed': 20261124, 'order_seed': 20261126,
    'datasets': list(DATASETS), 'workers': list(WORKERS), 'arms': list(REFERENCES),
    'batch_size': 32, 'warmup_steps': 8, 'steps': 100, 'measured_graphs': 3200,
    'audit_pairs': 32, 'hops': 2, 'cap': 200, 'prefetch_factor': 1,
    'n_positives': 4000, 'manifest_rows': 8000, 'max_neighbors': 20000000,
    'inner_cpu_memory_gib': 180, 'gpu_allocator_gib': 24,
    'job_timeout_seconds': 3600, 'stock_index_dtype': 'auto',
    'model': 'DGCNN k281 hidden32 layers3 max_z1000 Adam1e-4 features=false',
    'repetition_unit': 'independent fresh benchmark and training processes, fixed workload/model/extraction seed',
    'order': 'randomized dataset-worker blocks; cyclic Latin order balances each arm at every position once per configuration',
    'cache': 'OS caches are not evicted; on-disk Numba cache is shared, compilation and warmup stay outside the window',
    'instrumentation': 'unchanged inner V4 parent sampling at approximately0.25s; reviewed outer V3 process-tree guard at approximately2s; no added per-query observations or overhead subtraction',
    'scope': 'same-law finite-window execution variability; not quality, full-upstream SEAL, cold-I/O or time-to-quality',
    'reporting': 'all three wall times and descriptive mean/sample-SD/median/range; paired ratios by block; no p-values or population CIs',
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 32 << 20,
            'bounded regular JSON required: ' + str(path))
    return json.loads(path.read_text(), parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    temporary.replace(path)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def verify_sources():
    pins = read(HERE / 'SOURCE_PINS.json')
    for name, expected in {**pins['frozen_sources'], **{r['path']: r for r in pins['workloads'].values()}}.items():
        path = HERE / name
        require(path.is_file() and not path.is_symlink() and path.stat().st_size == expected['bytes']
                and sha(path) == expected['sha256'], 'frozen source/workload differs: ' + name)
    manifest = HERE / 'PACKET_MANIFEST.json'
    if manifest.exists():
        files = read(manifest)['files']
        observed = set()
        for path in HERE.rglob('*'):
            require(not path.is_symlink(), 'symlink inside frozen source packet')
            if path.is_file() and path != manifest:
                observed.add(str(path.relative_to(HERE)))
        require(observed == set(files), 'frozen source packet file inventory differs')
        for name, expected in files.items():
            path = HERE / name
            require(path.is_file() and not path.is_symlink() and path.stat().st_size == expected['bytes']
                    and sha(path) == expected['sha256'], 'prepared source packet differs: ' + name)
    return pins


def common_argv(args, dataset):
    graph = args.citation2_root / 'train_csr' if dataset == 'citation2' else args.friendster_root
    pairs = args.out / 'manifests' / (dataset + '_pairs.npy')
    argv = [sys.executable, '-u', str(BENCHMARK), '--graph-path', str(graph), '--file-endian', 'little',
            '--pairs-path', str(pairs), '--n-positives', '4000', '--negative-kind', 'uniform',
            '--batch-sizes', '32', '--steps', '100', '--warmup-steps', '8', '--prefetch-factor', '1',
            '--hops', '2', '--cap', '200', '--max-neighbors', '20000000', '--seed', '20261124',
            '--memory-budget-gib', '180', '--gpu-memory-budget-gib', '24', '--stock-index-dtype', 'auto',
            '--timeout-seconds', '3600', '--device', 'cuda']
    if dataset == 'citation2':
        argv += ['--train-pairs-csr', str(args.citation2_root / 'train_pairs_csr')]
    return argv


def make_plan(args):
    pins = verify_sources()
    jobs = []
    fixture = [sys.executable, '-u', str(BENCHMARK), '--fixture', '--device', 'cuda', '--workers', '1',
               '--batch-sizes', '2', '--steps', '2', '--warmup-steps', '1', '--cap', '3',
               '--n-positives', '64', '--seed', '20261124', '--memory-budget-gib', '8']
    for name, kind, budget, arms in (
            ('guard_refusal', 'expected_gpu_refusal', '1000000', [REFERENCES[1]]),
            ('gpu_fixture', 'gpu_fixture', '24', list(REFERENCES))):
        output = args.out / 'jobs' / name
        jobs.append(dict(name=name, kind=kind, gpu=True, dataset=None, out=str(output),
                         argv=fixture + ['--gpu-memory-budget-gib', budget, '--arms', *arms, '--out', str(output)]))
    for dataset in DATASETS:
        name = dataset + '_manifest'
        output = args.out / 'jobs' / name
        jobs.append(dict(name=name, kind='manifest', gpu=False, dataset=dataset, out=str(output),
                         pairs=str(args.out / 'manifests' / (dataset + '_pairs.npy')),
                         argv=common_argv(args, dataset) + ['--workers', '1', '8', '--manifest-only', '--out', str(output)]))
    rng = random.Random(PROTOCOL['order_seed'])
    configurations = [(dataset, workers) for dataset in DATASETS for workers in WORKERS]
    bases = {}
    for configuration in configurations:
        order = list(REFERENCES)
        rng.shuffle(order)
        bases[configuration] = order
    blocks = []
    for repeat in range(3):
        order = list(configurations)
        rng.shuffle(order)
        for dataset, workers in order:
            base = bases[(dataset, workers)]
            arms = base[repeat:] + base[:repeat]
            block_id = f'{dataset}_w{workers}_r{repeat + 1}'
            blocks.append(dict(id=block_id, dataset=dataset, workers=workers, repetition=repeat + 1, order=arms))
            for position, arm in enumerate(arms, 1):
                name = block_id + '_' + arm
                output = args.out / 'jobs' / name
                jobs.append(dict(name=name, kind='pipeline', gpu=True, dataset=dataset, workers=workers,
                                 repetition=repeat + 1, position=position, arm=arm, block=block_id, out=str(output),
                                 pairs=str(args.out / 'manifests' / (dataset + '_pairs.npy')),
                                 argv=common_argv(args, dataset) + ['--workers', str(workers), '--arms', arm, '--out', str(output)]))
    require(len(jobs) == 40 and sum(j['kind'] == 'pipeline' for j in jobs) == 36, 'complete40job/36timing grid required')
    return dict(schema_version=1, scope=PROTOCOL['scope'], protocol=PROTOCOL,
                protocol_sha256=hashlib.sha256(json.dumps(PROTOCOL, sort_keys=True).encode()).hexdigest(),
                blocks=blocks, jobs=jobs, historical_workload_pins=pins['workloads'],
                source_manifest_sha256=sha(HERE / 'SOURCE_PINS.json'),
                output_limits=dict(memory_budget_gib=args.memory_budget_gib, max_output_gib=args.max_output_gib,
                                   max_hours=args.max_hours), execute=args.execute)


def clean_exit(exit_data):
    return (type(exit_data.get('returncode')) is int and exit_data.get('returncode') == 0 and not exit_data.get('timed_out')
            and exit_data.get('owned_tree_quiescent') is True and not exit_data.get('orphan_cleanup')
            and exit_data.get('guard_abort_reason') is None)


def validate_job(job, exit_data, pins):
    output = Path(job['out'])
    report = read(output / 'report.json')
    require(report.get('benchmark_version') == 4, 'unchanged V4 report required')
    runs = report.get('runs', [])
    if job['kind'] == 'expected_gpu_refusal':
        logs = '\n'.join(p.read_text() for p in output.rglob('process.log') if p.stat().st_size <= 8 << 20)
        require(exit_data.get('returncode') != 0 and not exit_data.get('timed_out')
                and exit_data.get('owned_tree_quiescent') is True and not exit_data.get('orphan_cleanup')
                and exit_data.get('guard_abort_reason') is None and len(runs) == 1
                and runs[0].get('status') == 'failed' and 'GPU preflight refused' in logs,
                'only the declared impossible GPU-capacity refusal is expected')
        return report
    require(clean_exit(exit_data), 'owned job failed, timed out, was guarded or required orphan cleanup')
    require(len(report.get('source_sha256', {})) == 7, 'all seven scientific runtime bindings required')
    for name, actual in report['source_sha256'].items():
        require(actual == pins['frozen_sources']['frozen_src/' + name]['sha256'], 'scientific source changed')
    if job['kind'] == 'manifest':
        definition = pins['workloads'][job['dataset']]
        manifest = report['manifest']
        require(not runs and manifest['rows'] == 8000 and manifest['positive_occurrences'] == 4000
                and manifest['sha256'] == definition['sha256'] == sha(job['pairs']), 'original public workload differs')
        return report
    require(len(runs) == (3 if job['kind'] == 'gpu_fixture' else 1), 'exact intended arm count required')
    for run in runs:
        require(run.get('status') == 'complete' and run.get('returncode') == 0 and run.get('abort_reason') is None,
                'complete status cannot override nonzero exit or abort_reason')
        require(run['device']['type'] == 'cuda' and run['device']['visible_gpu_count'] == 1,
                'one CUDA device required, no CPU fallback')
        guard = run['gpu_memory']['guard']
        require(guard['status'] == 'configured_before_model_allocation'
                and guard['allocator_budget_bytes'] == 24 * GIB and 0 < guard['allocator_fraction'] <= 1
                and run['gpu_memory']['reserved_peak_gib'] <= 24, '24GiB allocator contract differs')
        require(run['model']['sortpool_k'] == 281 and run['model']['use_feature'] is False,
                'featureless DGCNN k281 required')
        if job['kind'] == 'pipeline':
            cfg, timing = run['config'], run['timing']
            require(cfg['arm'] == job['arm'] and cfg['num_workers'] == job['workers']
                    and cfg['batch_size'] == 32 and cfg['seed'] == 20261124 and cfg['audit_pairs'] == 32
                    and cfg['stock_index_dtype'] == 'auto' and timing['measured_steps'] == 100
                    and timing['measured_graphs'] == 3200 and timing['warmup_graphs_excluded'] == 256
                    and math.isfinite(timing['steady_seconds']) and timing['steady_seconds'] > 0
                    and abs(timing['graphs_per_second'] - 3200 / timing['steady_seconds']) < 1e-9
                    and cfg['manifest_sha256'] == pins['workloads'][job['dataset']]['sha256'], 'frozen finite-window protocol differs')
            require(report['runtime_versions'] == pins['original_runtime_versions'], 'original V4 software stack required')
            require(run['model']['initial_state_sha256'] == pins['original_initial_model_state_sha256'], 'initial model differs')
            require([a['index'] for a in run['audits']] == list(range(32)), 'all32 warmup-prefix audits required')
            if job['arm'] == REFERENCES[0]:
                require(run['stock_physical_route']['measured_temporary_row_dtype_counts'] == {'int32': 3200},
                        'measured native-auto stock row width must remain int32')
    if job['kind'] == 'gpu_fixture':
        audits = report['equality_audits']
        require(len(audits) == 3 and all(a['same_nodes_edges_drnl_and_initial_model'] for a in audits)
                and {(a['reference_arm'], a['candidate_arm']) for a in audits} == set(combinations(REFERENCES, 2)),
                'three-arm GPU fixture equality failed')
    return report


def compare_block(jobs, reports):
    result = []
    by_arm = {j['arm']: reports.get(j['name']) for j in jobs}
    for left_arm, right_arm in combinations(REFERENCES, 2):
        a, b = by_arm[left_arm], by_arm[right_arm]
        require(a is not None and b is not None, 'all three successful arms required for block comparison')
        a, b = a['runs'][0], b['runs'][0]
        require(a['audits'] == b['audits'] and a['model']['initial_state_sha256'] == b['model']['initial_state_sha256']
                and a['config']['manifest_sha256'] == b['config']['manifest_sha256'], 'matched-prefix equality failed')
        result.append(dict(block=jobs[0]['block'], reference_arm=left_arm, candidate_arm=right_arm,
                           audited_pairs=32, same_nodes_edges_drnl_and_initial_model=True,
                           scope='excluded warmup prefix only; not every measured graph'))
    return result


def summarize(plan, reports):
    checks = []
    for block in plan['blocks']:
        jobs = [j for j in plan['jobs'] if j.get('block') == block['id']]
        checks += compare_block(jobs, reports)
    rows = []
    ratios = []
    for dataset in DATASETS:
        for workers in WORKERS:
            arm_values = {}
            for arm in REFERENCES:
                jobs = sorted((j for j in plan['jobs'] if j.get('dataset') == dataset
                               and j.get('workers') == workers and j.get('arm') == arm), key=lambda j: j['repetition'])
                values = [reports[j['name']]['runs'][0]['timing']['steady_seconds'] for j in jobs]
                require(len(values) == 3 and all(math.isfinite(x) and x > 0 for x in values), 'all three measured windows required')
                arm_values[arm] = values
                rows.append(dict(dataset=dataset, workers=workers, arm=arm, steady_seconds=values,
                                 mean=statistics.mean(values), sample_sd=statistics.stdev(values),
                                 median=statistics.median(values), minimum=min(values), maximum=max(values)))
            for left, right in ((REFERENCES[0], REFERENCES[1]), (REFERENCES[0], REFERENCES[2]), (REFERENCES[1], REFERENCES[2])):
                values = [a / b for a, b in zip(arm_values[left], arm_values[right])]
                ratios.append(dict(dataset=dataset, workers=workers, numerator=left, denominator=right,
                                   paired_time_ratios=values, median=statistics.median(values),
                                   mean=statistics.mean(values), sample_sd=statistics.stdev(values)))
    require(len(checks) == 36 and len(rows) == 12 and len(ratios) == 12, 'complete declared result grid required')
    return dict(status='complete', timing_jobs=36, reference_prefix_comparisons=checks, rows=rows,
                paired_time_ratios=ratios, scope=PROTOCOL['scope'], reporting=PROTOCOL['reporting'])


class Runner:
    def __init__(self, args, plan):
        self.args, self.plan = args, plan
        self.pins = verify_sources()
        self.supervision = load_module(HERE / 'support/owned_supervision.py', 'pipeline_repeat_owned_supervision')
        self.memory = load_module(CODE / 'scripts/seal_revision/memory_benchmark_v3.py', 'pipeline_repeat_memory_guard')
        self.helpers = load_module(CODE / 'experiment_configs/seal_revision_2026-11/run_pipeline_suite_v4.py', 'frozen_v4_helpers')
        self.records, self.hardware, self.host, self.snapshots = [], None, None, {}
        self.env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                        NUMBA_NUM_THREADS='1', PYTHONPATH=str(CODE), PYTHONDONTWRITEBYTECODE='1',
                        NUMBA_CACHE_DIR=str(args.out / 'numba_cache'), PYTHONNOUSERSITE='1',
                        WANDB_MODE='disabled', HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false')

    def state(self, phase, **extra):
        save(self.args.out / 'status.json', dict(phase=phase, unix=time.time(), pid=os.getpid(), **extra))

    def check_output(self):
        size = 0
        for p in self.args.out.rglob('*'):
            require(not p.is_symlink(), 'output symlink refused')
            if p.is_file():
                size += p.stat().st_size
        require(size <= self.args.max_output_gib * GIB, 'output budget exceeded')
        require(shutil.disk_usage(self.args.out).free >= GIB, 'one GiB scratch reserve required')

    def freeze_inputs(self):
        definitions = {}
        c2 = read(HERE / 'pins/citation2_prepared_expected.json')['files']
        allowed = {'num_nodes.txt', 'train_csr/edge_starts.bin', 'train_csr/edge_ends.bin',
                   'train_pairs_csr/edge_starts.bin', 'train_pairs_csr/edge_ends.bin'}
        definitions.update({str(self.args.citation2_root / r['path']): r for r in c2 if r['path'] in allowed})
        fr = read(HERE / 'pins/friendster_full_expected.json')['files']
        definitions.update({str(self.args.friendster_root / name): row for name, row in fr.items()})
        require(len(definitions) == 8, 'exact five Citation2 plus three Friendster inputs required')
        for name, row in definitions.items():
            p = Path(name)
            require(p.is_file() and not p.is_symlink() and p.stat().st_size == row['bytes'] and sha(p) == row['sha256'],
                    'public numeric input identity differs: ' + name)
            st = p.stat()
            self.snapshots[name] = dict(bytes=st.st_size, mtime_ns=st.st_mtime_ns, inode=st.st_ino,
                                        device=st.st_dev, sha256=row['sha256'])
        save(self.args.out / 'input_identity.json', dict(files=self.snapshots, complete_used_input_hashes=True))
        directory = self.args.out / 'manifests'
        directory.mkdir()
        for dataset, row in self.pins['workloads'].items():
            p = directory / (dataset + '_pairs.npy')
            shutil.copyfile(HERE / row['path'], p)
            require(sha(p) == row['sha256'], 'copied historical workload differs')
        self.host = dict(hostname=platform.node(), boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                         cpu_affinity=sorted(os.sched_getaffinity(0)))
        versions = {n: importlib.metadata.version(n) for n in self.pins['original_runtime_versions']}
        require(versions == self.pins['original_runtime_versions'], 'original frozen software stack required')
        save(self.args.out / 'host.json', dict(**self.host, python=sys.version, versions=versions))

    def assert_inputs(self):
        require(dict(hostname=platform.node(), boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                     cpu_affinity=sorted(os.sched_getaffinity(0))) == self.host, 'host or affinity changed')
        for name, row in self.snapshots.items():
            st = Path(name).stat()
            require((st.st_size, st.st_mtime_ns, st.st_ino, st.st_dev) ==
                    (row['bytes'], row['mtime_ns'], row['inode'], row['device']), 'frozen numeric input changed')
        for dataset, row in self.pins['workloads'].items():
            require(sha(self.args.out / 'manifests' / (dataset + '_pairs.npy')) == row['sha256'], 'workload changed')

    def idle_gpu(self, name):
        h = self.helpers
        inventory = h.permitted_inventory(h.parse_inventory(h.nvidia_query('--query-gpu=index,uuid,name,memory.total,driver_version')), self.env)
        target = h.select_target(inventory, self.env)
        hardware = dict(permitted_inventory=inventory, target=target)
        require(self.hardware is None or self.hardware == hardware, 'GPU/driver inventory changed')
        self.hardware = hardware
        observations = []
        for i in range(2):
            apps = h.parse_compute_apps(h.nvidia_query('--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory'))
            require(not any(a['gpu_uuid'] in {r['uuid'] for r in inventory} for a in apps), 'permitted GPU is busy; no foreign signals sent')
            observations.append(dict(unix=time.time(), busy_apps=[]))
            if i == 0:
                require(time.monotonic() + 15 < self.deadline, 'suite deadline reached')
                time.sleep(15)
        save(self.args.out / 'hardware.json', self.hardware)
        self.env['CUDA_VISIBLE_DEVICES'] = target['uuid']
        return observations

    def execute(self, job):
        verify_sources()
        self.assert_inputs()
        self.check_output()
        require(time.monotonic() < self.deadline - 30, 'total execution deadline reached')
        idle = self.idle_gpu(job['name']) if job['gpu'] else []
        log = self.args.out / (job['name'] + '.log')
        abort, last, failures = [None], [0.0], [0]
        self.state('running', job=job['name'], completed_jobs=len(self.records))
        def clock():
            return time.monotonic() + (1e9 if abort[0] else 0)
        def groups(owner):
            observed = self.supervision.owned_groups(owner)
            if time.monotonic() - last[0] >= 2:
                last[0] = time.monotonic()
                try:
                    sample = self.memory.sample_tree(owner)
                    decision = self.memory.budget_decision(sample, int(self.args.memory_budget_gib * GIB), self.memory.cgroup_memory())
                    abort[0] = abort[0] or decision['abort_reason']
                    self.check_output()
                    failures[0] = 0
                except Exception as exc:
                    failures[0] += 1
                    if failures[0] >= 3:
                        abort[0] = 'three resource sampling failures: ' + type(exc).__name__
            if log.exists() and log.stat().st_size > 8 << 20:
                abort[0] = 'stdout exceeds8MiB'
            return observed
        def popen(*a, **kw):
            kw['env'] = self.env.copy()
            return subprocess.Popen(*a, **kw)
        result = self.supervision.run_owned(job['argv'], log, cwd=CODE,
            timeout=min(3600, self.deadline - time.monotonic() - 10), clock=clock,
            popen=popen, groups_reader=groups)
        result.update(guard_abort_reason=abort[0], gpu_idle_observations=idle, guard_policy=self.memory.GUARD_POLICY)
        validation_error, report = None, None
        try:
            report = validate_job(job, result, self.pins)
        except Exception as exc:
            validation_error = type(exc).__name__ + ': ' + str(exc)
        record = dict(name=job['name'], kind=job['kind'], exit=result, validation_error=validation_error,
                      report_sha256=sha(Path(job['out']) / 'report.json') if (Path(job['out']) / 'report.json').exists() else None)
        self.records.append(record)
        save(self.args.out / (job['name'] + '.exit.json'), record)
        save(self.args.out / 'job_records.json', self.records)
        if job['kind'] != 'pipeline':
            require(validation_error is None, 'required fixture/manifest failed: ' + str(validation_error))
        return report if validation_error is None else None

    def cleanup(self):
        for sig in (signal.SIGTERM, signal.SIGKILL):
            self.supervision.signal_confirmed(self.supervision.owned_groups(os.getpid()), sig)
            until = time.monotonic() + 5
            while time.monotonic() < until and self.supervision.owned_groups(os.getpid()):
                time.sleep(.1)
        return not self.supervision.owned_groups(os.getpid())

    def run(self):
        require(sys.platform.startswith('linux'), 'actual execution requires Linux; plan mode is portable')
        require((HERE / 'PACKET_MANIFEST.json').is_file(), 'frozen complete packet manifest required')
        require(not self.args.out.exists(), 'fresh output required; no automatic overwrite or resume')
        self.args.out.parent.mkdir(parents=True, exist_ok=True)
        require(shutil.disk_usage(self.args.out.parent).free >= 4 * GIB, 'four GiB scratch reserve required')
        coordination = self.args.coordination_dir or self.args.out.parent
        coordination.mkdir(parents=True, exist_ok=True)
        lock = (coordination / 'PUBLIC_SYSTEMS_RUNNING.lock').open('a+')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.supervision.enable_subreaper()
        self.args.out.mkdir()
        self.deadline = time.monotonic() + self.args.max_hours * 3600
        save(self.args.out / 'plan.json', self.plan)
        previous = signal.getsignal(signal.SIGTERM)
        def stop(signum, frame):
            raise KeyboardInterrupt('SIGTERM received by private repeat controller')
        signal.signal(signal.SIGTERM, stop)
        success, failure = False, None
        try:
            self.state('verifying_inputs')
            self.freeze_inputs()
            reports = {}
            for job in self.plan['jobs']:
                report = self.execute(job)
                if report is not None:
                    reports[job['name']] = report
            require(all(r['validation_error'] is None for r in self.records), 'one or more declared jobs failed; all failures retained')
            summary = summarize(self.plan, reports)
            save(self.args.out / 'scientific_summary.json', summary)
            success = True
        except BaseException as exc:
            failure = dict(type=type(exc).__name__, message=str(exc)[:2000])
            raise
        finally:
            quiescent = self.cleanup()
            terminal = dict(controller_finished=quiescent, success=success and quiescent,
                            owned_tree_quiescent=quiescent, records=self.records, failure=failure,
                            scope=PROTOCOL['scope'], completed_unix=time.time(), YT_writes=0)
            save(self.args.out / 'COMPLETE.json', terminal)
            self.state('complete' if terminal['success'] else 'failed', completed_jobs=len(self.records), failure=failure)
            signal.signal(signal.SIGTERM, previous)
            lock.close()
        require(quiescent, 'owned descendants remain after suite')


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--citation2-root', type=Path, required=True)
    p.add_argument('--friendster-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--coordination-dir', type=Path)
    p.add_argument('--memory-budget-gib', type=float, default=180)
    p.add_argument('--max-output-gib', type=float, default=2)
    p.add_argument('--max-hours', type=float, default=6)
    args = p.parse_args(argv)
    for name in ('citation2_root', 'friendster_root', 'out', 'coordination_dir'):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    require(all(math.isfinite(v) and v > 0 for v in (args.memory_budget_gib, args.max_output_gib, args.max_hours))
            and args.memory_budget_gib <= 180 and args.max_hours <= 6 and args.max_output_gib <= 2,
            'positive budgets bounded by180GiB/2GiB/6h required')
    require(not args.out.is_relative_to(HERE) and not HERE.is_relative_to(args.out), 'output must be disjoint from frozen packet')
    if args.coordination_dir is not None:
        require(not args.coordination_dir.is_relative_to(HERE) and not HERE.is_relative_to(args.coordination_dir),
                'timing lock directory must be disjoint from the source packet')
    for input_path in (args.citation2_root, args.friendster_root):
        require(not args.out.is_relative_to(input_path) and not input_path.is_relative_to(args.out), 'input/output trees must be disjoint')
        if args.coordination_dir is not None:
            require(not args.coordination_dir.is_relative_to(input_path) and not input_path.is_relative_to(args.coordination_dir),
                    'timing lock directory must be disjoint from graph input trees')
    return args


def main(argv=None):
    args = arguments(argv)
    plan = make_plan(args)
    print(json.dumps(plan, indent=2), flush=True)
    if args.execute:
        Runner(args, plan).run()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
