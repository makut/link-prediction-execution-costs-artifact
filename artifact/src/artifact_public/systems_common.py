"""Local, bounded execution shared by the public systems reproduction commands."""
from __future__ import annotations

from contextlib import contextmanager
import csv
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import signal
import shutil
import subprocess
import sys
import time

from . import owned_supervision as supervision

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / 'src'
GIB = 1 << 30


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path, limit=32 << 20):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= limit,
            'bounded regular JSON required: ' + str(path))
    result = json.loads(path.read_text(), parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
    require(isinstance(result, dict), 'JSON object required')
    return result


def save(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('x') as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    temporary.replace(path)


def common_arguments(parser, *, memory=180, output=2, hours=12):
    parser.add_argument('--out', type=Path, required=True, help='Fresh local output directory; failures are retained')
    parser.add_argument('--execute', action='store_true', help='Default only prints the plan and checks source pins')
    parser.add_argument('--coordination-dir', type=Path, help='Shared local timing lock directory; default OUT parent')
    parser.add_argument('--memory-budget-gib', type=float, default=memory)
    parser.add_argument('--max-output-gib', type=float, default=output)
    parser.add_argument('--max-hours', type=float, default=hours)


def validate_arguments(args, inputs=()):
    require(all(math.isfinite(v) and v > 0 for v in
                (args.memory_budget_gib, args.max_output_gib, args.max_hours)) and args.max_hours <= 12,
            'finite positive budgets and a total deadline of at most12hours required')
    args.out = args.out.resolve()
    require(not args.out.exists(), 'fresh output required; preserve existing and failed runs')
    require(not args.out.is_relative_to(ROOT), 'results must be outside the source artifact')
    for value in inputs:
        path = Path(value).resolve()
        require(not args.out.is_relative_to(path) and not path.is_relative_to(args.out),
                'output and input trees must be disjoint')


def coordinator_limits(args):
    return {'memory_budget_gib': args.memory_budget_gib, 'max_output_gib': args.max_output_gib,
            'total_hours': args.max_hours,
            'coordination_dir': str((args.coordination_dir or args.out.parent).resolve()),
            'scope': 'outer local supervision limits, separate from unchanged per-job preflight/allocator limits'}


def host_identity():
    models = []
    cpuinfo = Path('/proc/cpuinfo')
    if cpuinfo.is_file():
        with cpuinfo.open() as stream:
            text = stream.read((2 << 20) + 1)
        require(len(text) <= 2 << 20, 'bounded CPU description required')
        models = sorted({line.split(':', 1)[1].strip() for line in text.splitlines() if line.startswith('model name')})
    boot = Path('/proc/sys/kernel/random/boot_id')
    return {'hostname': platform.node(), 'platform': platform.platform(), 'cpu_models': models,
            'boot_id': boot.read_text().strip() if boot.is_file() else None,
            'cpu_affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None}


def frozen_sources():
    pins = read(ROOT / 'metadata/SYSTEMS_FROZEN_INPUTS.json')['files']
    for name, expected in pins.items():
        path = ROOT / name
        require(path.resolve().is_relative_to(ROOT) and path.is_file() and not path.is_symlink()
                and sha(path) == expected, 'frozen source/reference input differs: ' + name)
    # New public orchestration is pinned separately so it cannot be mistaken
    # for the historical measured source. Verify both source classes at runtime.
    selection = read(ROOT / 'metadata/SYSTEMS_SOURCE_SELECTION.json')['sources']
    for name, entry in selection.items():
        path = ROOT / name
        require(path.resolve().is_relative_to(ROOT) and path.is_file() and not path.is_symlink()
                and sha(path) == entry['packaged_sha256'], 'public systems source differs: ' + name)
    return pins


def verify_inputs(directory, expected, names=None):
    from .verify_payloads import verify
    definition = read(ROOT / expected)
    if names is not None:
        entries = definition['files']
        if isinstance(entries, dict):
            entries = [dict(path=name, **value) for name, value in entries.items()]
        definition = {'files': [entry for entry in entries if entry['path'] in names],
                      'scope': 'only explicitly used numeric inputs; not a new full-dataset reproduction claim'}
        require({entry['path'] for entry in definition['files']} == set(names), 'exact input allowlist required')
    return verify(Path(directory), definition)


def clean_exit(value):
    return (type(value.get('returncode')) is int and value.get('returncode') == 0 and value.get('timed_out') is False
            and value.get('owned_tree_quiescent') is True and not value.get('orphan_cleanup')
            and value.get('guard_abort_reason') is None)


def load_file(path, name):
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


def gpu_inventory():
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,driver_version',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=30, check=True)
    rows = []
    for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
        require(len(row) == 5 and row[0].isdigit() and row[1].startswith('GPU-'), 'whole GPU inventory required')
        rows.append(dict(index=int(row[0]), uuid=row[1], name=row[2], memory_total_mib=float(row[3]), driver_version=row[4]))
    visible = os.environ.get('NVIDIA_VISIBLE_DEVICES', 'all')
    if visible and visible != 'all':
        tokens = visible.split(',')
        require(not any(v.startswith('MIG-') for v in tokens), 'whole GPU protocol excludes MIG')
        rows = [r for r in rows if r['uuid'] in tokens or str(r['index']) in tokens]
        require(len(rows) == len(set(tokens)), 'permitted GPU inventory differs')
    require(rows, 'at least one permitted whole GPU required')
    return sorted(rows, key=lambda r: r['index'])


def select_gpu(rows, *, h100=False):
    selected = rows[0]
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')[0]
    if visible and visible != 'all':
        selected = next((r for i, r in enumerate(rows)
                         if visible == r['uuid'] or (visible.isdigit() and int(visible) == i)), None)
        require(selected is not None, 'CUDA_VISIBLE_DEVICES does not resolve to a permitted GPU')
    require(not h100 or 'H100' in selected['name'], 'the frozen V4 timing protocol requires H100')
    return selected


class Execution:
    """A private coordinator: no storage service, deployment, install or foreign signals."""
    def __init__(self, args, plan, *, inputs=(), min_disk_gib=4, gpu=False, h100=False):
        self.args, self.plan, self.inputs = args, plan, inputs
        self.min_disk = min_disk_gib * GIB
        self.gpu, self.h100 = gpu, h100
        self.records, self.failure = [], None
        self.hardware = None
        self.previous_sigterm = None

    def install_signals(self):
        self.previous_sigterm = signal.getsignal(signal.SIGTERM)
        def terminate(signum, frame):
            raise KeyboardInterrupt('SIGTERM received by public coordinator')
        signal.signal(signal.SIGTERM, terminate)

    def restore_signals(self):
        if self.previous_sigterm is not None:
            signal.signal(signal.SIGTERM, self.previous_sigterm)
            self.previous_sigterm = None

    def __enter__(self):
        require(sys.platform.startswith('linux'), 'execute on a Linux compute host; plan mode is portable')
        frozen_sources()
        validate_arguments(self.args, self.inputs)
        self.args.out.parent.mkdir(parents=True, exist_ok=True)
        require(shutil.disk_usage(self.args.out.parent).free >= self.min_disk, 'insufficient free local scratch disk')
        coordination = (self.args.coordination_dir or self.args.out.parent).resolve()
        coordination.mkdir(parents=True, exist_ok=True)
        self.lock = (coordination / 'PUBLIC_SYSTEMS_RUNNING.lock').open('a+')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.install_signals()
            supervision.enable_subreaper()
            self.deadline = time.monotonic() + self.args.max_hours * 3600
            self.args.out.mkdir()
            save(self.args.out / 'plan.json', self.plan)
            versions = {}
            for name in ('numpy', 'scipy', 'numba', 'torch', 'torch-geometric', 'torch-sparse', 'transformers'):
                try:
                    versions[name] = importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError:
                    versions[name] = None
            save(self.args.out / 'host.json', {'python': sys.version, **host_identity(),
                'cpu_affinity_count': len(os.sched_getaffinity(0)),
                'dependency_versions': versions, 'scope': 'actual local rerun host; timings are not expected to be byte-identical across hosts'})
            self.memory = load_file(CODE / 'scripts/seal_revision/memory_benchmark_v3.py', 'public_memory_v3')
            return self
        except BaseException as exc:
            try:
                if self.args.out.is_dir():
                    save(self.args.out / 'COMPLETE.json', {'controller_finished': True, 'success': False,
                        'owned_tree_quiescent': True, 'failure': {'type': type(exc).__name__, 'message': str(exc)[:500]},
                        'scope': 'failed local admission; no scientific job was launched'})
            finally:
                self.restore_signals()
                self.lock.close()
            raise

    def idle_gpu(self):
        rows = gpu_inventory()
        selected = select_gpu(rows, h100=self.h100)
        current = {'permitted_inventory': rows, 'target': selected}
        require(self.hardware is None or self.hardware == current, 'GPU inventory changed')
        self.hardware = current
        observations = []
        for _ in range(2):
            result = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory',
                                     '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=30, check=True)
            apps = list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))
            require(all(len(r) == 4 and r[0].startswith('GPU-') and r[1].isdigit() for r in apps), 'unrecognized compute-app inventory')
            require(not any(r[0] in {v['uuid'] for v in rows} for r in apps), 'timing window is busy; no foreign process is signaled')
            observations.append({'unix': time.time(), 'busy_apps': []})
            if len(observations) < 2:
                require(time.monotonic() + 15 < self.deadline, 'total deadline reached during GPU idle observations')
                time.sleep(15)
        save(self.args.out / 'hardware.json', self.hardware)
        return selected['uuid'], observations

    def run(self, job):
        frozen_sources()
        require(time.monotonic() < self.deadline - 15, 'total execution deadline reached')
        self.check_disk()
        name = job['name']
        require(name.replace('_', '').isalnum(), 'safe job name required')
        log = self.args.out / (name + '.log')
        device, idle = ('', [])
        if job.get('gpu'):
            device, idle = self.idle_gpu()
        abort, errors, last = [None], [0], [0.]
        def clock():
            return time.monotonic() + (1e9 if abort[0] else 0)
        def groups(owner):
            found = supervision.owned_groups(owner)
            if time.monotonic() - last[0] >= 2:
                last[0] = time.monotonic()
                try:
                    sample = self.memory.sample_tree(owner)
                    decision = self.memory.budget_decision(sample, int(self.args.memory_budget_gib * GIB), self.memory.cgroup_memory())
                    abort[0] = abort[0] or decision['abort_reason']
                    self.check_disk()
                    errors[0] = 0
                except Exception as exc:
                    errors[0] += 1
                    if errors[0] >= 3:
                        abort[0] = 'three consecutive local resource sampling failures: ' + type(exc).__name__
            if log.exists() and log.stat().st_size > 8 << 20:
                abort[0] = 'stdout exceeds8MiB'
            return found
        def popen(*a, **kw):
            kw['env'].update(CUDA_VISIBLE_DEVICES=device, WANDB_MODE='disabled', HF_HUB_OFFLINE='1',
                             HF_DATASETS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', PYTHONNOUSERSITE='1',
                             PYTHONPATH=str(CODE), PYTHONDONTWRITEBYTECODE='1',
                             NUMBA_CACHE_DIR=str(self.args.out / (name + '_numba_cache')),
                             HF_HOME=str(self.args.out / (name + '_hf_cache')))
            return subprocess.Popen(*a, **kw)
        try:
            result = supervision.run_owned(job['argv'], log, cwd=job.get('cwd', CODE),
                timeout=min(job.get('timeout_seconds', 1800), self.deadline - time.monotonic() - 10),
                clock=clock, popen=popen, groups_reader=groups)
        except BaseException:
            self.cleanup()
            raise
        result.update(guard_abort_reason=abort[0], gpu_idle_observations=idle,
                      guard_policy=self.memory.GUARD_POLICY, argv=job['argv'])
        save(self.args.out / (name + '.exit.json'), result)
        self.records.append({'name': name, 'exit': result})
        frozen_sources()
        return result

    def check_disk(self):
        size = 0
        for path in self.args.out.rglob('*'):
            require(not path.is_symlink(), 'symlink inside output is refused')
            if path.is_file():
                size += path.stat().st_size
        require(size <= self.args.max_output_gib * GIB, 'declared total output budget exceeded')
        require(shutil.disk_usage(self.args.out).free >= 1 * GIB, 'one GiB scratch reserve required')

    def cleanup(self):
        for sig in (signal.SIGTERM, signal.SIGKILL):
            supervision.signal_confirmed(supervision.owned_groups(os.getpid()), sig)
            until = time.monotonic() + 5
            while time.monotonic() < until and supervision.owned_groups(os.getpid()):
                time.sleep(.1)
        return not supervision.owned_groups(os.getpid())

    def __exit__(self, kind, value, traceback):
        try:
            quiescent = self.cleanup()
            success = kind is None and quiescent and all(clean_exit(r['exit']) or
                (r.get('expected_guard_refusal') is True and r['exit'].get('returncode') != 0
                 and not r['exit'].get('timed_out') and r['exit'].get('guard_abort_reason') is None
                 and not r['exit'].get('orphan_cleanup') and r['exit'].get('owned_tree_quiescent') is True)
                for r in self.records)
            save(self.args.out / 'COMPLETE.json', {'controller_finished': quiescent, 'success': success,
                 'owned_tree_quiescent': quiescent, 'jobs': self.records,
                 'failure': None if kind is None else {'type': kind.__name__, 'message': str(value)[:500]},
                 'new_remote_experiments': 0, 'YT_writes': 0,
                 'scope': self.plan['scope'], 'completed_unix': time.time()})
        finally:
            self.restore_signals()
            self.lock.close()
        if kind is None:
            require(success, 'owned cleanup or child status prevents scientific success')
        return False
