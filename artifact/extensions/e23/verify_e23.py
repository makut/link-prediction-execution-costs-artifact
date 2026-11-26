#!/usr/bin/env python3
"""Independent, bounded stdlib-only E23 source/archival verifier.

No benchmark modules are imported and no graph, model, GPU, network, process or
lock operations are performed. SOURCE_ONLY is distinct from recorded PASS.
"""
from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import random
import struct
import sys
import tarfile

HERE = Path(__file__).resolve().parent
PACKET = HERE / 'source/packet'
MIB, GIB = 1 << 20, 1 << 30
MAX_COMPRESSED, MAX_RAW, MAX_MEMBER = 128 * MIB, 384 * MIB, 32 * MIB
MAX_RETAINED, MAX_FILES, MAX_JSON_LINE = 96 * MIB, 4096, MIB
PACKET_SHA = '16b14211782d5c7c765e52ec36a533131e3bf559633229da19a874cdfb28c68f'
PROTOCOL_SHA = '6587e2055c5a77f4ffbe075fd7d70a6fcf648ad075e4d4cb8aa6574ae54061a1'
ARMS = ('reference_union_stock', 'reference_union_corrected', 'reference_union_compiled')
DATASETS, WORKERS = ('citation2', 'friendster'), (1, 8)
SOURCE_KEYS = ('scripts/seal_revision/pipeline_benchmark_v4.py',
               'scripts/seal_revision/exact_benchmark.py',
               'scripts/seal_revision/memory_benchmark.py', 'seal/models.py',
               'seal/utils.py', 'seal/graph_csr_kernels.py', 'seal/graph_csr_extraction.py')
MODEL = dict(name='DGCNN', sortpool_k=281, hidden_channels=32, num_layers=3,
             max_z=1000, learning_rate=1e-4, optimizer='Adam', use_feature=False,
             node_embedding=False)


class VerificationError(ValueError):
    pass


def need(condition, message):
    if not condition:
        raise VerificationError(message)


def reject_constant(value):
    raise VerificationError('Nonfinite JSON constant: ' + value)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        need(key not in result, 'Duplicate JSON key: ' + key)
        result[key] = value
    return result


def decode(data):
    need(len(data) <= MAX_MEMBER, 'JSON size cap')
    return json.loads(data, parse_constant=reject_constant, object_pairs_hook=unique_object)


def bounded_file(path, maximum=MAX_MEMBER):
    path = Path(path)
    need(path.is_file() and not path.is_symlink() and path.stat().st_size <= maximum,
         'Bounded regular file required: ' + str(path))
    return path


def digest_file(path, maximum=MAX_MEMBER):
    path = bounded_file(path, maximum)
    h, count = hashlib.sha256(), 0
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(MIB), b''):
            count += len(block)
            need(count <= maximum, 'Growing input exceeds cap')
            h.update(block)
    need(count == path.stat().st_size, 'Input changed while hashing')
    return dict(bytes=count, sha256=h.hexdigest())


def read_json(path):
    return decode(bounded_file(path).read_bytes())


def safe_name(name):
    path = PurePosixPath(name)
    need(isinstance(name, str) and name and not path.is_absolute()
         and '..' not in path.parts and '\\' not in name
         and str(path) == name and not name.endswith('/'), 'Unsafe member path')
    return name


def sha_hex(value):
    return isinstance(value, str) and len(value) == 64 and all(x in '0123456789abcdef' for x in value)


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def nonnegative(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def same_number(left, right):
    return (type(left) in (int, float) and type(right) in (int, float)
            and math.isfinite(left) and math.isfinite(right)
            and math.isclose(left, right, rel_tol=2e-12, abs_tol=1e-10))


def matching(actual, expected, label):
    if isinstance(expected, dict):
        need(isinstance(actual, dict) and set(actual) == set(expected), label + ': keys differ')
        for key in expected:
            matching(actual[key], expected[key], label + '.' + key)
    elif isinstance(expected, list):
        need(isinstance(actual, list) and len(actual) == len(expected), label + ': length differs')
        for index, value in enumerate(expected):
            matching(actual[index], value, label + '[' + str(index) + ']')
    elif type(expected) is float:
        need(same_number(actual, expected), label + ': numeric mismatch')
    else:
        need(type(actual) is type(expected) and actual == expected, label + ': mismatch')


def npy_rows(data, expected_rows):
    """Parse only the known, small little-endian int64 3-column public workloads."""
    need(data[:8] == b'\x93NUMPY\x01\x00' and len(data) >= 10, 'Expected NPY1 workload')
    size = struct.unpack('<H', data[8:10])[0]
    need(size <= 4096 and len(data) >= 10 + size, 'NPY header cap/truncation')
    header = ast.literal_eval(data[10:10 + size].decode('ascii'))
    need(header == dict(descr='<i8', fortran_order=False, shape=(expected_rows, 3)),
         'Only frozen int64[N,3] workloads permitted')
    need(len(data) == 10 + size + expected_rows * 24, 'NPY byte/row mismatch')
    rows = list(struct.iter_unpack('<qqq', data[10 + size:]))
    need(all(a >= 0 and b >= 0 and a != b and y in (0, 1) for a, b, y in rows),
         'Invalid workload endpoint/label')
    return rows


def manifest_rows(manifest):
    rows = manifest['files']
    if isinstance(rows, list):
        need(len({r['path'] for r in rows}) == len(rows), 'Duplicate source manifest path')
        rows = {r['path']: r for r in rows}
    need(isinstance(rows, dict) and len(rows) < MAX_FILES, 'Bounded source manifest required')
    for name, row in rows.items():
        safe_name(name)
        need(type(row['bytes']) is int and row['bytes'] >= 0 and sha_hex(row['sha256']),
             'Invalid manifest payload identity')
    return rows


def local_sources():
    need(digest_file(PACKET / 'PACKET_MANIFEST.json')['sha256'] == PACKET_SHA,
         'Immutable packet manifest changed')
    manifest = read_json(PACKET / 'PACKET_MANIFEST.json')
    rows = manifest_rows(manifest)
    observed = set()
    for path in PACKET.rglob('*'):
        need(not path.is_symlink(), 'Source symlink refused')
        if path.is_file() and path.name != 'PACKET_MANIFEST.json':
            observed.add(str(path.relative_to(PACKET)))
    need(observed == set(rows), 'Immutable packet inventory changed')
    for name, pin in rows.items():
        need(digest_file(PACKET / name) == {k: pin[k] for k in ('bytes', 'sha256')},
             'Source/workload changed: ' + name)
    pins, protocol = read_json(PACKET / 'SOURCE_PINS.json'), read_json(PACKET / 'PROTOCOL.json')
    need(hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest() == PROTOCOL_SHA,
         'Semantic protocol hash changed')
    for name, pin in pins['frozen_sources'].items():
        need(digest_file(PACKET / name) == {k: pin[k] for k in ('bytes', 'sha256')},
             'Original frozen source differs: ' + name)
    workloads = {}
    for dataset, pin in pins['workloads'].items():
        data = bounded_file(PACKET / pin['path'], MIB).read_bytes()
        need(len(data) == pin['bytes'] and hashlib.sha256(data).hexdigest() == pin['sha256'],
             'Frozen public workload differs')
        workloads[dataset] = npy_rows(data, 8000)
        need(sum(row[2] for row in workloads[dataset]) == 4000, 'Public positive count differs')
    for path in HERE.rglob('*.py'):
        ast.parse(bounded_file(path).read_text(), filename=str(path), feature_version=(3, 10))
    return pins, protocol, workloads, rows


def schedule():
    """Rebuild the protocol order from its declared seeds; do not import runner."""
    rng = random.Random(20261126)
    configs = [(dataset, workers) for dataset in DATASETS for workers in WORKERS]
    bases = {}
    for config in configs:
        bases[config] = list(ARMS)
        rng.shuffle(bases[config])
    blocks, jobs = [], []
    for name, kind in (('guard_refusal', 'expected_gpu_refusal'), ('gpu_fixture', 'gpu_fixture'),
                       ('citation2_manifest', 'manifest'), ('friendster_manifest', 'manifest')):
        jobs.append(dict(name=name, kind=kind))
    for repetition in (1, 2, 3):
        configs_now = configs[:]
        rng.shuffle(configs_now)
        for dataset, workers in configs_now:
            base = bases[(dataset, workers)]
            order = base[repetition - 1:] + base[:repetition - 1]
            block = f'{dataset}_w{workers}_r{repetition}'
            blocks.append(dict(id=block, dataset=dataset, workers=workers,
                               repetition=repetition, order=order))
            for position, arm in enumerate(order, 1):
                jobs.append(dict(name=block + '_' + arm, kind='pipeline', block=block,
                                 dataset=dataset, workers=workers, repetition=repetition,
                                 position=position, arm=arm))
    return blocks, jobs


class Archive:
    """Single bounded streaming pass. No extraction; bounded retained metadata."""
    def __init__(self, path, expected_sha):
        need(sha_hex(expected_sha), 'An independently supplied whole archive SHA256 is required')
        self.identity = digest_file(path, MAX_COMPRESSED)
        need(self.identity['sha256'] == expected_sha, 'Whole archive SHA differs')
        self.infos, self.json, self.npy, self.refusal_logs = {}, {}, {}, set()
        self.jsonl_counts, self.raw_bytes, retained = {}, 0, 0
        class BoundedDecompression:
            def __init__(self, stream):
                self.stream, self.count = stream, 0

            def read(self, size=-1):
                block = self.stream.read(min(MIB, size) if size >= 0 else MIB)
                self.count += len(block)
                need(self.count <= MAX_RAW + 16 * MIB, 'Whole decompressed tar/header cap')
                return block

        with gzip.open(path, 'rb') as compressed, tarfile.open(fileobj=BoundedDecompression(compressed), mode='r|') as tar:
            for member in tar:
                name = safe_name(member.name)
                need(member.isfile() and set(member.pax_headers) <= {'path'}
                     and member.pax_headers.get('path', name) == name and name not in self.infos
                     and len(self.infos) < MAX_FILES and 0 <= member.size <= MAX_MEMBER,
                     'Only bounded unique regular archive members permitted')
                need(name == 'SHA256_MANIFEST.json' or name.startswith('e23/'), 'Archive namespace differs')
                self.raw_bytes += member.size
                need(self.raw_bytes <= MAX_RAW, 'Decompressed archive payload cap')
                keep = name.endswith('.json') or name.endswith('.npy')
                need(not name.endswith('.npy') or name in {
                    'e23/sources/packet/workloads/citation2_pairs.npy',
                    'e23/sources/packet/workloads/friendster_pairs.npy',
                    'e23/results/manifests/citation2_pairs.npy',
                    'e23/results/manifests/friendster_pairs.npy',
                    'e23/results/jobs/gpu_fixture/pairs.npy',
                    'e23/results/jobs/guard_refusal/pairs.npy'}, 'Unexpected NPY/graph member')
                need(not any(part in ('numba_cache', '.venv', 'site-packages') for part in PurePosixPath(name).parts)
                     and not name.endswith(('.bin', '.pt', '.pth', '.ckpt', '.so')), 'Graph/model/environment payload refused')
                data, h, count, tail, found = bytearray(), hashlib.sha256(), 0, b'', False
                pending, line_count = b'', 0
                with tar.extractfile(member) as stream:
                    for block in iter(lambda: stream.read(MIB), b''):
                        count += len(block)
                        need(count <= member.size, 'Growing/truncated archive member')
                        h.update(block)
                        if keep:
                            retained += len(block)
                            need(retained <= MAX_RETAINED, 'Retained metadata memory cap')
                            data.extend(block)
                        if name.endswith('.log'):
                            found = found or b'GPU preflight refused' in tail + block
                            tail = block[-64:]
                        if name.endswith('.jsonl'):
                            pending += block
                            while b'\n' in pending:
                                line, pending = pending.split(b'\n', 1)
                                need(len(line) <= MAX_JSON_LINE, 'JSONL line cap')
                                if line:
                                    decode(line)
                                    line_count += 1
                            need(len(pending) <= MAX_JSON_LINE, 'JSONL line cap')
                need(count == member.size, 'Truncated archive member')
                self.infos[name] = dict(bytes=count, sha256=h.hexdigest())
                if found:
                    self.refusal_logs.add(name)
                if name.endswith('.jsonl'):
                    if pending:
                        decode(pending)
                        line_count += 1
                    self.jsonl_counts[name] = line_count
                elif name.endswith('.json'):
                    self.json[name] = decode(data)
                elif name.endswith('.npy'):
                    need(len(data) <= MIB, 'Small workload NPY cap')
                    self.npy[name] = bytes(data)
        need('SHA256_MANIFEST.json' in self.json, 'Whole-member manifest missing')
        manifest = self.json['SHA256_MANIFEST.json']
        expected = manifest_rows(manifest)
        need(manifest.get('graphs_included') is False and manifest.get('numba_cache_included') is False,
             'No-graph/no-cache archive declaration required')
        need({k: v for k, v in self.infos.items() if k != 'SHA256_MANIFEST.json'} == expected,
             'Every whole member must match the archive manifest')
        need(digest_file(path, MAX_COMPRESSED) == self.identity, 'Archive changed during verification')

    def get(self, name):
        need(name in self.json, 'Missing required JSON: ' + name)
        return self.json[name]

    def info(self, name):
        need(name in self.infos, 'Missing required member: ' + name)
        return self.infos[name]


def source_tree(archive, prefix, manifest_name, expected_sha, exact_inventory=True):
    manifest_path = prefix + manifest_name
    need(archive.info(manifest_path)['sha256'] == expected_sha, 'Bound source manifest SHA differs')
    rows = manifest_rows(archive.get(manifest_path))
    for name, pin in rows.items():
        need(archive.info(prefix + name) == {k: pin[k] for k in ('bytes', 'sha256')},
             'Archived source differs: ' + prefix + name)
    if exact_inventory:
        actual = {name[len(prefix):] for name in archive.infos if name.startswith(prefix)}
        need(actual == set(rows) | {manifest_name}, 'Archived source inventory differs')
    return rows


def check_clean_exit(value, expected_code=0, guard=True):
    need(type(value.get('returncode')) is int and
         (value['returncode'] == 0 if expected_code == 0 else value['returncode'] != 0), 'Actual integer exit differs')
    need(value.get('timed_out') is False and value.get('orphan_cleanup') is False
         and value.get('owned_tree_quiescent') is True and not value.get('owned_pgids_signaled'),
         'Timeout, orphan cleanup, signals or live owned process tree')
    if guard:
        need('guard_abort_reason' in value and value['guard_abort_reason'] is None, 'Outer resource guard abort')
    need(type(value.get('pid')) is int and value['pid'] > 0, 'Actual child PID required')
    need(positive(value.get('timeout_seconds')), 'Bounded supervision timeout required')


def validate_audits(audits, workload, count):
    need(isinstance(audits, list) and len(audits) == count, 'Complete audited prefix required')
    for index, audit in enumerate(audits):
        need(type(audit.get('index')) is int and audit['index'] == index, 'Audit order/index differs')
        need(audit['pair'] == list(workload[index][:2]) and audit['label'] == workload[index][2],
             'Audit pair/label differs from immutable public workload')
        need(all(sha_hex(audit.get(key)) for key in ('nodes_sha256', 'edges_sha256', 'drnl_sha256')),
             'Actual node/edge/DRNL digest required')


def validate_run(run, job, pins, workload, fixture=False):
    need(run.get('status') == 'complete' and type(run.get('returncode')) is int
         and run['returncode'] == 0 and 'abort_reason' in run and run['abort_reason'] is None,
         'Inner complete cannot override exit/abort')
    cfg, timing, device, model = run['config'], run['timing'], run['device'], run['model']
    need(device['type'] == 'cuda' and device['visible_gpu_count'] == 1
         and device['used_devices'] == ['cuda:0'] and 'H100' in device['name']
         and device['torch_version'] == pins['original_runtime_versions']['torch'],
         'One H100/CUDA device and exact Torch version required')
    for key, value in MODEL.items():
        need(type(model[key]) is type(value) and model[key] == value, 'Frozen DGCNN field differs: ' + key)
    need(model['initial_state_sha256'] == pins['original_initial_model_state_sha256']
         and sha_hex(model['final_state_sha256']), 'Initial/final model identity differs')
    guard, gpu = run['gpu_memory']['guard'], run['gpu_memory']
    need(guard['status'] == 'configured_before_model_allocation' and guard['cuda_api_called'] is True
         and guard['allocator_budget_bytes'] == 24 * GIB and guard['requested_budget_gib'] == 24
         and positive(guard['total_bytes']) and positive(guard['free_bytes_before_model'])
         and guard['free_bytes_before_model'] <= guard['total_bytes']
         and guard['allocator_budget_bytes'] <= int(.8 * guard['free_bytes_before_model'])
         and same_number(guard['allocator_fraction'], 24 * GIB / guard['total_bytes'])
         and 0 < guard['allocator_fraction'] <= 1 and gpu['reserved_peak_gib'] <= 24
         and nonnegative(gpu['reserved_peak_gib']) and nonnegative(gpu['allocated_peak_gib'])
         and gpu['allocated_peak_gib'] <= gpu['reserved_peak_gib'], '24 GiB allocator/preallocation guard differs')
    expected = dict(fixture=fixture, file_endian='little', seed=20261124, hops=2,
                    cap=3 if fixture else 200, warmup_pairs=2 if fixture else 256,
                    steps=2 if fixture else 100, prefetch_factor=1,
                    audit_pairs=2 if fixture else 32, device='cuda', torch_threads=1,
                    gpu_memory_budget_gib=24.0, stock_index_dtype='auto',
                    batch_size=2 if fixture else 32, num_workers=1 if fixture else job['workers'],
                    arm=job['arm'])
    for key, value in expected.items():
        type_ok = type(cfg[key]) is type(value) if type(value) is not float else type(cfg[key]) in (int, float)
        need(type_ok and cfg[key] == value, 'Fixed run config differs: ' + key)
    need(run['arm'] == cfg['arm'], 'Inner arm binding differs')
    if not fixture:
        need(cfg['max_neighbors'] == 20000000
             and cfg['manifest_sha256'] == pins['workloads'][job['dataset']]['sha256'],
             'Real workload/selected-neighbor guard differs')
    graphs, steps, warmup = (4, 2, 2) if fixture else (3200, 100, 256)
    need(timing['measured_graphs'] == graphs and timing['measured_steps'] == steps
         and timing['warmup_graphs_excluded'] == warmup
         and timing['warmup_steps'] == (1 if fixture else 8)
         and positive(timing['steady_seconds'])
         and same_number(timing['graphs_per_second'], graphs / timing['steady_seconds'])
         and timing['measured_positive_occurrences'] == sum(r[2] for r in workload[warmup:warmup + graphs]),
         'Measured denominator/window/throughput differs')
    need(timing['prefetch_batches_capacity'] == cfg['num_workers']
         and same_number(timing['prefetch_boundary_fraction'], cfg['num_workers'] / steps),
         'Prefetch finite-window boundary differs')
    need(len(run['losses']) == steps and all(nonnegative(v) for v in run['losses']), 'Finite measured losses required')
    validate_audits(run['audits'], workload, 2 if fixture else 32)
    if cfg['arm'] == ARMS[0]:
        need(run['stock_physical_route']['measured_temporary_row_dtype_counts'] == {'int32': graphs},
             'Native-auto stock row width must remain int32')


def check_plan(plan, contract, protocol):
    blocks, expected_jobs = schedule()
    need(plan['execute'] is True and plan['protocol'] == protocol
         and plan['protocol_sha256'] == PROTOCOL_SHA
         and plan['source_manifest_sha256'] == digest_file(PACKET / 'SOURCE_PINS.json')['sha256'],
         'Actual execute plan/protocol/source pin differs')
    need(plan['blocks'] == blocks and len(plan['jobs']) == 40, 'Predeclared order/grid differs')
    budgets = plan['output_limits']
    need(positive(budgets['memory_budget_gib']) and budgets['memory_budget_gib'] <= 180
         and positive(budgets['max_hours']) and budgets['max_hours'] <= 6
         and positive(budgets['max_output_gib']) and budgets['max_output_gib'] <= 2,
         'Frozen finite resource budgets differ')
    options = {'--seed': '20261124', '--batch-sizes': '32', '--steps': '100',
               '--warmup-steps': '8', '--prefetch-factor': '1', '--hops': '2', '--cap': '200',
               '--max-neighbors': '20000000', '--n-positives': '4000', '--negative-kind': 'uniform',
               '--stock-index-dtype': 'auto', '--device': 'cuda', '--memory-budget-gib': '180',
               '--gpu-memory-budget-gib': '24', '--timeout-seconds': '3600'}
    for actual, expected in zip(plan['jobs'], expected_jobs):
        need(all(actual.get(k) == v for k, v in expected.items()), 'Job identity/order differs')
        name, argv = actual['name'], actual['argv']
        output = contract['science_out'] + '/jobs/' + name
        need(actual['out'] == output and argv[:3] == [contract['python'], '-u',
             contract['packet'] + '/frozen_src/scripts/seal_revision/pipeline_benchmark_v4.py'],
             'Actual job executable/output binding differs')
        need(argv.count('--out') == 1 and argv[argv.index('--out') + 1] == output, 'Job output argv differs')
        if actual['kind'] == 'pipeline':
            for option, value in {**options, '--workers': str(actual['workers']), '--arms': actual['arm']}.items():
                need(argv.count(option) == 1 and argv[argv.index(option) + 1] == value, 'Fixed argv differs: ' + option)
            dataset = actual['dataset']
            need(actual['pairs'] == contract['science_out'] + '/manifests/' + dataset + '_pairs.npy',
                 'Job workload path differs')
            graph = contract['citation2_root'] + '/train_csr' if dataset == 'citation2' else contract['friendster_root']
            need(argv[argv.index('--graph-path') + 1] == graph and '--fixture' not in argv,
                 'Real graph binding differs')
    return expected_jobs


def descriptive(values):
    need(len(values) == 3 and all(positive(v) for v in values), 'Three positive observations required')
    mean = math.fsum(values) / 3
    return dict(mean=mean, sample_sd=math.sqrt(math.fsum((v - mean) ** 2 for v in values) / 2),
                median=sorted(values)[1], minimum=min(values), maximum=max(values))


def recompute(jobs, reports):
    """Independent elementary reaggregation; never calls runner summarize."""
    rows, ratios, comparisons = [], [], []
    blocks, _ = schedule()
    for block in blocks:
        by_arm = {j['arm']: reports[j['name']]['runs'][0] for j in jobs
                  if j.get('block') == block['id']}
        need(set(by_arm) == set(ARMS), 'All three block arms required')
        for i, left in enumerate(ARMS):
            for right in ARMS[i + 1:]:
                a, b = by_arm[left], by_arm[right]
                need(a['audits'] == b['audits']
                     and a['model']['initial_state_sha256'] == b['model']['initial_state_sha256']
                     and a['config']['manifest_sha256'] == b['config']['manifest_sha256'],
                     'Cross-arm warmup prefix/model/workload mismatch')
                comparisons.append(dict(block=block['id'], reference_arm=left, candidate_arm=right,
                    audited_pairs=32, same_nodes_edges_drnl_and_initial_model=True,
                    scope='excluded warmup prefix only; not every measured graph'))
    for dataset in DATASETS:
        for workers in WORKERS:
            values_by_arm = {}
            for arm in ARMS:
                chosen = sorted((j for j in jobs if j.get('dataset') == dataset
                                 and j.get('workers') == workers and j.get('arm') == arm),
                                key=lambda j: j['repetition'])
                need([j['repetition'] for j in chosen] == [1, 2, 3], 'Missing/duplicate process repetition')
                values = [reports[j['name']]['runs'][0]['timing']['steady_seconds'] for j in chosen]
                values_by_arm[arm] = values
                rows.append(dict(dataset=dataset, workers=workers, arm=arm,
                                 steady_seconds=values, **descriptive(values)))
            for i, left in enumerate(ARMS):
                for right in ARMS[i + 1:]:
                    values = [a / b for a, b in zip(values_by_arm[left], values_by_arm[right])]
                    stats = descriptive(values)
                    ratios.append(dict(dataset=dataset, workers=workers, numerator=left,
                        denominator=right, paired_time_ratios=values,
                        **{k: stats[k] for k in ('median', 'mean', 'sample_sd')}))
    need(len(rows) == len(ratios) == 12 and len(comparisons) == 36, 'Reaggregation grid differs')
    return rows, ratios, comparisons


def process_quiet(value, launch):
    need(value['quiescent'] is True and value['observation'] in ('absent', 'zombie')
         and value['pid'] == launch['pid'] and value['start_ticks'] == launch['start_ticks']
         and type(value['pid']) is int and value['pid'] > 0
         and type(value['start_ticks']) is int and value['start_ticks'] > 0,
         'Actual PID/start-time post-exit quiescence differs')


def identity_reference(archive, path, expected):
    need(archive.info(path) == {k: expected[k] for k in ('bytes', 'sha256')}, 'Receipt identity mismatch: ' + path)


def verify_lifecycle(archive, contract, records):
    root, env = 'e23/results/', 'e23/envelope/'
    complete, terminal = archive.get(root + 'COMPLETE.json'), archive.get(env + 'SCIENTIFIC_TERMINAL.json')
    need(all(complete.get(k) is True for k in ('controller_finished', 'success', 'owned_tree_quiescent'))
         and complete['failure'] is None and type(complete['YT_writes']) is int and complete['YT_writes'] == 0
         and complete['records'] == records, 'Scientific runner terminal differs')
    need(archive.get(root + 'status.json')['phase'] == 'complete', 'Final runner status differs')
    launch, exit_data = archive.get(env + 'child_launch.json'), archive.get(env + 'child_exit.json')
    check_clean_exit(exit_data, guard=False)
    need(exit_data['actual_wait_returned'] is True and exit_data['pid'] == launch['pid']
         and exit_data['start_ticks'] == launch['start_ticks'] and positive(exit_data['finished_unix'])
         and positive(launch['started_unix']) and exit_data['finished_unix'] >= launch['started_unix'],
         'Actual outer wait/child identity differs')
    need(launch['stdin_dependency'] is False and launch['start_new_session'] is True
         and launch['entrypoint'] == contract['packet'] + '/run_pipeline_repeats.py'
         and launch['source_sha256'] == archive.info('e23/sources/packet/run_pipeline_repeats.py')['sha256']
         and launch['packet_manifest_sha256'] == PACKET_SHA, 'Fresh detached child binding differs')
    need(terminal['scientific_success'] is True and terminal['timing_jobs'] == 36
         and terminal['declared_jobs'] == 40, 'Outer scientific terminal differs')
    identity_reference(archive, root + 'COMPLETE.json', terminal['complete_identity'])
    identity_reference(archive, root + 'scientific_summary.json', terminal['summary_identity'])
    process_quiet(terminal['process'], launch)
    lock = terminal['timing_lock']
    need(lock['path'] == contract['coordination'] + '/PUBLIC_SYSTEMS_RUNNING.lock'
         and lock['exclusive_flock_acquired_and_released'] is True
         and lock['checked_unix'] >= exit_data['finished_unix'], 'Actual post-wait flock receipt differs')
    outer_launch, outer_identity = archive.get(env + 'actual_outer_launch.json'), archive.get(env + 'controller_identity.json')
    for key in ('pid', 'start_ticks', 'entrypoint', 'source_sha256', 'source_manifest_sha256'):
        need(outer_launch[key] == outer_identity[key], 'Actual outer launch/source identity differs')
    need(outer_launch['stdin_dependency'] is False and outer_launch['detached_session'] is True
         and outer_launch['automatic_retry'] is False and outer_launch['output'] == contract['outer_out'],
         'Outer launch lifecycle differs')
    admission = archive.get(env + 'admission.json')
    need(admission['hostname_observed'] == contract['hostname']
         and admission['unix_uid_observed'] == contract['unix_uid']
         and admission['pod_uid_root_attested'] == contract['pod_uid']
         and admission['statefulset_uid_root_attested'] == contract['statefulset_uid']
         and admission['packet_manifest_sha256'] == PACKET_SHA
         and admission['outer_source_manifest_sha256'] == outer_launch['source_manifest_sha256'],
         'Host/UID/root-attested Kubernetes identity differs')
    return outer_launch


def verify_bootstrap(archive, contract):
    prefix, source = 'e23/bootstrap/', 'e23/sources/bootstrap/'
    source_tree(archive, source, 'SOURCE_MANIFEST.json', contract['bootstrap_source_manifest_sha256'])
    need(archive.info(source + 'input_bootstrap.py')['sha256'] == contract['bootstrap_source_sha256'],
         'Actual bootstrap source SHA differs')
    complete = archive.get(prefix + 'COMPLETE.json')
    for key in ('input_preparation_complete', 'all19_C2_numeric_payloads_exact', 'all3_Friendster_numeric_payloads_exact',
                'base_environment_unchanged', 'prepared_inputs_ready_for_separately_owned_scientific_controller',
                'preparation_child_reaped'):
        need(complete[key] is True, 'Bootstrap completion gate differs: ' + key)
    for key in ('package_installs', 'YT_writes', 'new_YT_nodes', 'new_YT_chunks'):
        need(type(complete[key]) is int and complete[key] == 0, 'Bootstrap no-install/no-write gate differs')
    need(archive.get(prefix + 'base_before.json') == archive.get(prefix + 'base_after.json')
         and archive.get(prefix + 'status.json')['phase'] == 'complete', 'Base environment/bootstrap status differs')
    for actual_name, expected_name, count, flag, root_key, expected_root in (
        ('ACTUAL_C2_19_IDENTITY.json', 'EXPECTED_C2_19.json', 19, 'all19_numeric_payloads_exact',
         'prepared_root', contract['citation2_root']),
        ('ACTUAL_FRIENDSTER_FULL3_IDENTITY.json', 'EXPECTED_FRIENDSTER_FULL3.json', 3,
         'all3_numeric_payloads_exact', 'root', contract['friendster_root'])):
        actual, expected = archive.get(prefix + actual_name), archive.get(source + expected_name)
        need(actual[flag] is True and len(actual['files']) == count
             and actual['files'] == expected['files'] and actual[root_key] == expected_root,
             'Prepared public numeric identities differ')
    launch = archive.get(prefix + 'actual_launch.json')
    need(launch['pid'] == contract['bootstrap_pid'] and launch['start_ticks'] == contract['bootstrap_start_ticks']
         and launch['source_sha256'] == contract['bootstrap_source_sha256']
         and launch['output'] == contract['bootstrap_out'] and launch['scientific_jobs_started'] == 0
         and launch['automatic_retry'] is False and launch['stdin_dependency'] is False
         and launch['detached_session'] is True, 'Actual bootstrap launch binding differs')
    admitted = archive.get('e23/envelope/bootstrap_admitted.json')
    need(admitted['actual_launch'] == launch, 'Admitted original launch differs')
    identity_reference(archive, prefix + 'actual_launch.json', admitted['actual_launch_identity'])
    process_quiet(admitted['process'], launch)
    for name, pin in admitted['receipt_files'].items():
        identity_reference(archive, prefix + name, pin)


def verify_inputs(archive, contract, pins):
    root = 'e23/results/'
    identity = archive.get(root + 'input_identity.json')
    c2 = read_json(PACKET / 'pins/citation2_prepared_expected.json')['files']
    allowed = {'num_nodes.txt', 'train_csr/edge_starts.bin', 'train_csr/edge_ends.bin',
               'train_pairs_csr/edge_starts.bin', 'train_pairs_csr/edge_ends.bin'}
    expected = {contract['citation2_root'] + '/' + r['path']: r for r in c2 if r['path'] in allowed}
    fr = read_json(PACKET / 'pins/friendster_full_expected.json')['files']
    expected.update({contract['friendster_root'] + '/' + k: v for k, v in fr.items()})
    need(identity['complete_used_input_hashes'] is True and len(expected) == 8
         and set(identity['files']) == set(expected), 'All eight used input hashes required')
    for name, row in expected.items():
        actual = identity['files'][name]
        need(all(actual[k] == row[k] for k in ('bytes', 'sha256'))
             and all(type(actual[k]) is int and actual[k] >= 0 for k in ('mtime_ns', 'inode', 'device')),
             'Used input hash/stat binding differs')
    host, hardware = archive.get(root + 'host.json'), archive.get(root + 'hardware.json')
    need(host['hostname'] == contract['hostname'] and host['versions'] == pins['original_runtime_versions']
         and '3.10.' in host['python'] and host['cpu_affinity'] and host['boot_id'], 'Actual host/software identity differs')
    target = hardware['target']
    need(target['uuid'] == contract['gpu_uuid'] and 'H100' in target['name'], 'Actual H100 target differs')


def verify_actual(path, expected_sha, expected_outer_sha, pins, protocol, workloads, packet_rows):
    archive = Archive(path, expected_sha)
    source_tree(archive, 'e23/sources/packet/', 'PACKET_MANIFEST.json', PACKET_SHA)
    for name, pin in packet_rows.items():
        need(archive.info('e23/sources/packet/' + name) == {k: pin[k] for k in ('bytes', 'sha256')},
             'Recorded immutable packet differs from independently pinned local packet')
    contract = archive.get('e23/sources/outer/CONTRACT.json')
    outer_launch = archive.get('e23/envelope/actual_outer_launch.json')
    source_tree(archive, 'e23/sources/outer/', 'SOURCE_MANIFEST.json', outer_launch['source_manifest_sha256'])
    need(sha_hex(expected_outer_sha)
         and archive.info('e23/sources/outer/outer_e23.py')['sha256'] == expected_outer_sha
         and outer_launch['source_sha256'] == expected_outer_sha, 'Independently supplied outer source SHA differs')
    need(contract['packet_manifest_sha256'] == PACKET_SHA and contract['protocol_sha256'] == PROTOCOL_SHA
         and contract['max_total_seconds'] == 21600 and contract['bootstrap_wait_seconds'] == 5400
         and contract['final_reserve_seconds'] == 900 and contract['scientific_protocol_changes'] == 0,
         'Actual envelope admission/protocol/resource bounds differ')
    plan = archive.get('e23/results/plan.json')
    expected_jobs = check_plan(plan, contract, protocol)
    need(plan['historical_workload_pins'] == pins['workloads'], 'Historical workload pins differ')
    verify_inputs(archive, contract, pins)
    verify_bootstrap(archive, contract)
    for dataset, pin in pins['workloads'].items():
        for prefix in ('e23/results/manifests/', 'e23/sources/packet/workloads/'):
            member = prefix + dataset + '_pairs.npy'
            need(archive.info(member) == {k: pin[k] for k in ('bytes', 'sha256')}
                 and npy_rows(archive.npy[member], 8000) == workloads[dataset], 'Actual public NPY differs')
    records = archive.get('e23/results/job_records.json')
    need(isinstance(records, list) and len(records) == 40
         and [r['name'] for r in records] == [j['name'] for j in expected_jobs], 'All ordered 40 job records required')
    reports = {}
    for job, record in zip(plan['jobs'], records):
        name, kind = job['name'], job['kind']
        need(record['kind'] == kind and record['validation_error'] is None,
             'Failed/guarded job cannot be omitted or relabeled')
        need(archive.get('e23/results/' + name + '.exit.json') == record, 'Per-job exit receipt differs')
        report_path = 'e23/results/jobs/' + name + '/report.json'
        need(record['report_sha256'] == archive.info(report_path)['sha256'], 'Raw report SHA differs')
        report = archive.get(report_path)
        reports[name] = report
        need(report['benchmark_version'] == 4 and report['runtime_versions'] == pins['original_runtime_versions']
             and set(report['source_sha256']) == set(SOURCE_KEYS)
             and all(report['source_sha256'][k] == pins['frozen_sources']['frozen_src/' + k]['sha256'] for k in SOURCE_KEYS),
             'All seven scientific source and exact runtime bindings required')
        check_clean_exit(record['exit'], expected_code=1 if kind == 'expected_gpu_refusal' else 0)
        need(record['exit']['guard_policy']['version'] == 'stable_pss_individual_rss_cgroup90_v3', 'Outer CPU guard policy differs')
        idle = record['exit']['gpu_idle_observations']
        need((len(idle) == 2 and all(i['busy_apps'] == [] and positive(i['unix']) for i in idle)
              and idle[1]['unix'] - idle[0]['unix'] >= 14) if job['gpu'] else idle == [], 'Pre-phase GPU idle admission differs')
        runs = report['runs']
        fixture_rows = None
        if kind in ('expected_gpu_refusal', 'gpu_fixture'):
            member = 'e23/results/jobs/' + name + '/pairs.npy'
            tiny_manifest = report['manifest']
            need(archive.info(member)['bytes'] <= 4096
                 and archive.info(member)['sha256'] == tiny_manifest['sha256']
                 and tiny_manifest['rows'] == 128 and tiny_manifest['positive_occurrences'] == 64
                 and tiny_manifest['columns'] == ['src', 'dst', 'label']
                 and tiny_manifest['path'] == contract['science_out'] + '/jobs/' + name + '/pairs.npy'
                 and report['graph']['path'] == 'fixture', 'Tiny fixture workload/report binding differs')
            fixture_rows = npy_rows(archive.npy[member], 128)
            need(sum(r[2] for r in fixture_rows) == 64, 'Tiny fixture positive count differs')
        if kind == 'manifest':
            pin = pins['workloads'][job['dataset']]
            need(runs == [] and report['manifest']['rows'] == 8000
                 and report['manifest']['positive_occurrences'] == 4000
                 and report['manifest']['sha256'] == pin['sha256'], 'Manifest gate differs')
            continue
        if kind == 'expected_gpu_refusal':
            need(len(runs) == 1 and runs[0]['status'] == 'failed'
                 and type(runs[0]['returncode']) is int and runs[0]['returncode'] != 0
                 and runs[0]['abort_reason'] is None
                 and any(p.startswith('e23/results/jobs/guard_refusal/') for p in archive.refusal_logs),
                 'Only the declared impossible-capacity refusal may fail')
            need(runs[0]['config']['gpu_memory_budget_gib'] == 1000000
                 and runs[0]['config']['arm'] == ARMS[1]
                 and runs[0]['config']['fixture'] is True
                 and runs[0]['config']['device'] == 'cuda'
                 and runs[0]['config']['seed'] == 20261124
                 and runs[0]['config']['num_workers'] == 1
                 and runs[0]['config']['batch_size'] == 2, 'Refusal fixture configuration differs')
            continue
        fixture = kind == 'gpu_fixture'
        need(len(runs) == (3 if fixture else 1), 'Intended run count differs')
        if fixture:
            need([r['config']['arm'] for r in runs] == list(ARMS), 'Three fixture arms differ')
        else:
            pin = pins['workloads'][job['dataset']]
            need(report['manifest']['sha256'] == pin['sha256']
                 and report['manifest']['rows'] == 8000 and report['manifest']['positive_occurrences'] == 4000,
                 'Real report manifest identity/counts differ')
        for run in runs:
            run_job = dict(job, arm=run['config']['arm']) if fixture else job
            validate_run(run, run_job, pins, fixture_rows if fixture else workloads[job['dataset']], fixture)
            subdir = f"{run['config']['arm']}.batch{run['config']['batch_size']}.workers{run['config']['num_workers']}"
            base = 'e23/results/jobs/' + name + '/' + subdir + '/'
            need(archive.get(base + 'config.json') == run['config'], 'Raw child configuration differs')
            child = archive.get(base + 'result.json')
            need(all(run.get(k) == v for k, v in child.items()), 'Raw child result differs from parent report')
            samples = archive.jsonl_counts.get(base + 'memory.samples.jsonl', 0)
            need(samples > 0 and samples == run['cpu_memory_sampled_peak']['samples'],
                 'Historical inner resource sampler evidence/count differs')
        if fixture:
            comparisons = report['equality_audits']
            pairs = {(ARMS[i], r) for i in range(3) for r in ARMS[i + 1:]}
            need(len(comparisons) == 3 and {(c['reference_arm'], c['candidate_arm']) for c in comparisons} == pairs
                 and all(c['same_nodes_edges_drnl_and_initial_model'] is True and c['query_audits'] == 2 for c in comparisons)
                 and all(r['audits'] == runs[0]['audits'] for r in runs), 'Tiny GPU fixture three-way prefix equality differs')
    rows, ratios, comparisons = recompute(plan['jobs'], reports)
    summary = archive.get('e23/results/scientific_summary.json')
    need(summary['status'] == 'complete' and summary['timing_jobs'] == 36
         and summary['scope'] == protocol['scope'] and summary['reporting'] == protocol['reporting'], 'Scientific summary scope differs')
    matching(summary['rows'], rows, '12 independent descriptive rows')
    matching(summary['paired_time_ratios'], ratios, '12 independently paired ratio groups')
    matching(summary['reference_prefix_comparisons'], comparisons, '36 independent prefix comparisons')
    verify_lifecycle(archive, contract, records)
    return dict(status='PASS', scope='Recorded E23 archive verification and independent numerical reaggregation only.',
        raw_archive=archive.identity, archive_members=len(archive.infos), raw_payload_bytes=archive.raw_bytes,
        source_manifest_sha256=PACKET_SHA, semantic_protocol_sha256=PROTOCOL_SHA,
        outer_source_sha256=expected_outer_sha, declared_job_gates=40, timing_jobs=36,
        expected_recorded_final_persistence_path=contract['yt_final'],
        descriptive_rows=rows, paired_time_ratios=ratios, warmup_prefix_comparisons=36,
        audited_queries_per_timing=32, measured_graphs_per_window=3200,
        actual_child_wait_zero_quiescence_and_flock_checked=True,
        full_graph_oracles_recomputed=False, fresh_science_executed=False,
        final_outer_completion_and_YT_persistence_checked=False,
        limitation='Prefix hashes are checked across arms and bound to sources/workloads; graph-derived node/edge/DRNL hashes are not regenerated without graph inputs. Outer final persistence receipts are generated after this archive and require separate verification.')


def check_external(report, outer_complete_path=None, pointer_path=None):
    need((outer_complete_path is None) == (pointer_path is None), 'Supply both external outer COMPLETE and archive pointer, or neither')
    if outer_complete_path is None:
        return
    complete, pointer = read_json(outer_complete_path), read_json(pointer_path)
    need(all(complete.get(k) is True for k in ('controller_finished', 'success', 'scientific_success', 'owned_tree_quiescent'))
         and complete['failure'] is None and complete['archive_pointer'] == pointer
         and complete['final_YT_write_file_calls'] == 1 and complete['automatic_retry'] is False
         and complete['statefulset_lifecycle_actions'] == 0 and complete['package_installs'] == 0,
         'External successful outer completion differs')
    need(all(pointer[k] == report['raw_archive'][k] for k in ('bytes', 'sha256'))
         and pointer['yt_path'] == report['expected_recorded_final_persistence_path']
         and pointer['readback_verified'] is True and pointer['transaction_committed'] is True
         and pointer['new_YT_file_nodes'] == pointer['chunk_count'] == pointer['write_file_calls'] == 1
         and pointer['automatic_retry'] is False, 'External one-file persistence pointer differs')
    report['final_outer_completion_and_YT_persistence_checked'] = True
    report['external_receipts'] = dict(outer_complete=digest_file(outer_complete_path), archive_pointer=digest_file(pointer_path))
    report['external_scope'] = 'Recorded final receipts checked; no live YT/network or outer PID observation performed.'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--archive-sha256')
    parser.add_argument('--outer-source-sha256')
    parser.add_argument('--outer-complete', type=Path)
    parser.add_argument('--archive-pointer', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    pins, protocol, workloads, packet_rows = local_sources()
    if args.archive is None:
        need(not any((args.archive_sha256, args.outer_source_sha256, args.outer_complete, args.archive_pointer)),
             'Actual-result arguments require an actual archive')
        blocks, jobs = schedule()
        report = dict(status='SOURCE_ONLY', actual_scientific_results_verified=False,
            packet_manifest_sha256=PACKET_SHA, semantic_protocol_sha256=PROTOCOL_SHA,
            immutable_packet_payloads=len(packet_rows), protocol_jobs=len(jobs), timing_jobs=36,
            randomized_blocks=len(blocks), expected_descriptive_rows=12, expected_paired_ratio_groups=12,
            expected_prefix_comparisons=36, public_workload_rows={k: len(v) for k, v in workloads.items()},
            source_python310_grammar=True, numerical_or_benchmark_modules_imported=False,
            graph_model_GPU_network_cluster_actions=0, fresh_Linux_GPU_replay_performed=False)
    else:
        report = verify_actual(args.archive, args.archive_sha256, args.outer_source_sha256,
                               pins, protocol, workloads, packet_rows)
        check_external(report, args.outer_complete, args.archive_pointer)
    data = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if args.output:
        need(not args.output.exists() and not args.output.is_symlink(), 'Fresh output receipt required; no overwrite')
        with args.output.open('x') as stream:
            stream.write(data)
    else:
        print(data, end='')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (VerificationError, KeyError, TypeError, json.JSONDecodeError, OSError, tarfile.TarError) as exc:
        print(json.dumps(dict(status='FAIL', error=type(exc).__name__, message=str(exc)), allow_nan=False), file=sys.stderr)
        raise SystemExit(1)
