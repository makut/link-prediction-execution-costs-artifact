#!/usr/bin/env python3
"""Independent stdlib numerical audit of a genuine completed E23 raw archive.

No runner, producer summary, canonical verifier or scientific module is imported.
The genuine input archive and its independently collected SHA are mandatory.
"""
from __future__ import annotations
import argparse
import ast
from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import random
import struct
import tarfile

MIB = 1 << 20
ARMS = ('reference_union_stock', 'reference_union_corrected', 'reference_union_compiled')
DATASETS, WORKERS = ('citation2', 'friendster'), (1, 8)
PACKET_SHA = '16b14211782d5c7c765e52ec36a533131e3bf559633229da19a874cdfb28c68f'
PINS_SHA = 'c0356232609a5aecf5784a9cd0842f54e0c487e577a6ab9e86d9b1a8d2f943f4'
PROTOCOL_SHA = '6587e2055c5a77f4ffbe075fd7d70a6fcf648ad075e4d4cb8aa6574ae54061a1'
MODEL_SHA = 'f88254079fe6c152b4c0153c857a63516348353d578f918622e7f6719559618e'
MODEL = dict(name='DGCNN', sortpool_k=281, hidden_channels=32, num_layers=3,
             max_z=1000, learning_rate=0.0001, optimizer='Adam', use_feature=False,
             node_embedding=False, parameters=596674)
SOURCES = ('scripts/seal_revision/pipeline_benchmark_v4.py', 'scripts/seal_revision/exact_benchmark.py',
           'scripts/seal_revision/memory_benchmark.py', 'seal/models.py', 'seal/utils.py',
           'seal/graph_csr_kernels.py', 'seal/graph_csr_extraction.py')


def need(ok, message):
    if not ok:
        raise ValueError(message)


def decode(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            need(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def pin(path, maximum):
    path = Path(path)
    need(path.is_file() and not path.is_symlink() and path.stat().st_size <= maximum, 'bounded regular file required')
    h, size = hashlib.sha256(), 0
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(MIB), b''):
            size += len(block)
            need(size <= maximum, 'growing input exceeds bound')
            h.update(block)
    need(size == path.stat().st_size, 'input changed during hash')
    return dict(bytes=size, sha256=h.hexdigest())


def schedule():
    rng = random.Random(20261126)
    configs = [(dataset, workers) for dataset in DATASETS for workers in WORKERS]
    orders = {}
    for config in configs:
        orders[config] = list(ARMS)
        rng.shuffle(orders[config])
    blocks, jobs = [], []
    for repeat in (1, 2, 3):
        current = configs[:]
        rng.shuffle(current)
        for dataset, workers in current:
            base = orders[dataset, workers]
            order = base[repeat - 1:] + base[:repeat - 1]
            block = f'{dataset}_w{workers}_r{repeat}'
            blocks.append(dict(id=block, dataset=dataset, workers=workers, repetition=repeat, order=order))
            for position, arm in enumerate(order, 1):
                jobs.append(dict(name=block + '_' + arm, kind='pipeline', block=block,
                    dataset=dataset, workers=workers, repetition=repeat, position=position, arm=arm))
    return blocks, jobs


def load_archive(path, expected_sha, jobs):
    identity = pin(path, 128 * MIB)
    need(identity['sha256'] == expected_sha, 'independently collected whole archive SHA differs')
    wanted = {'SHA256_MANIFEST.json', 'e23/results/plan.json', 'e23/results/job_records.json',
              'e23/results/COMPLETE.json', 'e23/results/scientific_summary.json',
              'e23/sources/packet/PACKET_MANIFEST.json', 'e23/sources/packet/SOURCE_PINS.json',
              'e23/sources/packet/PROTOCOL.json'}
    for dataset in DATASETS:
        wanted.add('e23/results/manifests/' + dataset + '_pairs.npy')
    for job in jobs:
        root = 'e23/results/jobs/' + job['name'] + '/'
        child = root + f"{job['arm']}.batch32.workers{job['workers']}/"
        wanted.update((root + 'report.json', child + 'result.json', child + 'config.json',
                       'e23/results/' + job['name'] + '.exit.json'))
    payloads, identities, total, kept = {}, {}, 0, 0
    with tarfile.open(path, 'r|gz') as tar:
        for member in tar:
            name, rel = member.name, PurePosixPath(member.name)
            need(member.isfile() and set(member.pax_headers) <= {'path'}
                 and member.pax_headers.get('path', name) == name and not rel.is_absolute()
                 and '..' not in rel.parts and '\\' not in name and str(rel) == name
                 and name not in identities and len(identities) < 4096 and member.size <= 32 * MIB,
                 'bounded unique regular canonical archive member required')
            total += member.size
            need(total <= 384 * MIB, 'raw archive payload cap')
            h, size, data = hashlib.sha256(), 0, bytearray()
            with tar.extractfile(member) as stream:
                for block in iter(lambda: stream.read(MIB), b''):
                    size += len(block);h.update(block)
                    if name in wanted:
                        kept += len(block)
                        need(kept <= 64 * MIB, 'retained numerical metadata cap')
                        data.extend(block)
            need(size == member.size, 'truncated raw member')
            identities[name] = dict(bytes=size, sha256=h.hexdigest())
            if name in wanted:
                payloads[name] = bytes(data)
    need(set(payloads) == wanted, 'all36 actual reports/children and required numerical provenance must exist')
    manifest = decode(payloads['SHA256_MANIFEST.json'])
    need(manifest['files'] == {k: v for k, v in identities.items() if k != 'SHA256_MANIFEST.json'}, 'whole-member archive manifest differs')
    need(pin(path, 128 * MIB) == identity, 'archive changed during reading')
    return payloads, identities, identity


def workload(raw):
    need(raw[:8] == b'\x93NUMPY\x01\x00', 'frozen NPY1 workload required')
    size = struct.unpack('<H', raw[8:10])[0]
    need(size <= 4096 and ast.literal_eval(raw[10:10 + size].decode('ascii')) ==
         dict(descr='<i8', fortran_order=False, shape=(8000, 3)) and len(raw) == 10 + size + 8000 * 24,
         'exact original int64[8000,3] workload required')
    rows = list(struct.iter_unpack('<qqq', raw[10 + size:]))
    need(sum(row[2] for row in rows) == 4000, 'original4000 positive occurrences required')
    return rows


def clean(exit):
    return (type(exit['returncode']) is int and exit['returncode'] == 0 and exit['timed_out'] is False
            and exit['orphan_cleanup'] is False and exit['owned_tree_quiescent'] is True
            and exit['guard_abort_reason'] is None and not exit['owned_pgids_signaled'])


def finite_positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def same(actual, expected, label):
    if isinstance(expected, dict):
        need(isinstance(actual, dict) and set(actual) == set(expected), label + ': keys')
        for key, value in expected.items():same(actual[key], value, label + '.' + key)
    elif isinstance(expected, list):
        need(isinstance(actual, list) and len(actual) == len(expected), label + ': length')
        for i, value in enumerate(expected):same(actual[i], value, label + '[' + str(i) + ']')
    elif type(expected) is float:
        need(type(actual) in (int, float) and math.isfinite(actual)
             and math.isclose(actual, expected, rel_tol=2e-12, abs_tol=1e-10), label + ': numerical value')
    else:
        need(type(actual) is type(expected) and actual == expected, label + ': identity')


def statistics3(values):
    need(len(values) == 3 and all(finite_positive(v) for v in values), 'all3 positive actual observations required')
    with localcontext() as context:
        context.prec = 50
        numbers = [Decimal(str(value)) for value in values]
        mean = sum(numbers) / Decimal(3)
        sd = (sum((value - mean) ** 2 for value in numbers) / Decimal(2)).sqrt()
        return dict(mean=float(mean), sample_sd=float(sd), median=float(sorted(numbers)[1]),
                    minimum=float(min(numbers)), maximum=float(max(numbers)))


def audit(payloads, infos, jobs, blocks):
    def get(name):return decode(payloads[name])
    root = 'e23/results/'
    need(infos['e23/sources/packet/PACKET_MANIFEST.json']['sha256'] == PACKET_SHA
         and infos['e23/sources/packet/SOURCE_PINS.json']['sha256'] == PINS_SHA, 'immutable packet/source pins differ')
    pins, protocol = get('e23/sources/packet/SOURCE_PINS.json'), get('e23/sources/packet/PROTOCOL.json')
    need(hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest() == PROTOCOL_SHA, 'canonical protocol differs')
    plan, records, done = get(root + 'plan.json'), get(root + 'job_records.json'), get(root + 'COMPLETE.json')
    need(plan['execute'] is True and plan['protocol'] == protocol and plan['protocol_sha256'] == PROTOCOL_SHA
         and plan['blocks'] == blocks and len(plan['jobs']) == 40 and len(records) == 40,
         'actual complete full-grid plan/records required')
    gate_names = ['guard_refusal', 'gpu_fixture', 'citation2_manifest', 'friendster_manifest'] + [j['name'] for j in jobs]
    need([r['name'] for r in records] == [j['name'] for j in plan['jobs']] == gate_names
         and all(r['validation_error'] is None for r in records), 'no omitted or invalid original gates allowed')
    need(done['success'] is True and done['controller_finished'] is True and done['owned_tree_quiescent'] is True
         and done['failure'] is None and done['records'] == records, 'genuine successful scientific terminal required')
    inputs = {}
    for dataset in DATASETS:
        path = root + 'manifests/' + dataset + '_pairs.npy'
        need(infos[path] == {k: pins['workloads'][dataset][k] for k in ('bytes', 'sha256')}, 'original workload SHA/bytes differ')
        inputs[dataset] = workload(payloads[path])
    reports, raw_values = {}, []
    for job, actual, record in zip(jobs, plan['jobs'][4:], records[4:]):
        need(all(actual[k] == value for k, value in job.items()), 'declared timing job order/identity differs')
        need(record['kind'] == 'pipeline' and clean(record['exit'])
             and get(root + job['name'] + '.exit.json') == record, 'actual clean timing exit required')
        parent = root + 'jobs/' + job['name'] + '/'
        report = get(parent + 'report.json')
        need(record['report_sha256'] == infos[parent + 'report.json']['sha256'] and len(report['runs']) == 1
             and report['benchmark_version'] == 4 and report['runtime_versions'] == pins['original_runtime_versions'],
             'actual report/source/runtime binding differs')
        need(set(report['source_sha256']) == set(SOURCES) and all(report['source_sha256'][name] ==
             pins['frozen_sources']['frozen_src/' + name]['sha256'] for name in SOURCES), 'all7 scientific runtime source hashes required')
        run = report['runs'][0];cfg, timing = run['config'], run['timing']
        need(run['status'] == 'complete' and type(run['returncode']) is int and run['returncode'] == 0
             and run['abort_reason'] is None, 'inner complete cannot override exit/abort')
        expected = dict(fixture=False, file_endian='little', seed=20261124, hops=2, cap=200,
            warmup_pairs=256, steps=100, prefetch_factor=1, audit_pairs=32, device='cuda',
            torch_threads=1, gpu_memory_budget_gib=24.0, stock_index_dtype='auto', batch_size=32,
            num_workers=job['workers'], arm=job['arm'], max_neighbors=20000000,
            manifest_sha256=pins['workloads'][job['dataset']]['sha256'])
        need(all(type(cfg[key]) is type(value) and cfg[key] == value for key, value in expected.items()), 'frozen timing config differs')
        need(all(type(run['model'][key]) is type(value) and run['model'][key] == value for key, value in MODEL.items())
             and run['model']['initial_state_sha256'] == MODEL_SHA == pins['original_initial_model_state_sha256'], 'DGCNN/model initialization differs')
        need(run['device']['type'] == 'cuda' and run['device']['visible_gpu_count'] == 1
             and run['device']['used_devices'] == ['cuda:0'] and 'H100' in run['device']['name'], 'one real H100 CUDA process required')
        need(timing['measured_graphs'] == 3200 and timing['measured_steps'] == 100
             and timing['warmup_steps'] == 8 and timing['warmup_graphs_excluded'] == 256
             and finite_positive(timing['steady_seconds'])
             and math.isclose(timing['graphs_per_second'], 3200 / timing['steady_seconds'], rel_tol=1e-12)
             and timing['measured_positive_occurrences'] == sum(row[2] for row in inputs[job['dataset']][256:3456]), 'actual measured window arithmetic differs')
        need(len(run['losses']) == 100 and all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in run['losses']), 'all100 finite measured losses required')
        audits = run['audits'];need(len(audits) == 32, 'exact32 excluded warmup-prefix audits required')
        for index, row in enumerate(audits):
            need(row['index'] == index and row['pair'] == list(inputs[job['dataset']][index][:2])
                 and row['label'] == inputs[job['dataset']][index][2], 'warmup audit workload/index differs')
            need(all(isinstance(row[key], str) and len(row[key]) == 64 and all(c in '0123456789abcdef' for c in row[key])
                     for key in ('nodes_sha256', 'edges_sha256', 'drnl_sha256')), 'real prefix hashes required')
        child = parent + f"{job['arm']}.batch32.workers{job['workers']}/"
        raw = get(child + 'result.json')
        need(all(key in raw for key in ('status', 'config', 'timing', 'model', 'device', 'audits', 'gpu_memory'))
             and all(run.get(key) == value for key, value in raw.items()) and get(child + 'config.json') == cfg,
             'complete original child result/config must match its parent report')
        reports[job['name']] = run
        raw_values.append(dict(**job, steady_seconds=timing['steady_seconds'], graphs_per_second=timing['graphs_per_second']))
    comparisons = []
    for block in blocks:
        by_arm = {j['arm']: reports[j['name']] for j in jobs if j['block'] == block['id']}
        need(set(by_arm) == set(ARMS), 'all3 matched block arms required')
        for index, left in enumerate(ARMS):
            for right in ARMS[index + 1:]:
                a, b = by_arm[left], by_arm[right]
                need(a['audits'] == b['audits'] and a['model']['initial_state_sha256'] == b['model']['initial_state_sha256']
                     and a['config']['manifest_sha256'] == b['config']['manifest_sha256'], 'matched-prefix/model/workload identity differs')
                comparisons.append(dict(block=block['id'], reference_arm=left, candidate_arm=right,
                    audited_pairs=32, same_nodes_edges_drnl_and_initial_model=True,
                    scope='excluded warmup prefix only; not every measured graph'))
    rows, ratios = [], []
    for dataset in DATASETS:
        for workers in WORKERS:
            values = {}
            for arm in ARMS:
                selected = sorted((j for j in jobs if (j['dataset'], j['workers'], j['arm']) == (dataset, workers, arm)), key=lambda j: j['repetition'])
                need([j['repetition'] for j in selected] == [1, 2, 3], 'all3 ordered repetitions required')
                times = [reports[j['name']]['timing']['steady_seconds'] for j in selected];values[arm] = times
                rows.append(dict(dataset=dataset, workers=workers, arm=arm, steady_seconds=times, **statistics3(times)))
            for index, left in enumerate(ARMS):
                for right in ARMS[index + 1:]:
                    with localcontext() as context:
                        context.prec = 50
                        paired = [float(Decimal(str(a)) / Decimal(str(b))) for a, b in zip(values[left], values[right])]
                    stats = statistics3(paired)
                    ratios.append(dict(dataset=dataset, workers=workers, numerator=left, denominator=right,
                        paired_time_ratios=paired, **{key: stats[key] for key in ('median', 'mean', 'sample_sd')}))
    need(len(raw_values) == len(comparisons) == 36 and len(rows) == len(ratios) == 12, 'full numerical comparison grid required')
    summary = get(root + 'scientific_summary.json')
    need(summary['status'] == 'complete' and summary['timing_jobs'] == 36 and summary['scope'] == protocol['scope']
         and summary['reporting'] == protocol['reporting'], 'recorded summary scope/count differs')
    same(summary['rows'], rows, 'independent Decimal descriptive rows')
    same(summary['paired_time_ratios'], ratios, 'independent Decimal paired ratios')
    same(summary['reference_prefix_comparisons'], comparisons, 'all36 matched prefix comparisons')
    return dict(raw_36_timing_observations=raw_values, descriptive_rows=rows, paired_time_ratios=ratios,
                reference_prefix_comparisons=comparisons, all36_original_child_results_and_configs_match=True,
                recorded_summary_independently_recomputed=True, all_seven_runtime_source_hashes_checked=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--archive-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    need(not args.output.exists() and not args.output.is_symlink(), 'fresh actual-audit receipt required')
    need(len(args.archive_sha256) == 64 and all(c in '0123456789abcdef' for c in args.archive_sha256), 'root-collected archive SHA required')
    blocks, jobs = schedule()
    payloads, identities, whole = load_archive(args.archive, args.archive_sha256, jobs)
    data = audit(payloads, identities, jobs, blocks)
    report = dict(kind='INDEPENDENT_ACTUAL_E23_RAW_NUMERICAL_PEER', status='PASS_ACTUAL_RAW_NUMERICAL_AUDIT',
        actual_archive=whole, archive_members_rehashed=len(identities), script=pin(Path(__file__), MIB),
        aggregation='Independent Decimal50 mean/sampleSD and per-repetition divisions from all36 actual raw steady times; no runner/producer/verifier summary function imported or called.',
        scope='Existing completed E23 archive only; finite-window same-workload process repetitions, descriptive uncertainty. Prefix equality covers32 excluded warmup queries, not every measured graph. This numerical peer does not replace the separate complete source/guard/bootstrap/persistence verifier or live outer lifecycle collection.',
        new_scientific_experiments=0, graph_model_GPU_network_cluster_YT_actions=0,
        canonical_verifier_or_runner_imported=False, **data)
    with args.output.open('x') as stream:json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status=report['status'], output=str(args.output), archive=whole,
        rows=report['descriptive_rows'], paired_ratios=report['paired_time_ratios']), indent=2))


if __name__ == '__main__':
    main()
