#!/usr/bin/env python3
"""Verify recorded E22 bytes and recompute its seven timing rows using stdlib only."""
from pathlib import Path, PurePosixPath
import argparse
import ast
import collections
import datetime
import gzip
import hashlib
import json
import math
import statistics
import struct
import tarfile

HERE = Path(__file__).resolve().parent
PREFIX = 'citation2_cpu_matched_v1_20261126/'
SOURCE_MANIFEST = 'd30698edff9d9d6b1d959e78bedc7ea226359fb667941e70f43a87945443012b'
N, M = 2927963, 60703502


def require(value, message):
    if not value:
        raise ValueError(message)


def decode(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def identity(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'regular payload required: ' + str(path))
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return {'bytes': path.stat().st_size, 'sha256': h.hexdigest()}


def safe(name):
    p = PurePosixPath(name)
    require(name and str(p) == name and not p.is_absolute() and '..' not in p.parts
            and '\\' not in name, 'safe canonical relative path required')
    return name


def load_archive(path):
    """Bound decompression and retain small JSON/source payloads, never extract or import."""
    payloads, total = {}, 0
    with gzip.open(path, 'rb') as stream, tarfile.open(fileobj=stream, mode='r|') as archive:
        for member in archive:
            require(member.name.startswith(PREFIX) and member.isfile() and not member.linkname,
                    'regular E22 archive members required')
            name = safe(member.name[len(PREFIX):])
            require(name not in payloads and 0 <= member.size <= 4 << 20, 'duplicate or oversized member')
            total += member.size
            require(total <= 16 << 20 and len(payloads) < 256, 'bounded raw archive required')
            with archive.extractfile(member) as source:
                raw = source.read(member.size + 1)
            require(len(raw) == member.size, 'member length differs')
            payloads[name] = raw
    manifest = decode(payloads['MANIFEST.json'])
    require(set(payloads) == set(manifest['files']) | {'MANIFEST.json'}, 'whole raw inventory differs')
    for name, pin in manifest['files'].items():
        raw = payloads[name]
        require({'bytes': len(raw), 'sha256': digest(raw)} == pin, 'raw payload differs: ' + name)
    require(digest(payloads['sources/SOURCE_MANIFEST.json']) == SOURCE_MANIFEST,
            'reviewed actual E22 source manifest differs')
    for row in decode(payloads['sources/SOURCE_MANIFEST.json'])['files']:
        raw = payloads['sources/' + safe(row['path'])]
        require({'bytes': len(raw), 'sha256': digest(raw)} == {k: row[k] for k in ('bytes', 'sha256')},
                'frozen scientific source differs')
    return payloads


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12)


def vector(values):
    require(isinstance(values, list) and len(values) == 3
            and all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in values),
            'three positive finite measured times required')


def clean(value):
    require(type(value['returncode']) is int and value['returncode'] == 0
            and value['timed_out'] is False and value['owned_tree_quiescent'] is True
            and value['orphan_cleanup'] is False and value['guard_abort_reason'] is None,
            'clean owned phase exit required')


def aggregate(records):
    return {
        'edge_seconds': math.fsum(statistics.median(r['edge_seconds']) for r in records),
        'edge_plus_drnl_seconds': math.fsum(statistics.median([a + b for a, b in
            zip(r['edge_seconds'], r['drnl_seconds'])]) for r in records),
    }


def verify(catalog):
    for name, pin in catalog['selected_payloads'].items():
        name = safe(name)
        require((HERE / name).resolve().is_relative_to(HERE), 'catalog payload escaped the extension root')
        require(identity(HERE / name) == pin, 'catalog payload differs: ' + name)
    raw_path = HERE / catalog['recorded']['archive']
    payloads = load_archive(raw_path)
    get = lambda name: decode(payloads['results/' + name])
    life = decode((HERE / catalog['recorded']['lifecycle']).read_bytes())
    expected_numbers = decode((HERE / catalog['recorded']['integration_numbers']).read_bytes())
    before = get('full_input_before.json')
    require(before == get('full_input_after.json') == get('admission.json')['full_original'],
            'whole public input before/after continuity differs')
    require((before['nodes'], before['stored_entries']) == (N, M), 'strict-clean Citation2 required')
    protocol_raw = payloads['sources/COHORT_PROTOCOL.json']
    protocol = decode(protocol_raw)
    require({Path(n).name: {k: row[k] for k in ('bytes', 'sha256')} for n, row in before['files'].items()}
            == protocol['input_files'], 'whole public numeric input hashes differ')
    cohort, plan, frozen = get('cohort/citation2_exact_int32.json'), get('cohort/FROZEN_128_SELECTION.json'), get('cohort/COHORT_READY.json')
    require(plan == get('native/replay_plan.json') and plan['count'] == len(cohort['records']) == 128,
            'same frozen 128 cases required')
    require(frozen['archive'] == {'bytes': len(payloads['results/cohort/citation2_exact_int32.json']),
                                'sha256': digest(payloads['results/cohort/citation2_exact_int32.json'])}
            and frozen['protocol']['sha256'] == digest(protocol_raw), 'frozen cohort/protocol binding differs')
    require(frozen['timing_or_model_results_consulted'] is False and frozen['replacement_on_guard_failure'] is False,
            'pre-timing cohort required')
    candidates = [(sampler, cap, i, 'positive' if i % 2 == 0 else 'twohop_nonedge')
                  for sampler in protocol['samplers'] for cap in protocol['caps'] for i in range(128)]
    groups = collections.defaultdict(list)
    for i, (sampler, cap, _, workload) in enumerate(candidates):
        groups[(sampler, cap, workload)].append(i)
    ordered = [sorted(groups[key], key=lambda i: digest(
        f'{digest(protocol_raw)}:{protocol["candidate_selection_seed"]}:{i}'.encode())) for key in sorted(groups, key=repr)]
    selected, at = [], 0
    while len(selected) < 128:
        for group in ordered:
            if at < len(group):
                selected.append(group[at])
            if len(selected) == 128:
                break
        at += 1
    require(sorted(selected) == cohort['configuration']['selected_candidate_indices'], 'deterministic selection differs')
    for i, (row, planned) in enumerate(zip(cohort['records'], plan['selected'])):
        nodes = row['nodes']
        require(len(set(nodes)) == len(nodes) and 2 <= len(nodes) <= 402 and row['pair'] == nodes[:2]
                and all(type(n) is int and 0 <= n < N for n in nodes), 'invalid ordered node set')
        require(digest(struct.pack('<' + 'q' * len(nodes), *nodes)) == row['node_set_sha256'], 'ordered S hash differs')
        require(planned['archive_record_index'] == i
                and all(row[k] == v for k, v in planned.items() if k != 'archive_record_index'), 'oracle/plan binding differs')
    base, isolated = get('base_environment_before.json'), get('isolated_environment.json')
    require(base == get('base_environment_after_install.json') == get('base_environment_after.json')
            and isolated == get('isolated_environment_after.json'), 'environment before/after differs')
    versions = {'torch': '2.2.2+cu121', 'numpy': '1.24.4', 'scipy': '1.10.1', 'numba': '0.58.0'}
    for env in (base, isolated):
        require(env['python'] == '3.10.12' and env['machine'] == 'x86_64' and env['platform'].startswith('Linux-')
                and env['packages'] == versions and env['CUDA_VISIBLE_DEVICES'] == ''
                and set(env['thread_environment'].values()) == {'1'}, 'actual target stack differs')
    tree = ast.parse(payloads['sources/cpu_environment_v1.py'].decode())
    wheels = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == 'WHEEL_MEMBERS' for t in n.targets))
    for pkg, ver in (('torch_sparse', '0.6.18+pt22cu121'), ('torch_scatter', '2.1.2+pt22cu121')):
        native = isolated['native'][pkg]
        require(native['installed'] is True and native['version'] == ver
                and native['source'] == wheels[pkg + '/__init__.py'], 'native package admission differs')
        for name, pin in native['binaries'].items():
            require({k: pin[k] for k in ('bytes', 'sha256')} == wheels[pkg + '/' + name], 'native binary fingerprint differs')
    conversion = get('native/native_conversion_receipt.json')
    require(conversion['status'] == 'complete' and conversion['actual_native_fixture']['status'] == 'pass'
            and conversion['runtime_native_content_verification']['status'] == 'pass', 'actual conversion/fixture required')
    meta = conversion['conversion']
    require(all(meta[k] == v for k, v in before['expected_native_hashes'].items())
            and meta['sentinel_added'] is True and meta['graph']['native_int64_storage_bytes'] == 509051728,
            'lossless independently admitted conversion differs')
    upstream = get('native/upstream_native_admission.json')
    require(upstream['status'] == 'complete' and upstream['all_executed_arms_exact'] is True
            and upstream['native_admission_fixture']['status'] == 'pass', 'genuine installed upstream fixture required')
    sums, fingerprints = {}, None
    for mode in ('persistent_touched_reset', 'reference_full_clear', 'upstream_torch_sparse_direct'):
        report = get('native/' + mode + '.json')
        require(report['status'] == 'complete' and report['record_count'] == 128
                and report['configuration']['warmup'] == 1 and report['configuration']['repeats'] == 3
                and report['archive_sha256'] == frozen['archive']['sha256'], 'native report scope differs')
        require(report['native_setup']['original_conversion_manifest'] == meta
                and report['actual_native_admission']['status'] == 'pass', 'native setup/ABI fixture differs')
        observed = []
        for i, row in enumerate(report['records']):
            require(row['archive_record_index'] == i and row['node_set_sha256'] == cohort['records'][i]['node_set_sha256']
                    and row['status'] == 'ok' and row['exact_edges'] is True and row['exact_drnl'] is True
                    and row['exact_ordered_rows_cols_and_original_eids'] is True, 'native cohort/oracle differs')
            for key in ('edge_seconds', 'drnl_seconds', 'raw_operator_seconds'):
                vector(row[key])
            observed.append(row['ordered_output_sha256'])
        require(fingerprints is None or fingerprints == observed, 'ordered rows/columns/edge-ID hashes differ')
        fingerprints = observed
        sums['native_' + mode] = aggregate(report['records'])
    for width in (32, 64):
        report = get(f'citation2_neighbors{width}.json')
        require(report['status'] == 'complete' and report['record_count'] == 128 and report['neighbor_width'] == width
                and report['numba_threads'] == 1 and report['torch_imported'] is False
                and report['host'] == get('host.json') and all(report['compiled_kernels'].values()), 'actual Numba route differs')
        for i, row in enumerate(report['records']):
            require(row['archive_record_index'] == i and row['exact_archived_oracles'] is True
                    and all(row[k] == cohort['records'][i][k] for k in
                            ('node_set_sha256', 'canonical_edges_sha256', 'drnl_sha256')), 'Numba oracle differs')
            for item in row['methods'].values():
                require(item['status'] == 'ok' and item['exact_edges'] is True and item['exact_drnl'] is True, 'Numba exactness differs')
                for key in ('edge_seconds', 'drnl_seconds', 'edge_plus_drnl_seconds'):
                    vector(item[key])
                require(all(close(a + b, c) for a, b, c in zip(item['edge_seconds'], item['drnl_seconds'], item['edge_plus_drnl_seconds'])),
                        'combined repeat time differs')
        for method in ('scan', 'pairwise'):
            sums[f'numba{width}_{method}'] = aggregate([r['methods'][method] for r in report['records']])
    for row in expected_numbers['table_rows']:
        require(all(close(sums[row['arm']][key], value) for key, value in row['exact_seconds']['citation2'].items()),
                'recomputed E22 manuscript row differs')
    for phase in get('stages.json')['stages']:
        clean(phase['exit'])
    terminal = get('SCIENTIFIC_TERMINAL.json')
    require(terminal['success'] is True and terminal['owned_tree_quiescent'] is True, 'scientific completion differs')
    final = life['results/citation2_cpu_matched_v1_20261126/COMPLETE.json']
    outer = life['results/citation2_cpu_matched_v1_20261126.outer_exit.json']
    require(final['success'] is True and final['timing_lock_released'] is True and final['owned_tree_quiescent'] is True
            and outer['returncode'] == 0 and outer['wait_returned'] is True
            and outer['child_pid'] == final['controller_identity']['pid']
            and life['pid17015_exists'] is False and life['pid17016_exists'] is False, 'actual outer completion differs')
    return {'status': 'PASS', 'checked_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'raw_archive': identity(raw_path), 'archive_members': len(payloads), 'source_manifest_sha256': SOURCE_MANIFEST,
            'cohort_cases': 128, 'distinct_physical_pairs': len({tuple(r['pair']) for r in cohort['records']}),
            'distinct_ordered_sets': len({r['node_set_sha256'] for r in cohort['records']}),
            'sum_of_128_per_case_medians_seconds': sums,
            'scope': 'Recorded E22 verification and numerical reaggregation only. No graph arrays, models, native libraries or scientific modules are loaded; no fresh timing is performed.',
            'fresh_Linux_execution_performed_by_verifier': False,
            'outside_source_paths_used': False, 'network_or_cluster_actions': 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='New JSON report; default prints to stdout')
    args = parser.parse_args(argv)
    catalog = decode((HERE / 'RESULT_CATALOG.json').read_bytes())
    result = verify(catalog)
    text = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.output is not None:
        with args.output.open('x') as stream:
            stream.write(text)
    print(text, end='')


if __name__ == '__main__':
    main()
