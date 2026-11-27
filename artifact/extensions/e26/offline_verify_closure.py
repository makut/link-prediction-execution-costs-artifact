#!/usr/bin/env python3
"""Replay a complete, externally SHA-pinned closure evidence index offline.

Uses only the standard library. No downloads, ML imports, cluster calls or
recorded remote paths are used. An incomplete/pending index is never accepted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys

MIB = 1 << 20
KIND = 'CLOSURE_V1_PUBLIC_OFFLINE_EVIDENCE_INDEX'
STATUS = 'COMPLETE_36_APPLICATIONS'
AUDITOR_BYTES = 93988
AUDITOR_SHA = '98ad346c6d6529878e859441df72ce9b33c226eb01229bf015adda1926b0834c'
SOURCE_HASHES = {
    'source_manifest_sha256': '0bb01dfd0524340e5f3f9f08ab1e0f2fc604e47a726db498f59ee1b376149bb6',
    'protocol_sha256': 'ea487e898ae9dd07113408990bb48e55e37dafeef3d23e5b8ef0ab1d88462566',
    'wrapper_sha256': 'c5b90c729ca3956b9c306967161f8b8a676733d565149f35a55f11757e85a5d8',
}
ORDER = [(dataset, block) for block in (1, 2, 3)
         for dataset in ('friendster', 'citation2')]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def whole_sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def exact_keys(value, keys, label):
    require(isinstance(value, dict) and set(value) == set(keys), label + ': exact keys required')


def plain_path(path, *, missing_final=False):
    """Reject symlinks in every component, including the supplied parent tree."""
    require(path.is_absolute(), 'absolute internal path required')
    current = Path(path.anchor)
    for number, part in enumerate(path.parts[1:], 1):
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            require(missing_final and number == len(path.parts) - 1,
                    'input or output parent is missing: ' + str(current))
            return
        require(not stat.S_ISLNK(mode), 'symlink path refused: ' + str(current))
        if number < len(path.parts) - 1:
            require(stat.S_ISDIR(mode), 'non-directory path parent: ' + str(current))


def signature(value):
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def stream_pin(path, maximum, *, capture=False):
    plain_path(path)
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and before.st_size <= maximum,
            'bounded regular file required: ' + str(path))
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
    digest, size, chunks = hashlib.sha256(), 0, []
    with os.fdopen(os.open(path, flags), 'rb') as stream:
        require(signature(os.fstat(stream.fileno())) == signature(before),
                'file changed before opening: ' + str(path))
        for chunk in iter(lambda: stream.read(MIB), b''):
            size += len(chunk)
            require(size <= maximum, 'growing input: ' + str(path))
            digest.update(chunk)
            if capture:
                chunks.append(chunk)
        require(signature(os.fstat(stream.fileno())) == signature(before),
                'file changed while hashing: ' + str(path))
    require(signature(path.lstat()) == signature(before) and size == before.st_size,
            'file changed after hashing: ' + str(path))
    return {'bytes': size, 'sha256': digest.hexdigest()}, b''.join(chunks) if capture else None


def unique_object(pairs):
    value = {}
    for key, item in pairs:
        require(key not in value, 'duplicate JSON key: ' + key)
        value[key] = item
    return value


def decode(raw):
    def bad_constant(value):
        raise ValueError('non-finite JSON constant: ' + value)
    return json.loads(raw, object_pairs_hook=unique_object, parse_constant=bad_constant)


def supplied_pin(base, value, maximum, used, label):
    exact_keys(value, ('path', 'bytes', 'sha256'), label)
    name = value['path']
    require(isinstance(name, str) and 0 < len(name) <= 4096 and
            not any(ord(c) < 32 for c in name) and '\\' not in name,
            label + ': bounded relative POSIX path required')
    relative = PurePosixPath(name)
    require(not relative.is_absolute() and '..' not in relative.parts and
            str(relative) == name and name != '.', label + ': unsafe relative path')
    require(name not in used, label + ': duplicate input path')
    used.add(name)
    require(type(value['bytes']) is int and 0 < value['bytes'] <= maximum and
            whole_sha(value['sha256']), label + ': positive bounded bytes and whole SHA required')
    path = base.joinpath(*relative.parts)
    actual, _ = stream_pin(path, maximum)
    require(actual == {k: value[k] for k in ('bytes', 'sha256')}, label + ': supplied file pin differs')
    return path


def validate_index(index_path, index_sha):
    require(whole_sha(index_sha), '--index-sha256 must be a lowercase whole SHA256')
    anchor, raw = stream_pin(index_path, MIB, capture=True)
    require(anchor['sha256'] == index_sha, 'index whole SHA differs from supplied trust anchor')
    value = decode(raw)
    exact_keys(value, ('kind', 'status', 'source_hashes', 'auditor', 'blocks', 'aggregates'), 'index')
    require(value['kind'] == KIND and value['status'] == STATUS,
            'only the fixed COMPLETE_36_APPLICATIONS index is accepted; pending is not evidence')
    require(value['source_hashes'] == SOURCE_HASHES, 'original frozen scientific source hashes required')
    base, used = index_path.parent, {index_path.name}
    auditor = supplied_pin(base, value['auditor'], AUDITOR_BYTES, used, 'auditor')
    require(value['auditor']['bytes'] == AUDITOR_BYTES and value['auditor']['sha256'] == AUDITOR_SHA,
            'exact original independent auditor required')
    require(isinstance(value['blocks'], list) and len(value['blocks']) == 6,
            'exactly six blocks required')
    blocks = []
    for row, (dataset, block) in zip(value['blocks'], ORDER):
        exact_keys(row, ('dataset_id', 'block', 'archive', 'root_collection', 'recorded_audit'), 'block')
        require(row['dataset_id'] == dataset and type(row['block']) is int and row['block'] == block,
                'block order must be F1,C1,F2,C2,F3,C3')
        paths = {key: supplied_pin(base, row[key], 128 * MIB if key == 'archive' else 32 * MIB,
                                   used, dataset + str(block) + '.' + key)
                 for key in ('archive', 'root_collection', 'recorded_audit')}
        blocks.append((row, paths))
    require(isinstance(value['aggregates'], list) and len(value['aggregates']) == 2,
            'exactly two recorded dataset aggregates required')
    aggregates = []
    for row, dataset in zip(value['aggregates'], ('friendster', 'citation2')):
        exact_keys(row, ('dataset_id', 'recorded_aggregate'), 'aggregate')
        require(row['dataset_id'] == dataset, 'aggregate order must be Friendster,Citation2')
        supplied_pin(base, row['recorded_aggregate'], 32 * MIB, used, dataset + '.recorded_aggregate')
        aggregates.append(row)
    actual, auditor_bytes = stream_pin(auditor, AUDITOR_BYTES, capture=True)
    require(actual == {'bytes': AUDITOR_BYTES, 'sha256': AUDITOR_SHA}, 'auditor changed before snapshot')
    require(stream_pin(index_path, MIB)[0] == anchor, 'index changed during input validation')
    return anchor, auditor_bytes, blocks, aggregates


def write_json(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def validate_block_report(report, row, seen_pods, seen_archives):
    """Bind a freshly replayed audit to its declared cell and fresh allocation."""
    require(isinstance(report, dict), 'generated block report must be an object')
    require(report.get('kind') == 'INDEPENDENT_CLOSURE_V1_SIX_APP_BLOCK_AUDIT' and
            report.get('status') == 'PASS_SIX_APP_DATASET_BLOCK' and
            report.get('dataset_id') == row['dataset_id'], 'generated block dataset/kind/status differs')
    for key, value in (('repetition_block', row['block']), ('block_jobs', 6),
                       ('global_protocol_jobs', 18), ('global_two_dataset_jobs', 36)):
        require(type(report.get(key)) is int and report[key] == value,
                'generated block identity/count differs: ' + key)
    require(report.get('global_dataset_complete') is False and
            report.get('all_three_complete_blocks_required') is True and
            report.get('source_hashes') == SOURCE_HASHES and
            report.get('script') == {'bytes': AUDITOR_BYTES, 'sha256': AUDITOR_SHA},
            'generated block completeness/source/auditor differs')
    for key in ('archive', 'root_collection'):
        require(report.get(key) == {k: row[key][k] for k in ('bytes', 'sha256')},
                'generated block input binding differs: ' + key)
    platform = report.get('platform')
    require(isinstance(platform, dict) and platform.get('dataset_id') == row['dataset_id'] and
            type(platform.get('block')) is int and platform['block'] == row['block'],
            'generated block actual platform dataset/block differs')
    pod = platform.get('pod_uid')
    require(isinstance(pod, str) and 0 < len(pod) <= 256, 'generated block actual Pod UID required')
    archive = report['archive']['sha256']
    require(pod not in seen_pods and archive not in seen_archives,
            'all six blocks require distinct actual Pod UIDs and raw archive SHAs')
    seen_pods.add(pod)
    seen_archives.add(archive)


def validate_aggregate_report(report, row):
    """Do not let an internally valid aggregate stand in for the other graph."""
    require(isinstance(report, dict) and
            report.get('kind') == 'CLOSURE_V1_THREE_BLOCK_DATASET_AGGREGATE' and
            report.get('status') == 'PASS_THREE_BLOCKS_18_APPS' and
            report.get('dataset_id') == row['dataset_id'],
            'generated aggregate dataset/kind/status differs')
    require(type(report.get('blocks')) is int and report['blocks'] == 3 and
            type(report.get('jobs')) is int and report['jobs'] == 18 and
            report.get('global_dataset_complete') is True and
            report.get('global_two_dataset_complete') is False and
            report.get('source_hashes') == SOURCE_HASHES and
            report.get('auditor_source_sha256') == AUDITOR_SHA,
            'generated aggregate counts/completeness/source/auditor differs')


def generated_report(path, expected):
    actual, raw = stream_pin(path, 32 * MIB, capture=True)
    require(actual == expected, 'generated report changed before semantic validation')
    return decode(raw)


def replay(snapshot, out, label, arguments, expected):
    output = out / (label + '.json')
    command = [sys.executable, '-B', '-S', str(snapshot), *arguments, '--output', str(output)]
    with (out / (label + '.log')).open('xb') as log:
        result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                cwd=out, env={'PATH': os.defpath, 'LANG': 'C'}, timeout=600, check=False)
    require(result.returncode == 0, label + ': auditor returned ' + str(result.returncode))
    actual, _ = stream_pin(output, 32 * MIB)
    require(actual == {k: expected[k] for k in ('bytes', 'sha256')},
            label + ': replay differs from original recorded bytes/SHA')
    return output, actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--index-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True, help='fresh directory with an existing parent')
    args = parser.parse_args()
    index_path = Path(os.path.abspath(args.index))
    out = Path(os.path.abspath(args.out))
    created, completed, anchor = False, [], None
    try:
        anchor, auditor_bytes, blocks, aggregates = validate_index(index_path, args.index_sha256)
        require(not out.is_relative_to(index_path.parent) and not index_path.parent.is_relative_to(out),
                'output must be outside and disjoint from the directory containing the index')
        plain_path(out, missing_final=True)
        require(not os.path.lexists(out), 'fresh output directory required')
        out.mkdir()
        created = True
        snapshot = out / 'auditor_snapshot.py'
        with snapshot.open('xb') as stream:
            stream.write(auditor_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        snapshot.chmod(0o444)
        require(stream_pin(snapshot, AUDITOR_BYTES)[0] == {'bytes': AUDITOR_BYTES, 'sha256': AUDITOR_SHA},
                'written auditor snapshot differs')
        regenerated, seen_pods, seen_archives = {}, set(), set()
        for row, paths in blocks:
            dataset, block = row['dataset_id'], row['block']
            label = dataset + '_block' + str(block)
            command = ['block', '--archive', str(paths['archive']), '--archive-sha256', row['archive']['sha256'],
                       '--root-collection', str(paths['root_collection']),
                       '--root-collection-sha256', row['root_collection']['sha256']]
            for key, digest in SOURCE_HASHES.items():
                command += ['--' + key.replace('_', '-'), digest]
            path, actual = replay(snapshot, out, label, command, row['recorded_audit'])
            validate_block_report(generated_report(path, actual), row, seen_pods, seen_archives)
            regenerated[dataset, block] = (path, actual)
            completed.append({'step': label, **actual})
        for row in aggregates:
            dataset = row['dataset_id']
            command = ['aggregate']
            for block in (1, 2, 3):
                path, actual = regenerated[dataset, block]
                command += ['--block-audit', str(path), actual['sha256']]
            path, actual = replay(snapshot, out, dataset + '_aggregate', command, row['recorded_aggregate'])
            validate_aggregate_report(generated_report(path, actual), row)
            completed.append({'step': dataset + '_aggregate', **actual})
        require(len(completed) == 8 and len(seen_pods) == len(seen_archives) == 6 and
                stream_pin(index_path, MIB)[0] == anchor,
                'all eight byte-identical replays and unchanged trust anchor required')
        write_json(out / 'OFFLINE_VERIFIED.json', {
            'kind': 'CLOSURE_V1_PUBLIC_OFFLINE_ROUNDTRIP', 'status': 'PASS_COMPLETE_36_APPLICATIONS',
            'index': anchor, 'auditor': {'bytes': AUDITOR_BYTES, 'sha256': AUDITOR_SHA},
            'source_hashes': SOURCE_HASHES, 'six_block_audits_and_two_aggregates': completed,
            'all_eight_byte_identical_to_recorded': True,
            'scope': 'Offline replay of supplied complete evidence only; no new experiments, downloads, '
                     'ML imports, remote APIs or recorded remote-path reads.'})
        print(json.dumps({'status': 'PASS_COMPLETE_36_APPLICATIONS', 'verified_replays': 8, 'out': str(out)}))
        return 0
    except Exception as error:
        failure = {'kind': 'CLOSURE_V1_PUBLIC_OFFLINE_ROUNDTRIP_FAILURE', 'status': 'FAIL',
                   'index': anchor, 'completed_byte_identical_replays': completed,
                   'error_type': type(error).__name__, 'error': str(error)[:2000]}
        if created:
            try:
                write_json(out / 'OFFLINE_FAILED.json', failure)
            except Exception:
                pass
        print(json.dumps({'status': 'FAIL', 'completed_replays': len(completed),
                          'error_type': failure['error_type'], 'error': failure['error']}), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
