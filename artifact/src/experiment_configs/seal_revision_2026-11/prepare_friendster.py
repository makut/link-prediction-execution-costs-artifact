#!/usr/bin/env python3
"""Prepare the colleague's EXACT Friendster 80/10/10 split, offline and bounded.

Reads immutable little-endian full GraphCSR; two Numba passes apply the archived
thresholds (no histogram, global edge sort or regeneration of evaluation queries).
Writes train_csr and canonical train_pairs_csr plus all original query arrays.

Repeated source IDs exist: evaluate by ORIGINAL QUERY ROW, never source grouping.
Flat SEAL files preserve rows but do not repair a source-grouping evaluator.

    python prepare_friendster.py --full-graph DATA/friendster/full_csr \
      --eval-bundle DATA/friendster/friendster_eval.tgz \
      --source-manifest DATA/input_download_manifest.json \
      --out DATA/friendster/prepared --threads 64

No network, cluster access, training or execution of archived scripts occurs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path, PurePosixPath
import resource
import shutil
import struct
import sys
import tarfile
import time
import uuid

import numba as nb
import numpy as np

SEED = 20261102
TRAIN_THRESHOLD = 14757470836727281938
VALID_THRESHOLD = 16602285075010369541
TRUSTED_REFERENCE_SHA256 = '8df41e2f06c2c7356b9b54e3e260451f1140703de87fc971cbbee5d2e5877c56'
HISTORICAL_PREPARE_SHA256 = '37b9a778038007e099dbe7b7f6e94ff0ad1e02f44ef0f088d7a5299d2d148939'
YT_SOURCE = '//home/ucp-research/projects/graph_baselines/friendster/csr'
LOG = logging.getLogger('prepare_friendster_revision')


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, payload):
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w') as f:
        json.dump(payload, f, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    temp.replace(path)


def memory_status():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {'peak_rss_bytes': int(peak if sys.platform == 'darwin' else peak * 1024)}


@nb.njit(inline='always')
def mix(x):
    # Verbatim splitmix64 arithmetic from the audited colleague preparer.
    x = x + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


@nb.njit(inline='always')
def key_for(u, v):
    if u > v:
        u, v = v, u
    return (np.uint64(u) << np.uint64(32)) | np.uint64(v)


@nb.njit(inline='always')
def row_end(starts, m, u):
    return starts[u + 1] if u + 1 < len(starts) else m


@nb.njit(inline='always')
def has_edge(starts, col, u, v):
    lo = starts[u]; end = row_end(starts, len(col), u); hi = end
    while lo < hi:
        mid = (lo + hi) // 2
        if col[mid] < v:
            lo = mid + 1
        else:
            hi = mid
    return lo < end and col[lo] == v


@nb.njit(cache=True, parallel=True)
def count_split(starts, col, seed, threshold0, threshold1, train_counts, pair_counts, blocks):
    """Pass1: counts and full touched-input sorted/simple/bounds checks, O(N)."""
    n = len(starts)
    stats = np.zeros((blocks, 4), dtype=np.int64)  # canonical train/valid/test, input errors
    for block in nb.prange(blocks):
        canonical_train, canonical_valid, canonical_test, invalid = 0, 0, 0, 0
        for u in range(n * block // blocks, n * (block + 1) // blocks):
            lo, hi = starts[u], row_end(starts, len(col), u)
            if lo < 0 or hi < lo or hi > len(col):
                invalid += 1
                continue
            kept, upper = 0, 0
            previous = -1
            for at in range(lo, hi):
                v = int(col[at])
                if v < 0 or v >= n or v == u or v <= previous:
                    invalid += 1
                previous = v
                if v < 0 or v >= n:
                    continue
                h = mix(key_for(u, v) ^ seed)
                if h <= threshold0:
                    kept += 1
                    if u < v:
                        upper += 1; canonical_train += 1
                elif u < v:
                    if h <= threshold1:
                        canonical_valid += 1
                    else:
                        canonical_test += 1
            train_counts[u] = kept
            pair_counts[u] = upper
        stats[block, 0] = canonical_train
        stats[block, 1] = canonical_valid
        stats[block, 2] = canonical_test
        stats[block, 3] = invalid
    return stats


@nb.njit(cache=True, parallel=True)
def fill_split(starts, col, seed, threshold, train_starts, pair_starts, train_col, pair_col, blocks):
    """Pass2: bounded writes to disjoint preallocated CSR rows; no edge sorting."""
    n = len(starts)
    mismatches = np.zeros(blocks, dtype=np.int64)
    for block in nb.prange(blocks):
        invalid = 0
        for u in range(n * block // blocks, n * (block + 1) // blocks):
            out, upper = train_starts[u], pair_starts[u]
            for at in range(starts[u], row_end(starts, len(col), u)):
                v = int(col[at])
                if mix(key_for(u, v) ^ seed) <= threshold:
                    train_col[out] = v; out += 1
                    if u < v:
                        pair_col[upper] = v; upper += 1
            if out != row_end(train_starts, len(train_col), u) or upper != row_end(pair_starts, len(pair_col), u):
                invalid += 1
        mismatches[block] = invalid
    return mismatches


@nb.njit(cache=True, parallel=True)
def check_queries(full_starts, full_col, train_starts, train_col, sources, targets, negatives, warm,
                  seed, threshold0, threshold1, split_index, regenerate):
    """Validate all query groups and exactly reproduce the archived negative RNG.

    Error bits: 1 invalid positive,2 missing full edge,4 train leakage,8 wrong
    heldout partition,16 invalid/neighbor negative,32 warm-mask mismatch,
    64 deterministic original negative generator mismatch.
    """
    n, k = len(full_starts), negatives.shape[1]
    errors = np.zeros(len(sources), dtype=np.int64)
    for i in nb.prange(len(sources)):
        u, v = int(sources[i]), int(targets[i])
        if u < 0 or u >= n or v < 0 or v >= n or u == v:
            errors[i] |= 1
            continue
        if not has_edge(full_starts, full_col, u, v) or not has_edge(full_starts, full_col, v, u):
            errors[i] |= 2
        if has_edge(train_starts, train_col, u, v) or has_edge(train_starts, train_col, v, u):
            errors[i] |= 4
        h = mix(key_for(u, v) ^ seed)
        if h <= threshold0 or (split_index == 0 and h > threshold1) or (split_index == 1 and h <= threshold1):
            errors[i] |= 8
        actual_warm = (row_end(train_starts, len(train_col), u) > train_starts[u]
                       and row_end(train_starts, len(train_col), v) > train_starts[v])
        if actual_warm != warm[i]:
            errors[i] |= 32
        table = np.full(2048, -1, dtype=np.int64)
        for j in range(k):
            saved = int(negatives[i, j])
            if saved < 0 or saved >= n or saved == u:
                errors[i] |= 16
            elif has_edge(full_starts, full_col, u, saved) or has_edge(full_starts, full_col, saved, u):
                errors[i] |= 16
            if regenerate:
                counter = np.uint64(i) * np.uint64(k) + np.uint64(j)
                attempt = np.uint64(0)
                while True:
                    candidate_hash = mix(seed + np.uint64(10 + split_index) + counter
                                         + attempt * np.uint64(0x9E3779B97F4A7C15))
                    candidate = int(candidate_hash % np.uint64(n))
                    attempt += np.uint64(1)
                    if candidate == u or has_edge(full_starts, full_col, u, candidate):
                        continue
                    slot = candidate % 2048
                    while table[slot] != -1 and table[slot] != candidate:
                        slot = (slot + 1) % 2048
                    if table[slot] == candidate:
                        continue
                    table[slot] = candidate
                    if candidate != saved:
                        errors[i] |= 64
                    break
            else:
                # Fixture mode can check arbitrary valid candidate arrays too.
                slot = saved % 2048
                while table[slot] != -1 and table[slot] != saved:
                    slot = (slot + 1) % 2048
                if table[slot] == saved:
                    errors[i] |= 16
                table[slot] = saved
    return errors


class CSR:
    def __init__(self, root):
        self.root = Path(root)
        offsets = self.root / 'edge_starts.bin'; ends = self.root / 'edge_ends.bin'
        if offsets.stat().st_size == 0 or offsets.stat().st_size % 8:
            raise ValueError('invalid raw int64 offset file')
        self.starts = np.memmap(offsets, dtype='<i8', mode='r')
        with ends.open('rb') as f:
            self.m = struct.unpack('<q', f.read(8))[0]
        if self.m < 0 or ends.stat().st_size != 8 + self.m * 4:
            raise ValueError('neighbor count header/file size mismatch')
        self.col = np.memmap(ends, dtype='<i4', offset=8, shape=(self.m,), mode='r') if self.m else np.empty(0, dtype=np.int32)
        self.n = int(self.starts.size)
        if self.starts[0] != 0:
            raise ValueError('first CSR offset must be zero')


def safe_bundle_extract(bundle, destination, *, trusted_reference_sha=TRUSTED_REFERENCE_SHA256):
    if destination.exists():
        raise FileExistsError(destination)
    with tarfile.open(bundle, 'r:gz') as archive:
        members = archive.getmembers()
        names = set()
        for member in members:
            path = PurePosixPath(member.name)
            if (path.is_absolute() or '..' in path.parts or not path.parts or '\\' in member.name
                    or ':' in path.parts[0] or not member.isfile() or member.name in names):
                raise ValueError(f'unsafe or duplicate eval bundle member: {member.name}')
            names.add(member.name)
        if sum(m.size for m in members) > 100 << 20 or len(members) > 100:
            raise ValueError('eval bundle exceeds bounded size/member budget')
        member = archive.getmember('BUNDLE_MANIFEST.json')
        if member.size > 2 << 20:
            raise ValueError('oversized bundle manifest')
        manifest = json.load(archive.extractfile(member))
        if set(manifest['files']) | {'BUNDLE_MANIFEST.json'} != names:
            raise ValueError('bundle file list differs from declared manifest layout')
        destination.mkdir(parents=True)
        for member in members:
            target = destination.joinpath(*PurePosixPath(member.name).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            h, written = hashlib.sha256(), 0
            with archive.extractfile(member) as src, target.open('xb') as out:
                for block in iter(lambda: src.read(4 << 20), b''):
                    out.write(block); h.update(block); written += len(block)
            if member.name != 'BUNDLE_MANIFEST.json':
                expected = manifest['files'][member.name]
                if written != expected['bytes'] or h.hexdigest() != expected['sha256']:
                    raise ValueError(f'eval bundle content hash mismatch: {member.name}')
    reference_path = destination / 'reference_manifest.json'
    if sha256_file(reference_path) != trusted_reference_sha or manifest['reference_manifest_sha256'] != trusted_reference_sha:
        raise ValueError('bundle reference is not the audited latest12Nov reference')
    reference = json.loads(reference_path.read_text())
    for name, expected in reference['evaluation_files'].items():
        if sha256_file(destination / name) != expected['sha256'] or (destination / name).stat().st_size != expected['bytes']:
            raise ValueError(f'evaluation array does not match latest reference: {name}')
    evidence = json.loads((destination / 'provenance/data_validation_latest.json').read_text())
    if not manifest['latest_series_evaluation_hashes_confirmed'] or not evidence['all_validation_test_arrays_sha256_equal']:
        raise ValueError('latest-series evaluation equality evidence absent')
    if trusted_reference_sha == TRUSTED_REFERENCE_SHA256:
        if sha256_file(destination / 'provenance/prepare_historical.py') != HISTORICAL_PREPARE_SHA256:
            raise ValueError('historical hash/RNG source provenance mismatch')
    return reference, manifest


def verify_full_source(full_root, download_manifest_path, reference):
    manifest = json.loads(Path(download_manifest_path).read_text())
    result = {'full_graph_path': str(Path(full_root).resolve()), 'download_manifest_sha256': sha256_file(download_manifest_path),
              'historical_source_revisions_equal': True, 'files': {}}
    for name, expected in reference['source_files'].items():
        source = YT_SOURCE + '/' + name
        matching = [value for value in manifest.values() if isinstance(value, dict) and value.get('source') == source]
        if len(matching) != 1:
            raise ValueError(f'need exactly one staged immutable source record: {source}')
        staged = matching[0]; path = Path(full_root) / name
        if int(staged['revision']) != expected['revision'] or staged['bytes'] != expected['bytes']:
            raise ValueError(f'full graph source revision/size differs from latest colleague reference: {name}')
        if path.stat().st_size != staged['bytes'] or sha256_file(path) != staged['sha256']:
            raise ValueError(f'full graph staged checksum mismatch: {name}')
        result['files'][name] = {'bytes': path.stat().st_size, 'sha256': staged['sha256'], 'source': source,
                                'revision': int(staged['revision']), 'mtime_ns': path.stat().st_mtime_ns}
    graph = CSR(full_root)
    expected = reference['split_metadata']
    if graph.n != expected['num_nodes'] or graph.m != expected['original_stored_entries']:
        raise ValueError('full graph N/M differs from archived split metadata')
    return result


def exclusive_prefix_in_place(array, chunk_nodes=1_000_000):
    carry, isolates = 0, 0
    for lo in range(0, len(array), chunk_nodes):
        hi = min(len(array), lo + chunk_nodes)
        counts = np.array(array[lo:hi], dtype=np.int64, copy=True)
        isolates += int(np.count_nonzero(counts == 0))
        prefix = np.empty_like(counts); prefix[0] = carry
        if counts.size > 1:
            np.cumsum(counts[:-1], out=prefix[1:]); prefix[1:] += carry
        carry += int(counts.sum())
        array[lo:hi] = prefix
    array.flush()
    return carry, isolates


def create_columns(root, count):
    path = root / 'edge_ends.bin'
    with path.open('xb') as f:
        f.write(struct.pack('<q', count)); f.truncate(8 + count * 4)
    return np.memmap(path, dtype='<i4', offset=8, shape=(count,), mode='r+') if count else np.empty(0, dtype=np.int32)


def flatten_queries(eval_dir, prepared, *, chunk_queries=1024):
    result = {}
    for name in ('valid', 'test'):
        source = np.load(eval_dir / name / 'source_node.npy', mmap_mode='r', allow_pickle=False)
        target = np.load(eval_dir / name / 'target_node.npy', mmap_mode='r', allow_pickle=False)
        negatives = np.load(eval_dir / name / 'target_node_neg.npy', mmap_mode='r', allow_pickle=False)
        q, k = negatives.shape
        pos = np.lib.format.open_memmap(prepared / f'{name}_edge.npy', mode='w+', dtype=np.int64, shape=(q, 2))
        pos[:, 0] = source; pos[:, 1] = target; pos.flush()
        flat = np.lib.format.open_memmap(prepared / f'{name}_edge_neg.npy', mode='w+', dtype=np.int64, shape=(q * k, 2))
        for lo in range(0, q, chunk_queries):
            hi = min(q, lo + chunk_queries)
            block = flat[lo * k:hi * k].reshape(hi - lo, k, 2)
            block[:, :, 0] = source[lo:hi, None]
            block[:, :, 1] = negatives[lo:hi]
        flat.flush()
        np.save(prepared / f'{name}_query_row_ids.npy', np.arange(q, dtype=np.int64))
        unique, counts = np.unique(source, return_counts=True)
        result[name] = {'query_rows': q, 'negatives_per_query': k, 'unique_sources': int(unique.size),
                        'repeated_source_ids': int(np.count_nonzero(counts > 1)), 'max_source_multiplicity': int(counts.max()),
                        'query_id_definition': 'original archived row index, 0..query_rows-1',
                        'source_group_evaluator_compatible': bool(unique.size == q)}
    return result


def verify_queries(full, train, eval_root, metadata, *, regenerate=True):
    results = {}
    for split_index, name in enumerate(('valid', 'test')):
        folder = eval_root / name
        source = np.load(folder / 'source_node.npy', mmap_mode='r', allow_pickle=False)
        target = np.load(folder / 'target_node.npy', mmap_mode='r', allow_pickle=False)
        neg = np.load(folder / 'target_node_neg.npy', mmap_mode='r', allow_pickle=False)
        warm = np.load(folder / 'warm_query.npy', mmap_mode='r', allow_pickle=False)
        q, k = metadata['queries_per_split'], metadata['negatives_per_query']
        if source.shape != (q,) or target.shape != (q,) or warm.shape != (q,) or neg.shape != (q, k) or k > 1000:
            raise ValueError('archived evaluation shape mismatch')
        if source.dtype != np.int64 or target.dtype != np.int64 or neg.dtype != np.int32 or warm.dtype != np.bool_:
            raise ValueError('archived evaluation dtype mismatch')
        errors = check_queries(full.starts, full.col, train.starts, train.col, source, target, neg, warm,
                               np.uint64(metadata['seed']), np.uint64(metadata['threshold_train']),
                               np.uint64(metadata['threshold_validation']), split_index, regenerate)
        if np.any(errors):
            failed = np.flatnonzero(errors)
            raise ValueError(f'{name} query verification failed: first rows/error bits={[(int(i), int(errors[i])) for i in failed[:10]]}')
        actual_warm = int(np.count_nonzero(warm))
        if actual_warm != metadata[name]['warm_queries'] or q - actual_warm != metadata[name]['cold_queries']:
            raise ValueError('warm/cold query counts differ from historical metadata')
        results[name] = {'queries': q, 'negative_occurrences_checked': q * k,
                         'positive_in_full_both_orientations': True, 'positive_absent_train_both_orientations': True,
                         'correct_hash_partition': True, 'negatives_are_full_graph_nonneighbors_both_orientations': True,
                         'no_negative_self': True, 'unique_negatives_per_query': True,
                         'exact_archived_negative_generator_reproduced': regenerate,
                         'negative_distribution': 'historical splitmix64 counter proposals modulo N, rejection of self/full neighbors/repeats; not a new IID claim',
                         'warm_queries': actual_warm, 'cold_queries': q - actual_warm}
    return results


def output_hashes(root):
    return {str(path.relative_to(root)): {'bytes': path.stat().st_size, 'sha256': sha256_file(path)}
            for path in sorted(root.rglob('*')) if path.is_file() and path.name not in ('manifest.json', 'PREPARED_COMPLETE.json')}


def build_prepared(full_root, bundle_dir, out, reference, source_audit, *, blocks=64, chunk_nodes=1_000_000,
                   regenerate=True, strict_official=True, eval_bundle_sha256=None):
    if out.exists():
        raise FileExistsError(f'existing output retained; use explicit verified reuse in main: {out}')
    metadata = reference['split_metadata']
    bundle_metadata = json.loads((bundle_dir / 'BUNDLE_MANIFEST.json').read_text())
    if strict_official:
        if not (source_audit.get('historical_source_revisions_equal') or source_audit.get('public_numeric_payload_hashes_equal_verified')) or not bundle_metadata.get('latest_series_evaluation_hashes_confirmed'):
            raise ValueError('strict preparation requires verified full-graph revisions and latest-array identity')
        if metadata['seed'] != SEED or metadata['threshold_train'] != TRAIN_THRESHOLD or metadata['threshold_validation'] != VALID_THRESHOLD:
            raise ValueError('hash seed/thresholds differ from pinned colleague protocol')
    full = CSR(full_root)
    if full.n != metadata['num_nodes'] or full.m != metadata['original_stored_entries']:
        raise ValueError('input graph dimensions differ from split')
    required = (metadata['train_stored_entries'] + metadata['train_undirected_edges']) * 4 + full.n * 16 + (1 << 30)
    if shutil.disk_usage(out.parent).free < required:
        raise RuntimeError('insufficient free disk for train/pairs CSR and bounded workspace')
    temp = out.with_name(out.name + '.partial-' + uuid.uuid4().hex[:8]); temp.mkdir()
    timings = []
    train_root, pair_root = temp / 'train_csr', temp / 'train_pairs_csr'
    train_root.mkdir(); pair_root.mkdir()
    train_starts = np.memmap(train_root / 'edge_starts.bin', dtype='<i8', mode='w+', shape=(full.n,))
    pair_starts = np.memmap(pair_root / 'edge_starts.bin', dtype='<i8', mode='w+', shape=(full.n,))
    start = time.perf_counter(); LOG.info('PASS1 count/sorted-simple validation START %s', memory_status())
    totals = count_split(full.starts, full.col, np.uint64(metadata['seed']), np.uint64(metadata['threshold_train']),
                         np.uint64(metadata['threshold_validation']), train_starts, pair_starts, blocks).sum(axis=0)
    if totals[3]:
        raise ValueError(f'input sorted/simple/range/offset validation errors={totals[3]}')
    expected_partition = np.array([metadata['train_undirected_edges'], metadata['validation_undirected_edges'], metadata['test_undirected_edges']])
    if not np.array_equal(totals[:3], expected_partition):
        raise ValueError(f'exact threshold partition count mismatch: actual={totals[:3].tolist()}, expected={expected_partition.tolist()}')
    train_m, isolates = exclusive_prefix_in_place(train_starts, chunk_nodes)
    pair_m, _ = exclusive_prefix_in_place(pair_starts, chunk_nodes)
    if train_m != metadata['train_stored_entries'] or pair_m != metadata['train_undirected_edges'] or isolates != metadata['train_isolated_nodes']:
        raise ValueError(f'train/pairs/isolates mismatch: {train_m}/{pair_m}/{isolates}')
    timings.append({'stage': 'pass1_counts_and_prefix', 'seconds': time.perf_counter() - start, **memory_status()})
    LOG.info('PASS1 done %s', timings[-1])
    train_col, pair_col = create_columns(train_root, train_m), create_columns(pair_root, pair_m)
    start = time.perf_counter(); LOG.info('PASS2 fill START')
    errors = fill_split(full.starts, full.col, np.uint64(metadata['seed']), np.uint64(metadata['threshold_train']),
                        train_starts, pair_starts, train_col, pair_col, blocks)
    if np.any(errors):
        raise ValueError(f'pass2 row-count mismatches: {errors.tolist()}')
    for array in (train_starts, pair_starts, train_col, pair_col):
        if isinstance(array, np.memmap):
            array.flush()
    for folder in (train_root, pair_root):
        (folder / 'timestamps.bin').write_bytes(b'')
        (folder / 'num_nodes.txt').write_text(str(full.n) + '\n')
    (temp / 'num_nodes.txt').write_text(str(full.n) + '\n')
    timings.append({'stage': 'pass2_fill', 'seconds': time.perf_counter() - start, **memory_status()})
    LOG.info('PASS2 done %s', timings[-1])
    shutil.copytree(bundle_dir / 'evaluation', temp / 'evaluation')
    shutil.copytree(bundle_dir / 'provenance', temp / 'provenance')
    shutil.copyfile(bundle_dir / 'reference_manifest.json', temp / 'reference_manifest.json')
    shutil.copyfile(bundle_dir / 'BUNDLE_MANIFEST.json', temp / 'BUNDLE_MANIFEST.json')
    start = time.perf_counter(); LOG.info('ALL QUERY checks START')
    train = CSR(train_root)
    checks = verify_queries(full, train, temp / 'evaluation', metadata, regenerate=regenerate)
    groups = flatten_queries(temp / 'evaluation', temp)
    timings.append({'stage': 'all_queries_and_flat_files', 'seconds': time.perf_counter() - start, **memory_status()})
    LOG.info('ALL QUERY checks done %s', timings[-1])
    # Do not modify or silently accept a changing full source during preparation.
    for name, item in source_audit.get('files', {}).items():
        path = Path(full_root) / name
        if path.stat().st_size != item['bytes'] or path.stat().st_mtime_ns != item['mtime_ns']:
            raise ValueError(f'immutable source file changed during preparation: {name}')
    start = time.perf_counter(); LOG.info('OUTPUT HASHES START')
    hashes = output_hashes(temp)
    timings.append({'stage': 'output_hashes', 'seconds': time.perf_counter() - start, **memory_status()})
    manifest = {'schema_version': 1, 'dataset': 'friendster', 'variant': 'exact_colleague_80_10_10',
                'file_endian': 'little', 'features': False, 'num_nodes': full.n,
                'seed': metadata['seed'], 'threshold_train': metadata['threshold_train'],
                'threshold_validation': metadata['threshold_validation'], 'train_stored_entries': train_m,
                'train_undirected_edges': pair_m, 'train_isolated_nodes': isolates,
                'split_metadata': metadata, 'source_graph': source_audit,
                'reference_manifest_sha256': sha256_file(bundle_dir / 'reference_manifest.json'),
                'eval_bundle_sha256': eval_bundle_sha256,
                'historical_hash_source_sha256': HISTORICAL_PREPARE_SHA256,
                'preparer_sha256': sha256_file(Path(__file__)), 'python': sys.version,
                'numpy': np.__version__, 'numba': nb.__version__, 'numba_threads': nb.get_num_threads(),
                'evaluation': {'layout': 'original_query_row',
                               'latest_series_arrays_sha256_equal': bool(bundle_metadata.get('latest_series_evaluation_hashes_confirmed')),
                               'locations': {'valid': 'evaluation/valid', 'test': 'evaluation/test'},
                               'query_id': 'original archived row index 0..9999; retain all rows including repeated source IDs',
                               'group_statistics': groups, 'checks': checks,
                               'source_group_evaluator_forbidden': any(not v['source_group_evaluator_compatible'] for v in groups.values()),
                               'flat_files_scope': 'byte-preserving candidate rows for adapters; do NOT merge/group by source ID'},
                'all_heldout_exclusion': 'construction keeps iff splitmix64(canonical_edge_key xor seed) <= threshold_train, hence excludes every heldout edge in both orientations; all provided queries independently checked',
                'timestamps': 'omitted; original full graph timestamps are zero and neither sampler uses them',
                'files': hashes, 'timings': timings, 'memory': memory_status(), 'completed_unix': time.time()}
    atomic_json(temp / 'manifest.json', manifest)
    atomic_json(temp / 'PREPARED_COMPLETE.json', {'manifest_sha256': sha256_file(temp / 'manifest.json'), 'completed_unix': time.time()})
    temp.rename(out)
    return manifest


def verify_existing(out, bundle_sha, source_audit):
    manifest_path, marker_path = out / 'manifest.json', out / 'PREPARED_COMPLETE.json'
    if not manifest_path.is_file() or not marker_path.is_file():
        raise RuntimeError('existing prepared output is unmanifested/incomplete; refusing overwrite')
    manifest, marker = json.loads(manifest_path.read_text()), json.loads(marker_path.read_text())
    if sha256_file(manifest_path) != marker['manifest_sha256'] or manifest.get('eval_bundle_sha256') != bundle_sha:
        raise ValueError('existing manifest/bundle identity mismatch')
    def identity(audit):
        return {name: {key: value for key, value in item.items() if key != 'mtime_ns'}
                for name, item in audit['files'].items()}
    if identity(manifest['source_graph']) != identity(source_audit):
        raise ValueError('existing source identity differs from current verified input')
    for name, expected in manifest['files'].items():
        path = out / name
        if not path.is_file() or path.stat().st_size != expected['bytes'] or sha256_file(path) != expected['sha256']:
            raise ValueError(f'existing output checksum mismatch: {name}')
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--full-graph', type=Path, required=True)
    parser.add_argument('--eval-bundle', type=Path, required=True)
    parser.add_argument('--eval-bundle-sha256')
    parser.add_argument('--source-manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=min(32, nb.config.NUMBA_NUM_THREADS))
    parser.add_argument('--blocks', type=int, default=64)
    parser.add_argument('--chunk-nodes', type=int, default=1_000_000)
    args = parser.parse_args(argv)
    if args.threads < 1 or args.threads > nb.config.NUMBA_NUM_THREADS or args.blocks < 1 or args.chunk_nodes < 1:
        parser.error('invalid thread/block/chunk budget')
    nb.set_num_threads(args.threads)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    args.out = args.out.resolve(); args.out.parent.mkdir(parents=True, exist_ok=True)
    lock = args.out.parent / 'FRIENDSTER_PREPARATION_RUNNING.lock'
    with lock.open('x') as f:
        f.write(json.dumps({'pid': os.getpid(), 'started_unix': time.time()}))
    try:
        if args.out.exists() and not (args.out / 'PREPARED_COMPLETE.json').is_file():
            raise RuntimeError('existing unmanifested output preserved; no overwrite')
        bundle_sha = sha256_file(args.eval_bundle)
        if args.eval_bundle_sha256 and args.eval_bundle_sha256 != bundle_sha:
            raise ValueError('eval bundle outer SHA256 mismatch')
        extracted = args.out.parent / ('eval_bundle.partial-' + uuid.uuid4().hex[:8])
        reference, bundle_manifest = safe_bundle_extract(args.eval_bundle, extracted)
        if reference['split_metadata']['seed'] != SEED:
            raise ValueError('wrong split seed')
        source_audit = verify_full_source(args.full_graph.resolve(), args.source_manifest.resolve(), reference)
        atomic_json(args.out.parent / 'friendster_input_audit.json', {'source_graph': source_audit, 'eval_bundle_sha256': bundle_sha,
                    'reference_manifest_sha256': TRUSTED_REFERENCE_SHA256, 'all_latest_evaluation_arrays_sha256_equal': True})
        if args.out.exists():
            manifest = verify_existing(args.out, bundle_sha, source_audit)
        else:
            manifest = build_prepared(args.full_graph.resolve(), extracted, args.out, reference, source_audit,
                                      blocks=args.blocks, chunk_nodes=args.chunk_nodes, eval_bundle_sha256=bundle_sha)
        shutil.rmtree(extracted)  # only our own verified extracted copy, not source/bundle
        LOG.info('READY prepared=%s manifest=%s %s', args.out, args.out / 'manifest.json', memory_status())
        return 0
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    raise SystemExit(main())
