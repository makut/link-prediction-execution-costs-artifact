#!/usr/bin/env python3
"""Prepare original and strictly clean citation2 from a supplied LOCAL OGB ZIP.

No downloads or installation are performed. The original cache and prepared
variant are retained. Quality experiments must use ``prepared`` (strict clean),
not ``prepared_original``. Completion is atomic at the directory level. Existing
unmanifested outputs are never overwritten.

    python experiment_configs/seal_revision_2026-11/prepare_citation2.py \
        --zip /home/jovyan/seal_revision_20261124/data/citation2/citation-v2.zip \
        --root /home/jovyan/seal_revision_20261124/data/citation2

The known official split has 62 validation and 67 test reverse-oriented train
collisions: 129 directed arcs, or 258 symmetric adjacency entries. Counts are
asserted rather than silently assumed; both original and clean audits are saved.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import csv
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path, PurePosixPath
import resource
import shutil
import socket
import stat
import struct
import sys
import time
from unittest import mock
import uuid
import zipfile

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CONTRACT_VERSION = 1
LOG = logging.getLogger('prepare_citation2_revision')


def sha256_file(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            result.update(block)
    return result.hexdigest()


def source_provenance() -> dict:
    paths = [Path(__file__), PROJECT_ROOT / 'scripts/prepare_ogb_dataset.py',
             PROJECT_ROOT / 'scripts/transpose_graph_csr.py', PROJECT_ROOT / 'scripts/convert_graph_csr_to_upper.py']
    return {'python': sys.version, 'numpy': np.__version__,
            'source_sha256': {str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in paths}}


def atomic_json(path: Path, payload: dict) -> None:
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w') as f:
        json.dump(payload, f, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def memory_status() -> dict:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = int(peak if sys.platform == 'darwin' else peak * 1024)
    current_bytes = None
    status = Path('/proc/self/status')
    if status.exists():
        for line in status.read_text().splitlines():
            if line.startswith('VmRSS:'):
                current_bytes = int(line.split()[1]) * 1024
    return {'peak_rss_bytes': peak_bytes, 'current_rss_bytes': current_bytes}


@contextmanager
def stage(name: str, timings: list[dict]):
    start = time.perf_counter()
    LOG.info('START %s %s', name, memory_status())
    try:
        yield
    finally:
        item = {'stage': name, 'seconds': time.perf_counter() - start, **memory_status()}
        timings.append(item)
        LOG.info('END %s', item)


@contextmanager
def offline_only():
    """Fail fast on an accidental network/download or interactive OGB update."""
    message = 'Network and interactive prompts are disabled: use the supplied local ZIP/cache'
    with ExitStack() as stack:
        for target in ('socket.socket.connect', 'socket.create_connection',
                       'socket.getaddrinfo', 'urllib.request.urlopen', 'builtins.input'):
            stack.enter_context(mock.patch(target, side_effect=RuntimeError(message)))
        yield


def safe_extract(archive: Path, destination: Path, *, max_uncompressed_bytes: int = 32 << 30,
                 disk_reserve_bytes: int = 1 << 30) -> list[dict]:
    """Validate every name/type/budget before streaming ZIP members into a NEW directory."""
    if destination.exists():
        raise FileExistsError(f'refusing existing extraction directory: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        entries, seen, total = [], set(), 0
        for member in z.infolist():
            name = member.filename
            rel = PurePosixPath(name)
            if ('\\' in name or rel.is_absolute() or '..' in rel.parts or not rel.parts
                    or ':' in rel.parts[0]):
                raise ValueError(f'unsafe ZIP member path: {name!r}')
            normalized = str(rel)
            if normalized in seen:
                raise ValueError(f'duplicate ZIP member: {normalized}')
            seen.add(normalized)
            mode = member.external_attr >> 16
            kind = stat.S_IFMT(mode)
            if kind not in (0, stat.S_IFREG, stat.S_IFDIR) or stat.S_ISLNK(mode):
                raise ValueError(f'non-regular ZIP member: {name}')
            if member.flag_bits & 1:
                raise ValueError(f'encrypted ZIP member is unsupported: {name}')
            total += member.file_size
            entries.append(member)
        if len(entries) > 100000 or total > max_uncompressed_bytes:
            raise ValueError('ZIP extraction exceeds member-count or uncompressed-size budget')
        if shutil.disk_usage(destination.parent).free < total + disk_reserve_bytes:
            raise RuntimeError('insufficient disk space for safe ZIP extraction')
        destination.mkdir()
        manifest = []
        for member in entries:
            target = destination.joinpath(*PurePosixPath(member.filename).parts)
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            checksum, size = hashlib.sha256(), 0
            with z.open(member) as src, target.open('xb') as dst:
                for block in iter(lambda: src.read(4 << 20), b''):
                    dst.write(block); checksum.update(block); size += len(block)
            if size != member.file_size:
                raise ValueError(f'short ZIP member: {member.filename}')
            manifest.append({'path': member.filename, 'bytes': size, 'sha256': checksum.hexdigest(), 'zip_crc32': member.CRC})
        return manifest


def ogb_release_version() -> str:
    spec = importlib.util.find_spec('ogb')
    if spec is None or spec.origin is None:
        raise RuntimeError('OGB must already be installed; no automatic network installation is performed')
    master = Path(spec.origin).parent / 'linkproppred/master.csv'
    with master.open(newline='') as f:
        for row in csv.DictReader(f):
            if row.get('') == 'version':
                return str(row['ogbl-citation2'])
    raise RuntimeError('cannot determine installed OGB citation2 release version')


def make_cache(archive: Path, cache: Path, source: dict, *, max_uncompressed_bytes: int) -> dict:
    """Normalize the discovered archive root to OGB's ogbl_citation2 directory."""
    if cache.exists():
        marker = cache / 'CACHE_COMPLETE.json'
        if not marker.is_file():
            raise RuntimeError(f'existing cache lacks completion manifest: {cache}')
        prior = json.loads(marker.read_text())
        if prior['source_zip']['sha256'] != source['sha256']:
            raise RuntimeError('existing cache originates from a different ZIP')
        for item in prior['files']:
            file = cache / item['path']
            if not file.is_file() or file.stat().st_size != item['bytes'] or sha256_file(file) != item['sha256']:
                raise RuntimeError(f'existing cache failed integrity check: {file}')
        return prior
    temp = cache.with_name(cache.name + '.partial-' + uuid.uuid4().hex[:8])
    extracted = temp / 'extracted'
    temp.mkdir()
    members = safe_extract(archive, extracted, max_uncompressed_bytes=max_uncompressed_bytes)
    candidates = {f.parent.parent for f in extracted.rglob('edge.csv.gz') if f.parent.name == 'raw'}
    candidates |= {f.parent.parent for f in extracted.rglob('data.npz') if f.parent.name == 'raw'}
    if len(candidates) != 1:
        raise RuntimeError(f'expected one OGB raw dataset root, found {len(candidates)}; extraction retained at {extracted}')
    found = candidates.pop()
    if not (found / 'split').is_dir():
        raise RuntimeError('ZIP raw root has no official split directory')
    target = temp / 'ogbl_citation2'
    found.rename(target)
    version = ogb_release_version()
    release = target / f'RELEASE_v{version}.txt'
    if not release.exists():
        release.write_text('Local supplied official citation2 ZIP; release version from installed OGB metadata.\n')
    # OGB writes only its derived processed cache; the extracted raw/split files remain unchanged.
    (target / 'processed').mkdir(exist_ok=True)
    files = []
    for file in sorted(target.rglob('*')):
        if file.is_file():
            files.append({'path': str(file.relative_to(temp)), 'bytes': file.stat().st_size, 'sha256': sha256_file(file)})
    payload = {'contract_version': CONTRACT_VERSION, 'source_zip': source, 'archive_members': members,
               'ogb_dataset_directory': 'ogbl_citation2', 'files': files, 'completed_unix': time.time()}
    atomic_json(temp / 'CACHE_COMPLETE.json', payload)
    # Preserve any archive files outside the dataset root too, under extracted/.
    temp.rename(cache)
    return payload


class CSR:
    def __init__(self, root: Path):
        self.root = root
        self.starts = np.memmap(root / 'edge_starts.bin', dtype='<i8', mode='r')
        with (root / 'edge_ends.bin').open('rb') as f:
            self.m = struct.unpack('<q', f.read(8))[0]
        if self.m < 0 or (root / 'edge_ends.bin').stat().st_size != 8 + self.m * 4:
            raise ValueError('CSR neighbor-count/file-size mismatch')
        self.indices = np.memmap(root / 'edge_ends.bin', dtype='<i4', offset=8, shape=(self.m,), mode='r') if self.m else np.empty(0, dtype=np.int32)
        self.n = int(self.starts.size)

    def position(self, u: int, v: int) -> int | None:
        if not 0 <= u < self.n or not 0 <= v < self.n:
            raise ValueError('heldout endpoint outside graph')
        start = int(self.starts[u]); end = int(self.starts[u + 1]) if u + 1 < self.n else self.m
        row = self.indices[start:end]
        at = int(np.searchsorted(row, v))
        return start + at if at < row.size and int(row[at]) == v else None


def audit_heldout(graph_root: Path, positives: dict[str, np.ndarray]) -> tuple[dict, np.ndarray]:
    graph, positions, result = CSR(graph_root), set(), {}
    for name, pairs in positives.items():
        counts, collisions = [0, 0], []
        for u, v in pairs:
            found = []
            for direction, (a, b) in enumerate(((int(u), int(v)), (int(v), int(u)))):
                pos = graph.position(a, b)
                if pos is not None:
                    counts[direction] += 1; positions.add(pos); found.append(direction)
            if found:
                collisions.append({'source': int(u), 'target': int(v), 'present_orientations': found})
        result[name] = {'queries': int(pairs.shape[0]), 'forward': counts[0], 'reverse': counts[1], 'collisions': collisions}
    return result, np.asarray(sorted(positions), dtype=np.int64)


def filter_csr(source: Path, destination: Path, remove: np.ndarray, *, chunk_edges: int = 2_000_000) -> dict:
    """Delete known sorted entry positions, preserving row order and aligned timestamps."""
    if destination.exists():
        raise FileExistsError(destination)
    graph = CSR(source)
    remove = np.asarray(remove, dtype=np.int64)
    if remove.size and (remove[0] < 0 or remove[-1] >= graph.m or np.any(remove[1:] <= remove[:-1])):
        raise ValueError('removal positions must be sorted unique valid CSR positions')
    destination.mkdir(parents=True)
    with (destination / 'edge_starts.bin').open('xb') as out:
        for lo in range(0, graph.n, chunk_edges):
            offsets = np.asarray(graph.starts[lo:lo + chunk_edges], dtype=np.int64)
            clean = offsets - np.searchsorted(remove, offsets, side='left')
            out.write(clean.astype('<i8', copy=False).tobytes())
    timestamps = source / 'timestamps.bin'
    has_ts = timestamps.is_file() and timestamps.stat().st_size > 0
    ts = None
    if has_ts:
        if timestamps.stat().st_size != 8 + graph.m * 4:
            raise ValueError('timestamp count/file-size mismatch')
        with timestamps.open('rb') as f:
            if struct.unpack('<q', f.read(8))[0] != graph.m:
                raise ValueError('timestamp header differs from neighbor count')
        ts = np.memmap(timestamps, dtype='<i4', offset=8, shape=(graph.m,), mode='r') if graph.m else np.empty(0, dtype=np.int32)
    count = graph.m - remove.size
    with (destination / 'edge_ends.bin').open('xb') as edges, (destination / 'timestamps.bin').open('xb') as times:
        edges.write(struct.pack('<q', count))
        if has_ts:
            times.write(struct.pack('<q', count))
        for lo in range(0, graph.m, chunk_edges):
            hi = min(graph.m, lo + chunk_edges)
            removed = remove[np.searchsorted(remove, lo):np.searchsorted(remove, hi)] - lo
            block = graph.indices[lo:hi]
            if removed.size:
                keep = np.ones(hi - lo, dtype=bool); keep[removed] = False
                block = block[keep]
                time_block = ts[lo:hi][keep] if has_ts else None
            else:
                time_block = ts[lo:hi] if has_ts else None
            edges.write(np.asarray(block, dtype='<i4').tobytes())
            if has_ts:
                times.write(np.asarray(time_block, dtype='<i4').tobytes())
    return {'nodes': graph.n, 'original_entries': graph.m, 'removed_entries': int(remove.size), 'clean_entries': int(count),
            'timestamps_preserved': has_ts, 'chunk_edges': chunk_edges}


def files_manifest(root: Path) -> list[dict]:
    result = []
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.name not in ('manifest.json', 'PREPARED_COMPLETE.json'):
            result.append({'path': str(path.relative_to(root)), 'bytes': path.stat().st_size, 'sha256': sha256_file(path)})
    return result


def complete_directory(temp: Path, destination: Path, manifest: dict) -> None:
    if destination.exists():
        raise FileExistsError(f'refusing overwrite: {destination}')
    manifest = dict(manifest, contract_version=CONTRACT_VERSION, files=files_manifest(temp), completed_unix=time.time())
    atomic_json(temp / 'manifest.json', manifest)
    atomic_json(temp / 'PREPARED_COMPLETE.json', {'manifest_sha256': sha256_file(temp / 'manifest.json'), 'completed_unix': time.time()})
    temp.rename(destination)


def verify_existing(root: Path, source_sha: str, variant: str) -> dict:
    manifest_path, marker_path = root / 'manifest.json', root / 'PREPARED_COMPLETE.json'
    if not manifest_path.is_file() or not marker_path.is_file():
        raise RuntimeError(f'existing output is unmanifested/incomplete; preserved without changes: {root}')
    marker, manifest = json.loads(marker_path.read_text()), json.loads(manifest_path.read_text())
    if marker['manifest_sha256'] != sha256_file(manifest_path):
        raise RuntimeError(f'manifest completion hash mismatch: {root}')
    if manifest['source_zip']['sha256'] != source_sha or manifest['variant'] != variant or manifest['contract_version'] != CONTRACT_VERSION:
        raise RuntimeError(f'existing output has different source or contract: {root}')
    for item in manifest['files']:
        path = root / item['path']
        if not path.is_file() or path.stat().st_size != item['bytes'] or sha256_file(path) != item['sha256']:
            raise RuntimeError(f'existing output failed integrity check: {path}')
    return manifest


def link_immutable(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copyfile(source, destination)


def validate_official_artifacts(original: Path, *, expected_shapes: bool) -> dict:
    """Check all saved official candidate rows in bounded blocks; do not resample."""
    n = int((original / 'num_nodes.txt').read_text())
    features = np.load(original / 'node_feat.npy', mmap_mode='r', allow_pickle=False)
    years = np.load(original / 'node_year.npy', mmap_mode='r', allow_pickle=False)
    if features.ndim != 2 or features.shape[0] != n or features.dtype != np.float32 or years.shape != (n,):
        raise ValueError('feature/year shape or feature dtype mismatch')
    if expected_shapes and (n != 2927963 or features.shape != (2927963, 128)):
        raise ValueError('wrong official citation2 feature shape')
    result = {'nodes': n, 'feature_shape': list(features.shape), 'feature_dtype': str(features.dtype), 'splits': {}}
    for name in ('valid', 'test'):
        pos = np.load(original / f'{name}_edge.npy', mmap_mode='r', allow_pickle=False)
        neg = np.load(original / f'{name}_edge_neg.npy', mmap_mode='r', allow_pickle=False)
        if pos.ndim != 2 or pos.shape[1] != 2 or pos.shape[0] == 0 or neg.ndim != 2 or neg.shape[1] != 2:
            raise ValueError('invalid official positive/negative array layout')
        if pos.dtype != np.int64 or neg.dtype != np.int64 or neg.shape[0] % pos.shape[0]:
            raise ValueError('invalid official candidate dtype/count')
        k = neg.shape[0] // pos.shape[0]
        if k < 1 or (expected_shapes and (pos.shape != (86596, 2) or k != 1000)):
            raise ValueError('wrong official split shape')
        if np.min(pos) < 0 or np.max(pos) >= n or np.any(pos[:, 0] == pos[:, 1]):
            raise ValueError('invalid heldout positive endpoints')
        for lo in range(0, pos.shape[0], 1024):
            hi = min(pos.shape[0], lo + 1024)
            block = neg[lo * k:hi * k].reshape(hi - lo, k, 2)
            if not np.all(block[:, :, 0] == pos[lo:hi, 0, None]):
                raise ValueError('negative candidates are not aligned to official positive source rows')
            if np.min(block[:, :, 1]) < 0 or np.max(block[:, :, 1]) >= n:
                raise ValueError('negative target outside graph')
        result['splits'][name] = {'queries': int(pos.shape[0]), 'negatives_per_query': int(k),
                                  'source_group_alignment_verified': True,
                                  'unique_positive_sources': int(np.unique(pos[:, 0]).size)}
    return result


def build_clean(original: Path, clean: Path, source: dict, *, expected_reverse: int = 129,
                chunk_edges: int = 2_000_000, expected_shapes: bool = True) -> dict:
    """Build atomic strict-clean variant from immutable prepared_original."""
    if clean.exists():
        return verify_existing(clean, source['sha256'], 'strict_clean')
    artifact_validation = validate_official_artifacts(original, expected_shapes=expected_shapes)
    positives = {name: np.load(original / f'{name}_edge.npy', mmap_mode='r') for name in ('valid', 'test')}
    symmetric_audit, symmetric_remove = audit_heldout(original / 'train_csr', positives)
    directed_audit, directed_remove = audit_heldout(original / 'train_directed_csr', positives)
    atomic_json(original.parent / 'citation2_overlap_audit.json', {
        'source_zip': source, 'symmetric': symmetric_audit, 'directed': directed_audit,
        'distinct_symmetric_positions': int(symmetric_remove.size), 'distinct_directed_positions': int(directed_remove.size)})
    forward_count = sum(item['forward'] for item in directed_audit.values())
    reverse_count = sum(item['reverse'] for item in directed_audit.values())
    if forward_count != 0 or reverse_count != expected_reverse:
        raise AssertionError(f'unexpected official directed overlap: forward={forward_count}, reverse={reverse_count}, expected reverse={expected_reverse}')
    if symmetric_remove.size != 2 * expected_reverse or directed_remove.size != expected_reverse:
        raise AssertionError('unexpected distinct removal count; inspect overlap audits')
    if expected_shapes:
        assert directed_audit['valid']['reverse'] == 62 and directed_audit['test']['reverse'] == 67
        assert CSR(original / 'train_csr').m == 60703760
    temp = clean.with_name(clean.name + '.partial-' + uuid.uuid4().hex[:8]); temp.mkdir()
    timings = []
    with stage('filter_symmetric', timings):
        sym_stats = filter_csr(original / 'train_csr', temp / 'train_csr', symmetric_remove, chunk_edges=chunk_edges)
    with stage('filter_directed', timings):
        dir_stats = filter_csr(original / 'train_directed_csr', temp / 'train_directed_csr', directed_remove, chunk_edges=chunk_edges)
    from scripts.transpose_graph_csr import transpose_graph_csr
    from scripts.convert_graph_csr_to_upper import convert_graph_to_upper
    with stage('clean_transpose', timings):
        transpose_stats = transpose_graph_csr(temp / 'train_directed_csr', temp / 'train_directed_csr_in', file_endian='little')
    with stage('clean_canonical_pairs', timings):
        convert_graph_to_upper(graph_dir=temp / 'train_csr', out_dir=temp / 'train_pairs_csr', use_mmap=True,
                               file_endian='little', writable=False, allow_non_native=True, chunk_bytes=64 << 20,
                               node_chunk_size=250000, use_numba=True, log_every_chunks=10, show_progress=False)
    for name in ('num_nodes.txt', 'node_feat.npy', 'node_year.npy', 'valid_edge.npy', 'valid_edge_neg.npy', 'test_edge.npy', 'test_edge_neg.npy'):
        if not (original / name).is_file():
            raise RuntimeError(f'required original artifact missing: {name}')
        link_immutable(original / name, temp / name)
    after = {}
    with stage('verify_clean_no_heldout', timings):
        for variant in ('train_csr', 'train_directed_csr', 'train_directed_csr_in', 'train_pairs_csr'):
            audit, removed = audit_heldout(temp / variant, positives)
            if removed.size:
                raise AssertionError(f'heldout link remains in {variant}: {audit}')
            after[variant] = audit
    if expected_shapes:
        assert sym_stats['clean_entries'] == 60703502
        assert dir_stats['clean_entries'] == 30387866
    manifest = {'variant': 'strict_clean', 'source_zip': source, 'code_provenance': source_provenance(),
                'original_variant': str(original.resolve()),
                'original_manifest_sha256': sha256_file(original / 'manifest.json') if (original / 'manifest.json').exists() else None,
                'quality_dataset': True, 'file_endian': 'little', 'expected_reverse_collisions': expected_reverse,
                'historical_symmetric_overlap': symmetric_audit, 'historical_directed_overlap': directed_audit,
                'filter_symmetric': sym_stats, 'filter_directed': dir_stats, 'transpose': transpose_stats,
                'clean_heldout_audits': after, 'official_splits': 'unchanged positive and all 1000 negative candidate rows; byte-identical files linked/copied from prepared_original',
                'official_artifact_validation': artifact_validation,
                'feature_policy': 'original float32 node features and node years preserved',
                'positive_training_source': 'train_directed_csr; destination-only negative corruption',
                'timings': timings, 'memory': memory_status()}
    with stage('hash_clean_and_complete', timings):
        complete_directory(temp, clean, manifest)
    return json.loads((clean / 'manifest.json').read_text())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--zip', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source-manifest', type=Path, help='optional download manifest; its hash/content are embedded')
    parser.add_argument('--chunk-edges', type=int, default=2_000_000)
    parser.add_argument('--max-uncompressed-gib', type=float, default=32)
    parser.add_argument('--expected-reverse-overlaps', type=int, default=129)
    args = parser.parse_args(argv)
    if args.chunk_edges < 1 or args.max_uncompressed_gib <= 0:
        parser.error('chunk size and extraction budget must be positive')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    root, archive = args.root.resolve(), args.zip.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = root / 'PREPARATION_RUNNING.lock'
    with lock.open('x') as f:
        f.write(json.dumps({'pid': os.getpid(), 'started_unix': time.time(), 'source_zip': str(archive)}))
    timings = []
    try:
        # Refuse incomplete outputs before spending time extracting/converting.
        for name in ('prepared', 'prepared_original'):
            target = root / name
            if target.exists() and not (target / 'PREPARED_COMPLETE.json').is_file():
                raise RuntimeError(f'preserving unmanifested existing output without overwrite: {target}')
        with stage('hash_source_zip', timings):
            before = archive.stat()
            source = {'path': str(archive), 'bytes': before.st_size, 'sha256': sha256_file(archive)}
            after = archive.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError('source ZIP changed while hashing')
            if args.source_manifest:
                source['download_manifest_sha256'] = sha256_file(args.source_manifest)
                source['download_manifest'] = json.loads(args.source_manifest.read_text())
        with stage('local_cache', timings):
            cache_manifest = make_cache(archive, root / 'ogb_cache', source,
                                       max_uncompressed_bytes=int(args.max_uncompressed_gib * 2**30))
        original = root / 'prepared_original'
        if original.exists():
            original_manifest = verify_existing(original, source['sha256'], 'official_original')
        else:
            temp = root / ('prepared_original.partial-' + uuid.uuid4().hex[:8])
            # Fresh derived-cache overlay avoids trusting an unmanifested processed
            # pickle left by an interrupted conversion, while preserving raw cache.
            if shutil.disk_usage(root).free < 16 * 2**30:
                raise RuntimeError('need at least 16 GiB free before original conversion')
            conversion_cache = root / ('ogb_conversion_cache.partial-' + uuid.uuid4().hex[:8])
            conversion_dataset = conversion_cache / 'ogbl_citation2'
            conversion_dataset.mkdir(parents=True)
            for entry in (root / 'ogb_cache/ogbl_citation2').iterdir():
                if entry.name != 'processed':
                    (conversion_dataset / entry.name).symlink_to(entry, target_is_directory=entry.is_dir())
            (conversion_dataset / 'processed').mkdir()
            with stage('existing_ogb_converter_offline', timings), offline_only():
                from scripts.prepare_ogb_dataset import prepare_ogb_dataset
                prepare_ogb_dataset(dataset='ogbl-citation2', out_dir=temp, ogb_root=conversion_cache,
                                    file_endian='little', layout='prepared')
            # Delete only our newly created derived cache; symlink targets (raw
            # and official split files) remain in the immutable ogb_cache.
            shutil.rmtree(conversion_cache)
            original_manifest = {'variant': 'official_original', 'quality_dataset': False, 'source_zip': source,
                                 'code_provenance': source_provenance(),
                                 'cache_manifest_sha256': sha256_file(root / 'ogb_cache/CACHE_COMPLETE.json'),
                                 'cache_source_verified': cache_manifest['source_zip']['sha256'] == source['sha256'],
                                 'layout': 'existing scripts.prepare_ogb_dataset(layout=prepared)', 'file_endian': 'little',
                                 'converter_sha256': sha256_file(PROJECT_ROOT / 'scripts/prepare_ogb_dataset.py'),
                                 'wrapper_sha256': sha256_file(Path(__file__)), 'timings': timings}
            with stage('hash_original_and_complete', timings):
                complete_directory(temp, original, original_manifest)
        with stage('strict_clean_variant', timings):
            clean_manifest = build_clean(original, root / 'prepared', source,
                                         expected_reverse=args.expected_reverse_overlaps, chunk_edges=args.chunk_edges)
        run = {'source_zip': source, 'prepared': str(root / 'prepared'), 'prepared_original': str(original),
               'clean_manifest_sha256': sha256_file(root / 'prepared/manifest.json'),
               'original_manifest_sha256': sha256_file(original / 'manifest.json'),
               'timings': timings, 'memory': memory_status(), 'completed_unix': time.time()}
        atomic_json(root / 'PREPARATION_COMPLETE.json', run)
        LOG.info('READY strict-clean quality dataset=%s; manifest=%s', root / 'prepared', root / 'prepared/manifest.json')
        return 0
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    raise SystemExit(main())
