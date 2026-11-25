from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tarfile
import tempfile
import unittest

import numpy as np

SPEC = importlib.util.spec_from_file_location('prepare_friendster_revision', Path(__file__).with_name('prepare_friendster.py'))
prep = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = prep
SPEC.loader.exec_module(prep)

MASK = (1 << 64) - 1
GOLDEN = 0x9E3779B97F4A7C15


def mix_python(x):
    x = (x + GOLDEN) & MASK
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK
    return x ^ (x >> 31)


def key(u, v):
    return (min(u, v) << 32) | max(u, v)


def write_graph(root, n, edges):
    root.mkdir(parents=True)
    starts, columns = [], []
    for u in range(n):
        starts.append(len(columns))
        columns.extend(sorted(v for a, v in edges if a == u))
    np.asarray(starts, dtype='<i8').tofile(root / 'edge_starts.bin')
    with (root / 'edge_ends.bin').open('wb') as f:
        f.write(struct.pack('<q', len(columns))); f.write(np.asarray(columns, dtype='<i4').tobytes())
    (root / 'timestamps.bin').write_bytes(b'')
    (root / 'num_nodes.txt').write_text(str(n) + '\n')
    (root / 'meta.json').write_text('{}')


def fixture(root):
    n = 64
    undirected = set((min(i, (i + 1) % n), max(i, (i + 1) % n)) for i in range(n))
    undirected |= {(0, i) for i in range(1, 41)}
    edges = sorted(undirected)
    ranks = sorted(mix_python(key(u, v) ^ prep.SEED) for u, v in edges)
    ntrain, nvalid_end = len(edges) * 8 // 10, len(edges) * 9 // 10
    t0, t1 = ranks[ntrain - 1], ranks[nvalid_end - 1]
    train = [(u, v) for u, v in edges if mix_python(key(u, v) ^ prep.SEED) <= t0]
    sets = [set() for _ in range(n)]; train_sets = [set() for _ in range(n)]
    for u, v in edges:
        sets[u].add(v); sets[v].add(u)
    for u, v in train:
        train_sets[u].add(v); train_sets[v].add(u)
    full = root / 'full'
    write_graph(full, n, edges + [(v, u) for u, v in edges])
    bundle = root / 'bundle'; bundle.mkdir()
    metadata = {'seed': prep.SEED, 'num_nodes': n, 'original_stored_entries': len(edges) * 2,
                'train_stored_entries': len(train) * 2, 'train_undirected_edges': len(train),
                'validation_undirected_edges': nvalid_end - ntrain, 'test_undirected_edges': len(edges) - nvalid_end,
                'threshold_train': t0, 'threshold_validation': t1, 'train_isolated_nodes': sum(not s for s in train_sets),
                'queries_per_split': 2, 'negatives_per_query': 4}
    for split, name in enumerate(('valid', 'test')):
        candidates = [(u, v) for u, v in edges if (t0 < mix_python(key(u, v) ^ prep.SEED) <= t1 if split == 0
                                                   else mix_python(key(u, v) ^ prep.SEED) > t1)]
        # Deliberately choose two query rows of one source when possible.
        hub = [(u, v) for u, v in candidates if u == 0]
        chosen = hub[:2] if len(hub) >= 2 else candidates[:2]
        sources = np.array([u for u, v in chosen], dtype=np.int64)
        targets = np.array([v for u, v in chosen], dtype=np.int64)
        negatives = np.empty((2, 4), dtype=np.int32)
        for i, u in enumerate(sources):
            used = set()
            for j in range(4):
                attempt = 0
                while True:
                    h = mix_python((prep.SEED + 10 + split + i * 4 + j + attempt * GOLDEN) & MASK)
                    v = h % n; attempt += 1
                    if v == u or v in sets[u] or v in used:
                        continue
                    negatives[i, j] = v; used.add(v); break
        warm = np.array([bool(train_sets[u]) and bool(train_sets[v]) for u, v in chosen], dtype=bool)
        folder = bundle / 'evaluation' / name; folder.mkdir(parents=True)
        for filename, arr in [('source_node', sources), ('target_node', targets), ('target_node_neg', negatives), ('warm_query', warm)]:
            np.save(folder / (filename + '.npy'), arr)
        metadata[name] = {'warm_queries': int(warm.sum()), 'cold_queries': int((~warm).sum())}
    reference = {'split_metadata': metadata, 'evaluation_files': {}, 'source_files': {}}
    for path in (bundle / 'evaluation').rglob('*.npy'):
        reference['evaluation_files'][str(path.relative_to(bundle))] = {'bytes': path.stat().st_size, 'sha256': prep.sha256_file(path)}
    (bundle / 'reference_manifest.json').write_text(json.dumps(reference))
    (bundle / 'BUNDLE_MANIFEST.json').write_text('{}')
    (bundle / 'provenance').mkdir()
    return full, bundle, reference


class PrepareFriendsterTests(unittest.TestCase):
    def test_mix_canonical_hash_matches_independent_unsigned_python(self):
        for u, v in [(0, 1), (655_000, 17), (65_608_365, 65_000_000)]:
            expected = mix_python(key(u, v) ^ prep.SEED)
            self.assertEqual(int(prep.mix(np.uint64(int(prep.key_for(u, v)) ^ prep.SEED))), expected)
            self.assertEqual(int(prep.key_for(u, v)), int(prep.key_for(v, u)))

    def test_streaming_two_pass_split_and_exact_query_regeneration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); full, bundle, reference = fixture(root)
            out = root / 'prepared'
            manifest = prep.build_prepared(full, bundle, out, reference, {'files': {}}, blocks=2,
                                           chunk_nodes=7, strict_official=False, eval_bundle_sha256='fixture')
            self.assertEqual(manifest['train_stored_entries'], reference['split_metadata']['train_stored_entries'])
            self.assertEqual(manifest['train_undirected_edges'], reference['split_metadata']['train_undirected_edges'])
            graph = prep.CSR(out / 'train_csr'); pairs = prep.CSR(out / 'train_pairs_csr')
            canonical = set()
            for u in range(graph.n):
                end = graph.starts[u + 1] if u + 1 < graph.n else graph.m
                for v in graph.col[graph.starts[u]:end]:
                    self.assertTrue(prep.has_edge(graph.starts, graph.col, int(v), u))
                    self.assertLessEqual(mix_python(key(u, int(v)) ^ prep.SEED), reference['split_metadata']['threshold_train'])
                    canonical.add((min(u, int(v)), max(u, int(v))))
                end2 = pairs.starts[u + 1] if u + 1 < pairs.n else pairs.m
                self.assertTrue(all(v > u for v in pairs.col[pairs.starts[u]:end2]))
            self.assertEqual(len(canonical), pairs.m)
            self.assertTrue(manifest['evaluation']['checks']['valid']['exact_archived_negative_generator_reproduced'])
            for split in ('valid', 'test'):
                neg = np.load(out / 'evaluation' / split / 'target_node_neg.npy')
                flat = np.load(out / f'{split}_edge_neg.npy').reshape(2, 4, 2)
                np.testing.assert_array_equal(flat[:, :, 1], neg)
                np.testing.assert_array_equal(np.load(out / f'{split}_query_row_ids.npy'), [0, 1])
            complete = prep.verify_existing(out, 'fixture', {'files': {}})
            self.assertEqual(complete['variant'], 'exact_colleague_80_10_10')
            (out / 'num_nodes.txt').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                prep.verify_existing(out, 'fixture', {'files': {}})

    def test_query_checks_reject_neighbor_negative_and_duplicate_source_stays_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); full, bundle, reference = fixture(root)
            out = root / 'prepared'
            prep.build_prepared(full, bundle, out, reference, {'files': {}}, blocks=2,
                                chunk_nodes=9, strict_official=False)
            folder = out / 'evaluation/valid'
            source = np.load(folder / 'source_node.npy')
            self.assertEqual(len(np.unique(source)), 1)  # repeated source IDs deliberately retained
            neg = np.load(folder / 'target_node_neg.npy'); neg[0, 0] = source[0]
            np.save(folder / 'target_node_neg.npy', neg)
            with self.assertRaisesRegex(ValueError, 'query verification failed'):
                prep.verify_queries(prep.CSR(full), prep.CSR(out / 'train_csr'), out / 'evaluation', reference['split_metadata'])

    def test_count_pass_rejects_non_simple_or_unsorted_input(self):
        starts = np.array([0, 2, 3], dtype=np.int64)
        col = np.array([2, 1, 0], dtype=np.int32)
        train, pairs = np.zeros(3, np.int64), np.zeros(3, np.int64)
        stats = prep.count_split(starts, col, np.uint64(prep.SEED), np.uint64(prep.TRAIN_THRESHOLD),
                                 np.uint64(prep.VALID_THRESHOLD), train, pairs, 2)
        self.assertGreater(stats[:, 3].sum(), 0)

    def test_trusted_bundle_roundtrip_and_full_source_revision_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); full, bundle, reference = fixture(root)
            download = {}
            for i, name in enumerate(('edge_starts.bin', 'edge_ends.bin', 'num_nodes.txt', 'meta.json')):
                path = full / name
                reference['source_files'][name] = {'revision': 1234 + i, 'bytes': path.stat().st_size}
                download[str(path)] = {'source': prep.YT_SOURCE + '/' + name, 'revision': 1234 + i,
                                       'bytes': path.stat().st_size, 'sha256': prep.sha256_file(path)}
            ref_path = bundle / 'reference_manifest.json'; ref_path.write_text(json.dumps(reference))
            ref_sha = prep.sha256_file(ref_path)
            evidence = bundle / 'provenance/data_validation_latest.json'
            evidence.write_text(json.dumps({'all_validation_test_arrays_sha256_equal': True}))
            file_manifest = {str(path.relative_to(bundle)): {'bytes': path.stat().st_size, 'sha256': prep.sha256_file(path)}
                             for path in bundle.rglob('*') if path.is_file() and path.name != 'BUNDLE_MANIFEST.json'}
            (bundle / 'BUNDLE_MANIFEST.json').write_text(json.dumps({'files': file_manifest,
                'reference_manifest_sha256': ref_sha, 'latest_series_evaluation_hashes_confirmed': True}))
            archive = root / 'eval.tgz'
            with tarfile.open(archive, 'w:gz') as tar:
                for path in sorted(bundle.rglob('*')):
                    if path.is_file():
                        tar.add(path, arcname=str(path.relative_to(bundle)), recursive=False)
            recovered_ref, recovered_manifest = prep.safe_bundle_extract(archive, root / 'extracted', trusted_reference_sha=ref_sha)
            self.assertEqual(recovered_ref, reference)
            self.assertTrue(recovered_manifest['latest_series_evaluation_hashes_confirmed'])
            input_manifest = root / 'downloads.json'; input_manifest.write_text(json.dumps(download))
            audit = prep.verify_full_source(full, input_manifest, reference)
            self.assertTrue(audit['historical_source_revisions_equal'])
            first = next(iter(download.values())); first['revision'] += 1
            input_manifest.write_text(json.dumps(download))
            with self.assertRaisesRegex(ValueError, 'revision'):
                prep.verify_full_source(full, input_manifest, reference)

    def test_tar_extract_rejects_escape_before_writing(self):
        import io
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path = root / 'bad.tgz'
            with tarfile.open(path, 'w:gz') as archive:
                info = tarfile.TarInfo('../outside'); info.size = 1
                archive.addfile(info, io.BytesIO(b'x'))
            with self.assertRaises(ValueError):
                prep.safe_bundle_extract(path, root / 'out')
            self.assertFalse((root / 'out').exists())


if __name__ == '__main__':
    prep.nb.set_num_threads(min(2, prep.nb.config.NUMBA_NUM_THREADS))
    unittest.main()
