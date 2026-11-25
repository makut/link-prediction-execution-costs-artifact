from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import socket
import stat
import struct
import tempfile
import unittest
from unittest import mock
import zipfile

import numpy as np

SPEC = importlib.util.spec_from_file_location('prepare_citation2_revision', Path(__file__).with_name('prepare_citation2.py'))
prep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prep)


def write_graph(path: Path, n: int, edges: list[tuple[int, int]]):
    path.mkdir(parents=True)
    starts, ends, timestamps = [], [], []
    for u in range(n):
        starts.append(len(ends))
        for v in sorted(v for a, v in edges if a == u):
            ends.append(v); timestamps.append(100 * u + v)
    np.asarray(starts, dtype='<i8').tofile(path / 'edge_starts.bin')
    for name, values in [('edge_ends.bin', ends), ('timestamps.bin', timestamps)]:
        with (path / name).open('wb') as f:
            f.write(struct.pack('<q', len(values))); f.write(np.asarray(values, dtype='<i4').tobytes())


class PrepareCitation2Tests(unittest.TestCase):
    def test_safe_zip_rejects_traversal_symlinks_and_duplicate_members(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i, name in enumerate(('../escape', '/absolute', 'C:/drive', r'bad\path')):
                archive = root / f'bad{i}.zip'
                with zipfile.ZipFile(archive, 'w') as z:
                    z.writestr(name, b'not extracted')
                with self.assertRaises(ValueError):
                    prep.safe_extract(archive, root / f'out{i}', disk_reserve_bytes=0)
                self.assertFalse((root / f'out{i}').exists())
            archive = root / 'symlink.zip'
            with zipfile.ZipFile(archive, 'w') as z:
                info = zipfile.ZipInfo('link'); info.create_system = 3
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
                z.writestr(info, '../outside')
            with self.assertRaises(ValueError):
                prep.safe_extract(archive, root / 'links', disk_reserve_bytes=0)

    def test_zip_size_budget_and_valid_streamed_file_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); archive = root / 'good.zip'
            with zipfile.ZipFile(archive, 'w') as z:
                z.writestr('citation-v2/raw/edge.csv.gz', b'payload')
            with self.assertRaises(ValueError):
                prep.safe_extract(archive, root / 'too_big', max_uncompressed_bytes=1, disk_reserve_bytes=0)
            files = prep.safe_extract(archive, root / 'valid', disk_reserve_bytes=0)
            self.assertEqual(files[0]['sha256'], prep.sha256_file(root / 'valid/citation-v2/raw/edge.csv.gz'))
            with self.assertRaises(FileExistsError):
                prep.safe_extract(archive, root / 'valid', disk_reserve_bytes=0)

    def test_cache_normalizes_archive_root_and_checks_reuse_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); archive = root / 'fixture.zip'
            with zipfile.ZipFile(archive, 'w') as z:
                z.writestr('citation-v2/raw/edge.csv.gz', b'raw-input')
                z.writestr('citation-v2/split/time/train.pt', b'fixture-split')
            source = {'sha256': prep.sha256_file(archive)}
            with mock.patch.object(prep, 'ogb_release_version', return_value='1'):
                manifest = prep.make_cache(archive, root / 'ogb_cache', source, max_uncompressed_bytes=10000)
            self.assertTrue((root / 'ogb_cache/ogbl_citation2/raw/edge.csv.gz').is_file())
            self.assertTrue((root / 'ogb_cache/ogbl_citation2/RELEASE_v1.txt').is_file())
            self.assertEqual(prep.make_cache(archive, root / 'ogb_cache', source, max_uncompressed_bytes=10000), manifest)
            (root / 'ogb_cache/ogbl_citation2/raw/edge.csv.gz').write_bytes(b'corrupt')
            with self.assertRaisesRegex(RuntimeError, 'integrity'):
                prep.make_cache(archive, root / 'ogb_cache', source, max_uncompressed_bytes=10000)

    def test_streaming_deletion_at_chunk_boundaries_preserves_timestamps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_graph(root / 'source', 5, [(0, 1), (0, 2), (1, 0), (1, 3), (2, 4), (4, 0)])
            original = prep.CSR(root / 'source')
            before_hash = prep.sha256_file(root / 'source/edge_ends.bin')
            positions = np.array([0, 2, 5], dtype=np.int64)
            result = prep.filter_csr(root / 'source', root / 'clean', positions, chunk_edges=2)
            clean = prep.CSR(root / 'clean')
            np.testing.assert_array_equal(clean.indices, np.delete(original.indices, positions))
            np.testing.assert_array_equal(clean.starts, original.starts - np.searchsorted(positions, original.starts))
            source_ts = np.fromfile(root / 'source/timestamps.bin', '<i4', offset=8)
            clean_ts = np.fromfile(root / 'clean/timestamps.bin', '<i4', offset=8)
            np.testing.assert_array_equal(clean_ts, np.delete(source_ts, positions))
            self.assertEqual(result['removed_entries'], 3)
            self.assertEqual(before_hash, prep.sha256_file(root / 'source/edge_ends.bin'))

    def test_reverse_overlap_audit_and_complete_clean_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); original = root / 'prepared_original'; original.mkdir()
            directed = [(1, 0), (0, 2), (2, 3), (3, 4), (4, 1)]
            write_graph(original / 'train_directed_csr', 6, directed)
            write_graph(original / 'train_csr', 6, directed + [(v, u) for u, v in directed])
            (original / 'num_nodes.txt').write_text('6\n')
            np.save(original / 'node_feat.npy', np.arange(12, dtype=np.float32).reshape(6, 2))
            np.save(original / 'node_year.npy', np.arange(6, dtype=np.int32))
            np.save(original / 'valid_edge.npy', np.array([[0, 1]], dtype=np.int64))
            np.save(original / 'test_edge.npy', np.array([[4, 5]], dtype=np.int64))
            for name, u in [('valid', 0), ('test', 4)]:
                np.save(original / f'{name}_edge_neg.npy', np.array([[u, 3], [u, 5]], dtype=np.int64))
            audit, positions = prep.audit_heldout(original / 'train_directed_csr',
                                                  {'valid': np.array([[0, 1]]), 'test': np.array([[4, 5]])})
            self.assertEqual((audit['valid']['forward'], audit['valid']['reverse']), (0, 1))
            self.assertEqual(positions.size, 1)
            source = {'sha256': 'tiny_fixture_source'}
            manifest = prep.build_clean(original, root / 'prepared', source, expected_reverse=1,
                                        chunk_edges=2, expected_shapes=False)
            self.assertEqual(manifest['filter_symmetric']['removed_entries'], 2)
            self.assertEqual(manifest['filter_directed']['removed_entries'], 1)
            for directory in ('train_csr', 'train_directed_csr', 'train_directed_csr_in', 'train_pairs_csr'):
                graph = prep.CSR(root / 'prepared' / directory)
                self.assertIsNone(graph.position(0, 1)); self.assertIsNone(graph.position(1, 0))
            for name in ('valid_edge_neg.npy', 'test_edge_neg.npy', 'node_feat.npy'):
                self.assertEqual(prep.sha256_file(original / name), prep.sha256_file(root / 'prepared' / name))
            completed = prep.verify_existing(root / 'prepared', source['sha256'], 'strict_clean')
            self.assertEqual(completed['variant'], 'strict_clean')
            with (root / 'prepared/num_nodes.txt').open('w') as f:
                f.write('invalid')
            with self.assertRaisesRegex(RuntimeError, 'integrity'):
                prep.verify_existing(root / 'prepared', source['sha256'], 'strict_clean')

    def test_offline_guard_and_unmanifested_output_refusal(self):
        with prep.offline_only():
            with self.assertRaisesRegex(RuntimeError, 'Network'):
                socket.create_connection(('example.com', 443))
            with self.assertRaises(RuntimeError):
                input('must not prompt')
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, 'unmanifested'):
                prep.verify_existing(Path(tmp), 'source', 'strict_clean')


if __name__ == '__main__':
    unittest.main()
