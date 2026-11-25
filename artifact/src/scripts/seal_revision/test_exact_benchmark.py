"""Small contract tests; do not allocate padded global graph arrays or use Torch."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from scipy import sparse

from scripts.seal_revision import exact_benchmark as bench


class ExactBenchmarkTests(unittest.TestCase):
    def test_directed_and_undirected_match_independent_edge_oracle(self):
        for directed in (False, True):
            with self.subTest(directed=directed):
                graph = bench.fixture_graph(directed)
                incoming = bench.transpose_small(graph, 1000) if directed else None
                nodes = np.array([2, 1, 0, 3, 4, 5, 6, 7, 8], dtype=np.int64)
                record = bench.benchmark_pair(graph, nodes, graph.n, incoming=incoming, repeats=2)
                self.assertEqual(set(record['methods']), set(bench.METHODS))
                for arm in record['methods'].values():
                    self.assertEqual(arm['status'], 'ok')
                    self.assertTrue(arm['exact_edges'])
                    self.assertTrue(arm['exact_drnl'])
                    self.assertEqual(len(arm['edge_seconds']), 2)
                adjacency = bench.oracle_induced(graph, nodes)
                self.assertEqual(adjacency[0, 1], 0)
                self.assertEqual(adjacency[1, 0], 0)
                if directed:
                    self.assertGreater((adjacency != adjacency.T).nnz, 0)

    def test_drnl_common_neighbor_bridge_and_disconnected_nodes(self):
        # Endpoints 0,1; common neighbor2 -> label2; node3 reaches only through
        # endpoint0 -> unreachable after the opposite endpoint is deleted.
        rows = [0, 2, 1, 2, 0, 3]
        cols = [2, 0, 2, 1, 3, 0]
        a = sparse.csr_matrix((np.ones(6), (rows, cols)), shape=(5, 5))
        np.testing.assert_array_equal(bench.drnl_labels(a), [1, 1, 2, 0, 0])
        # Direction is ignored for DRNL, while retained in GNN edge semantics.
        np.testing.assert_array_equal(bench.drnl_labels(sparse.triu(a).tocsr()), [1, 1, 2, 0, 0])

    def test_virtual_padding_preserves_frozen_structure_without_global_arrays(self):
        graph = bench.fixture_graph()
        nodes = np.array([0, 1, 2, 3, 4], dtype=np.int64)
        original_shapes = (graph.starts.shape, graph.indices.shape)
        base = bench.benchmark_pair(graph, nodes, graph.n, repeats=1)
        padded = bench.benchmark_pair(graph, nodes, 10009, repeats=1)
        self.assertEqual(base['node_set_sha256'], padded['node_set_sha256'])
        self.assertEqual(base['canonical_edges_sha256'], padded['canonical_edges_sha256'])
        self.assertEqual(base['drnl_sha256'], padded['drnl_sha256'])
        self.assertEqual(original_shapes, (graph.starts.shape, graph.indices.shape))
        self.assertGreater(padded['stock_workspace_upper_estimate_bytes'], base['stock_workspace_upper_estimate_bytes'])
        self.assertEqual(bench.gather_selected_rows(graph, nodes, 10009, np.dtype('int64')).indptr.size, nodes.size + 1)

    def test_large_virtual_n_skips_stock_before_allocation(self):
        graph = bench.fixture_graph()
        nodes = np.array([0, 1, 2], dtype=np.int64)
        with mock.patch.object(bench, 'stock_scipy', side_effect=AssertionError('must not allocate')):
            record = bench.benchmark_pair(graph, nodes, 180_000_000, repeats=1, workspace_bytes=1024)
        self.assertEqual(record['methods']['stock_scipy']['status'], 'skipped_workspace_guard')
        self.assertTrue(record['methods']['ours_pairwise']['exact_edges'])

    def test_rejects_self_loops_unsorted_rows_and_incorrect_transpose(self):
        for graph in (bench.CSRGraph(np.array([0, 1], dtype=np.int64), np.array([0], dtype=np.int32)),
                      bench.CSRGraph(np.array([0, 2, 2], dtype=np.int64), np.array([2, 1], dtype=np.int32))):
            with self.assertRaises(ValueError):
                bench.benchmark_pair(graph, np.array([0, 1], dtype=np.int64), graph.n, repeats=1)
        directed = bench.fixture_graph(True)
        with self.assertRaisesRegex(ValueError, 'transpose'):
            bench.benchmark_pair(directed, np.array([0, 1, 2], dtype=np.int64), directed.n,
                                 incoming=directed, repeats=1)

    def test_mmap_format_and_pair_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            graph = bench.fixture_graph()
            graph.starts.astype('<i8').tofile(path / 'edge_starts.bin')
            with (path / 'edge_ends.bin').open('wb') as f:
                f.write(graph.m.to_bytes(8, 'little', signed=True))
                f.write(graph.indices.astype('<i4').tobytes())
            loaded = bench.load_graph(path)
            self.assertIsInstance(loaded.starts, np.memmap)
            self.assertIsInstance(loaded.indices, np.memmap)
            np.testing.assert_array_equal(loaded.indices, graph.indices)
            pairs = np.array([[0, 1], [1, 3], [2, 5]], dtype=np.int64)
            np.save(path / 'pairs.npy', pairs)
            self.assertEqual(bench.read_pairs(path / 'pairs.npy', 2), [(0, 1), (1, 3)])

    def test_sampling_freezes_deterministic_capped_sets_and_limits_work(self):
        graph = bench.fixture_graph()
        for sampler in ('reference_union', 'production_union', 'per_vertex'):
            first = bench.sample_node_set(graph, (0, 1), 2, 1, np.random.default_rng(42), sampler)
            second = bench.sample_node_set(graph, (0, 1), 2, 1, np.random.default_rng(42), sampler)
            np.testing.assert_array_equal(first, second)
            self.assertLessEqual(first.size, 4)
            np.testing.assert_array_equal(first[:2], [0, 1])
        with self.assertRaises(bench.WorkspaceLimit):
            bench.sample_node_set(graph, (0, 1), 2, None, np.random.default_rng(42), max_neighbors=1)

    def test_twohop_sampling_and_index_width_controls(self):
        graph = bench.fixture_graph()
        pairs = bench.deterministic_pairs(graph, 8, 42, 'mixed_twohop')
        self.assertEqual(pairs, bench.deterministic_pairs(graph, 8, 42, 'mixed_twohop'))
        for i, (u, v) in enumerate(pairs):
            self.assertEqual(graph.has_edge(u, v), i % 2 == 0)
            if i % 2:
                self.assertTrue(any(graph.has_edge(int(mid), v) for mid in graph.neighbors(u)))
        nodes = np.array([0, 1, 2, 3, 4], dtype=np.int64)
        for dtype in ('int32', 'int64'):
            record = bench.benchmark_pair(graph, nodes, graph.n, stock_index_dtype=dtype, repeats=1)
            self.assertTrue(record['methods']['stock_scipy']['exact_edges'])
        record = bench.benchmark_pair(graph, nodes, graph.n, max_pairwise_checks=1, repeats=1)
        self.assertEqual(record['methods']['ours_pairwise']['status'], 'skipped_pairwise_work_guard')
        with self.assertRaises(bench.WorkspaceLimit):
            bench.benchmark_pair(graph, nodes, graph.n, max_working_bytes=1, repeats=1)

    def test_subprocess_cli_provenance_no_torch_and_single_s_for_padding(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'bench.json'
            run = subprocess.run([sys.executable, '-m', 'scripts.seal_revision.exact_benchmark',
                                  '--fixture', '--directed', '--limit', '2', '--caps', '2',
                                  '--pad-isolates', '0,1000', '--repeats', '1', '--warmup', '1',
                                  '--output', str(output)], cwd=bench.ROOT, text=True, capture_output=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertFalse(json.loads(run.stdout)['torch_imported'])
            payload = json.loads(output.read_text())
            self.assertEqual(len(payload['records']), 4)
            self.assertIn('source_sha256', payload['provenance'])
            for left, right in zip(payload['records'][::2], payload['records'][1::2]):
                self.assertEqual(left['node_set_sha256'], right['node_set_sha256'])
                self.assertEqual(left['canonical_edges_sha256'], right['canonical_edges_sha256'])


if __name__ == '__main__':
    unittest.main()
