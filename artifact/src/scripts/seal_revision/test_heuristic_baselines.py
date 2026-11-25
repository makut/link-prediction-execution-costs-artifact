from __future__ import annotations

import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from scripts.seal_revision import heuristic_baselines as heur


def graph_arrays(n, undirected):
    neighbors = [set() for _ in range(n)]
    for u, v in undirected:
        neighbors[u].add(v); neighbors[v].add(u)
    starts, col = [], []
    for row in neighbors:
        starts.append(len(col)); col.extend(sorted(row))
    return np.array(starts, np.int64), np.array(col, np.int32), neighbors


def oracle_scores(neighbors, u, v):
    common = sorted(neighbors[u] & neighbors[v])
    return np.array([len(neighbors[v]), len(neighbors[u]) * len(neighbors[v]), len(common),
                     sum(1 / math.log(len(neighbors[w])) for w in common),
                     sum(1 / len(neighbors[w]) for w in common)], dtype=np.float64)


def oracle_ranks(neighbors, sources, targets, negative, ids):
    ranks = []
    for q in ids:
        positive = oracle_scores(neighbors, int(sources[q]), int(targets[q]))
        scores = np.stack([oracle_scores(neighbors, int(sources[q]), int(v)) for v in negative[q]])
        ranks.append(1 + (scores > positive).sum(axis=0) + .5 * (scores == positive).sum(axis=0))
    return np.asarray(ranks)


def write_fixture(root):
    prepared = root / 'prepared'; graph = prepared / 'train_csr'; graph.mkdir(parents=True)
    starts, col, neighbors = graph_arrays(8, [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4), (5, 6)])
    starts.astype('<i8').tofile(graph / 'edge_starts.bin')
    with (graph / 'edge_ends.bin').open('wb') as f:
        f.write(struct.pack('<q', len(col))); f.write(col.astype('<i4').tobytes())
    sources = np.array([0, 0, 7, 6], np.int64)
    targets = np.array([4, 3, 1, 2], np.int64)
    negative = np.array([[5, 6, 7], [4, 5, 6], [2, 3, 4], [0, 3, 7]], np.int32)
    warm = np.array([bool(neighbors[u]) and bool(neighbors[v]) for u, v in zip(sources, targets)])
    for split in ('valid', 'test'):
        folder = prepared / 'evaluation' / split; folder.mkdir(parents=True)
        for name, data in [('source_node', sources), ('target_node', targets), ('target_node_neg', negative), ('warm_query', warm)]:
            np.save(folder / (name + '.npy'), data, allow_pickle=False)
    files = {str(p.relative_to(prepared)): {'bytes': p.stat().st_size, 'sha256': heur.sha256_file(p)}
             for p in prepared.rglob('*') if p.is_file()}
    manifest = {'dataset': 'friendster', 'num_nodes': 8, 'train_stored_entries': len(col), 'files': files}
    (prepared / 'manifest.json').write_text(json.dumps(manifest))
    (prepared / 'PREPARED_COMPLETE.json').write_text(json.dumps({'manifest_sha256': heur.sha256_file(prepared / 'manifest.json')}))
    return prepared, starts, col, neighbors, sources, targets, negative, warm


def quality_fixture(root, prepared):
    quality = root / 'quality'; quality.mkdir()
    jobs, frozen_jobs = [], []
    subsets = {}
    for split in ('val', 'test'):
        path = quality / (split + '_query_ids.npy')
        np.save(path, np.array([0, 1, 2], np.int64))
        subsets[split] = {'outputs': {'query_ids': {'path': str(path), 'sha256': heur.sha256_file(path)}}}
    for seed in (42, 43, 44):
        for variant in ('union', 'pv'):
            name = f'{variant}_seed{seed}'
            config = {'dataset': {'graph_csr_root': str(prepared / 'train_csr')}, 'model': {'k': 370},
                      'training': {'seed': seed, 'max_steps': 20000},
                      'train_loader': {'seed': seed, 'negative_sampling': {'seed': seed}},
                      'seal': {'graph_csr_use_per_vertex_sampling': variant == 'pv'}, 'output': {'dir': name}}
            path = quality / (name + '.json'); path.write_text(json.dumps(config))
            checkpoint = quality / (name + '.pth'); checkpoint.write_bytes(name.encode())
            job = {'name': name, 'seed': seed, 'variant': variant, 'config': str(path), 'config_sha256': heur.sha256_file(path)}
            jobs.append(job)
            frozen_jobs.append({**job, 'checkpoint': str(checkpoint), 'checkpoint_sha256': heur.sha256_file(checkpoint),
                                'selection': {'selection_split': 'validation', 'metric': 'ogb_mrr'},
                                'training': {'test_scored': False, 'steps': 20000}, 'initial_model_sha256': str(seed)})
    graph_stats = {str(p): {'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns}
                   for p in (prepared / 'train_csr').iterdir()}
    plan = {'protocol': {'dataset': 'friendster_colleague_80_10_10', 'max_steps': 20000},
            'jobs': jobs, 'subsets': subsets, 'graph_root': str(prepared / 'train_csr'), 'graph_input_stats': graph_stats,
            'dataset_provenance': {'sha256': heur.sha256_file(prepared / 'manifest.json')}}
    path = quality / 'plan.json'; path.write_text(json.dumps(plan))
    frozen = {'plan_sha256': heur.sha256_file(path), 'selection_uses_test': False, 'jobs': frozen_jobs}
    freeze = quality / 'FROZEN_CHECKPOINTS.json'; freeze.write_text(json.dumps(frozen))
    return quality, path, freeze


class HeuristicTests(unittest.TestCase):
    def test_pair_scores_merge_binary_and_bound_against_set_oracle(self):
        starts, col, neighbors = graph_arrays(110, [(0, v) for v in range(1, 101)] + [(1, v) for v in (101, 102, 103)])
        for u, v in [(2, 1), (1, 2), (0, 1), (108, 1), (1, 109), (2, 3)]:
            scores, actual, bad = heur.score_pair(starts, col, u, v)
            np.testing.assert_allclose(scores, oracle_scores(neighbors, u, v), rtol=0, atol=1e-14)
            self.assertEqual(bad, 0)
            self.assertLessEqual(actual, heur.pair_operation_bound(len(neighbors[u]), len(neighbors[v])))

    def test_duplicate_sources_cold_and_midrank_queries_match_oracle(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, starts, col, neighbors, source, target, neg, _ = write_fixture(Path(tmp))
            ids = np.arange(4, dtype=np.int64)
            ranks, actual, bad = heur.score_query_rows(starts, col, source, target, neg, ids)
            np.testing.assert_allclose(ranks, oracle_ranks(neighbors, source, target, neg, ids), rtol=0, atol=0)
            self.assertEqual(bad.sum(), 0)
            self.assertNotEqual(ranks[0, 2], ranks[1, 2])  # same source, distinct independent queries
            np.testing.assert_array_equal(ranks[2, 1:], [2.5] * 4)  # cold source, all PA/CN/AA/RA ties
            bounds, invalid = heur.estimate_query_operations(starts, len(col), source, target, neg, ids)
            self.assertFalse(invalid.any()); self.assertTrue(np.all(actual <= bounds))

    def test_test_gate_precedes_input_loader_and_all_scoring(self):
        with mock.patch.object(heur, 'load_inputs', side_effect=AssertionError('must not load test input')):
            with self.assertRaisesRegex(ValueError, 'before ANY test scoring'):
                heur.run('/unused', 'test', '/unused-output', frozen_path=None)

    def test_six_checkpoint_gate_and_exact_subset_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); prepared, *_ = write_fixture(root)
            _, plan_path, frozen = quality_fixture(root, prepared)
            plan, gate = heur.verify_test_gate(frozen)
            self.assertEqual(gate['verified_checkpoints'], 6)
            qids = Path(plan['subsets']['test']['outputs']['query_ids']['path'])
            selected, provenance = heur.resolve_query_ids(4, qids, plan=plan)
            np.testing.assert_array_equal(selected, [0, 1, 2]); self.assertTrue(provenance['matches_quality_subset'])
            wrong = root / 'wrong.npy'; np.save(wrong, np.array([0, 2, 3], np.int64))
            with self.assertRaisesRegex(ValueError, 'EXACT frozen'):
                heur.resolve_query_ids(4, wrong, plan=plan)
            content = json.loads(frozen.read_text()); content['jobs'].pop(); frozen.write_text(json.dumps(content))
            with self.assertRaisesRegex(ValueError, 'all six'):
                heur.verify_test_gate(frozen)

    def test_gate_rejects_same_hash_consistent_but_changed_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); prepared, *_ = write_fixture(root)
            _, plan_path, freeze = quality_fixture(root, prepared)
            plan, frozen = json.loads(plan_path.read_text()), json.loads(freeze.read_text())
            config = Path(plan['jobs'][0]['config']); cfg = json.loads(config.read_text()); cfg['model']['k'] = 371
            config.write_text(json.dumps(cfg)); digest = heur.sha256_file(config)
            plan['jobs'][0]['config_sha256'] = digest; frozen['jobs'][0]['config_sha256'] = digest
            plan_path.write_text(json.dumps(plan)); frozen['plan_sha256'] = heur.sha256_file(plan_path)
            freeze.write_text(json.dumps(frozen))
            with self.assertRaisesRegex(ValueError, 'one normalized protocol'):
                heur.verify_test_gate(freeze)

    def test_valid_before_freeze_and_test_run_preserve_query_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); prepared, _, _, neighbors, source, target, neg, warm = write_fixture(root)
            torch_imported_before = 'torch' in sys.modules
            valid = heur.run(prepared, 'valid', root / 'valid', threads=1, chunk_queries=2)
            self.assertEqual(valid['queries'], 4)
            self.assertEqual(valid['unique_sources'], 3)
            self.assertIsNone(valid['gate'])
            self.assertEqual(valid['torch_imported'], torch_imported_before)
            self.assertEqual('torch' in sys.modules, torch_imported_before)
            quality, _, freeze = quality_fixture(root, prepared)
            qids_path = quality / 'test_query_ids.npy'
            result = heur.run(prepared, 'test', root / 'test', frozen_path=freeze, query_ids_path=qids_path,
                              threads=1, chunk_queries=2)
            ids = np.load(root / 'test/query_ids.npy')
            expected = oracle_ranks(neighbors, source, target, neg, ids)
            np.testing.assert_allclose(np.load(root / 'test/reciprocal_ranks.npy'), 1 / expected)
            self.assertEqual(result['metrics']['common_neighbors']['mrr'], float((1 / expected[:, 2]).mean()))
            self.assertTrue(result['selection']['matches_quality_subset'])
            self.assertTrue((root / 'test/COMPLETE.json').is_file())

    def test_operation_budget_fails_before_intersection_without_silent_subset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); prepared, *_ = write_fixture(root)
            with mock.patch.object(heur, 'score_query_rows', side_effect=AssertionError('no scoring allowed')):
                with self.assertRaises(heur.OperationBudgetExceeded):
                    heur.run(prepared, 'valid', root / 'result', threads=1, max_operations=1)
            self.assertFalse((root / 'result').exists())
            partial = next(root.glob('result.partial-*'))
            failure = json.loads((partial / 'FAILED.json').read_text())
            self.assertEqual(failure['processed_queries'], 0)
            self.assertFalse((partial / 'COMPLETE.json').exists())

    def test_cli_tiny_valid_imports_no_torch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); prepared, *_ = write_fixture(root)
            result = subprocess.run([sys.executable, '-m', 'scripts.seal_revision.heuristic_baselines',
                                     '--prepared-root', str(prepared), '--split', 'valid', '--threads', '1',
                                     '--out', str(root / 'cli')], capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(json.loads(result.stdout)['torch_imported'])


if __name__ == '__main__':
    heur.nb.set_num_threads(min(2, heur.nb.config.NUMBA_NUM_THREADS))
    unittest.main()
