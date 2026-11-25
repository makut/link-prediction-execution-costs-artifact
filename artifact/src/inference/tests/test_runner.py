from __future__ import annotations

import unittest

import numpy as np

from inference.edge_scorer import EdgeScorer
from inference.runner import (
    per_rank_count,
    run_scoring,
    run_scoring_distributed,
    shard_pairs_for_rank,
)


class _SumScorer(EdgeScorer):
    """Returns src + dst per edge, and records the batch sizes it saw."""

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def score_batch(self, edges: np.ndarray) -> np.ndarray:
        self.batch_sizes.append(int(edges.shape[0]))
        return (edges[:, 0] + edges[:, 1]).astype(np.float32)


class _WrongLengthScorer(EdgeScorer):
    def score_batch(self, edges: np.ndarray) -> np.ndarray:
        return np.zeros(edges.shape[0] + 1, dtype=np.float32)


class RunScoringTests(unittest.TestCase):
    def test_run_scoring_aligns_and_batches(self) -> None:
        pairs = np.arange(20, dtype=np.int64).reshape(10, 2)
        scorer = _SumScorer()
        scores = run_scoring(scorer, pairs, batch_size=3, show_progress=False)

        self.assertEqual(scores.shape, (10,))
        self.assertEqual(scores.dtype, np.float32)
        np.testing.assert_array_equal(scores, (pairs[:, 0] + pairs[:, 1]).astype(np.float32))
        self.assertEqual(scorer.batch_sizes, [3, 3, 3, 1])

    def test_run_scoring_empty(self) -> None:
        scores = run_scoring(_SumScorer(), np.empty((0, 2), dtype=np.int64), show_progress=False)
        self.assertEqual(scores.shape, (0,))

    def test_run_scoring_rejects_bad_shape(self) -> None:
        with self.assertRaises(ValueError):
            run_scoring(_SumScorer(), np.zeros((3, 3), dtype=np.int64), show_progress=False)

    def test_run_scoring_detects_length_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            run_scoring(_WrongLengthScorer(), np.zeros((4, 2), dtype=np.int64), batch_size=2, show_progress=False)


class DistributedShardingTests(unittest.TestCase):
    """Cover the rank-sharding math without a real process group.

    ``gather_eval_arrays`` is a no-op passthrough when distributed is not
    initialised, so we reconstruct the global result by concatenating each
    simulated rank's slice — which is exactly what rank 0 does after the gather.
    """

    def _reference(self, pairs: np.ndarray) -> np.ndarray:
        return (pairs[:, 0] + pairs[:, 1]).astype(np.float32)

    def test_shards_are_equal_size_and_reassemble_in_order(self) -> None:
        for num_pairs in (0, 1, 5, 23, 64):
            for world_size in (1, 2, 3, 4, 8):
                pairs = np.stack(
                    [np.arange(num_pairs), np.arange(num_pairs) + 1000], axis=1
                ).astype(np.int64).reshape(num_pairs, 2)
                per_rank = per_rank_count(num_pairs, world_size)
                slices = [
                    shard_pairs_for_rank(pairs, rank=r, world_size=world_size)
                    for r in range(world_size)
                ]
                # Every rank gets the same padded length -> identical batch counts.
                for sl in slices:
                    self.assertEqual(sl.shape, (per_rank, 2))
                # Concatenation (rank order) truncated to N reproduces the input.
                if num_pairs > 0:
                    stitched = np.concatenate([s[:, 0] for s in slices])[:num_pairs]
                    np.testing.assert_array_equal(stitched, pairs[:, 0])

    def test_distributed_scores_match_single_process_via_concat(self) -> None:
        num_pairs, world_size = 23, 4
        pairs = np.stack(
            [np.arange(num_pairs), np.arange(num_pairs) + 1000], axis=1
        ).astype(np.int64)
        reference = self._reference(pairs)

        # Simulate each rank scoring its slice; concatenate as the gather would.
        per_rank = per_rank_count(num_pairs, world_size)
        stitched = np.empty(per_rank * world_size, dtype=np.float32)
        for r in range(world_size):
            local = shard_pairs_for_rank(pairs, rank=r, world_size=world_size)
            stitched[r * per_rank : (r + 1) * per_rank] = self._reference(local)
        np.testing.assert_array_equal(stitched[:num_pairs], reference)

    def test_world_size_one_matches_run_scoring(self) -> None:
        pairs = np.arange(20, dtype=np.int64).reshape(10, 2)
        scores = run_scoring_distributed(
            _SumScorer(), pairs, rank=0, world_size=1, batch_size=3, show_progress=False
        )
        np.testing.assert_array_equal(scores, (pairs[:, 0] + pairs[:, 1]).astype(np.float32))


if __name__ == "__main__":
    unittest.main()
