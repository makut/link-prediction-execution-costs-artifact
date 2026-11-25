from __future__ import annotations

import unittest

import numpy as np

from dataset_utils.eval_data import LabeledEdgeDataset


class LabeledEdgeDatasetTests(unittest.TestCase):
    def test_eval_order_shuffles_labels_within_source_group(self) -> None:
        src = 7
        pos = np.asarray([[src, 10_000 + i] for i in range(64)], dtype=np.int64)
        neg = np.asarray([[src, 20_000 + i] for i in range(64)], dtype=np.int64)

        dataset = LabeledEdgeDataset(pos, neg)

        labels = dataset.labels[dataset.edges[:, 0] == src].astype(np.int64)
        self.assertEqual(int(labels.sum()), 64)
        self.assertEqual(int(labels.size), 128)
        self.assertLess(int(np.flatnonzero(labels == 0)[0]), int(np.flatnonzero(labels == 1)[-1]))

    def test_eval_order_keeps_sources_contiguous(self) -> None:
        pos = np.asarray([[1, 10], [2, 20], [1, 11], [3, 30]], dtype=np.int64)
        neg = np.asarray([[2, 21], [1, 12], [3, 31], [2, 22]], dtype=np.int64)

        dataset = LabeledEdgeDataset(pos, neg)
        srcs = dataset.edges[:, 0]

        seen: set[int] = set()
        current = None
        for src in srcs.tolist():
            if src != current:
                self.assertNotIn(src, seen)
                if current is not None:
                    seen.add(current)
                current = src


if __name__ == "__main__":
    unittest.main()
