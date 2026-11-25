from __future__ import annotations

import unittest

import numpy as np

from dataset_utils.metrics import PerSourceMetricAccumulator


class PerSourceMetricAccumulatorTests(unittest.TestCase):
    def test_streaming_ranking_metrics(self) -> None:
        acc = PerSourceMetricAccumulator(
            metrics=("roc_auc@k", "hits@k", "recall@k", "ndcg@k", "mrr", "mrr@k"),
            metrics_at_k=(1, 2),
            apply_sigmoid=False,
        )
        acc.update(
            srcs=np.array([1, 1, 2, 2]),
            labels=np.array([1, 0, 0, 1]),
            scores=np.array([0.9, 0.1, 0.8, 0.7]),
        )
        metrics = acc.compute()
        self.assertEqual(metrics["hits@1"], 0.5)
        self.assertEqual(metrics["hits@2"], 1.0)
        self.assertEqual(metrics["recall@1"], 0.5)
        self.assertEqual(metrics["mrr"], 0.75)
        self.assertEqual(metrics["mrr@1"], 0.5)
        self.assertEqual(metrics["mrr@2"], 0.75)
        self.assertEqual(metrics["roc_auc@2"], 0.5)

    def test_boundary_groups_are_dropped(self) -> None:
        acc = PerSourceMetricAccumulator(
            metrics=("hits@k", "num_sources", "num_sources_with_positives"),
            metrics_at_k=(1,),
            apply_sigmoid=False,
            drop_first_src=True,
            drop_last_src=True,
        )
        acc.update(
            srcs=np.array([1, 1, 2, 2, 3, 3]),
            labels=np.array([1, 0, 0, 1, 1, 0]),
            scores=np.array([0.9, 0.1, 0.8, 0.7, 0.9, 0.1]),
        )
        metrics = acc.compute()
        self.assertEqual(metrics["num_sources"], 1.0)
        self.assertEqual(metrics["num_sources_with_positives"], 1.0)
        self.assertEqual(metrics["hits@1"], 0.0)

    def test_source_class_count_diagnostics(self) -> None:
        acc = PerSourceMetricAccumulator(
            metrics=(
                "num_sources",
                "num_sources_with_positives",
                "num_sources_with_negatives",
                "num_sources_with_pos_and_neg",
                "mean_dsts_per_src",
            ),
            metrics_at_k=(1,),
            apply_sigmoid=False,
        )
        acc.update(
            srcs=np.array([1, 2, 2, 3]),
            labels=np.array([1, 1, 0, 0]),
            scores=np.array([0.9, 0.8, 0.1, 0.2]),
        )
        metrics = acc.compute()
        self.assertEqual(metrics["num_sources"], 3.0)
        self.assertEqual(metrics["num_sources_with_positives"], 2.0)
        self.assertEqual(metrics["num_sources_with_negatives"], 2.0)
        self.assertEqual(metrics["num_sources_with_pos_and_neg"], 1.0)
        self.assertAlmostEqual(metrics["mean_dsts_per_src"], 4.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
