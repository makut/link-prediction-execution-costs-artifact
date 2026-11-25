from __future__ import annotations

import unittest

from dataset_utils.node_sampling import NodeDistributedSampler, NodeRangeDataset


class NodeDistributedSamplerTests(unittest.TestCase):
    def test_single_rank_visits_every_node_once(self) -> None:
        sampler = NodeDistributedSampler(NodeRangeDataset(5), num_replicas=1, rank=0, seed=42)

        self.assertEqual(set(iter(sampler)), set(range(5)))

    def test_shards_and_pads_like_distributed_sampler(self) -> None:
        dataset = NodeRangeDataset(5)
        rank_0 = list(NodeDistributedSampler(dataset, num_replicas=2, rank=0, seed=42))
        rank_1 = list(NodeDistributedSampler(dataset, num_replicas=2, rank=1, seed=42))

        self.assertEqual(len(rank_0), 3)
        self.assertEqual(len(rank_1), 3)
        self.assertEqual(set(rank_0 + rank_1), set(range(5)))

    def test_set_epoch_changes_shuffle(self) -> None:
        sampler = NodeDistributedSampler(NodeRangeDataset(10), num_replicas=1, rank=0, seed=42)
        epoch_0 = list(sampler)

        sampler.set_epoch(1)

        self.assertNotEqual(epoch_0, list(sampler))


if __name__ == "__main__":
    unittest.main()
