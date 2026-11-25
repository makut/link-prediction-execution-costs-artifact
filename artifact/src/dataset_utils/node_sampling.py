from __future__ import annotations

import math

import torch
from torch.utils.data import Dataset, Sampler


class NodeRangeDataset(Dataset[int]):
    def __init__(self, num_nodes: int) -> None:
        self.num_nodes = int(num_nodes)

    def __len__(self) -> int:
        return self.num_nodes

    def __getitem__(self, idx: int) -> int:
        return int(idx)


class NodeDistributedSampler(Sampler[int]):
    """Shard shuffled node ids across ranks like DistributedSampler."""

    def __init__(
        self,
        dataset: NodeRangeDataset,
        *,
        num_replicas: int,
        rank: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

        n = len(dataset)
        if self.drop_last:
            self.num_samples = n // self.num_replicas
        else:
            self.num_samples = int(math.ceil(n / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        n = len(self.dataset)
        if self.shuffle:
            indices = torch.randperm(n, generator=g).tolist()
        else:
            indices = list(range(n))

        if self.drop_last:
            indices = indices[: self.total_size]
        elif len(indices) < self.total_size:
            indices = indices + indices[: (self.total_size - len(indices))]

        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
