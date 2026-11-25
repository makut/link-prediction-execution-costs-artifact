from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from seal.graph_csr_kernels import _sample_neighbors_per_vertex_numba
from seal.models import DGCNN
from seal.ogb_metrics import Citation2MetricAccumulator
from seal.reproducibility import (configure_seeded_loader, edge_extraction_seed,
                                  model_state_sha256, seed_everything)


class RNGDataset(Dataset):
    def __len__(self):
        return 6

    def __getitem__(self, index):
        sampled, _ = _sample_neighbors_per_vertex_numba(
            np.array([0], dtype=np.int64), np.arange(100, dtype=np.int32), 100,
            np.array([0], dtype=np.int64), 5)
        return np.r_[index, random.random(), np.random.random(), torch.rand(1).item(), sampled]


def _worker_draws(seed):
    loader = DataLoader(RNGDataset(), batch_size=1, num_workers=2, multiprocessing_context="spawn")
    configure_seeded_loader(loader, seed)
    return torch.cat(list(loader)).numpy()


def test_two_worker_python_numpy_torch_and_numba_streams_repeat():
    first = _worker_draws(42)
    second = _worker_draws(42)
    changed = _worker_draws(43)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first[:, 1:], changed[:, 1:])
    assert not np.array_equal(first[0, 1:], first[1, 1:])  # distinct worker streams


def test_seed_covers_actual_fixed_k_dgcnn_initialization():
    def build(seed):
        seed_everything(seed)
        return DGCNN(hidden_channels=8, num_layers=2, max_z=20, k=20,
                     train_dataset=SimpleNamespace(num_features=4), dynamic_train=True,
                     use_feature=True, node_embedding=None)
    assert model_state_sha256(build(42)) == model_state_sha256(build(42))
    assert model_state_sha256(build(42)) != model_state_sha256(build(43))


def test_per_pair_eval_seed_is_order_and_worker_assignment_independent():
    seed = edge_extraction_seed(20261126, 120, 400)
    assert seed == edge_extraction_seed(20261126, 120, 400)
    assert seed != edge_extraction_seed(20261126, 400, 120)
    seed_everything(seed)
    first = RNGDataset()[0]
    seed_everything(999)  # arbitrary other pairs/worker work cannot alter this pair
    _ = RNGDataset()[0]
    seed_everything(seed)
    np.testing.assert_array_equal(first, RNGDataset()[0])


def test_ogb_midrank_ties_across_batch_boundaries():
    metric = Citation2MetricAccumulator(expected_negatives=3)
    metric.update(srcs=[7, 7], labels=[0, 1], scores=[0.9, 0.5])
    metric.update(srcs=[7, 7, 11], labels=[0, 0, 1], scores=[0.5, 0.5, 1.0])
    metric.update(srcs=[11, 11, 11], labels=[0, 0, 0], scores=[0.0, 0.0, 0.0])
    result = metric.compute()
    assert result["ogb_mrr"] == pytest.approx((1 / 3 + 1) / 2)
    assert result["ogb_hits@1"] == 0.5
    assert result["ogb_num_sources"] == 2


def test_ogb_metric_refuses_incomplete_group_instead_of_silent_drop():
    metric = Citation2MetricAccumulator(expected_negatives=3)
    metric.update(srcs=[7, 7], labels=[1, 0], scores=[1.0, 0.0])
    with pytest.raises(ValueError, match="expected 1 positive and 3 negatives"):
        metric.compute()


def test_ogb_accepts_hashed_source_order_but_rejects_split_groups():
    metric = Citation2MetricAccumulator(expected_negatives=1)
    metric.update(srcs=[11, 11, 7, 7], labels=[1, 0, 1, 0], scores=[2, 0, 1, 0])
    assert metric.compute()["ogb_mrr"] == 1.0
    with pytest.raises(ValueError, match="contiguous complete groups"):
        metric.update(srcs=[11, 11], labels=[1, 0], scores=[2, 0])
