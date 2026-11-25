from pathlib import Path
from types import SimpleNamespace
import copy
import json

import numpy as np
import pytest
import torch
import yaml
from torch_geometric.data import Data

from query_data import QueryRowDataset, freeze_query_subset
from train_entry import QueryEvalCollator
from seal.models import DGCNN
from seal.ogb_metrics import Citation2MetricAccumulator
from seal.reproducibility import model_state_sha256, seed_everything
from runner import validate_dataset_manifest
from query_data import sha256


def _provided(tmp_path):
    directory = tmp_path / "provided"
    directory.mkdir()
    arrays = {"source_node": np.array([5, 5, 20, 30], dtype=np.int64),
              "target_node": np.array([10, 11, 21, 31], dtype=np.int64),
              "target_node_neg": np.array([[1, 2, 3], [10, 12, 13], [4, 5, 6], [7, 8, 9]], dtype=np.int64),
              "warm_query": np.array([True, False, True, True])}
    for key, value in arrays.items():
        np.save(directory / f"{key}.npy", value)
    return directory, arrays


def test_complete_queries_with_repeated_sources_and_cold_flags_survive(tmp_path):
    directory, arrays = _provided(tmp_path)
    manifest = freeze_query_subset(input_dir=directory, output_dir=tmp_path / "subset", name="valid",
        count=4, seed=20261124, candidate_order_seed=20261127, expected_queries=4,
        expected_negatives=3, chunk_queries=1)
    ids = np.load(manifest["outputs"]["query_ids"]["path"])
    pos = np.load(manifest["outputs"]["positive"]["path"])
    neg = np.load(manifest["outputs"]["negative"]["path"]).reshape(4, 3, 2)
    np.testing.assert_array_equal(ids, np.arange(4))
    np.testing.assert_array_equal(pos[:, 0], arrays["source_node"])
    np.testing.assert_array_equal(pos[:, 1], arrays["target_node"])
    np.testing.assert_array_equal(neg[:, :, 1], arrays["target_node_neg"])
    np.testing.assert_array_equal(np.load(manifest["outputs"]["warm_query"]["path"]), arrays["warm_query"])
    assert manifest["unique_source_count"] == 3
    assert manifest["query_count"] == 4
    assert not manifest["is_official_ogb_benchmark"]


class RecordingExtractor:
    def __init__(self):
        self.pairs = []

    def build_graph(self, source, target, label):
        self.pairs.append((source, target))
        return Data(z=torch.tensor([1, 1]), edge_index=torch.tensor([[0, 1], [1, 0]]),
                    y=torch.tensor([label]), num_nodes=2)


def test_query_identity_drives_metrics_while_real_endpoints_drive_extraction():
    positive = np.array([[5, 10], [5, 11]], dtype=np.int64)
    negative = np.array([[5, 1], [5, 2], [5, 3], [5, 10], [5, 12], [5, 13]], dtype=np.int64)
    dataset = QueryRowDataset(positive, negative, np.array([1717, 1740]), candidate_order_seed=20261127)
    extractor = RecordingExtractor()
    batch = QueryEvalCollator(extractor, seed=20261126)([dataset[i] for i in range(len(dataset))])
    assert [s for s, _ in extractor.pairs] == [5] * 8
    assert set(batch.edge_src.tolist()) == {1717, 1740}
    assert batch.edge_original_src.tolist() == [5] * 8
    # One fixed scorer on real endpoints: pair(5,10) is positive in query1717
    # and a provided negative in query1740. It must receive the same score.
    value = {10: 0.9, 11: 0.1, 1: 0.0, 2: 0.1, 3: 0.2, 12: 0.2, 13: 0.0}
    scores = np.array([value[int(d)] for d in batch.edge_dst])
    metric = Citation2MetricAccumulator(expected_negatives=3)
    for lo, hi in ((0, 3), (3, 5), (5, 8)):
        metric.update(srcs=batch.edge_src.numpy()[lo:hi], labels=batch.y.numpy()[lo:hi], scores=scores[lo:hi])
    assert metric.compute()["ogb_mrr"] == pytest.approx((1 + 1 / 3) / 2)
    # Full candidate permutation, with no source merging or candidate duplication.
    for q in range(2):
        slots = [dataset[q * 4 + i][3] for i in range(4)]
        assert sorted(slots) == [0, 1, 2, 3]


def test_subsets_are_selected_by_original_query_id_reproducibly(tmp_path):
    directory, arrays = _provided(tmp_path)
    first = freeze_query_subset(input_dir=directory, output_dir=tmp_path / "first", name="test", count=3,
        seed=20261125, candidate_order_seed=20261127, expected_queries=4, expected_negatives=3)
    second = freeze_query_subset(input_dir=directory, output_dir=tmp_path / "again", name="test", count=3,
        seed=20261125, candidate_order_seed=20261127, expected_queries=4, expected_negatives=3)
    ids = np.load(first["outputs"]["query_ids"]["path"])
    np.testing.assert_array_equal(ids, np.sort(np.random.default_rng(20261125).choice(4, 3, replace=False)))
    assert first["outputs"]["negative"]["sha256"] == second["outputs"]["negative"]["sha256"]
    actual = np.load(first["outputs"]["negative"]["path"]).reshape(3, 3, 2)
    np.testing.assert_array_equal(actual[:, :, 1], arrays["target_node_neg"][ids])


def test_wrong_query_candidate_alignment_fails():
    with pytest.raises(ValueError, match="aligned"):
        QueryRowDataset(np.array([[5, 1], [5, 2]]), np.array([[5, 3], [8, 4]]),
                        np.array([1717, 1740]), candidate_order_seed=1)


def test_friendster_fixed_capacity_pair_initialization():
    def model(seed):
        seed_everything(seed)
        return DGCNN(hidden_channels=32, num_layers=3, max_z=1000, k=370,
                     train_dataset=SimpleNamespace(num_features=0), dynamic_train=True,
                     use_feature=False, node_embedding=None)
    assert model_state_sha256(model(42)) == model_state_sha256(model(42))
    assert model_state_sha256(model(42)) != model_state_sha256(model(43))


def test_manifest_requires_exact_archived_thresholds_and_query_provenance(tmp_path):
    protocol = yaml.safe_load(Path(__file__).with_name("protocol.yaml").read_text())
    content = {"dataset": "friendster", "variant": "exact_colleague_80_10_10", "num_nodes": protocol["num_nodes"],
               "seed": protocol["split_seed"], "threshold_train": protocol["threshold_train"],
               "threshold_validation": protocol["threshold_validation"],
               "train_stored_entries": protocol["expected_train_entries"],
               "train_undirected_edges": protocol["expected_canonical_positive_entries"],
               "source_graph": {"historical_source_revisions_equal": True},
               "evaluation": {"layout": "original_query_row", "latest_series_arrays_sha256_equal": True,
                              "source_group_evaluator_forbidden": True,
                              "group_statistics": {name: {"query_rows": 10000, "negatives_per_query": 1000}
                                                   for name in ("valid", "test")}}}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(content))
    path.with_name("PREPARED_COMPLETE.json").write_text(json.dumps({"manifest_sha256": sha256(path)}))
    validate_dataset_manifest(content, protocol, path)
    wrong = copy.deepcopy(content)
    wrong["threshold_train"] = int(0.8 * 2**64)  # not the colleague's archived histogram threshold
    with pytest.raises(ValueError, match="threshold_train"):
        validate_dataset_manifest(wrong, protocol, path)
    wrong = copy.deepcopy(content)
    wrong["evaluation"]["latest_series_arrays_sha256_equal"] = False
    with pytest.raises(ValueError, match="query-row identity"):
        validate_dataset_manifest(wrong, protocol, path)
