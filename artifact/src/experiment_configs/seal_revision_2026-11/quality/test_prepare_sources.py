from pathlib import Path

import numpy as np
import pytest

from prepare_sources import freeze_source_subset


def _fixtures(tmp_path):
    sources = np.array([90, 10, 70, 30, 50, 110], dtype=np.int64)
    positive = np.column_stack((sources, sources + 1))
    # Original row ordering is deliberately mixed, not contiguous by source.
    negative = np.array([(int(src), int(src) + 10 + j) for j in range(4) for src in sources], dtype=np.int64)
    pos_path, neg_path = tmp_path / "positive.npy", tmp_path / "negative.npy"
    np.save(pos_path, positive)
    np.save(neg_path, negative)
    return pos_path, neg_path, positive, negative


def test_fixed_random_whole_source_subset_keeps_every_provided_negative(tmp_path):
    pos_path, neg_path, positive, negative = _fixtures(tmp_path)
    manifests = [freeze_source_subset(pos_path=pos_path, neg_path=neg_path, output_dir=tmp_path / out,
                    name="valid", count=3, seed=20261124, expected_negatives=4, chunk_rows=5)
                 for out in ("first", "again")]
    chosen = np.load(manifests[0]["outputs"]["sources"]["path"])
    np.testing.assert_array_equal(chosen, np.load(manifests[1]["outputs"]["sources"]["path"]))
    assert not np.array_equal(chosen, np.sort(positive[:, 0])[:3])
    expected_pos = positive[np.isin(positive[:, 0], chosen)]
    expected_neg = negative[np.isin(negative[:, 0], chosen)]
    np.testing.assert_array_equal(np.load(manifests[0]["outputs"]["positive"]["path"]), expected_pos)
    np.testing.assert_array_equal(np.load(manifests[0]["outputs"]["negative"]["path"]), expected_neg)
    assert manifests[0]["negative_rows"] == 12
    assert manifests[0]["all_provided_negatives_retained"]
    assert manifests[0]["outputs"]["negative"]["sha256"] == manifests[1]["outputs"]["negative"]["sha256"]


def test_preparation_refuses_a_missing_candidate_in_any_selected_group(tmp_path):
    pos_path, neg_path, _, negative = _fixtures(tmp_path)
    np.save(neg_path, negative[:-1])
    with pytest.raises(ValueError, match="Incomplete valid groups"):
        freeze_source_subset(pos_path=pos_path, neg_path=neg_path, output_dir=tmp_path / "subset",
                             name="valid", count=6, seed=42, expected_negatives=4, chunk_rows=3)


def test_preparation_rejects_multiple_positives_for_citation2_source(tmp_path):
    pos_path, neg_path, positive, _ = _fixtures(tmp_path)
    np.save(pos_path, np.concatenate([positive, positive[:1]]))
    with pytest.raises(ValueError, match="exactly one positive"):
        freeze_source_subset(pos_path=pos_path, neg_path=neg_path, output_dir=tmp_path / "subset",
                             name="valid", count=3, seed=42, expected_negatives=4)
